"""합성 데이터로 충돌 제외·그룹 분할·원본 보존 계약을 검증한다."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from scripts.wm_sandbox.export_canonical import export_frame
from src.wafer_dl.split import SPLITS, build_groups, choose_assignment, load_split_config, run_split
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


# ==========================================
# 실제 pickle·사용자 데이터·외부 API 없이 분할 검증
# - 클래스마다 독립적인 합성 맵과 Lot을 생성
# - 충돌 격리·완료 표식·미라벨의 평가 구간 보호를 확인
# ==========================================
class WaferSplitTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'canonical'
        self.config_path = Path(__file__).resolve().parents[1] / 'configs/wm811k/split.json'
        self.config = load_split_config(self.config_path)

    def create_bundle(self):
        # 비트 패턴으로 맵을 서로 다르게 만들어 의도하지 않은 중복 충돌을 피한다.
        rows = []
        for label in range(9):
            for number in range(6):
                position = len(rows)
                array = np.ones((3, 3), dtype=np.uint8)
                for bit in range(6):
                    array.flat[bit] = 2 if position & (1 << bit) else 1
                rows.append({'waferMap': array, 'failureType': CLASS_NAMES[label],
                             'lotName': f'lot_{position}'})
        # 충돌 맵의 라벨·미라벨 행 전체가 함께 격리되어야 한다.
        conflict = np.full((4, 4), 2, dtype=np.uint8)
        for label in ('Center', 'Donut', []):
            rows.append({'waferMap': conflict.copy(), 'failureType': label,
                         'lotName': f'conflict_lot_{len(rows)}'})
        # 별도 Lot의 미라벨 행도 동일 맵의 그룹 배정을 보존해야 한다.
        rows.append({'waferMap': rows[0]['waferMap'].copy(), 'failureType': [], 'lotName': 'extra_lot'})
        export_frame(pd.DataFrame(rows), self.root, 'wm811k_' + 'a' * 64, shard_rows=20)
        (self.root / 'SUCCESS.json').write_text(json.dumps(
            {'status': 'canonical_export_complete', 'training_ready': False}), encoding='utf-8')
        (self.root / 'provenance.json').write_text(json.dumps(
            {'source_sha256': 'a' * 64, 'source_bytes': 100}), encoding='utf-8')

    def test_config_rejects_invalid_values(self):
        """알 수 없는 키·비율 오류·bool seed·후보 수 초과를 거부한다."""
        for changes in ({'extra': 1}, {'ratios': [0.7, 0.2, 0.2]},
                        {'seed': True}, {'max_candidates': 33}, {'max_group_ratio': float('nan')}):
            path = self.base / 'bad.json'
            path.write_text(json.dumps(dict(self.config, **changes)), encoding='utf-8')
            with self.assertRaises(ValidationError):
                load_split_config(path)

    def test_candidate_determinism_and_coverage(self):
        """동일 설정은 동일 배정이며 최소 3그룹에서도 각 구간의 클래스를 보존한다."""
        counts = np.repeat(np.eye(9, dtype=np.int64), 3, axis=0)
        first, candidates = choose_assignment(counts, self.config)
        second, _ = choose_assignment(counts, self.config)
        self.assertEqual(first[:2], second[:2])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertTrue((first[3] > 0).all())
        self.assertEqual(len(candidates), 32)
        with self.assertRaisesRegex(ValidationError, 'INSUFFICIENT_CLASS_GROUPS'):
            choose_assignment(np.eye(9, dtype=np.int64), self.config)

    def test_unlabeled_bridge_groups(self):
        """미라벨 Lot을 거친 연결도 같은 그룹으로 묶는다."""
        database = sqlite3.connect(':memory:')
        self.addCleanup(database.close)
        database.execute('CREATE TABLE rows(position INTEGER, lot TEXT, wafer TEXT, hash TEXT, label INTEGER, accepted INTEGER)')
        database.executemany('INSERT INTO rows VALUES(?,?,?,?,?,?)', [
            (0, 'A', None, 'map1', 0, 1), (1, 'bridge', None, 'map1', None, 1),
            (2, 'bridge', None, 'map2', None, 1), (3, 'B', None, 'map2', 1, 1)])
        counts, _, _, _ = build_groups(database)
        self.assertEqual(counts.shape, (1, 9))
        self.assertEqual(counts[0, :2].tolist(), [1, 1])

    def test_impossible_coverage_is_not_forced(self):
        """사전 그룹 수를 만족해도 실제 지원 조건이 불가능하면 자동 재분할하지 않는다."""
        # 어느 세 그룹도 세 구간을 덮어야 하는 모순된 지원 구성이다.
        counts = np.ones((4, 9), dtype=np.int64)
        counts[:, :4] -= np.eye(4, dtype=np.int64)
        with self.assertRaisesRegex(ValidationError, 'NO_FEASIBLE_SPLIT_CANDIDATE'):
            choose_assignment(counts, self.config)

    def test_full_export_preserves_and_protects(self):
        """전체 행·충돌 격리·클래스 지원·Lot/맵 비중첩과 원본 보존을 확인한다."""
        self.create_bundle()
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        output = self.base / 'split'
        self.assertEqual(run_split(self.root, output, self.config_path, approved=True),
                         {'status': 'split_complete', 'training_ready': False})
        records = [json.loads(line) for line in (output / 'split_manifest.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 58)
        self.assertTrue(all(row['split'] == 'quarantined' and row['exclusion_reasons'] == ['MAP_LABEL_CONFLICT'] for row in records[54:57]))
        self.assertEqual(records[-1]['split'], 'unlabeled')
        self.assertEqual(records[-1]['group_split'], records[0]['group_split'])
        self.assertEqual(records[-1]['split_group_id'], records[0]['split_group_id'])
        for field in ('lot_id', 'map_hash'):
            membership = {}
            for row in records:
                if row['group_split'] is not None:
                    membership.setdefault(row[field], set()).add(row['group_split'])
            self.assertTrue(all(len(values) == 1 for values in membership.values()))
        for split in SPLITS:
            self.assertEqual({row['class_index'] for row in records if row['split'] == split}, set(range(9)))
        report = json.loads((output / 'split.json').read_text(encoding='utf-8'))
        self.assertEqual(report['excluded_rows'], 3)
        self.assertEqual(sum(report['split_counts'].values()), 54)
        self.assertTrue((output / 'SUCCESS.json').is_file())
        self.assertFalse(report['training_ready'])
        self.assertTrue((output / 'split_summary.csv').is_file())
        # 새 출력 경로에서도 같은 원본·설정은 같은 프로토콜·배정을 만든다.
        repeated = self.base / 'split_repeated'
        run_split(self.root, repeated, self.config_path, approved=True)
        self.assertEqual((output / 'split_manifest.jsonl').read_bytes(),
                         (repeated / 'split_manifest.jsonl').read_bytes())
        for path, content in before.items():
            self.assertEqual((self.root / path).read_bytes(), content)
        with self.assertRaisesRegex(ValidationError, 'INVALID_SPLIT_OUTPUT_DIRECTORY'):
            run_split(self.root, output, self.config_path, approved=True)

    def test_approval_and_corruption_blocks_completion(self):
        """승인이 없거나 배열이 손상되면 완료 폴더를 만들지 않는다."""
        with self.assertRaisesRegex(ValidationError, 'SPLIT_POLICY_APPROVAL_REQUIRED'):
            run_split(self.root, self.base / 'split', self.config_path)
        self.create_bundle()
        shard = next((self.root / 'maps').glob('*.npz'))
        shard.write_bytes(b'corrupt')
        with self.assertRaises(zipfile.BadZipFile):
            run_split(self.root, self.base / 'split', self.config_path, approved=True)
        self.assertFalse((self.base / 'split/SUCCESS.json').exists())


    def test_large_group_blocks_completion(self):
        """거대 그룹 검토 상한을 넘으면 완료 표식을 만들지 않는다."""
        self.create_bundle()
        path = self.base / 'strict.json'
        path.write_text(json.dumps(dict(self.config, max_group_ratio=0.001)), encoding='utf-8')
        with self.assertRaisesRegex(ValidationError, 'SPLIT_PRECONDITIONS_NOT_PASSED'):
            run_split(self.root, self.base / 'blocked', path, approved=True)
        self.assertFalse((self.base / 'blocked/SUCCESS.json').exists())


if __name__ == '__main__':
    unittest.main()
