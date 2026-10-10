"""합성 SQLite 행으로 충돌 그룹 제외 시나리오를 검증한다."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.wafer_dl.diagnose_split import compare_conflict_exclusion, write_diagnostic_report


# ==========================================
# 실제 데이터 없이 가상 제외의 집계·복원·보고서 검사
# - 9개 클래스에 독립 그룹 3개씩 있는 합성 기준선 사용
# - 충돌 맵 전체 제외와 미라벨 동반 제외를 확인
# ==========================================
class SplitDiagnosisTest(unittest.TestCase):
    def setUp(self):
        self.database = sqlite3.connect(':memory:')
        self.addCleanup(self.database.close)
        self.database.execute('CREATE TABLE rows (position INTEGER, lot TEXT, wafer TEXT, '
                              'hash TEXT, label INTEGER, accepted INTEGER)')
        for index in range(27):
            self.database.execute('INSERT INTO rows VALUES (?, ?, ?, ?, ?, 1)',
                                  (index, f'private_lot_{index}', '1', f'private_hash_{index}', index // 3))

    def test_no_conflicts(self):
        """충돌이 없으면 전후 표본·그룹과 상태가 동일하다."""
        result = compare_conflict_exclusion(self.database)
        self.assertEqual(result['candidate_excluded_manifest_rows'], 0)
        self.assertEqual(result['baseline'], result['hypothetical_exclusion'])
        self.assertEqual(result['baseline']['split_assessment']['status'], 'preconditions_passed')
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['arrays_revalidated'])

    def test_whole_hash_exclusion_and_restore(self):
        """충돌 해시의 라벨·미라벨 행을 함께 제외하고 원래 임시 테이블을 복원한다."""
        self.database.execute('UPDATE rows SET hash=? WHERE position=3', ('private_hash_0',))
        self.database.execute('INSERT INTO rows VALUES (27, ?, ?, ?, NULL, 1)',
                              ('private_unlabeled_lot', '1', 'private_hash_0'))
        result = compare_conflict_exclusion(self.database)
        self.assertEqual(result['candidate_conflict_hash_groups'], 1)
        self.assertEqual(result['candidate_excluded_manifest_rows'], 3)
        self.assertEqual(result['hypothetical_exclusion']['duplicate_audit']['map_label_conflicts'], 0)
        for index in (0, 1):
            row = result['class_comparison'][index]
            self.assertEqual(row['samples_before'], 3)
            self.assertEqual(row['samples_after'], 2)
            self.assertEqual(row['groups_after'], 2)
            self.assertFalse(row['minimum_groups_met_after'])
        self.assertEqual(result['hypothetical_exclusion']['split_assessment']['status'], 'review_required')
        self.assertEqual(self.database.execute('SELECT COUNT(*) FROM rows').fetchone()[0], 28)
        # 같은 연결로 다시 진단해도 임시 테이블 이름 충돌 없이 동일해야 한다.
        self.assertEqual(result, compare_conflict_exclusion(self.database))

    def test_report_contract(self):
        """로컬 표와 JSON을 보존하되 원본 식별자는 보고서에 포함하지 않는다."""
        result = compare_conflict_exclusion(self.database)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'report'
            write_diagnostic_report(result, output)
            text = (output / 'diagnosis.json').read_text(encoding='utf-8')
            self.assertNotIn('private_lot_', text)
            self.assertNotIn('private_hash_', text)
            self.assertFalse(json.loads(text)['data_modified'])
            self.assertTrue((output / 'class_comparison.csv').is_file())
            self.assertIn('None', (output / 'diagnosis.md').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                write_diagnostic_report(result, output)


if __name__ == '__main__':
    unittest.main()
