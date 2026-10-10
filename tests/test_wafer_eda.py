"""원본 없이 합성 canonical 파일로 공통 EDA·노트북 계약 검사."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from scripts.wm_sandbox.export_canonical import export_frame
from src.wafer_dl.eda import (
    load_eda_metadata, summarize_metadata, load_example_maps,
    select_example_positions, run_eda, summarize_correlations,
)
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


# ==========================================
# 에이전트와 노트북이 공유하는 EDA 코어 검증
# - 실제 원본·사용자 NPZ·저장 모델·외부 API는 사용하지 않음
# - 집계·예시 로드·로컬 보고서·최소 반환 상태를 확인
# ==========================================
class WaferEdaTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'canonical'
        rows = []
        for index, name in enumerate(CLASS_NAMES):
            array = np.ones((2, 3), dtype=np.uint8)
            for bit in range(4):
                array.flat[bit] = 2 if index & (1 << bit) else 1
            rows.append({'waferMap': array, 'failureType': name, 'lotName': f'private_lot_{index}'})
        for label in ([], 'unknown'):
            rows.append({'waferMap': np.ones((2, 3), dtype=np.uint8),
                         'failureType': label, 'lotName': 'private_extra_lot'})
        export_frame(pd.DataFrame(rows), self.root, 'wm811k_' + 'a' * 64, shard_rows=5)
        (self.root / 'SUCCESS.json').write_text(json.dumps(
            {'status': 'canonical_export_complete', 'training_ready': False}), encoding='utf-8')
        (self.root / 'provenance.json').write_text(json.dumps(
            {'source_sha256': 'a' * 64, 'source_bytes': 100}), encoding='utf-8')

    def test_summary_and_no_private_keys(self):
        """미라벨·격리와 9개 클래스를 구분하며 집계에 원본 ID를 남기지 않는다."""
        frame = load_eda_metadata(self.root)
        tables = summarize_metadata(frame)
        self.assertEqual(tables['statistics']['row_count'], 11)
        self.assertEqual(tables['classes']['accepted_labeled_samples'].tolist(), [1] * 9)
        self.assertEqual(tables['statuses'].loc[('unlabeled', 'accepted'), 'rows'], 1)
        self.assertEqual(tables['statuses'].loc[('invalid', 'quarantined'), 'rows'], 1)
        self.assertNotIn('private_', str(tables))

    def test_numeric_examples_and_reference_guard(self):
        """9개 예시를 숫자로 읽고 폴더 밖 참조는 거부한다."""
        frame = load_eda_metadata(self.root)
        positions = select_example_positions(frame)
        examples = load_example_maps(self.root, positions)
        self.assertEqual(len(examples), 9)
        self.assertTrue(all(array.dtype == np.uint8 and array.shape == (2, 3) for _, array in examples))
        manifest = self.root / 'manifest.jsonl'
        lines = manifest.read_text().splitlines()
        row = json.loads(lines[0])
        row['map_ref']['shard'] = '../outside.npz'
        lines[0] = json.dumps(row)
        manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValidationError, 'INVALID_MAP_REFERENCE'):
            load_example_maps(self.root, [0])

    def test_shared_runner_and_minimal_result(self):
        """함수 호출만으로 보고서를 생성하고 상세 데이터 없는 상태만 반환한다."""
        output = Path(self.temporary.name) / 'eda'
        before = (self.root / 'manifest.jsonl').read_bytes()
        self.assertEqual(run_eda(self.root, output), {'status': 'eda_complete'})
        for name in ('distributions.png', 'die_statistics.png', 'examples.png',
                     'correlations.png', 'correlation_scatter.png', 'correlation_pearson.csv',
                     'correlation_spearman.csv', 'correlation_by_class.csv', 'correlation_support.csv',
                     'classes.csv', 'statuses.csv', 'shapes.csv', 'eda.json', 'eda.md', 'SUCCESS.json'):
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual(before, (self.root / 'manifest.jsonl').read_bytes())
        with self.assertRaisesRegex(ValidationError, 'INVALID_REPORT_DIRECTORY'):
            run_eda(self.root, output)

    def test_old_converter_rejected(self):
        """이전 라벨 변환 묶음을 조용히 EDA 대상으로 채택하지 않는다."""
        path = self.root / 'conversion.json'
        summary = json.loads(path.read_text())
        summary.pop('label_normalization_version')
        path.write_text(json.dumps(summary), encoding='utf-8')
        with self.assertRaisesRegex(ValidationError, 'OUTDATED_LABEL_NORMALIZATION'):
            load_eda_metadata(self.root)

    def test_correlations_and_eligibility(self):
        """단조 관계·상수·결측·격리 제외와 클래스별 표본 수를 합성 값으로 확인한다."""
        frame = pd.DataFrame({
            'map_hash': ['synthetic'] * 6, 'quality_status': ['accepted'] * 5 + ['quarantined'],
            'pattern_label': ['Center'] * 6, 'map_height': [2] * 6,
            'map_width': [1, 2, 3, 4, np.nan, 100],
            'valid_die_count': [1, 2, 3, 4, 5, 1],
            'failed_die_count': [1, 4, 9, 16, 25, 100],
            'failed_die_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 1],
            'class_index': [1] * 6, 'lot_id': ['private'] * 6,
        })
        tables = summarize_correlations(frame)
        self.assertEqual(tables['support'].iloc[0, 0], 4)
        self.assertAlmostEqual(tables['spearman'].loc['valid_die_count', 'failed_die_count'], 1)
        self.assertLess(tables['pearson'].loc['valid_die_count', 'failed_die_count'], 1)
        self.assertTrue(tables['pearson']['map_height'].isna().all())
        self.assertNotIn('class_index', tables['pearson'].columns)
        self.assertNotIn('private', str(tables))
        self.assertTrue(tables['by_class'].loc[tables['by_class']['pattern_label'].eq('Center'), 'rows'].eq(4).all())
        self.assertTrue(tables['by_class'].loc[tables['by_class']['pattern_label'].eq('Donut'), 'correlation'].isna().all())

    def test_notebook_is_valid_python(self):
        """사용자가 실행한 셀·출력을 보존하고 셀 문법과 공통 상관분석 호출을 검사한다."""
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'reports/wm811k/analysis.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        correlation_found = False
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                if 'summarize_correlations(metadata)' in ''.join(cell['source']):
                    correlation_found = True
        self.assertTrue(correlation_found)


if __name__ == '__main__':
    unittest.main()
