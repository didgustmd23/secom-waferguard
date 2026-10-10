"""범주형 맵 변환과 개발 구간 패턴 검사를 합성 데이터로 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from src.wafer_dl import check_transform
from src.wafer_dl.split import run_split
from src.wafer_dl.transform import (
    load_transform_config, make_channels, preservation_metrics, resize_map,
)
from src.wafer_dl.validate_canonical import ValidationError
from tests import test_wafer_split as fixtures


# ==========================================
# 범주·좌표·여백·채널 계약 검사
# - 실제 데이터나 모델을 읽지 않고 작은 배열로 정답을 직접 비교
# - nearest와 픽셀 중심 최근접 방식의 차이가 드러나는 축소 사례 포함
# ==========================================
class WaferTransformTest(unittest.TestCase):
    def test_identity_and_no_mutation(self):
        """크기가 같으면 상태와 위치가 같고 원본 배열을 수정하지 않는다."""
        original = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        before = original.copy()
        output, geometry = resize_map(original, (2, 3))
        np.testing.assert_array_equal(output, original)
        np.testing.assert_array_equal(original, before)
        self.assertFalse(np.shares_memory(output, original))
        self.assertEqual(geometry['scale'], 1)

    def test_half_pixel_coordinates(self):
        """4→2 축소에서 원본 1·3 위치를 선택하고 보간된 범주값을 만들지 않는다."""
        original = np.array([[0, 1, 2, 0]], dtype=np.uint8)
        output, geometry = resize_map(original, (1, 2))
        np.testing.assert_array_equal(output, [[1, 0]])
        self.assertEqual(geometry['resized_width'], 2)
        enlarged, _ = resize_map(np.array([[1, 2]], dtype=np.uint8), (2, 4))
        np.testing.assert_array_equal(enlarged, [[1, 1, 2, 2], [1, 1, 2, 2]])

    def test_rounding_and_asymmetric_padding(self):
        """반올림을 명시하고 홀수 여백의 추가 픽셀은 아래·오른쪽에 둔다."""
        output, geometry = resize_map(np.ones((2, 4), dtype=np.uint8), (5, 5))
        self.assertEqual((geometry['resized_height'], geometry['resized_width']), (3, 5))
        self.assertEqual((geometry['top'], geometry['bottom']), (1, 1))
        output, geometry = resize_map(np.ones((2, 4), dtype=np.uint8), (4, 4))
        self.assertEqual((geometry['top'], geometry['bottom']), (1, 1))
        output, geometry = resize_map(np.ones((1, 4), dtype=np.uint8), (4, 4))
        self.assertEqual((geometry['top'], geometry['bottom']), (1, 2))
        self.assertTrue((output[0] == 0).all() and (output[2:] == 0).all())

    def test_channels_and_preservation(self):
        """외부·정상·불량의 채널값과 불량·유효 영역 소실을 검출한다."""
        array = np.array([[0, 1, 2]], dtype=np.uint8)
        channels = make_channels(array)
        np.testing.assert_array_equal(channels, [[[0, 0, 1]], [[0, 1, 1]]])
        self.assertEqual(channels.dtype, np.float32)
        self.assertTrue(channels.flags.c_contiguous)
        self.assertTrue((channels[0] <= channels[1]).all())
        original = np.ones((3, 3), dtype=np.uint8)
        original[0, 0] = 2
        output, _ = resize_map(original, (1, 1))
        metrics = preservation_metrics(original, output)
        self.assertTrue(metrics['failed_region_lost'])
        self.assertFalse(metrics['valid_region_lost'])
        original = np.zeros((3, 3), dtype=np.uint8)
        original[0, 0] = 1
        output, _ = resize_map(original, (1, 1))
        metrics = preservation_metrics(original, output)
        self.assertTrue(metrics['valid_region_lost'])
        self.assertIsNone(metrics['absolute_ratio_delta'])

    def test_invalid_maps_and_sizes(self):
        """dtype·차원·상태·빈 영역과 잘못된 목표 크기를 거부한다."""
        for array in (np.ones((2, 2)), np.ones((2, 2, 2), dtype=np.uint8),
                      np.zeros((2, 2), dtype=np.uint8), np.array([[3]], dtype=np.uint8)):
            with self.assertRaises(ValidationError):
                resize_map(array)
        for size in ((True, 128), (0, 128), (2048, 128), (128,)):
            with self.assertRaises(ValidationError):
                resize_map(np.ones((2, 2), dtype=np.uint8), size)

    def test_config_is_explicit(self):
        """명시된 NumPy 좌표 규칙만 허용하며 지원하지 않는 보간으로 대체하지 않는다."""
        path = Path(__file__).resolve().parents[1] / 'configs/wm811k/transform.json'
        config = load_transform_config(path)
        self.assertEqual(config['target_size'], [128, 128])
        with tempfile.TemporaryDirectory() as temporary:
            for changes in ({'extra': 1}, {'resize_backend': 'nearest'},
                            {'target_size': [True, 128]}, {'padding_value': False}):
                invalid = Path(temporary) / 'bad.json'
                invalid.write_text(json.dumps(dict(config, **changes)), encoding='utf-8')
                with self.assertRaises(ValidationError):
                    load_transform_config(invalid)


# ==========================================
# 파생 분할과 실제 읽기 범위의 통합 검사
# - 기존 합성 canonical 생성 도우미만 재사용
# - Test NPY 구성원을 읽으면 실패하도록 경계를 감시
# - 전체 소실이 검토 필요로 기록되고 원본/기존 결과는 유지되는지 검사
# ==========================================
class TransformCheckTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WaferSplitTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.create_bundle()
        self.split_dir = self.fixture.base / 'split'
        run_split(self.fixture.root, self.split_dir, self.fixture.config_path, approved=True)
        self.config_path = Path(__file__).resolve().parents[1] / 'configs/wm811k/transform.json'

    def test_full_check_without_test_arrays(self):
        """개발 구간만 복원하고 결과·그림·최소 반환값과 원본 보존을 확인한다."""
        records = [json.loads(line) for line in (self.split_dir / 'split_manifest.jsonl').read_text().splitlines()]
        allowed = {row['map_ref']['key'] + '.npy' for row in records if row['split'] in ('train', 'validation')}
        read_numeric = check_transform.read_numeric_member
        opened = []

        def guarded_read(archive, member):
            self.assertIn(member.filename, allowed)
            opened.append(member.filename)
            return read_numeric(archive, member)

        before = (self.fixture.root / 'manifest.jsonl').read_bytes()
        output = self.fixture.base / 'check'
        with patch.object(check_transform, 'read_numeric_member', side_effect=guarded_read):
            result = check_transform.run_transform_check(self.split_dir, output, self.config_path)
        self.assertEqual(result, {'status': 'transform_check_passed', 'training_ready': False})
        self.assertEqual(set(opened), allowed)
        report = json.loads((output / 'transform_check.json').read_text(encoding='utf-8'))
        self.assertEqual(report['rows_checked'], len(allowed))
        self.assertFalse(report['test_maps_read'])
        self.assertTrue(report['visual_review_required'])
        for name in ('preservation.csv', 'examples.png', 'transform_check.md', 'SUCCESS.json'):
            self.assertTrue((output / name).is_file())
        self.assertEqual(before, (self.fixture.root / 'manifest.jsonl').read_bytes())
        with self.assertRaisesRegex(ValidationError, 'INVALID_TRANSFORM_OUTPUT_DIRECTORY'):
            check_transform.run_transform_check(self.split_dir, output, self.config_path)

    def test_loss_requires_review(self):
        """작은 불량이 축소 중 사라지면 통과로 표시하지 않고 자동 보정하지 않는다."""
        config = load_transform_config(self.config_path)
        config['target_size'] = [1, 1]
        path = self.fixture.base / 'small.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        output = self.fixture.base / 'review'
        with patch.object(check_transform, 'save_examples'):
            result = check_transform.run_transform_check(self.split_dir, output, path)
        self.assertEqual(result['status'], 'transform_review_required')
        self.assertTrue(json.loads((output / 'SUCCESS.json').read_text())['review_required'])

    def test_tampered_split_blocks_completion(self):
        """분할 manifest에서 원본 라벨을 변경하면 배열 검사·완료 전에 차단한다."""
        path = self.split_dir / 'split_manifest.jsonl'
        lines = path.read_text().splitlines()
        row = json.loads(lines[0])
        row['class_index'] = 8
        lines[0] = json.dumps(row)
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        output = self.fixture.base / 'bad_check'
        with self.assertRaisesRegex(ValidationError, 'SPLIT_SOURCE_ROW_CHANGED'):
            check_transform.run_transform_check(self.split_dir, output, self.config_path)
        self.assertFalse((output / 'SUCCESS.json').exists())


if __name__ == '__main__':
    unittest.main()
