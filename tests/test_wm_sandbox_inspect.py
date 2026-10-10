"""독립 합성 입력만 사용해 최초 구조 검사를 확인한다. 실제 pickle은 읽지 않는다."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.wm_sandbox.inspect_pickle import file_digest, inspect_frame, within_root


# ==========================================
# 최초 검사 스크립트의 정상·예외 입력 검증
# - 실제 원본·Sandbox·네트워크·역직렬화는 사용하지 않음
# - 이 테스트 통과가 OS 격리나 원본 안전성 증명은 아님
# ==========================================
class SandboxInspectionTest(unittest.TestCase):
    def make_frame(self, maps=None):
        # 원본에서 가져오지 않은 작은 배열과 가상 라벨을 만든다.
        if maps is None:
            maps = [np.array([[0, 1], [2, 1]], dtype=np.uint8)]
        return pd.DataFrame({"waferMap": maps,
                             "failureType": [np.array([["None"]])] * len(maps),
                             "lotName": ["SYNTHETIC_CANARY"] * len(maps)})

    def test_valid_sample_and_no_raw_export(self):
        # 정상 구조 집계는 만들지만 샘플·Lot 식별값을 결과에 복사하지 않는다.
        result = inspect_frame(self.make_frame())
        self.assertEqual(result["status"], "inspection_complete")
        self.assertEqual(result["sample_map_status"], {"valid_structure": 1})
        self.assertFalse(result["full_quality_audit"])
        self.assertFalse(result["raw_samples_exported"])
        self.assertNotIn("SYNTHETIC_CANARY", str(result))

    def test_missing_and_duplicate_columns(self):
        # 필수 필드 누락은 차단 상태이며 중복 컬럼은 해석하지 않고 거부한다.
        missing = inspect_frame(self.make_frame().drop(columns="lotName"))
        self.assertEqual(missing["status"], "blocked_schema")
        frame = self.make_frame()
        frame.columns = ["waferMap", "waferMap", "lotName"]
        with self.assertRaises(ValueError):
            inspect_frame(frame)

    def test_invalid_maps_are_counted_without_fix(self):
        # 비정상 값·빈 유효 영역·객체 dtype·차원 오류를 구분한다.
        maps = [np.array([[3]]), np.zeros((2, 2)),
                np.array([["upload"]], dtype=object), np.array([1, 2])]
        result = inspect_frame(self.make_frame(maps))
        self.assertEqual(sum(result["sample_map_status"].values()), 4)
        self.assertNotIn("valid_structure", result["sample_map_status"])
        self.assertEqual(maps[0][0, 0], 3)

    def test_sample_limit_and_types(self):
        # 일부 행 검사임을 명시하고 비정상 한도를 자동 보정하지 않는다.
        result = inspect_frame(self.make_frame([np.ones((1, 1))] * 3), limit=1)
        self.assertEqual(result["sampled_rows"], 1)
        self.assertEqual(result["row_count"], 3)
        for limit in (0, 1001, True, 1.5):
            with self.assertRaises(ValueError):
                inspect_frame(self.make_frame(), limit=limit)
        with self.assertRaises(ValueError):
            inspect_frame({"waferMap": []})

    def test_local_path_boundary_and_digest(self):
        # 테스트 임시 폴더만 사용하며 원본 경로를 열지 않는다.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "allowed"
            allowed.mkdir()
            inside = allowed / "sample.bin"
            inside.write_bytes(b"synthetic fixture")
            outside = root / "outside.bin"
            outside.write_bytes(b"other fixture")
            self.assertEqual(within_root(inside, allowed), inside.resolve())
            with self.assertRaises(ValueError):
                within_root(outside, allowed)
            self.assertEqual(file_digest(inside), file_digest(inside))
            self.assertNotEqual(file_digest(inside), file_digest(outside))


if __name__ == "__main__":
    unittest.main()
