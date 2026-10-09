# ==========================================
# V2 비교 평가의 합성 입력 검증
# - 실제 Test·저장 모델은 사용하지 않음
# - 경로 이동과 데이터 정의 변경을 구분하고 평가 행 변경을 차단
# ==========================================
from copy import deepcopy
import unittest
import pandas as pd
from src.sensor_ml.evaluation.evaluate_v2_test import dataset_contract, validate_v1_rows
from src.split_contract import SOURCE_ROW_ID


class V2TestComparisonTest(unittest.TestCase):
    def test_relocation(self):
        """경로만 달라지면 허용하고 기준 변경은 구별하며 원본을 보존한다."""
        old = {"input_path": "old/data", "missing_ratio_threshold": 0.5,
               "ingestion": {"sources": [{"path": "old/raw"}]}}
        moved = deepcopy(old)
        moved["input_path"] = "new/data"
        moved["ingestion"]["sources"][0]["path"] = "new/raw"
        self.assertEqual(dataset_contract(old), dataset_contract(moved))
        self.assertEqual(old["input_path"], "old/data")
        moved["missing_ratio_threshold"] = 0.4
        self.assertNotEqual(dataset_contract(old), dataset_contract(moved))

    def test_rows(self):
        """순서 변경만 허용하고 행 누락·중복·정답 변경을 거부한다."""
        test = pd.DataFrame({SOURCE_ROW_ID: [10, 11], "label": [-1, 1]})
        previous = pd.DataFrame({"source_row_id": [11, 10], "label": [1, -1]})
        validate_v1_rows(test, previous)
        for changed in (test.iloc[:1], test.assign(label=[1, 1]),
                        test.assign(**{SOURCE_ROW_ID: [10, 10]})):
            with self.assertRaises(ValueError):
                validate_v1_rows(changed, previous)


if __name__ == "__main__":
    unittest.main()
