# ==========================================
# 전체 센서 비교 모델의 출처·OOF·품질 필터 합성 검증
# - 실제 SECOM 학습이나 Test 평가는 하지 않음
# - 저장 V2의 블록 재사용과 미래 정답의 문턱 선택 격리를 확인
# ==========================================
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.sensor_ml.experiments.all_sensor_reference import reference_blocks, prepare_reference
from src.split_contract import SOURCE_ROW_ID, temporal_folds
from tests.test_fixed_sensor_compare import synthetic_frame


class AllSensorReferenceTest(unittest.TestCase):
    def setUp(self):
        """합성 Train과 저장 V2 형식의 OOF 블록을 만든다."""
        self.frame = synthetic_frame()
        folds = list(temporal_folds(self.frame, MODELING_CONFIG.dataset, 3))
        self.fit_i, self.future_i = folds[2]
        initial = list(folds[1][0])
        post = [i for i in self.fit_i if i not in set(initial)]
        half = len(post) // 2
        self.blocks = [(1, initial, post[:half]), (2, initial + post[:half], post[half:])]
        self.oof = self.frame.iloc[post][[SOURCE_ROW_ID, "label"]].rename(
            columns={SOURCE_ROW_ID: "source_row_id"}).reset_index(drop=True)
        self.oof["inner_fold"] = [1] * half + [2] * (len(post) - half)
        self.future = pd.DataFrame({"source_row_id": self.frame.iloc[self.future_i][SOURCE_ROW_ID]})
        self.execution = {
            "config": json.loads(json.dumps(asdict(MODELING_CONFIG), default=str)),
            "oof_scope": "after_selection_only", "inner_splits": 2,
            "candidate_execution": {"selection_fold": 2, "n_splits": 3,
                                    "classifier_max_depth": 2, "classifier_reg_lambda": 1,
                                    "class_weight_mode": "ratio", "n_estimators": 3},
        }

    def read_blocks(self):
        """파일 읽기만 대체하고 원본 행·시간 경계 검사는 실제로 수행한다."""
        with patch.object(Path, "read_text", return_value=json.dumps(self.execution)), patch(
            "src.sensor_ml.experiments.all_sensor_reference.pd.read_csv",
            side_effect=[self.oof.copy(), self.future.copy()],
        ):
            return reference_blocks(self.frame, MODELING_CONFIG, Path("합성기준"))

    def test_reference_rows_and_labels(self):
        """같은 OOF 행은 허용하고 정답 변경은 거부한다."""
        blocks, fit_i, _ = self.read_blocks()
        self.assertEqual(blocks, self.blocks)
        self.assertEqual(list(fit_i), list(self.fit_i))
        self.oof.loc[0, "label"] *= -1
        with self.assertRaisesRegex(ValueError, "OOF 행·정답"):
            self.read_blocks()

    def test_quality_filter_and_future_isolation(self):
        """품질 제거는 학습 범위에서만 계산하고 미래 정답은 OOF에 영향을 주지 않는다."""
        self.frame["sensor_empty"] = float("nan")
        self.frame["sensor_constant"] = 1.0
        contract = (self.blocks, self.fit_i, self.execution)
        with patch("src.sensor_ml.experiments.all_sensor_reference.reference_blocks", return_value=contract):
            first = prepare_reference(self.frame, MODELING_CONFIG, Path("합성기준"), 1)
            changed = self.frame.copy()
            changed.loc[self.future_i, "label"] *= -1
            second = prepare_reference(changed, MODELING_CONFIG, Path("합성기준"), 1)
        pd.testing.assert_frame_equal(first[1], second[1])
        pd.testing.assert_frame_equal(first[3], second[3])
        for quality in first[4]:
            self.assertIn("sensor_empty", quality["high_missing_features"])
            self.assertIn("sensor_constant", quality["constant_features"])
        self.assertEqual(first[0].named_steps["model"].max_depth, 2)
        self.assertEqual(len(first[3]), 2)
        self.assertFalse(set(first[1].source_row_id) & set(self.future.source_row_id))


if __name__ == "__main__":
    unittest.main()
