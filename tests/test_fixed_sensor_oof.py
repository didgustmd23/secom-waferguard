# ==========================================
# 고정 센서 선택 이후 OOF의 시간·출처·문턱 분리 검증
# - 실제 데이터 대신 합성 데이터와 트리 3개 사용
# - 선택 행과 미래 평가 행은 OOF 정답에 포함하지 않음
# - 미래 label 변경이 OOF와 선택 문턱에 영향을 주지 않음
# ==========================================
import unittest
from unittest.mock import patch
from pathlib import Path
import json
from dataclasses import asdict

import pandas as pd

from src.sensor_ml.experiments.fixed_sensor_oof import compare_fixed_oof, validate_candidate
from src.sensor_ml.experiments.fixed_sensor_compare import _json_value
from src.modeling_config import MODELING_CONFIG
from src.split_contract import temporal_folds, SOURCE_ROW_ID
from tests.test_fixed_sensor_compare import synthetic_frame


class FixedSensorOofTest(unittest.TestCase):
    def test_oof_scope_and_future_isolation(self):
        """선택 이후 행만 예측하고 미래 정답과 문턱 선정을 분리한다."""
        frame = synthetic_frame()
        names = [f"sensor_{i}" for i in range(20)]
        folds = list(temporal_folds(frame, MODELING_CONFIG.dataset, 3))
        result = compare_fixed_oof(frame, MODELING_CONFIG, names, n_estimators=3)
        selection_ids = set(frame.iloc[folds[1][0]][SOURCE_ROW_ID])
        future_ids = set(frame.iloc[folds[2][1]][SOURCE_ROW_ID])
        ids = set(result["oof_predictions"].source_row_id)
        self.assertFalse(ids & selection_ids)
        self.assertFalse(ids & future_ids)
        self.assertEqual(ids, set(frame.iloc[folds[2][0]][SOURCE_ROW_ID]) - selection_ids)
        blocks = result["inner_folds"]
        self.assertTrue((pd.to_datetime(blocks.train_end) < pd.to_datetime(blocks.evaluation_start)).all())
        changed = frame.copy()
        changed.loc[folds[2][1], "label"] *= -1
        other = compare_fixed_oof(changed, MODELING_CONFIG, names, n_estimators=3)
        pd.testing.assert_frame_equal(result["threshold_compare"], other["threshold_compare"])
        pd.testing.assert_frame_equal(result["oof_predictions"], other["oof_predictions"])
        pd.testing.assert_series_equal(result["scenario_results"].threshold, other["scenario_results"].threshold)

    def test_candidate_contract_and_invalid_inputs(self):
        """후보 설정·선택 출처와 목록 오류를 학습 전에 검증한다."""
        frame = synthetic_frame()
        names = [f"sensor_{i}" for i in range(20)]
        folds = list(temporal_folds(frame, MODELING_CONFIG.dataset, 3))
        record = {"config": json.loads(json.dumps(asdict(MODELING_CONFIG), default=_json_value)),
                  "selection_fold": 2, "n_splits": 3, "classifier_max_depth": 2,
                  "classifier_reg_lambda": 1, "class_weight_mode": "ratio"}
        tables = [pd.DataFrame({"sensor_order": range(20), "feature": names}),
                  frame.iloc[folds[1][0]][[SOURCE_ROW_ID]].rename(columns={SOURCE_ROW_ID: "source_row_id"})]
        with patch.object(Path, "read_text", return_value=json.dumps(record)), \
             patch("src.sensor_ml.experiments.fixed_sensor_oof.pd.read_csv", side_effect=tables):
            loaded, _ = validate_candidate(frame, MODELING_CONFIG, Path("candidate"))
            self.assertEqual(loaded, names)
        record["classifier_reg_lambda"] = 5
        with patch.object(Path, "read_text", return_value=json.dumps(record)):
            with self.assertRaisesRegex(ValueError, "분류기 설정"):
                validate_candidate(frame, MODELING_CONFIG, Path("candidate"))
        for bad in (names[:-1], names[:-1] + [names[0]], names[:-1] + ["unknown"]):
            with self.assertRaises(ValueError):
                compare_fixed_oof(frame, MODELING_CONFIG, bad, n_estimators=3)
        with self.assertRaises(ValueError):
            compare_fixed_oof(frame, MODELING_CONFIG, names, inner_splits=1, n_estimators=3)


if __name__ == "__main__":
    unittest.main()
