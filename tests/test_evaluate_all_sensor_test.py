# ==========================================
# 전체 센서 동결 평가의 합성 검증
# - 실제 저장 모델이나 Test 데이터를 사용하지 않음
# - 두 문턱에서 확률은 한 번만 계산하고 재학습하지 않음을 확인
# ==========================================
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.sensor_ml.evaluation.evaluate_all_sensor_test import evaluate_scenarios, validate_pipeline


class EvaluateAllSensorTest(unittest.TestCase):
    def test_shared_scores_and_frozen_thresholds(self):
        """한 번의 확률 계산으로 두 동결 문턱을 적용하고 fit은 호출하지 않는다."""
        pipeline = Mock()
        pipeline.fit.side_effect = AssertionError("평가 중 학습 금지")
        labels = pd.Series([-1, -1, 1, 1])
        scenarios = pd.DataFrame({"target_recall": [0.8, 0.9], "threshold": [0.5, 0.1]})
        scores = np.array([0.05, 0.2, 0.4, 0.9])
        with patch("src.sensor_ml.evaluation.evaluate_all_sensor_test.positive_scores", return_value=scores) as predict:
            results = evaluate_scenarios(pipeline, pd.DataFrame({"sensor_1": range(4)}),
                                         labels, MODELING_CONFIG, scenarios)
        predict.assert_called_once()
        pipeline.fit.assert_not_called()
        self.assertEqual(results[0][1]["false_negative"], 1)
        self.assertEqual(results[1][1]["false_negative"], 0)
        self.assertEqual(results[1][1]["false_positive"], 1)
        self.assertEqual([result[1]["threshold"] for result in results], [0.5, 0.1])
        np.testing.assert_array_equal(results[0][2].positive_score, results[1][2].positive_score)
        self.assertFalse(results[0][1]["recall_target_met"])

    def test_restored_pipeline_contract(self):
        """복원 센서·분류기·전처리 설정 불일치를 거부한다."""
        quality = SimpleNamespace(retained_features_=("sensor_1",), missing_ratio_threshold=0.5,
                                  drop_zero_variance=True)
        classifier = SimpleNamespace(get_params=lambda: {"max_depth": 2})
        pipeline = SimpleNamespace(named_steps={"quality_filter": quality,
                                   "imputer": SimpleNamespace(strategy="median"), "model": classifier})
        record = {"retained_sensors": ["sensor_1"], "model_parameters": {"max_depth": 2},
                  "config": {"dataset": {"missing_ratio_threshold": 0.5, "drop_zero_variance": True}}}
        validate_pipeline(pipeline, record)
        record["model_parameters"]["max_depth"] = 3
        with self.assertRaisesRegex(ValueError, "분류기 설정"):
            validate_pipeline(pipeline, record)
        record["model_parameters"]["max_depth"] = 2
        quality.retained_features_ = ("sensor_other",)
        with self.assertRaisesRegex(ValueError, "유지 센서"):
            validate_pipeline(pipeline, record)


if __name__ == "__main__":
    unittest.main()
