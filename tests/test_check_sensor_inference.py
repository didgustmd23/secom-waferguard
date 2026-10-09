# ==========================================
# 실제 실행용 정합성 검사기의 합성 데이터 테스트
# - 학습 행 비교를 성능 평가로 기록하지 않는지 확인
# - 모든 선택 센서가 결측인 행의 거부 개수를 보존
# - 확률 불일치가 정상 통과로 기록되지 않는지 확인
# ==========================================

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.sensor_ml.verification.check_sensor_inference import check_inference_paths, compare_prediction_paths
from src.modeling_preprocessing import SensorQualityFilter
from src.sensor_inference import build_sensor_inference


class CheckSensorInferenceTest(unittest.TestCase):
    def setUp(self):
        """프로젝트 파일을 읽지 않고 세 센서 중 두 개를 선택하는 모델을 학습한다."""
        rng = np.random.RandomState(42)
        self.frame = pd.DataFrame(rng.normal(size=(24, 3)), columns=["a", "b", "c"])
        labels = np.where(self.frame.a > 0, "fail", "pass")
        self.pipeline = Pipeline([
            ("quality_filter", SensorQualityFilter()),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("selector", SelectFromModel(RandomForestClassifier(n_estimators=3, random_state=42),
                                         threshold=-np.inf, max_features=2)),
            ("model", RandomForestClassifier(n_estimators=3, random_state=43)),
        ]).fit(self.frame, labels)
        self.options = dict(positive_label="fail", threshold=0.3, max_sensors=2)

    def test_all_cases_pass_without_changing_input(self):
        """정상·NaN·단일 행·배치 비교를 구현 검증으로 기록한다."""
        original = self.frame.copy(deep=True)
        result = check_inference_paths(self.pipeline, self.frame, **self.options)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["checks"]), 5)
        self.assertFalse(result["test_used"])
        self.assertFalse(result["validation_used"])
        self.assertFalse(result["is_performance_evaluation"])
        pd.testing.assert_frame_equal(self.frame, original)

    def test_all_missing_rows_are_counted(self):
        """원본 입력에 전체 결측 행이 있으면 제외 개수를 명시한다."""
        inference = build_sensor_inference(self.pipeline, **self.options)
        frame = self.frame.copy()
        frame.loc[0, list(inference.sensors)] = np.nan
        result = check_inference_paths(self.pipeline, frame, **self.options)
        self.assertEqual(result["all_missing_rejected_rows"], 1)
        self.assertEqual(result["compared_rows"], 23)
        self.assertEqual(result["status"], "passed")
        frame.loc[:, list(inference.sensors)] = np.nan
        with self.assertRaisesRegex(ValueError, "한 행 이상"):
            check_inference_paths(self.pipeline, frame, **self.options)

    def test_probability_difference_fails(self):
        """확률이 다르면 같은 판정이 나오더라도 통과시키지 않는다."""
        inference = build_sensor_inference(self.pipeline, **self.options)
        # 축소 모델의 확률만 바꿔서 원본과의 차이를 반드시 검출하도록 한다.
        with patch.object(inference._model, "predict_proba",
                          return_value=np.tile([0.123, 0.877], (24, 1))):
            result = compare_prediction_paths(self.pipeline, inference, self.frame, case="changed")
        self.assertFalse(result["passed"])
        self.assertFalse(result["probability_match"])


if __name__ == "__main__":
    unittest.main()
