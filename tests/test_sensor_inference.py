# ==========================================
# 축소 센서 추론의 정합성과 입력 계약 검증
# - 실제 SECOM 데이터 대신 작은 합성 수치형 데이터만 사용
# - 결측 대치·표준화·컬럼 재정렬 이후 기존 Pipeline과 확률 비교
# - 잘못된 입력은 명시적으로 거부하고 추론에서 fit 호출 금지
# ==========================================

import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling_preprocessing import SensorQualityFilter
from src.sensor_inference import build_sensor_inference


class SensorInferenceTest(unittest.TestCase):
    def setUp(self):
        """변동 센서 4개·제거될 상수 센서 1개로 작은 학습 자료를 만든다."""
        rng = np.random.RandomState(42)
        self.frame = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
        self.frame["constant"] = 1.0
        self.frame.loc[[1, 5], "a"] = np.nan
        self.labels = np.where(self.frame.b > 0, "fail", "pass")

    def _pipeline(self, scale=False):
        """실제 코어와 같은 수치형 단계 순서로 두 센서를 선택한다."""
        steps = [("quality_filter", SensorQualityFilter()),
                 ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.extend([
            ("selector", SelectFromModel(
                RandomForestClassifier(n_estimators=3, random_state=42),
                threshold=-np.inf, max_features=2)),
            ("model", RandomForestClassifier(n_estimators=3, random_state=43)),
        ])
        return Pipeline(steps).fit(self.frame, self.labels)

    def test_probabilities_and_decisions_match(self):
        """결측·열 순서 변경에도 원본 확률과 후보 문턱 판정이 일치한다."""
        for scale in (False, True):
            with self.subTest(scale=scale):
                fitted = self._pipeline(scale)
                inference = build_sensor_inference(
                    fitted, positive_label="fail", threshold=0.3, max_sensors=2)
                full = self.frame.iloc[:6].copy()
                full.loc[full.index[0], inference.sensors[0]] = np.nan
                reduced = full.loc[:, list(reversed(inference.sensors))]
                before = reduced.copy(deep=True)
                result = inference.predict(reduced)
                position = list(fitted.classes_).index("fail")
                expected = fitted.predict_proba(full)[:, position]
                np.testing.assert_allclose(result.positive_score, expected, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(result.predicted_positive, expected >= 0.3)
                np.testing.assert_array_equal(result.predicted_label,
                                              np.where(expected >= 0.3, "fail", "pass"))
                pd.testing.assert_frame_equal(reduced, before)
                # 같은 행을 단독으로 넣어도 배치의 median에 따라 결과가 바뀌지 않는다.
                single = inference.predict(reduced.iloc[[0]])
                self.assertEqual(single.positive_score.iloc[0], result.positive_score.iloc[0])

    def test_invalid_inputs_are_rejected(self):
        """컬럼 누락과 NaN을 구분하고 타입·중복·전체 결측을 검증한다."""
        inference = build_sensor_inference(self._pipeline(), positive_label="fail",
                                            threshold=0.3, max_sensors=2)
        frame = self.frame.loc[:2, list(inference.sensors)].copy()
        invalid = [frame.iloc[:, :1], frame.assign(extra=1),
                   pd.concat([frame, frame.iloc[:, :1]], axis=1),
                   frame.astype(str), frame.astype(bool), frame * np.inf,
                   frame * np.nan, frame.iloc[:0], frame.to_numpy()]
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    inference.predict(value)

    def test_no_refit_and_independent_model_copy(self):
        """변환·추론에서 fit하지 않고 원본 모델과 독립적인 객체를 사용한다."""
        fitted = self._pipeline()
        with patch.object(SimpleImputer, "fit", side_effect=AssertionError("재학습 금지")), \
                patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("재학습 금지")):
            inference = build_sensor_inference(fitted, positive_label="fail",
                                                threshold=0.3, max_sensors=2)
            frame = self.frame.loc[:2, list(inference.sensors)]
            before = inference.predict(frame)
            self.assertIsNot(inference._model, fitted.named_steps["model"])
            # 파일 저장 기능은 보류하되 메모리 직렬화로 상태가 유지되는지 확인한다.
            restored = pickle.loads(pickle.dumps(inference))
            pd.testing.assert_frame_equal(before, restored.predict(frame))

    def test_invalid_conversion_is_rejected(self):
        """센서 수 초과·알 수 없는 양성 label·잘못된 문턱과 구조를 거부한다."""
        fitted = self._pipeline()
        for options in ({"max_sensors": 1}, {"threshold": np.nan},
                        {"threshold": True}, {"positive_label": "unknown"}):
            arguments = dict(positive_label="fail", threshold=0.3, max_sensors=2)
            arguments.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_sensor_inference(fitted, **arguments)
        with self.assertRaises(ValueError):
            build_sensor_inference(Pipeline(fitted.steps[1:]), positive_label="fail",
                                    threshold=0.3, max_sensors=2)

    def test_xgboost_adapter_matches_pandas_pipeline(self):
        """현재 XGBoost adapter와 pandas 출력 경로도 축소 추론과 일치한다."""
        from src.modeling_models import XGBoostClassifierAdapter

        pipeline = self._pipeline()
        # 실제 RF Top-K 경로와 같이 대치 결과에 센서 이름을 유지한다.
        pipeline.named_steps["imputer"].set_output(transform="pandas")
        pipeline.set_params(model=XGBoostClassifierAdapter(
            positive_label="fail", negative_label="pass", n_estimators=3,
            max_depth=2, reg_lambda=5, n_jobs=1))
        pipeline.fit(self.frame, self.labels)
        inference = build_sensor_inference(pipeline, positive_label="fail",
                                            threshold=0.3, max_sensors=2)
        full = self.frame.iloc[:5].copy()
        full.loc[full.index[0], inference.sensors[0]] = np.nan
        result = inference.predict(full.loc[:, list(inference.sensors)])
        np.testing.assert_allclose(result.positive_score,
                                   pipeline.predict_proba(full)[:, 1], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
