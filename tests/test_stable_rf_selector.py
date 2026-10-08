# ==========================================
# S3 반복 센서 선택의 재현성·입력 계약·CV/시간순 연결 확인
# - 실제 SECOM 실험은 실행하지 않고 작은 합성 데이터만 학습
# - 최종 XGBoost 설정 유지와 bootstrap별 전처리 기록을 확인
# ==========================================
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.experiments.feature_time_compare import build_feature_pipelines, compare_feature_time
from src.modeling_config import MODELING_CONFIG
from src.modeling_models import build_pipeline
from src.experiments.step6_feature_compare import compare_features
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID


class StableRFTest(unittest.TestCase):
    @staticmethod
    def frame():
        # 20개 선택이 가능한 수치형 센서와 양쪽 클래스를 준비한다.
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(48, 30)),
                             columns=[f"sensor_{index}" for index in range(30)])
        frame["label"] = [-1, 1] * 24
        frame["timestamp"] = [date.strftime("%d/%m/%Y %H:%M:%S")
                              for date in pd.date_range("2008-01-01", periods=48)]
        frame[SOURCE_ROW_ID] = np.arange(48)
        frame[SPLIT_ROLE] = "train"
        frame[PROTOCOL_ID] = "time:stable-toy"
        return frame

    def test_reproducible_selection_and_validation(self):
        names = ("xgboost_all", "xgboost_rf_top_20")
        pipelines = build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                           n_estimators=3, rf_stability_repeats=2)
        # 모델 설정은 반복 선택을 사용하지 않는 기준과 같아야 한다.
        baseline = build_feature_pipelines(MODELING_CONFIG, experiment_names=names, n_estimators=3)
        self.assertEqual(pipelines[names[1]].named_steps["model"].get_params(),
                         baseline[names[1]].named_steps["model"].get_params())
        selector = pipelines[names[1]].named_steps["selector"]
        frame = self.frame()
        features = frame.filter(regex="^sensor_")
        first, second = clone(selector).fit(features, frame.label), clone(selector).fit(features, frame.label)
        np.testing.assert_array_equal(first.get_support(), second.get_support())
        np.testing.assert_array_equal(first.selection_frequency_, second.selection_frequency_)
        self.assertEqual(first.get_support().sum(), 20)
        self.assertEqual(len(first.bootstrap_log_), 2)
        # 새로운 입력 값이 바뀌어도 저장된 선택 센서 목록은 변경하지 않는다.
        original = first.get_feature_names_out().copy()
        first.transform(features * 100)
        np.testing.assert_array_equal(first.get_feature_names_out(), original)
        with self.assertRaises(ValueError):
            first.transform(features.drop(columns=original[0]))
        for repeats in (-1, 1, True):
            with self.assertRaises(ValueError):
                build_feature_pipelines(MODELING_CONFIG, experiment_names=names, rf_stability_repeats=repeats)

    def test_temporal_and_cv_record_stability(self):
        # 시간순 내부 OOF에도 별도 반복 선택을 적용하고 기록한다.
        names = ("xgboost_all", "xgboost_rf_top_20")
        result = compare_feature_time(self.frame(), MODELING_CONFIG, experiment_names=names,
                                      n_estimators=3, rf_stability_repeats=2)
        selected = result["selected_features"].query("model_name == 'xgboost_rf_top_20'")
        self.assertTrue(selected.groupby("fit_id").feature.nunique().eq(20).all())
        self.assertTrue(selected.selection_method.eq("rf_stability").all())
        self.assertTrue(selected.bootstrap_selection_frequency.between(0, 1).all())
        self.assertEqual(len(result["bootstrap_records"]), 20)
        # CV 테스트의 RF·XGBoost는 트리 3개로 제한해 실제 실험과 구분한다.
        def small_pipeline(config, name, **kwargs):
            pipeline = build_pipeline(config, name, **kwargs)
            pipeline.set_params(**{key: 3 for key in pipeline.get_params() if key.endswith("__n_estimators")})
            return pipeline
        config = replace(MODELING_CONFIG, experiment=replace(MODELING_CONFIG.experiment,
                         cv=replace(MODELING_CONFIG.experiment.cv, n_splits=2, n_repeats=1)))
        with patch("src.experiments.step6_feature_compare.build_pipeline", side_effect=small_pipeline):
            cv = compare_features(self.frame(), config, experiment_names=names, rf_stability_repeats=2)
        self.assertEqual(len(cv.attrs["bootstrap_records"]), 4)
        self.assertIn("bootstrap_mean_rank", cv.attrs["selected_features"].columns)
        self.assertEqual(cv.attrs["reduction_assessment"].iloc[0].sensor_status, "충족")


if __name__ == "__main__":
    unittest.main()
