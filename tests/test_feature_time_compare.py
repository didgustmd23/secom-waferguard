# ==========================================
# 시간순 축소 비교의 센서 수·선택 근거·빈도 분모 검증
# - RF 중요도와 LightGBM 중요도를 구분하고 후보를 최종으로 표시하지 않음
# - XGBoost 실험을 지정하면 해당 모델만 같은 시간 검증 코어에서 실행
# ==========================================

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from src.feature_time_compare import compare_feature_time, selection_rows, build_feature_pipelines
from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID


class FeatureTimeComparisonTest(unittest.TestCase):
    @staticmethod
    def frame():
        # 작은 데이터에서도 Top-50 선택이 가능하도록 변동 센서 60개를 만든다.
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(48, 60)),
                             columns=[f"sensor_{i}" for i in range(60)])
        frame["sensor_empty"] = np.nan
        frame["label"] = [-1, 1] * 24
        frame["timestamp"] = [date.strftime("%d/%m/%Y %H:%M:%S")
                              for date in pd.date_range("2008-01-01", periods=24) for _ in range(2)]
        frame[SOURCE_ROW_ID] = np.arange(48)
        frame[SPLIT_ROLE] = "train"
        frame[PROTOCOL_ID] = "time:feature-toy"
        return frame

    def test_selection_names_and_counts_match_pipeline(self):
        frame = self.frame()
        features = frame.filter(regex="^sensor_")
        for name, pipeline in build_feature_pipelines(MODELING_CONFIG, n_estimators=3).items():
            pipeline.fit(features, frame.label)
            rows = selection_rows(pipeline, {"model_name": name})
            expected = 60 if name == "lightgbm_all" else int(name.rsplit("_", 1)[1])
            self.assertEqual(len(rows), expected)
            self.assertNotIn("sensor_empty", [row["feature"] for row in rows])
            self.assertEqual([row["selection_rank"] for row in rows], list(range(1, expected + 1)))
            self.assertTrue(all(row["feature_space"] == "raw_sensor" for row in rows))
            if "rf_top" in name:
                self.assertTrue(all(row["selection_method"] == "rf_importance" for row in rows))

    def test_temporal_records_frequency_and_nonfinal_candidates(self):
        results = compare_feature_time(self.frame(), MODELING_CONFIG, n_estimators=3)
        self.assertEqual(len(results["fold_results"]), 30)
        self.assertEqual(len(results["policy_selection"]), 15)
        selected = results["selected_features"]
        frequency = results["feature_frequency"]
        self.assertTrue(frequency.selection_frequency.between(0, 1).all())
        self.assertTrue(frequency.loc[frequency.fit_role == "outer_fit", "total_fit_count"].eq(3).all())
        self.assertTrue(frequency.loc[frequency.fit_role == "inner_oof", "total_fit_count"].eq(6).all())
        self.assertFalse(results["candidate_features"].is_final.any())
        self.assertEqual(len(results["quality_filter"]), 45)
        for (name, fit_id), group in selected.groupby(["model_name", "fit_id"]):
            expected = 60 if name == "lightgbm_all" else int(name.rsplit("_", 1)[1])
            self.assertEqual(group.feature.nunique(), expected)

    # ==========================================
    # XGBoost 전체·RF Top-20의 시간순 검증 연결
    # - 최종 분류기 설정과 선택기 병렬·트리 수 전달을 확인
    # - 과거 학습에만 센서 선택을 적용하고 외부·내부 기록 수를 확인
    # ==========================================
    def test_xgboost_time_comparison_and_selection(self):
        names = ("xgboost_all", "xgboost_rf_top_20")
        pipelines = build_feature_pipelines(
            MODELING_CONFIG, n_estimators=3, experiment_names=names,
        )
        self.assertEqual(set(pipelines), set(names))
        self.assertEqual(pipelines[names[0]].named_steps["model"].get_params(),
                         pipelines[names[1]].named_steps["model"].get_params())
        self.assertEqual(pipelines[names[1]].named_steps["selector"].estimator.n_estimators, 3)
        results = compare_feature_time(
            self.frame(), MODELING_CONFIG, n_estimators=3, experiment_names=names,
        )
        self.assertEqual(set(results["summary"].model_name), set(names))
        self.assertEqual(len(results["fold_results"]), 12)
        self.assertEqual(len(results["policy_selection"]), 6)
        selected = results["selected_features"]
        top = selected[selected.model_name.eq(names[1])]
        self.assertTrue(top.groupby("fit_id").feature.nunique().eq(20).all())
        self.assertTrue(top.selection_method.eq("rf_importance").all())
        self.assertFalse(results["candidate_features"].is_final.any())
        chronological = results["quality_filter"]
        self.assertTrue((pd.to_datetime(chronological.train_end)
                         < pd.to_datetime(chronological.evaluation_start)).all())

    def test_xgboost_gain_method_and_invalid_names(self):
        name = "xgboost_top_20"
        pipeline = build_feature_pipelines(
            MODELING_CONFIG, n_estimators=3, experiment_names=(name,),
        )[name]
        frame = self.frame()
        pipeline.fit(frame.filter(regex="^sensor_"), frame.label)
        rows = selection_rows(pipeline, {"model_name": name})
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["selection_method"] == "xgboost_gain" for row in rows))
        for names in ((), (name, name), ("unknown",)):
            with self.assertRaises(ValueError):
                build_feature_pipelines(MODELING_CONFIG, experiment_names=names)


if __name__ == "__main__":
    unittest.main()
