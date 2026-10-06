# ==========================================
# 시간순 축소 비교의 센서 수·선택 근거·빈도 분모 검증
# - RF 중요도와 LightGBM 중요도를 구분하고 후보를 최종으로 표시하지 않음
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


if __name__ == "__main__":
    unittest.main()
