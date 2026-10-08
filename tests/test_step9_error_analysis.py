# ==========================================
# Time Validation 오류 사례 분석 단위 테스트
# - TP/FN/FP/TN 분류와 feature 비교 결과의 기본 구조를 확인
# ==========================================

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.diagnostics.step9_error_analysis import assign_error_groups, compare_error_features, summarize_error_groups


class ErrorAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = pd.Series([1, 1, -1, -1])
        self.scores = np.array([0.9, 0.1, 0.8, 0.2])
        self.groups = assign_error_groups(
            self.labels, self.scores, positive_label=1, threshold=0.5
        )
        self.features = pd.DataFrame(
            {
                "sensor_0": [1.0, 2.0, 3.0, 4.0],
                "sensor_1": [np.nan, 1.0, 1.0, 1.0],
            }
        )
        self.cases = pd.DataFrame(
            {
                "positive_score": self.scores,
                "error_group": self.groups,
            }
        )

    def test_assigns_all_confusion_matrix_groups(self) -> None:
        self.assertEqual(self.groups.tolist(), ["TP", "FN", "FP", "TN"])

    def test_summarizes_missing_ratio_for_each_group(self) -> None:
        summary = summarize_error_groups(self.cases, self.features)

        self.assertEqual(summary["error_group"].tolist(), ["TP", "FN", "FP", "TN"])
        self.assertEqual(summary.loc[summary["error_group"] == "FN", "sample_count"].item(), 1)

    def test_joins_drift_priority_to_feature_comparison(self) -> None:
        drift_report = pd.DataFrame(
            {
                "feature": ["sensor_0", "sensor_1"],
                "drift_priority": ["높음", "낮음"],
                "drift_priority_score": [10.0, 1.0],
            }
        )
        result = compare_error_features(self.features, self.groups, drift_report)

        self.assertEqual(set(result["feature"]), {"sensor_0", "sensor_1"})
        self.assertIn("drift_priority", result.columns)

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "0.0 이상 1.0 이하"):
            assign_error_groups(
                self.labels, self.scores, positive_label=1, threshold=-0.1
            )


if __name__ == "__main__":
    unittest.main()
