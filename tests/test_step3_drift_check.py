# ==========================================
# Time-based 분포 변화 진단 테스트
# - 수치형 결측률·분포 변화와 범주형 새 범주 비율을 작은 가상 split으로 확인
# - 서로 다른 Train/Validation feature 구성은 진단 전에 차단하는지 확인
# ==========================================

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.modeling_config import DatasetSpec
from src.sensor_ml.diagnostics.step3_drift_check import (
    align_split_features,
    build_label_drift_summary,
    build_temporal_drift_report,
)


class TemporalDriftCheckTest(unittest.TestCase):
    # 수치형 및 범주형 feature를 함께 선언한 범용 Dataset Profile을 사용한다.
    dataset = DatasetSpec(
        dataset_id="toy_drift",
        input_path=Path("unused.csv"),
        label_column="target",
        positive_label="fail",
        negative_label="pass",
        timestamp_column=None,
        timestamp_format=None,
        feature_selection_mode="all_except_metadata",
        feature_column_prefix=None,
        missing_ratio_threshold=0.8,
        drop_zero_variance=False,
        categorical_feature_columns=("machine",),
    )

    # 수치형 위치·결측률 변화와 Validation의 새 범주를 한 보고서에서 기록하는지 확인한다.
    def test_builds_numeric_and_categorical_drift_report(self) -> None:
        train_frame = pd.DataFrame(
            {
                "sensor": [0.0, 0.2, 0.4, 0.6, None],
                "machine": ["A", "A", "B", "B", "A"],
                "target": ["pass", "pass", "fail", "fail", "pass"],
            }
        )
        validation_frame = pd.DataFrame(
            {
                "sensor": [10.0, 10.2, None, None],
                "machine": ["C", "C", "A", "C"],
                "target": ["fail", "fail", "pass", "fail"],
            }
        )

        report = build_temporal_drift_report(
            train_frame, validation_frame, self.dataset
        )
        label_summary = build_label_drift_summary(
            train_frame, validation_frame, self.dataset
        )
        numeric = report.loc[report["feature"] == "sensor"].iloc[0]
        categorical = report.loc[report["feature"] == "machine"].iloc[0]

        self.assertEqual(numeric["feature_type"], "numeric")
        self.assertGreater(numeric["missing_ratio_delta"], 0.0)
        self.assertGreater(numeric["absolute_standardized_mean_difference"], 1.0)
        self.assertEqual(numeric["drift_priority"], "높음")
        self.assertEqual(categorical["feature_type"], "categorical")
        self.assertEqual(categorical["unseen_category_ratio"], 0.75)
        self.assertGreater(categorical["category_distribution_tvd"], 0.0)
        self.assertEqual(len(label_summary), 2)
        self.assertAlmostEqual(label_summary.loc[1, "ratio_delta"], 0.35)

    # feature 집합이 다르면 잘못된 column 비교가 일어나기 전에 중단해야 한다.
    def test_rejects_mismatched_train_validation_features(self) -> None:
        train_frame = pd.DataFrame(
            {
                "sensor": [0.0, 1.0],
                "machine": ["A", "B"],
                "target": ["pass", "fail"],
            }
        )
        validation_frame = pd.DataFrame(
            {
                "sensor": [0.1, 0.9],
                "machine": ["A", "B"],
                "other_sensor": [0.2, 0.8],
                "target": ["pass", "fail"],
            }
        )

        with self.assertRaisesRegex(ValueError, "feature column 구성이 일치하지"):
            align_split_features(train_frame, validation_frame, self.dataset)
