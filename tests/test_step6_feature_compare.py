# ==========================================
# 특징 선택 비교 테스트
# - 작은 수치형 데이터에서 PCA·L1·LightGBM 선택 실험 결과를 생성하는지 확인
# - 단일 class, 과도한 Top-K 설정을 실행 전 차단하는지 확인
# ==========================================

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.modeling_config import (
    CrossValidationConfig,
    DatasetSpec,
    ExperimentProtocol,
    MODELING_CONFIG,
)
from src.step6_feature_compare import compare_features


class FeatureComparisonTest(unittest.TestCase):
    dataset = DatasetSpec(
        dataset_id="toy_feature_compare",
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
    )
    config = replace(
        MODELING_CONFIG,
        dataset=dataset,
        experiment=ExperimentProtocol(
            default_threshold=0.5,
            pca_explained_variance=0.9,
            cv=CrossValidationConfig(n_splits=2, n_repeats=1, random_state=42),
        ),
        top_k_feature_counts=(3, 2),
    )

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sensor_a": [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.8, 0.9, 1.0, 0.7, 0.9, 0.8],
                "sensor_b": [1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 0.2, 0.1, 0.0, 0.3, 0.1, 0.2],
                "sensor_c": [0.2, 0.1, 0.3, 0.2, 0.4, 0.3, 0.7, 0.8, 0.6, 0.7, 0.9, 0.8],
                "target": ["pass"] * 6 + ["fail"] * 6,
            }
        )

    def test_runs_pca_l1_and_lightgbm_feature_comparisons(self) -> None:
        result = compare_features(self._frame(), self.config, n_jobs=1)

        self.assertEqual(
            set(result["experiment"]),
            {
                "l1_balanced_all",
                "l1_balanced_pca90",
                "l1_balanced_l1_select",
                "lightgbm_all",
                "lightgbm_top_3",
                "lightgbm_top_2",
            },
        )
        self.assertIn("selected_feature_count_mean", result.columns)
        self.assertIn("average_precision_std", result.columns)

    def test_rejects_single_class_train_data(self) -> None:
        frame = self._frame().assign(target="pass")

        with self.assertRaisesRegex(ValueError, "정상과 Fail label"):
            compare_features(frame, self.config)

    def test_rejects_top_k_larger_than_feature_count(self) -> None:
        invalid_config = replace(self.config, top_k_feature_counts=(4,))

        with self.assertRaisesRegex(ValueError, "Top-K feature 수가 입력 feature 수보다 큽니다"):
            compare_features(self._frame(), invalid_config)


if __name__ == "__main__":
    unittest.main()
