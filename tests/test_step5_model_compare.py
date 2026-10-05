# ==========================================
# 후보 모델 비교 테스트
# - 작은 Profile 호환 데이터에서 L1 후보 CV 결과가 생성되는지 확인
# - 단일 class와 CV fold 수 부족을 명확한 오류로 차단하는지 확인
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
from src.step5_model_compare import compare_candidates


class CandidateComparisonTest(unittest.TestCase):
    dataset = DatasetSpec(
        dataset_id="toy_model_compare",
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
        candidate_models=("logistic_regression_l1",),
    )

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sensor_a": [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.8, 0.9, 1.0, 0.7, 0.9, 0.8],
                "sensor_b": [1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 0.2, 0.1, 0.0, 0.3, 0.1, 0.2],
                "target": ["pass"] * 6 + ["fail"] * 6,
            }
        )

    def test_compares_l1_and_balanced_l1_with_cv_metrics(self) -> None:
        result = compare_candidates(self._frame(), self.config)

        self.assertEqual(
            set(result["model_name"]),
            {"logistic_regression_l1", "logistic_regression_l1_balanced"},
        )
        self.assertTrue((result["n_splits"] == 2).all())
        self.assertIn("average_precision_mean", result.columns)
        self.assertIn("roc_auc_std", result.columns)

    def test_rejects_single_class_train_data(self) -> None:
        frame = self._frame().assign(target="pass")

        with self.assertRaisesRegex(ValueError, "정상과 Fail label"):
            compare_candidates(frame, self.config)

    def test_rejects_too_few_samples_for_cv_fold_count(self) -> None:
        frame = self._frame().iloc[:7].copy()
        frame.loc[6, "target"] = "fail"

        with self.assertRaisesRegex(ValueError, "CV fold 수보다 작습니다"):
            compare_candidates(frame, self.config)


if __name__ == "__main__":
    unittest.main()
