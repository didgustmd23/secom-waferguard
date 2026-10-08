# ==========================================
# 특징 선택 비교 테스트
# - 작은 수치형 데이터에서 PCA·L1·LightGBM 선택 실험 결과를 생성하는지 확인
# - 단일 class, 과도한 Top-K 설정을 실행 전 차단하는지 확인
# - XGBoost 분류기를 유지한 채 센서 선택 방식만 달라지는지 확인
# ==========================================

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.modeling_config import (
    CrossValidationConfig,
    DatasetSpec,
    ExperimentProtocol,
    MODELING_CONFIG,
)
from src.step6_feature_compare import compare_features
from src.modeling_models import build_pipeline, fit_pipeline


class FeatureComparisonTest(unittest.TestCase):
    # ==========================================
    # 작은 합성 데이터로 CV의 M1~M3 옵션 전달을 검증
    # - 실제 SECOM 실험 대신 Top-2·트리 3개로 연결만 확인
    # - 전체 기준 깊이 3, 축소 분류기 깊이 2, 선택 RF는 깊이 제한 없음
    # ==========================================
    def test_m1_cv_keeps_baseline_and_selector(self):
        templates = {}

        def small_pipeline(config, name, *, n_jobs=1):
            model = build_pipeline(config, name, n_jobs=n_jobs)
            model.set_params(**{key: 3 for key in model.get_params()
                                if key.endswith("__n_estimators")})
            templates[name] = model
            return model

        # M1·M2·M3을 같은 작은 데이터에서 각각 독립적으로 검증한다.
        for depth, reg_lambda in ((2, None), (None, 5), (2, 5)):
            with self.subTest(depth=depth, reg_lambda=reg_lambda):
                with patch("src.step6_feature_compare.build_pipeline", side_effect=small_pipeline):
                    result = compare_features(self._frame(), self.config,
                                              experiment_names=("xgboost_all", "xgboost_rf_top_2"),
                                              topk_xgb_max_depth=depth, topk_xgb_reg_lambda=reg_lambda)
                self.assertEqual(templates["xgboost_all"].named_steps["model"].max_depth, 3)
                self.assertEqual(templates["xgboost_all"].named_steps["model"].reg_lambda, 1)
                top = templates["xgboost_rf_top_2"]
                self.assertEqual(top.named_steps["model"].max_depth, 3 if depth is None else depth)
                self.assertEqual(top.named_steps["model"].reg_lambda, 1 if reg_lambda is None else reg_lambda)
                self.assertIsNone(top.named_steps["selector"].estimator.max_depth)
                self.assertEqual(top.named_steps["selector"].estimator.min_samples_leaf, 1)
        self.assertEqual(templates["xgboost_all"].named_steps["model"].max_depth, 3)
        top = templates["xgboost_rf_top_2"]
        self.assertEqual(top.named_steps["model"].max_depth, 2)
        self.assertIsNone(top.named_steps["selector"].estimator.max_depth)
        self.assertEqual(top.named_steps["selector"].estimator.min_samples_leaf, 1)
        self.assertEqual(len(result.attrs["fold_results"]), 4)
        self.assertTrue(result.attrs["selected_features"].groupby("cv_fold").size().eq(2).all())

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

    # ==========================================
    # 전체·XGBoost 중요도·RF 중요도 비교의 공정성 검증
    # - 최종 분류기 설정은 같고 선택용 모델만 다르게 구성
    # - 작은 데이터의 Top-2 실행으로 fold별 결과와 센서 기록을 확인
    # ==========================================
    def test_xgboost_selectors_keep_same_classifier(self) -> None:
        names = ("xgboost_all", "xgboost_top_2", "xgboost_rf_top_2")
        pipelines = [build_pipeline(self.config, name) for name in names]
        expected = pipelines[0].named_steps["model"].get_params()
        for pipeline in pipelines[1:]:
            self.assertEqual(pipeline.named_steps["model"].get_params(), expected)
        self.assertEqual(
            pipelines[1].named_steps["selector"].estimator.importance_type, "gain",
        )
        self.assertEqual(
            type(pipelines[2].named_steps["selector"].estimator).__name__,
            "RandomForestClassifier",
        )

    def test_xgboost_comparison_records_each_fold(self) -> None:
        names = ("xgboost_all", "xgboost_top_2", "xgboost_rf_top_2")
        result = compare_features(self._frame(), self.config, experiment_names=names)
        counts = result.set_index("experiment")["selected_feature_count_mean"]
        self.assertEqual(counts.to_dict(), dict(zip(names, (3.0, 2.0, 2.0))))
        self.assertEqual(len(result.attrs["fold_results"]), 6)
        selected = result.attrs["selected_features"]
        self.assertEqual(len(selected), 8)
        self.assertTrue(selected["feature_space"].eq("raw_sensor").all())
        self.assertTrue(selected.groupby(["experiment", "cv_fold"]).size().eq(2).all())

    def test_xgboost_selection_ignores_heldout_values(self) -> None:
        # 검증 데이터의 값이 아니라 학습 fold만으로 제거할 센서를 판단한다.
        config = replace(self.config, dataset=replace(self.dataset, drop_zero_variance=True))
        train = self._frame().assign(sensor_c=1.0)
        features = train.drop(columns="target")
        template = build_pipeline(config, "xgboost_top_2")
        model = fit_pipeline(template, features, train["target"], config, "xgboost_top_2")
        # 검증 입력에 변동이 있어도 학습 시 값이 하나뿐이었던 센서는 복원하지 않는다.
        heldout = features.assign(sensor_c=range(len(features)))
        model.predict_proba(heldout)
        self.assertNotIn("sensor_c", model.named_steps["quality_filter"].retained_features_)
        self.assertEqual(model.named_steps["model"].n_features_in_, 2)
        self.assertFalse(hasattr(template.named_steps["model"], "estimator_"))


if __name__ == "__main__":
    unittest.main()
