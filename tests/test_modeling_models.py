# ==========================================
# 공통 모델 코어의 리팩터링 회귀 테스트
# - 기존 실험 이름·label 방향·독립 fold 학습을 보존하는지 확인
# - XGBoost의 OOF 병렬 설정과 오류 분석 모델 선택을 검증
# - 실제 프로젝트 데이터 대신 작은 임시 DataFrame만 사용
# ==========================================

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.modeling_config import MODELING_CONFIG
from src.modeling_models import (
    XGBoostClassifierAdapter, build_pipeline, fit_pipeline, positive_scores,
)
from src.step8_threshold_oof import generate_oof_scores
from src.step9_error_analysis import analyze_time_validation_errors


class ModelingModelsTest(unittest.TestCase):
    config = replace(
        MODELING_CONFIG,
        dataset=replace(
            MODELING_CONFIG.dataset, label_column="target",
            positive_label="fail", negative_label="pass",
            timestamp_column=None, timestamp_format=None,
            feature_selection_mode="all_except_metadata", feature_column_prefix=None,
            drop_zero_variance=False,
        ),
        experiment=replace(
            MODELING_CONFIG.experiment,
            cv=replace(MODELING_CONFIG.experiment.cv, n_splits=2, n_repeats=1),
        ),
        top_k_feature_counts=(1,),
    )

    @staticmethod
    def _frame():
        # 문자열 양성 label이 정렬상 먼저 오더라도 불량 확률 열을 올바르게 선택해야 한다.
        return pd.DataFrame({
            "sensor_a": np.linspace(0, 1, 18),
            "sensor_b": np.linspace(1, 0, 18),
            "target": ["pass"] * 12 + ["fail"] * 6,
        })

    def test_aliases_keep_classifier_settings(self):
        aliases = {
            "lightgbm_all": "lightgbm",
            "l1_balanced_all": "logistic_regression_l1_balanced",
        }
        for alias, candidate in aliases.items():
            with self.subTest(alias=alias):
                first = build_pipeline(self.config, alias).named_steps["model"]
                second = build_pipeline(self.config, candidate).named_steps["model"]
                self.assertEqual(first.get_params(), second.get_params())

    def test_fit_clones_and_sets_training_weight(self):
        frame = self._frame()
        template = build_pipeline(self.config, "xgboost_scale_pos_weight")
        model = fit_pipeline(
            template, frame.drop(columns="target"), frame.target, self.config,
            "xgboost_scale_pos_weight", n_jobs=2,
        )
        # 가중치는 학습 자료의 12/6에서 계산하고 원본 Pipeline에는 상태가 없어야 한다.
        self.assertEqual(model.named_steps["model"].scale_pos_weight, 2.0)
        self.assertEqual(model.named_steps["model"].n_jobs, 2)
        self.assertFalse(hasattr(template.named_steps["model"], "estimator_"))
        self.assertFalse(hasattr(template.named_steps["quality_filter"], "retained_features_"))

    def test_positive_scores_follow_configured_class(self):
        frame = self._frame()
        features = frame.drop(columns="target")
        model = fit_pipeline(
            build_pipeline(self.config, "logistic_regression_l1"),
            features, frame.target, self.config, "logistic_regression_l1",
        )
        # sklearn의 문자열 class 정렬에서는 fail이 첫 열이다.
        self.assertEqual(model.named_steps["model"].classes_[0], "fail")
        np.testing.assert_allclose(
            positive_scores(model, features, self.config.dataset),
            model.predict_proba(features)[:, 0],
        )

    def test_rejects_invalid_training_labels(self):
        frame = self._frame()
        for labels in (frame.target.replace("fail", "unknown"), frame.target.mask(frame.index == 0)):
            with self.subTest(labels=labels.tolist()):
                with self.assertRaisesRegex(ValueError, "정상·Fail label"):
                    fit_pipeline(
                        build_pipeline(self.config, "lightgbm"),
                        frame.drop(columns="target"), labels, self.config, "lightgbm",
                    )

    def test_adapter_preserves_original_labels_and_clone(self):
        frame = self._frame()
        adapter = XGBoostClassifierAdapter(
            positive_label="fail", negative_label="pass", n_estimators=2,
        )
        cloned = clone(adapter).fit(frame.drop(columns="target"), frame.target)
        self.assertEqual(cloned.classes_.tolist(), ["pass", "fail"])
        self.assertTrue(set(cloned.predict(frame.drop(columns="target"))) <= {"pass", "fail"})
        self.assertFalse(hasattr(adapter, "estimator_"))
        with self.assertRaisesRegex(ValueError, "결측값"):
            clone(adapter).fit(frame.drop(columns="target"), frame.target.mask(frame.index == 0))

    def test_xgboost_oof_propagates_parallel_setting(self):
        # 공통 fit으로 전달된 옵션뿐 아니라 실제 학습된 분류기의 값을 확인한다.
        fitted_models = []
        def record_fit(*args, **kwargs):
            model = fit_pipeline(*args, **kwargs)
            fitted_models.append(model)
            return model
        with patch("src.step8_threshold_oof.fit_pipeline", side_effect=record_fit):
            oof, _ = generate_oof_scores(
                self._frame(), self.config, experiment_name="xgboost", n_jobs=2,
            )
        self.assertEqual(len(oof), 18)
        self.assertTrue(oof.oof_prediction_count.eq(1).all())
        self.assertEqual(len(fitted_models), 2)
        self.assertTrue(all(model.named_steps["model"].n_jobs == 2 for model in fitted_models))

    def test_error_analysis_uses_selected_experiment(self):
        validation = pd.DataFrame({
            "sensor_a": [0.11, 0.87], "sensor_b": [0.88, 0.14],
            "target": ["pass", "fail"],
        })
        with patch("src.step9_error_analysis.fit_pipeline", wraps=fit_pipeline) as fit:
            cases, summary, features = analyze_time_validation_errors(
                self._frame(), validation, self.config,
                threshold=0.5, experiment_name="xgboost", n_jobs=2,
            )
        self.assertEqual(fit.call_args.args[4], "xgboost")
        self.assertIsInstance(fit.call_args.args[0].named_steps["model"], XGBoostClassifierAdapter)
        self.assertEqual(len(cases), 2)
        self.assertFalse(summary.empty)
        self.assertFalse(features.empty)

    def test_rejects_unknown_experiment(self):
        with self.assertRaisesRegex(ValueError, "정의되지 않은 후보 모델"):
            build_pipeline(self.config, "unknown_model")


if __name__ == "__main__":
    unittest.main()
