# ==========================================
# 팀원 제안 후보의 합성 입력 테스트
# - 실제 SECOM 학습·Test 평가 없이 가중치와 Pipeline 구성을 확인
# - 작은 문턱을 임의로 제외하지 않는지 검사
# ==========================================
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.modeling_config import load_modeling_config
from src.sensor_ml.experiments.lightgbm_gain_candidate import (
    FoldGainImportance, build_gain_candidate, recall_scenarios,
)


class GainCandidateTest(unittest.TestCase):
    def test_weight_uses_current_labels(self):
        """-1/1 label 합계 대신 fit별 정상/불량 건수로 가중치를 계산한다."""
        with patch("lightgbm.LGBMClassifier") as model:
            estimator = FoldGainImportance()
            estimator.fit(pd.DataFrame({"sensor_a": [1., 2., 3., 4.]}), [-1, -1, -1, 1])
            self.assertEqual(model.call_args.kwargs["scale_pos_weight"], 3.)
            self.assertEqual(model.return_value.fit.call_args.args[1].tolist(), [0, 0, 0, 1])
            estimator.fit(pd.DataFrame({"sensor_a": [1., 2.]}), [-1, 1])
            self.assertEqual(estimator.effective_scale_pos_weight_, 1.)
        with self.assertRaises(ValueError):
            FoldGainImportance().fit(pd.DataFrame({"sensor_a": [1.]}), [-1])

    def test_selection_is_inside_pipeline(self):
        """품질 필터와 선택기를 fold마다 독립적으로 복제할 수 있는지 확인한다."""
        pipeline = clone(build_gain_candidate(load_modeling_config(), n_jobs=2))
        self.assertIn("quality_filter", pipeline.named_steps)
        self.assertIn("selector", pipeline.named_steps)
        self.assertFalse(hasattr(pipeline.named_steps["selector"], "estimator_"))
        self.assertEqual(pipeline.named_steps["model"].class_weight_mode, "ratio")

    def test_small_threshold_is_preserved(self):
        """작은 문턱도 유효 후보로 유지하고 목표 미달은 별도로 기록한다."""
        table = pd.DataFrame({"threshold": [1e-7, .5], "recall": [.8, .2],
                              "precision": [.1, .5], "f1": [.18, .28]})
        result = recall_scenarios(table, targets=(.8, .9))
        self.assertEqual(result.iloc[0].threshold, 1e-7)
        self.assertEqual(result.iloc[1].status, "infeasible")

    def test_synthetic_oof_selects_per_fold(self):
        """합성 데이터에서 실제 두 fold를 실행해 선택 기록과 미학습 예측 수를 확인한다."""
        from src.sensor_ml.experiments.step8_threshold_oof import generate_oof_scores
        config = load_modeling_config()
        config = replace(config, experiment=replace(config.experiment,
                         cv=replace(config.experiment.cv, n_splits=2, n_repeats=1)))
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(40, 24)),
                             columns=[f"sensor_{i}" for i in range(24)])
        frame["label"] = [-1, 1] * 20
        frame["timestamp"] = "01/01/2008 00:00:00"
        result, _ = generate_oof_scores(frame, config, experiment_name="xgboost_lightgbm_gain_top20")
        self.assertTrue(result.oof_prediction_count.eq(1).all())
        selected = pd.DataFrame(result.attrs["selected_features"])
        self.assertEqual(selected.groupby("cv_fold").size().tolist(), [20, 20])


if __name__ == "__main__":
    unittest.main()
