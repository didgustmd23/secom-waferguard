# ==========================================
# 이상 탐지 정상 전용 학습·점수 방향·시간 누수 방어 검증
# - 모델·전처리에 불량이 섞이지 않는지 확인
# - 미래 label이 OOF와 선택 threshold에 영향을 주지 않는지 확인
# ==========================================

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from src.anomaly_compare import fit_detector, anomaly_scores, compare_anomalies
from src.modeling_config import MODELING_CONFIG, ThresholdPolicy
from src.modeling_metrics import evaluate_binary_scores, evaluate_anomaly_scores
from src.threshold_policy import select_policy_threshold
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID


class AnomalyComparisonTest(unittest.TestCase):
    config = replace(MODELING_CONFIG, threshold_policy=ThresholdPolicy())

    @staticmethod
    def frame():
        return pd.DataFrame({"sensor_a": np.arange(48) * 0.1,
            "sensor_normal_constant": [2., 100.] * 24,
            "sensor_b": np.sin(np.arange(48)), "sensor_empty": [np.nan] * 48,
            "label": [-1, 1] * 24,
            "timestamp": [date.strftime("%d/%m/%Y %H:%M:%S")
                          for date in pd.date_range("2008-01-01", periods=24) for _ in range(2)],
            SOURCE_ROW_ID: np.arange(48), SPLIT_ROLE: "train", PROTOCOL_ID: "time:anomaly-toy"})

    def test_normal_only_fit_and_score_definition(self):
        frame = self.frame()
        for name in ("isolation_forest", "pca_reconstruction"):
            model = fit_detector(frame, self.config, name, n_estimators=10)
            self.assertIn("sensor_normal_constant", model.named_steps["quality_filter"].constant_features_)
            # 불량 행의 센서값 변경이 학습된 점수에 영향을 주지 않아야 한다.
            changed = frame.copy()
            changed.loc[changed.label == 1, "sensor_a"] = 1e9
            other = fit_detector(changed, self.config, name, n_estimators=10)
            scores = anomaly_scores(model, frame, self.config.dataset, name)
            np.testing.assert_allclose(scores, anomaly_scores(other, frame, self.config.dataset, name))
            transformed = model[:-1].transform(frame[["sensor_a", "sensor_normal_constant", "sensor_b", "sensor_empty"]])
            estimator = model.named_steps["model"]
            if name == "isolation_forest":
                expected = -estimator.score_samples(transformed)
            else:
                expected = np.mean((transformed - estimator.inverse_transform(estimator.transform(transformed))) ** 2, axis=1)
            np.testing.assert_allclose(scores, expected)
        with self.assertRaisesRegex(ValueError, "정상 표본"):
            fit_detector(frame.loc[frame.label == 1], self.config, "isolation_forest")

    def test_raw_scores_and_probability_contract(self):
        metrics = evaluate_anomaly_scores([-1, 1], [-3, 8], positive_label=1,
                                          negative_label=-1, threshold=4)
        self.assertEqual(metrics.recall, 1)
        table = pd.DataFrame([metrics.to_dict()])
        self.assertEqual(select_policy_threshold(table, ThresholdPolicy(max_reinspection_ratio=0.5),
                                                score_kind="anomaly"), 4)
        with self.assertRaises(ValueError):
            evaluate_binary_scores([-1, 1], [-3, 8], positive_label=1, negative_label=-1, threshold=0.5)

    def test_temporal_oof_and_future_label_independence(self):
        frame = self.frame()
        before = compare_anomalies(frame, self.config, n_estimators=10)
        changed = frame.copy()
        changed.loc[36:, "label"] *= -1
        after = compare_anomalies(changed, self.config, n_estimators=10)
        pd.testing.assert_frame_equal(before["oof_predictions"], after["oof_predictions"])
        pd.testing.assert_frame_equal(before["threshold_compare"], after["threshold_compare"])
        selected = ["outer_fold", "model_name", "policy_status", "threshold"]
        pd.testing.assert_frame_equal(before["policy_selection"][selected], after["policy_selection"][selected])
        self.assertTrue(before["quality_filter"].fail_fit_samples.eq(0).all())
        self.assertTrue((pd.to_datetime(before["quality_filter"].fit_end) <
                         pd.to_datetime(before["quality_filter"].evaluation_start)).all())
        self.assertTrue(before["oof_coverage"].excluded_initial_samples.gt(0).all())
        for outer in (1, 2, 3):
            past = before["oof_predictions"].query("outer_fold == @outer")
            future = before["predictions"].query("outer_fold == @outer")
            self.assertFalse(set(past.source_row_id) & set(future.source_row_id))
        with self.assertRaisesRegex(ValueError, "train 역할"):
            compare_anomalies(frame.assign(**{SPLIT_ROLE: "test"}), self.config)


if __name__ == "__main__":
    unittest.main()
