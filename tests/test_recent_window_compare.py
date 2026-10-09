# ==========================================
# 최근 기간 시간 검증의 평가 행·학습 기간·fold 내부 선택 검증
# - 합성 데이터와 작은 트리만 사용하여 실제 SECOM 실험은 실행하지 않음
# - 같은 미래 행과 과거 경계, 전체·Top-K 센서 수를 확인
# ==========================================

import unittest
import numpy as np
import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID
from src.sensor_ml.experiments.recent_window_compare import compare_recent_windows, recent_training_rows, analyze_window_folds, compare_window_sensors


class RecentWindowTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(np.random.default_rng(42).normal(size=(60, 30)),
                                  columns=[f"sensor_{i}" for i in range(30)])
        self.frame["label"] = [-1, 1] * 30
        self.frame["timestamp"] = pd.date_range("2008-01-01", periods=60).strftime("%d/%m/%Y %H:%M:%S")
        self.frame[SOURCE_ROW_ID] = range(60)
        self.frame[SPLIT_ROLE] = "train"
        self.frame[PROTOCOL_ID] = "time:recent-toy"

    def test_recent_window_uses_past_end_only(self):
        recent = recent_training_rows(self.frame, np.arange(30), MODELING_CONFIG.dataset, 10)
        self.assertEqual(recent[SOURCE_ROW_ID].tolist(), list(range(19, 30)))
        self.assertEqual(len(recent_training_rows(self.frame, np.arange(30), MODELING_CONFIG.dataset, 0)), 30)

    def test_models_evaluate_same_rows_and_select_inside_window(self):
        original = self.frame.copy(deep=True)
        tables = compare_recent_windows(self.frame, MODELING_CONFIG, windows=(0, 10), n_splits=2, n_estimators=2)
        rows = tables["fold_results"]
        self.assertEqual(len(rows), 8)
        self.assertTrue(rows.status.eq("evaluated").all())
        self.assertEqual(set(rows.query("model_name == 'm3_all'").selected_feature_count), {30})
        self.assertEqual(set(rows.query("model_name == 's0_m3_topk'").selected_feature_count), {20})
        self.assertTrue(rows.threshold.eq(MODELING_CONFIG.experiment.default_threshold).all())
        self.assertTrue((pd.to_datetime(rows.train_end) < pd.to_datetime(rows.evaluation_start)).all())
        predictions = tables["predictions"]
        for _, group in predictions.groupby("outer_fold"):
            ids = [set(part.source_row_id) for _, part in group.groupby(["model_name", "window_days"])]
            self.assertTrue(all(value == ids[0] for value in ids))
        self.assertTrue(rows.query("window_days == 10").window_truncated.all())
        pd.testing.assert_frame_equal(self.frame, original)

    def test_invalid_windows_and_test_role_are_rejected(self):
        for windows in ((), (0, 0), (-1,), (True,), (1.5,)):
            with self.assertRaises(ValueError):
                compare_recent_windows(self.frame, MODELING_CONFIG, windows=windows, n_estimators=2)
        with self.assertRaises(ValueError):
            compare_recent_windows(self.frame.assign(**{SPLIT_ROLE: "test"}), MODELING_CONFIG, n_estimators=2)

    # 저장된 로그 분석은 학습 없이 AP 증감과 미평가 구간을 구분한다.
    def test_saved_fold_analysis_and_missing_results(self):
        common = {"model_name": "toy", "outer_fold": 1, "status": "evaluated",
                  "train_end": "2008-02-01", "evaluation_start": "2008-02-02",
                  "evaluation_end": "2008-02-10", "evaluation_samples": 10, "evaluation_fail": 2}
        rows = pd.DataFrame([
            {**common, "window_days": 0, "train_start": "2008-01-01", "train_samples": 100,
             "train_fail": 10, "average_precision": 0.2, "roc_auc": 0.7},
            {**common, "window_days": 10, "train_start": "2008-01-22", "train_samples": 30,
             "train_fail": 3, "average_precision": 0.1, "roc_auc": 0.6},
            {**common, "window_days": 60, "train_start": "2008-01-01", "train_samples": 100,
             "train_fail": 10, "average_precision": 0.2, "roc_auc": 0.7},
        ])
        compared = analyze_window_folds(rows)
        self.assertEqual(compared.removed_train_fail.tolist(), [7, 0])
        self.assertEqual(compared.same_training_period.tolist(), [False, True])
        self.assertEqual(compared.ap_relative_change.tolist(), [-0.5, 0.0])
        skipped = rows.copy()
        skipped.loc[1, "status"] = "skipped"
        self.assertTrue(pd.isna(analyze_window_folds(skipped).ap_delta.iloc[0]))
        changed = rows.copy()
        changed.loc[1, "evaluation_fail"] = 3
        with self.assertRaisesRegex(ValueError, "미래 평가"):
            analyze_window_folds(changed)
        with self.assertRaises(ValueError):
            analyze_window_folds(pd.concat([rows, rows.iloc[[0]]]))

    # 저장된 센서명으로 교집합·추가·제외를 계산하고 누락된 로그를 거부한다.
    def test_sensor_overlap_and_changed_membership(self):
        paired = pd.DataFrame({"model_name": ["toy", "toy"], "outer_fold": [1, 1],
                               "window_days": [30, 60], "status": ["evaluated"] * 2,
                               "baseline_status": ["evaluated"] * 2, "selected_feature_count": [2, 2],
                               "baseline_selected_feature_count": [2, 2], "ap_delta": [-0.1, 0.0],
                               "removed_train_fail": [7, 0]})
        selected = pd.DataFrame({"model_name": ["toy"] * 6, "outer_fold": [1] * 6,
                                 "window_days": [0, 0, 30, 30, 60, 60],
                                 "feature": ["a", "b", "b", "c", "a", "b"]})
        summary, changes = compare_window_sensors(selected, paired)
        self.assertEqual(summary.common_count.tolist(), [1, 2])
        self.assertEqual(summary.same_sensor_set.tolist(), [False, True])
        self.assertAlmostEqual(summary.jaccard.iloc[0], 1 / 3)
        self.assertEqual(changes.query("window_days == 30 and sensor_status == 'removed'").feature.tolist(), ["a"])
        self.assertEqual(changes.query("window_days == 30 and sensor_status == 'added'").feature.tolist(), ["c"])
        with self.assertRaises(ValueError):
            compare_window_sensors(selected.iloc[1:], paired)
        with self.assertRaises(ValueError):
            compare_window_sensors(pd.concat([selected, selected.iloc[[0]]]), paired)

    # ==========================================
    # 각 fold의 과거 센서만 고정하며 0일 경로의 예측 정합성 확인
    # - 최근 기간의 sensor 목록은 해당 fold의 전체 과거 목록과 같아야 함
    # - 선택에 사용한 과거 표본 수와 분류기 학습 표본 수를 구분
    # ==========================================
    def test_fixed_sensor_control_preserves_columns_and_baseline(self):
        tables = compare_recent_windows(self.frame, MODELING_CONFIG, windows=(0, 10), n_splits=2,
                                         n_estimators=2, fixed_sensor_control=True)
        rows = tables["fold_results"]
        self.assertEqual(set(rows.model_name), {"s0_m3_topk", "s0_m3_fixed_sensors"})
        self.assertTrue(rows.selected_feature_count.eq(20).all())
        fixed = rows[rows.model_name.eq("s0_m3_fixed_sensors")]
        self.assertTrue((fixed.selection_train_samples >= fixed.train_samples).all())
        selected = tables["selected_features"]
        predictions = tables["predictions"]
        for fold in (1, 2):
            sensor_rows = selected.query("outer_fold == @fold and model_name == 's0_m3_fixed_sensors'")
            full = sensor_rows.query("window_days == 0").feature.tolist()
            self.assertEqual(full, sensor_rows.query("window_days == 10").feature.tolist())
            baseline = predictions.query("outer_fold == @fold and window_days == 0")
            original = baseline.query("model_name == 's0_m3_topk'").positive_score.to_numpy()
            control = baseline.query("model_name == 's0_m3_fixed_sensors'").positive_score.to_numpy()
            np.testing.assert_allclose(original, control, rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            compare_recent_windows(self.frame, MODELING_CONFIG, windows=(10,), n_estimators=2, fixed_sensor_control=True)


if __name__ == "__main__":
    unittest.main()
