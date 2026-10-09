# ==========================================
# 시간순 내부 검증의 경계·OOF 범위·누수 방어 테스트
# - 동일 timestamp 보호 및 그룹 경계·Test 입력 차단 확인
# - 초기 미예측 구간 제외와 학습별 실제 센서 제거 기록 검증
# - 외부 평가 점수가 내부 OOF threshold 선택에 쓰이지 않는지 확인
# ==========================================

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.modeling_models import build_candidate_pipelines
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID
from src.sensor_ml.experiments.temporal_validation import temporal_folds, compare_temporal


class TemporalValidationTest(unittest.TestCase):
    config = replace(MODELING_CONFIG, experiment=replace(
        MODELING_CONFIG.experiment, cv=replace(MODELING_CONFIG.experiment.cv, n_splits=2, n_repeats=1)))

    @staticmethod
    def _frame():
        # 각 timestamp에 정상·Fail 두 행을 두어 모든 작은 시간 구간에서 학습 가능하게 한다.
        return pd.DataFrame({
            "sensor_value": [float(i % 2) + i * 0.01 for i in range(32)],
            "sensor_empty": [np.nan] * 32,
            "label": [-1, 1] * 16,
            "timestamp": [date.strftime("%d/%m/%Y %H:%M:%S")
                          for date in pd.date_range("2008-01-01", periods=16) for _ in range(2)],
            SOURCE_ROW_ID: np.arange(100, 132), SPLIT_ROLE: "train", PROTOCOL_ID: "time:toy",
        })

    def test_preserves_timestamp_and_chronological_boundaries(self):
        frame = self._frame().iloc[::-1]
        for fit_i, eval_i in temporal_folds(frame, self.config.dataset, 3):
            train, evaluation = frame.iloc[fit_i], frame.iloc[eval_i]
            fit_time = pd.to_datetime(train.timestamp, format=self.config.dataset.timestamp_format)
            eval_time = pd.to_datetime(evaluation.timestamp, format=self.config.dataset.timestamp_format)
            self.assertLess(fit_time.max(), eval_time.min())
            self.assertFalse(set(train.timestamp) & set(evaluation.timestamp))

    def test_rf_variants_use_requested_candidates(self):
        """RF 가중치·leaf 후보가 같은 시간 분할·OOF 경로로 실행된다."""
        names = ("random_forest", "random_forest_balanced",
                 "random_forest_leaf3", "random_forest_leaf5")
        # 실제 데이터 대신 합성 데이터와 트리 2개로 연결 경로만 확인한다.
        def small_candidates(config, n_jobs=1, *, names=None):
            pipelines = build_candidate_pipelines(config, n_jobs=n_jobs, names=names)
            for pipeline in pipelines.values():
                pipeline.set_params(model__n_estimators=2)
            return pipelines

        with patch("src.sensor_ml.experiments.temporal_validation.build_candidate_pipelines",
                   side_effect=small_candidates) as build:
            results = compare_temporal(self._frame(), self.config, model_names=names,
                                       oof_modes=("temporal_oof",))
        self.assertEqual(build.call_args.kwargs["names"], names)
        self.assertEqual(set(results["summary"].model_name), set(names))
        self.assertEqual(len(results["fold_results"]), len(names) * 3 * 2)
        self.assertEqual(len(results["quality_filter"]), len(names) * 3 * 3)

    def test_rejects_group_crossing_time_boundary(self):
        frame = self._frame().assign(machine="same_group")
        dataset = replace(self.config.dataset, group_columns=("machine",))
        with self.assertRaisesRegex(ValueError, "동일한 그룹"):
            temporal_folds(frame, dataset, 3)

    def test_rejects_test_role_and_invalid_timestamp(self):
        with self.assertRaisesRegex(ValueError, "train 역할"):
            compare_temporal(self._frame().assign(**{SPLIT_ROLE: "test"}), self.config)
        frame = self._frame()
        frame.loc[0, "timestamp"] = "invalid"
        with self.assertRaisesRegex(ValueError, "timestamp 파싱"):
            temporal_folds(frame, self.config.dataset, 3)

    def test_records_oof_coverage_and_fold_quality(self):
        results = compare_temporal(self._frame(), self.config,
                                   model_names=("logistic_regression_l1_balanced",))
        self.assertEqual(len(results["fold_results"]), 9)
        self.assertEqual(len(results["policy_selection"]), 6)
        policy_rows = results["policy_selection"]
        self.assertTrue(policy_rows.loc[policy_rows.policy_status == "infeasible", "threshold"].isna().all())
        coverage = results["oof_coverage"]
        temporal = coverage[coverage["mode"] == "temporal_oof"]
        stratified = coverage[coverage["mode"] == "stratified_oof"]
        self.assertTrue((temporal.excluded_initial_samples > 0).all())
        self.assertTrue(stratified.excluded_initial_samples.eq(0).all())
        oof = results["oof_predictions"]
        self.assertFalse(oof.duplicated(["outer_fold", "model_name", "mode", "source_row_id"]).any())
        for outer_fold in (1, 2, 3):
            past_ids = set(oof.loc[oof.outer_fold == outer_fold, "source_row_id"])
            future_ids = set(results["predictions"].loc[
                results["predictions"].outer_fold == outer_fold, "source_row_id"])
            self.assertFalse(past_ids & future_ids)
        quality = results["quality_filter"]
        self.assertTrue(quality.high_missing_removed_count.eq(1).all())
        self.assertTrue(quality.retained_feature_count.eq(1).all())
        chronological = quality[quality["mode"].isin(["outer_fit", "temporal_oof"])]
        self.assertTrue((pd.to_datetime(chronological.train_end)
                         < pd.to_datetime(chronological.evaluation_start)).all())
        # 저장한 후보 표의 F1 최대값과 실제 외부 평가 threshold가 일치하는지 확인한다.
        for _, row in results["fold_results"].iterrows():
            if row["mode"] == "default":
                continue
            table = results["threshold_compare"]
            table = table[(table.outer_fold == row.outer_fold) & (table["mode"] == row["mode"])]
            self.assertEqual(row.threshold, table.loc[table.f1.idxmax(), "threshold"])
        self.assertTrue(results["summary"].true_positive.add(results["summary"].false_negative).eq(12).all())

    def test_future_labels_do_not_change_threshold_selection(self):
        # 마지막 평가 구간의 label을 바꿔도 과거 학습·OOF·threshold는 같아야 한다.
        original = self._frame()
        changed = original.copy()
        changed.loc[24:, "label"] *= -1
        options = {"model_names": ("logistic_regression_l1_balanced",)}
        before = compare_temporal(original, self.config, **options)
        after = compare_temporal(changed, self.config, **options)
        pd.testing.assert_frame_equal(before["threshold_compare"], after["threshold_compare"])
        pd.testing.assert_frame_equal(before["oof_predictions"], after["oof_predictions"])
        # 정책 후보의 선택 여부·값도 미래 label 변경과 무관해야 한다.
        columns = ["outer_fold", "model_name", "mode", "policy_status", "threshold"]
        pd.testing.assert_frame_equal(before["policy_selection"][columns],
                                      after["policy_selection"][columns])


if __name__ == "__main__":
    unittest.main()
