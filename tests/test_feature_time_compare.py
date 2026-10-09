# ==========================================
# 시간순 축소 비교의 센서 수·선택 근거·빈도 분모 검증
# - RF 중요도와 LightGBM 중요도를 구분하고 후보를 최종으로 표시하지 않음
# - XGBoost 실험을 지정하면 해당 모델만 같은 시간 검증 코어에서 실행
# ==========================================

import unittest
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from src.sensor_ml.experiments.feature_time_compare import compare_feature_time, selection_rows, build_feature_pipelines
from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID
from src.sensor_ml.experiments.time_weight_compare import DEFAULT_PRESET, build_weight_candidates


class FeatureTimeComparisonTest(unittest.TestCase):
    # ==========================================
    # gain Top-K 최종 분류기의 깊이·규제 변경 범위 검증
    # - 단일·반복 선택기의 파라미터는 변경하지 않음
    # - 전체 센서 기준 모델도 유지해 기존 결과와 비교 가능
    # ==========================================
    def test_gain_topk_classifier_overrides(self):
        """gain 선택기를 보존하고 최종 분류기의 지정 값만 변경한다."""
        names = ("xgboost_all", "xgboost_top_20")
        for repeats in (0, 5):
            options = dict(experiment_names=names, xgb_weight_mode="ratio",
                           xgb_selector_weight_mode="ratio", xgb_stability_repeats=repeats)
            base = build_feature_pipelines(MODELING_CONFIG, **options)
            changed = build_feature_pipelines(MODELING_CONFIG, **options,
                                              topk_xgb_max_depth=2, topk_xgb_reg_lambda=5)
            self.assertEqual(base[names[0]].named_steps["model"].get_params(),
                             changed[names[0]].named_steps["model"].get_params())
            # 선택용 XGBoost의 깊이 3과 규제 1은 그대로 유지한다.
            for key, value in base[names[1]].named_steps["selector"].get_params().items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    actual = changed[names[1]].named_steps["selector"].get_params()[key]
                    # 결측 표시용 NaN은 값의 동일성 대신 둘 다 NaN인지 확인한다.
                    if isinstance(value, float) and np.isnan(value):
                        self.assertTrue(np.isnan(actual))
                    else:
                        self.assertEqual(value, actual)
            expected = base[names[1]].named_steps["model"].get_params()
            expected.update(max_depth=2, reg_lambda=5)
            self.assertEqual(expected, changed[names[1]].named_steps["model"].get_params())

    def test_gain_weight_changes_only_selector_and_is_fold_local(self):
        """선택기 가중치만 변경하고 실제 fit별 계산과 Top-20 선택을 확인한다."""
        names = ("xgboost_all", "xgboost_top_20", "xgboost_rf_top_20")
        base = build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                       xgb_weight_mode="ratio")
        weighted = build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                           xgb_weight_mode="ratio", xgb_selector_weight_mode="ratio")
        for name in names:
            self.assertEqual(base[name].named_steps["model"].get_params(),
                             weighted[name].named_steps["model"].get_params())
        self.assertEqual(base[names[2]].named_steps["selector"].estimator.get_params(),
                         weighted[names[2]].named_steps["selector"].estimator.get_params())
        expected = base[names[1]].named_steps["selector"].estimator.get_params()
        expected["class_weight_mode"] = "ratio"
        self.assertEqual(weighted[names[1]].named_steps["selector"].estimator.get_params(), expected)
        # 불균형 합성 label로 무가중치 1과 실제 정상/불량 비율을 구분한다.
        frame = self.frame()
        frame["label"] = [-1, -1, -1, 1] * 12
        results = compare_feature_time(frame, MODELING_CONFIG, n_estimators=3,
                                       experiment_names=names[:2], xgb_weight_mode="ratio",
                                       xgb_selector_weight_mode="ratio")
        weights = results["fit_weights"].query("model_name == 'xgboost_top_20'")
        self.assertEqual(len(weights), 9)
        self.assertTrue(weights.selector_class_weight_mode.eq("ratio").all())
        np.testing.assert_allclose(weights.selector_effective_scale_pos_weight,
                                   weights.train_negative / weights.train_positive)
        self.assertTrue(weights.selector_effective_scale_pos_weight.gt(1).all())
        top = results["selected_features"].query("model_name == 'xgboost_top_20'")
        self.assertTrue(top.groupby("fit_id").feature.nunique().eq(20).all())
        self.assertTrue(top.selection_method.eq("xgboost_gain").all())
        for options in ({"xgb_selector_weight_mode": "invalid"},
                        {"xgb_selector_weight_mode": "ratio"},
                        {"xgb_selector_weight_mode": "ratio", "experiment_names": names[2:]},
                        {"xgb_selector_weight_mode": "ratio", "experiment_names": names[:1]}):
            with self.assertRaises(ValueError):
                build_feature_pipelines(MODELING_CONFIG, **options)

    def test_p0_weight_matches_baseline_and_stays_in_folds(self):
        """P0 설정을 전체·Top-20에 동일 적용하고 실제 학습별 가중치를 확인한다."""
        names = ("xgboost_all", "xgboost_rf_top_20")
        preset = json.loads(DEFAULT_PRESET.read_text(encoding="utf-8"))
        reference = build_weight_candidates(MODELING_CONFIG, preset)["v2_m0_ratio"]
        chosen = build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                        xgb_weight_mode="ratio")
        for pipeline in chosen.values():
            self.assertEqual(pipeline.named_steps["model"].get_params(),
                             reference.named_steps["model"].get_params())
        # RF 센서 선택용 모델은 S0의 무가중치·leaf=1을 보존한다.
        selector = chosen[names[1]].named_steps["selector"].estimator
        self.assertIsNone(selector.class_weight)
        self.assertEqual(selector.min_samples_leaf, 1)
        results = compare_feature_time(self.frame(), MODELING_CONFIG, n_estimators=3,
                                       experiment_names=names, xgb_weight_mode="ratio")
        weights = results["fit_weights"]
        self.assertEqual(len(weights), 18)
        self.assertTrue(weights.class_weight_mode.eq("ratio").all())
        np.testing.assert_allclose(weights.effective_scale_pos_weight,
                                   weights.train_negative / weights.train_positive)
        selected = results["selected_features"]
        top = selected[selected.model_name.eq(names[1])]
        self.assertTrue(top.groupby("fit_id").feature.nunique().eq(20).all())
        self.assertTrue(top.selection_method.eq("rf_importance").all())
        for options in ({"xgb_weight_mode": "invalid"}, {"xgb_weight_mode": "ratio"},
                        {"xgb_weight_mode": "ratio", "experiment_names": ("lightgbm_all",)}):
            with self.assertRaises(ValueError):
                build_feature_pipelines(MODELING_CONFIG, **options)

    # ==========================================
    # M2 단독 정규화·M3 깊이와 정규화 결합의 적용 범위 검증
    # - 전체 센서 M0와 RF 선택기는 보존하고 최종 분류기만 변경
    # - 작은 합성 데이터에서 시간순 내부 OOF까지 설정 전달 확인
    # ==========================================
    def test_m2_m3_options_and_temporal_records(self):
        names = ("xgboost_all", "xgboost_rf_top_20")
        base = build_feature_pipelines(MODELING_CONFIG, experiment_names=names)
        for depth in (None, 2):
            with self.subTest(depth=depth):
                chosen = build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                                topk_xgb_max_depth=depth, topk_xgb_reg_lambda=5)
                self.assertEqual(chosen[names[0]].named_steps["model"].get_params(),
                                 base[names[0]].named_steps["model"].get_params())
                self.assertEqual(chosen[names[1]].named_steps["selector"].estimator.get_params(),
                                 base[names[1]].named_steps["selector"].estimator.get_params())
                expected = base[names[1]].named_steps["model"].get_params()
                expected["reg_lambda"] = 5
                if depth is not None:
                    expected["max_depth"] = depth
                self.assertEqual(chosen[names[1]].named_steps["model"].get_params(), expected)
                results = compare_feature_time(self.frame(), MODELING_CONFIG, n_estimators=3,
                                               experiment_names=names, topk_xgb_max_depth=depth,
                                               topk_xgb_reg_lambda=5)
                for _, record in results["quality_filter"].iterrows():
                    params = record.model_parameters
                    params = json.loads(params) if isinstance(params, str) else params
                    is_top = record.model_name == names[1]
                    self.assertEqual(params["reg_lambda"], 5 if is_top else 1)
                    self.assertEqual(params["max_depth"], depth if is_top and depth is not None else 3)
        for value in (True, -1, float("nan"), float("inf"), "5"):
            with self.assertRaisesRegex(ValueError, "유한한 0 이상"):
                build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                        topk_xgb_reg_lambda=value)
        # 정규화 0도 유효한 입력이며 문자열·결측값으로 오인하지 않는다.
        chosen = build_feature_pipelines(MODELING_CONFIG, experiment_names=names, topk_xgb_reg_lambda=0)
        self.assertEqual(chosen[names[1]].named_steps["model"].reg_lambda, 0)
        with self.assertRaisesRegex(ValueError, "실험이 없습니다"):
            build_feature_pipelines(MODELING_CONFIG, experiment_names=(names[0],), topk_xgb_reg_lambda=5)

    # ==========================================
    # M1 깊이 옵션이 RF 선택기와 전체 센서 M0를 바꾸지 않는지 확인
    # - 옵션 생략은 기존 동작, 깊이 2는 축소 경로만 변경
    # - 잘못된 깊이와 적용 대상 없는 요청은 학습 전에 차단
    # ==========================================
    def test_m1_changes_only_topk_classifier(self):
        names = ("xgboost_all", "xgboost_rf_top_20")
        base = build_feature_pipelines(MODELING_CONFIG, experiment_names=names)
        chosen = build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                        topk_xgb_max_depth=2)
        self.assertEqual(chosen[names[0]].named_steps["model"].get_params(),
                         base[names[0]].named_steps["model"].get_params())
        self.assertEqual(chosen[names[1]].named_steps["selector"].estimator.get_params(),
                         base[names[1]].named_steps["selector"].estimator.get_params())
        expected = base[names[1]].named_steps["model"].get_params()
        expected["max_depth"] = 2
        self.assertEqual(chosen[names[1]].named_steps["model"].get_params(), expected)
        for value in (True, 0, -1, 2.5):
            with self.assertRaisesRegex(ValueError, "양의 정수"):
                build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                        topk_xgb_max_depth=value)
        with self.assertRaisesRegex(ValueError, "실험이 없습니다"):
            build_feature_pipelines(MODELING_CONFIG, experiment_names=("xgboost_all",),
                                    topk_xgb_max_depth=2)

    def test_rf_limits_leave_xgboost_unchanged(self):
        # S1·S2 모두 선택용 RF만 달라지고 최종 분류기는 기준과 같아야 한다.
        names = ("xgboost_all", "xgboost_rf_top_20")
        base = build_feature_pipelines(MODELING_CONFIG, experiment_names=names)
        for depth in (None, 8):
            chosen = build_feature_pipelines(MODELING_CONFIG, experiment_names=names,
                                            rf_min_samples_leaf=2, rf_max_depth=depth)
            selector = chosen[names[1]].named_steps["selector"].estimator
            self.assertEqual(selector.min_samples_leaf, 2)
            self.assertEqual(selector.max_depth, depth)
            for name in names:
                self.assertEqual(chosen[name].named_steps["model"].get_params(),
                                 base[name].named_steps["model"].get_params())
        for value in (True, 0, -1):
            with self.assertRaises(ValueError):
                build_feature_pipelines(MODELING_CONFIG, experiment_names=names, rf_min_samples_leaf=value)

    @staticmethod
    def frame():
        # 작은 데이터에서도 Top-50 선택이 가능하도록 변동 센서 60개를 만든다.
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(48, 60)),
                             columns=[f"sensor_{i}" for i in range(60)])
        frame["sensor_empty"] = np.nan
        frame["label"] = [-1, 1] * 24
        frame["timestamp"] = [date.strftime("%d/%m/%Y %H:%M:%S")
                              for date in pd.date_range("2008-01-01", periods=24) for _ in range(2)]
        frame[SOURCE_ROW_ID] = np.arange(48)
        frame[SPLIT_ROLE] = "train"
        frame[PROTOCOL_ID] = "time:feature-toy"
        return frame

    def test_selection_names_and_counts_match_pipeline(self):
        frame = self.frame()
        features = frame.filter(regex="^sensor_")
        for name, pipeline in build_feature_pipelines(MODELING_CONFIG, n_estimators=3).items():
            pipeline.fit(features, frame.label)
            rows = selection_rows(pipeline, {"model_name": name})
            expected = 60 if name == "lightgbm_all" else int(name.rsplit("_", 1)[1])
            self.assertEqual(len(rows), expected)
            self.assertNotIn("sensor_empty", [row["feature"] for row in rows])
            self.assertEqual([row["selection_rank"] for row in rows], list(range(1, expected + 1)))
            self.assertTrue(all(row["feature_space"] == "raw_sensor" for row in rows))
            if "rf_top" in name:
                self.assertTrue(all(row["selection_method"] == "rf_importance" for row in rows))

    def test_temporal_records_frequency_and_nonfinal_candidates(self):
        results = compare_feature_time(self.frame(), MODELING_CONFIG, n_estimators=3)
        self.assertEqual(len(results["fold_results"]), 30)
        self.assertEqual(len(results["policy_selection"]), 15)
        selected = results["selected_features"]
        frequency = results["feature_frequency"]
        self.assertTrue(frequency.selection_frequency.between(0, 1).all())
        self.assertTrue(frequency.loc[frequency.fit_role == "outer_fit", "total_fit_count"].eq(3).all())
        self.assertTrue(frequency.loc[frequency.fit_role == "inner_oof", "total_fit_count"].eq(6).all())
        self.assertFalse(results["candidate_features"].is_final.any())
        self.assertEqual(len(results["quality_filter"]), 45)
        for (name, fit_id), group in selected.groupby(["model_name", "fit_id"]):
            expected = 60 if name == "lightgbm_all" else int(name.rsplit("_", 1)[1])
            self.assertEqual(group.feature.nunique(), expected)

    # ==========================================
    # XGBoost 전체·RF Top-20의 시간순 검증 연결
    # - 최종 분류기 설정과 선택기 병렬·트리 수 전달을 확인
    # - 과거 학습에만 센서 선택을 적용하고 외부·내부 기록 수를 확인
    # ==========================================
    def test_xgboost_time_comparison_and_selection(self):
        names = ("xgboost_all", "xgboost_rf_top_20")
        pipelines = build_feature_pipelines(
            MODELING_CONFIG, n_estimators=3, experiment_names=names,
        )
        self.assertEqual(set(pipelines), set(names))
        self.assertEqual(pipelines[names[0]].named_steps["model"].get_params(),
                         pipelines[names[1]].named_steps["model"].get_params())
        self.assertEqual(pipelines[names[1]].named_steps["selector"].estimator.n_estimators, 3)
        results = compare_feature_time(
            self.frame(), MODELING_CONFIG, n_estimators=3, experiment_names=names,
            topk_xgb_max_depth=2,
        )
        self.assertEqual(set(results["summary"].model_name), set(names))
        self.assertEqual(len(results["fold_results"]), 12)
        self.assertEqual(len(results["policy_selection"]), 6)
        selected = results["selected_features"]
        top = selected[selected.model_name.eq(names[1])]
        self.assertTrue(top.groupby("fit_id").feature.nunique().eq(20).all())
        self.assertTrue(top.selection_method.eq("rf_importance").all())
        self.assertFalse(results["candidate_features"].is_final.any())
        chronological = results["quality_filter"]
        # 시간순 외부 학습과 내부 OOF에서도 M1 설정이 실제 학습 기록에 남아야 한다.
        for _, record in chronological.iterrows():
            params = record.model_parameters
            params = json.loads(params) if isinstance(params, str) else params
            self.assertEqual(params["max_depth"], 3 if record.model_name == names[0] else 2)
        self.assertTrue((pd.to_datetime(chronological.train_end)
                         < pd.to_datetime(chronological.evaluation_start)).all())

    def test_xgboost_gain_method_and_invalid_names(self):
        name = "xgboost_top_20"
        pipeline = build_feature_pipelines(
            MODELING_CONFIG, n_estimators=3, experiment_names=(name,),
        )[name]
        frame = self.frame()
        pipeline.fit(frame.filter(regex="^sensor_"), frame.label)
        rows = selection_rows(pipeline, {"model_name": name})
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["selection_method"] == "xgboost_gain" for row in rows))
        for names in ((), (name, name), ("unknown",)):
            with self.assertRaises(ValueError):
                build_feature_pipelines(MODELING_CONFIG, experiment_names=names)


if __name__ == "__main__":
    unittest.main()
