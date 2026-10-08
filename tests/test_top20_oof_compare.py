# ==========================================
# M0·M3 단일 OOF 비교의 설정·샘플 정합성과 시나리오 검증
# - 작은 합성 데이터·트리 3개로 코드 연결만 확인
# - 실제 SECOM 학습·외부 Validation·최종 Test는 실행하지 않음
# ==========================================

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.modeling_models import build_pipeline
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID
from src.experiments.top20_oof_compare import compare_top20_oof, scenario_rows


class Top20OOFTest(unittest.TestCase):
    @staticmethod
    def frame():
        # 빈 센서는 각 학습 fold의 품질 필터에서 제거되어야 한다.
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(40, 30)),
                             columns=[f"sensor_{i}" for i in range(30)])
        frame["sensor_empty"] = np.nan
        frame["label"] = [-1, 1] * 20
        frame["timestamp"] = pd.date_range("2008-01-01", periods=40).strftime("%d/%m/%Y %H:%M:%S")
        frame[SOURCE_ROW_ID] = np.arange(40)
        frame[SPLIT_ROLE] = "train"
        frame[PROTOCOL_ID] = "time:oof-toy"
        return frame

    # ==========================================
    # 각 행의 미학습 예측 한 번과 M0·M3의 동일 fold·센서 선택 확인
    # - 최종 분류기만 깊이·정규화가 바뀌고 RF는 동일
    # - 같은 Recall 목표에서 최소 양성 비율 행이 선택되는지 대조
    # ==========================================
    def test_single_oof_alignment_and_parameters(self):
        config = replace(MODELING_CONFIG, experiment=replace(MODELING_CONFIG.experiment,
                         cv=replace(MODELING_CONFIG.experiment.cv, n_splits=2)))

        def small_pipeline(config, name, *, n_jobs=1):
            pipeline = build_pipeline(config, name, n_jobs=n_jobs)
            pipeline.set_params(model__n_estimators=3, selector__estimator__n_estimators=3)
            return pipeline

        with patch("src.experiments.top20_oof_compare.build_pipeline", side_effect=small_pipeline):
            tables, records = compare_top20_oof(self.frame(), config)
        oof = tables["oof_predictions"]
        self.assertEqual(len(oof), 80)
        self.assertTrue(oof.oof_prediction_count.eq(1).all())
        self.assertTrue(oof.oof_score_method.eq("single").all())
        m0 = oof[oof.model_variant.eq("M0")].reset_index(drop=True)
        m3 = oof[oof.model_variant.eq("M3")].reset_index(drop=True)
        pd.testing.assert_frame_equal(m0[["source_row_id", "cv_fold", "label"]],
                                      m3[["source_row_id", "cv_fold", "label"]])
        self.assertEqual(records[0]["model_parameters"]["max_depth"], 3)
        self.assertEqual(records[0]["model_parameters"]["reg_lambda"], 1)
        self.assertEqual(records[1]["model_parameters"]["max_depth"], 2)
        self.assertEqual(records[1]["model_parameters"]["reg_lambda"], 5)
        self.assertEqual(records[0]["rf_selector_parameters"], records[1]["rf_selector_parameters"])
        selected = tables["selected_features"]
        self.assertTrue(selected.groupby(["model_variant", "cv_fold"]).size().eq(20).all())
        self.assertNotIn("sensor_empty", selected.feature.tolist())
        for fold in (1, 2):
            self.assertEqual(set(selected.query("model_variant == 'M0' and cv_fold == @fold").feature),
                             set(selected.query("model_variant == 'M3' and cv_fold == @fold").feature))
        self.assertEqual(len(tables["scenario_compare"]), 8)
        self.assertFalse(tables["scenario_compare"].is_final_threshold.any())
        for _, row in tables["scenario_compare"].query("scenario == 'recall_target'").iterrows():
            candidates = tables["threshold_compare"]
            candidates = candidates[(candidates.model_variant == row.model_variant)
                                    & (candidates.recall >= row.target)]
            self.assertEqual(row.reinspection_ratio, candidates.reinspection_ratio.min())

    def test_scenario_ties_and_infeasible(self):
        # 목표 미달은 후보 없음으로 남기고 진단 최고값으로 바꾸지 않는다.
        table = pd.DataFrame({"threshold": [0.1, 0.2, 0.3], "recall": [0.8, 0.8, 0.5],
                              "reinspection_ratio": [0.4, 0.4, 0.1]})
        rows = scenario_rows(table, (0.8, 0.9), 0.2)
        self.assertEqual(rows[0]["threshold"], 0.2)
        self.assertEqual(rows[1]["status"], "infeasible")
        self.assertIsNone(rows[1]["threshold"])
        self.assertEqual(rows[2]["recall"], 0.5)

    def test_rejects_invalid_scenario_and_evaluation_rows(self):
        for targets in ((), (0.7, 0.7), (float("nan"),), (True,), (1.1,)):
            with self.assertRaises(ValueError):
                compare_top20_oof(self.frame(), MODELING_CONFIG, recall_targets=targets)
        with self.assertRaises(ValueError):
            compare_top20_oof(self.frame(), MODELING_CONFIG, alarm_limit=-0.1)
        with self.assertRaises(ValueError):
            compare_top20_oof(self.frame().assign(**{SPLIT_ROLE: "test"}), MODELING_CONFIG)


if __name__ == "__main__":
    unittest.main()
