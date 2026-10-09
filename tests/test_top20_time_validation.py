# ==========================================
# S0·M3 OOF 재사용과 시간 검증 연결 테스트
# - 합성 데이터만 사용하고 실제 SECOM 실험은 실행하지 않음
# - 모델 복원·정책 변경 허용·출처 및 문턱 불일치 차단 확인
# ==========================================

import copy
import json
import unittest
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.modeling_models import build_pipeline, configure_topk_xgb
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID
from src.sensor_ml.experiments.top20_time_validation import prepare_candidate, compare_all_sensors, save_sensor_comparison
from pathlib import Path
from tempfile import TemporaryDirectory
from src.sensor_ml.experiments.step7_time_validation import compare_time_validation


class Top20TimeValidationTest(unittest.TestCase):
    def setUp(self):
        # 특징은 30개, label은 두 종류이며 미래 검증 행은 다른 원본 ID를 가진다.
        self.config = replace(MODELING_CONFIG, threshold_policy=replace(
            MODELING_CONFIG.threshold_policy, min_recall=0.8, max_reinspection_ratio=0.6))
        self.train = pd.DataFrame(np.random.default_rng(42).normal(size=(20, 30)),
                                  columns=[f"sensor_{i}" for i in range(30)])
        self.train["label"] = [-1, 1] * 10
        self.train["timestamp"] = "01/01/2008 00:00:00"
        self.train[SOURCE_ROW_ID] = np.arange(20)
        self.train[SPLIT_ROLE] = "train"
        self.train[PROTOCOL_ID] = "time:toy-validation"
        self.experiment = "s0_m3_rf_top_20"
        self.oof = pd.DataFrame({"source_row_index": np.arange(20), "source_row_id": np.arange(20),
                                "label": self.train.label, "oof_positive_score": [0.1, 0.9] * 10,
                                "oof_prediction_count": 1, "model_variant": "M3",
                                "experiment": self.experiment, "training_protocol_id": "time:toy-validation",
                                "oof_score_method": "single", "threshold_use": "candidate"})
        pipeline = build_pipeline(self.config, "xgboost_rf_top_20")
        configure_topk_xgb({"xgboost_rf_top_20": pipeline}, max_depth=2, reg_lambda=5)
        # 트리 3개로 연결만 검사하여 테스트 시간을 줄인다.
        pipeline.set_params(model__n_estimators=3, selector__estimator__n_estimators=3)
        self.record = {"config": json.loads(json.dumps(asdict(self.config), default=str)),
                       "oof_score_method": "single", "n_repeats": 1,
                       "training_protocol_id": "time:toy-validation",
                       "models": [{"model_variant": "M3", "experiment": self.experiment,
                                   "model_parameters": pipeline.named_steps["model"].get_params(),
                                   "rf_selector_parameters": pipeline.named_steps["selector"].estimator.get_params()}]}

    def test_restores_parameters_and_evaluates_fixed_threshold(self):
        # 과거 정책만 달랐던 OOF는 재학습 없이 현재 정책에 사용할 수 있다.
        self.record["config"]["threshold_policy"]["min_recall"] = 0.9
        name, pipeline, table = prepare_candidate(self.train, self.oof, self.record, self.config,
                                                  threshold=0.9, n_jobs=1)
        self.assertEqual(pipeline.named_steps["model"].max_depth, 2)
        self.assertEqual(pipeline.named_steps["model"].reg_lambda, 5)
        self.assertEqual(pipeline.named_steps["selector"].estimator.min_samples_leaf, 1)
        validation = self.train.copy()
        validation["timestamp"] = "02/01/2008 00:00:00"
        validation[SOURCE_ROW_ID] += 20
        validation[SPLIT_ROLE] = "validation"
        result = compare_time_validation(self.train, validation, self.config, experiment_names=(name,),
                                         threshold=0.9, threshold_report=table,
                                         pipeline_templates={name: pipeline})
        self.assertEqual(result.threshold.tolist(), [0.9])
        self.assertEqual(len(result.attrs["selected_features"]), 20)
        self.assertEqual(result.selected_feature_count.tolist(), [20])
        predictions = pd.DataFrame(result.attrs["predictions"])
        self.assertEqual(len(predictions), len(validation))
        self.assertEqual(predictions.source_row_id.tolist(), validation[SOURCE_ROW_ID].tolist())
        self.assertEqual(predictions.predicted_positive.tolist(), (predictions.positive_score >= 0.9).tolist())

    def test_rejects_changed_config_and_threshold(self):
        changed = copy.deepcopy(self.record)
        changed["config"]["experiment"]["default_threshold"] = 0.4
        with self.assertRaisesRegex(ValueError, "실행 설정"):
            prepare_candidate(self.train, self.oof, changed, self.config, threshold=0.9)

    # ==========================================
    # 전체 모델은 RF 선택만 제거하고 M3 분류기 설정을 그대로 유지
    # - 같은 미래 구간에서 AP 등 지표와 행별 예측을 생성
    # - 문턱은 전체 모델의 운영값이 아니라 공통 진단값임을 기록
    # ==========================================
    def test_all_sensor_comparison_preserves_m3_parameters(self):
        name, pipeline, table = prepare_candidate(self.train, self.oof, self.record, self.config,
                                                  threshold=0.9, n_jobs=1)
        validation = self.train.copy()
        validation["timestamp"] = "02/01/2008 00:00:00"
        validation[SOURCE_ROW_ID] += 20
        validation[SPLIT_ROLE] = "validation"
        top20 = compare_time_validation(self.train, validation, self.config, experiment_names=(name,),
                                        threshold=0.9, threshold_report=table,
                                        pipeline_templates={name: pipeline})
        whole, params = compare_all_sensors(self.train, validation, self.config, pipeline, threshold=0.9)
        self.assertEqual(params, pipeline.named_steps["model"].get_params())
        self.assertEqual(whole.selected_feature_count.tolist(), [30])
        self.assertEqual(top20.selected_feature_count.tolist(), [20])
        self.assertEqual(whole.threshold.tolist(), [0.9])
        self.assertEqual(whole.threshold_source.tolist(), ["top20_oof_diagnostic_only"])
        self.assertEqual(len(whole.attrs["predictions"]), len(validation))
        with TemporaryDirectory() as directory:
            output = Path(directory)
            comparison = save_sensor_comparison(top20, whole, output)
            self.assertEqual(comparison.comparison_role.tolist(), ["all_sensors", "rf_top20"])
            self.assertTrue((output / "all_validation_predictions.csv").is_file())
            self.assertIn("운영 문턱이 아니므로", (output / "sensor_compare.md").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "선택값"):
            prepare_candidate(self.train, self.oof, self.record, self.config, threshold=0.8)
        changed = copy.deepcopy(self.record)
        changed["models"][0]["model_parameters"]["max_depth"] = 3
        with self.assertRaisesRegex(ValueError, "선택기·모델 설정"):
            prepare_candidate(self.train, self.oof, changed, self.config, threshold=0.9)

    def test_rejects_changed_rows_and_oof_source(self):
        changed = self.oof.copy()
        changed.loc[0, "label"] = 1
        with self.assertRaisesRegex(ValueError, "일치하지"):
            prepare_candidate(self.train, changed, self.record, self.config, threshold=0.9)
        changed = self.oof.assign(oof_score_method="repeated_mean")
        with self.assertRaisesRegex(ValueError, "출처 또는 방식"):
            prepare_candidate(self.train, changed, self.record, self.config, threshold=0.9)


if __name__ == "__main__":
    unittest.main()
