# ==========================================
# 범용 전처리 회귀 테스트
# - 모든 후보와 특징 선택 실험이 Profile 범주형 입력을 처리하는지 확인
# - 전부 결측인 컬럼 제거 및 변환 후 실제 특징 수 기록을 확인
# - OOF 생성 횟수와 원본 행 ID 전달을 실제 Pipeline으로 검증
# ==========================================

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from dataclasses import replace

import numpy as np
import pandas as pd

from src.modeling_config import CrossValidationConfig, MODELING_CONFIG
from src.modeling_preprocessing import fitted_feature_count
from src.sensor_ml.experiments.step5_model_compare import build_candidate_pipelines, compare_candidates
from src.sensor_ml.experiments.step6_feature_compare import build_experiments, compare_features
from src.sensor_ml.experiments.step8_threshold_oof import generate_oof_scores, run_threshold_oof_from_file
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID


class GenericPreprocessingTest(unittest.TestCase):
    config = replace(MODELING_CONFIG,
                     dataset=replace(MODELING_CONFIG.dataset, label_column="target", positive_label="fail",
                                     negative_label="pass", timestamp_column=None, timestamp_format=None,
                                     feature_selection_mode="all_except_metadata", feature_column_prefix=None),
                     experiment=replace(MODELING_CONFIG.experiment, cv=CrossValidationConfig(2, 1, 42)),
                     top_k_feature_counts=(3, 2))

    @staticmethod
    def _frame():
        # 충분한 class 신호를 넣어 L1 선택기가 빈 특징 집합을 만들지 않게 한다.
        return pd.DataFrame({"sensor_a": [0., .1, .2, .3, .2, .1, .8, .9, 1., .7, .9, .8],
                             "sensor_b": [1., .9, .8, .7, .8, .9, .2, .1, 0., .3, .1, .2],
                             "sensor_c": [.2, .1, .3, .2, .4, .3, .7, .8, .6, .7, .9, .8],
                             "target": ["pass"] * 6 + ["fail"] * 6})

    def test_all_candidates_handle_declared_categories(self):
        frame = self._frame().assign(machine=["A"] * 6 + ["B"] * 6)
        config = replace(self.config, dataset=replace(self.config.dataset, categorical_feature_columns=("machine",)))
        features = frame.drop(columns="target")
        for name, pipeline in build_candidate_pipelines(config).items():
            with self.subTest(model=name):
                pipeline.fit(features, frame.target)
                unseen = features.iloc[[0]].assign(machine="new_machine")
                self.assertEqual(pipeline.predict_proba(unseen).shape, (1, 2))

    def test_categorical_cv_and_transformed_top_k(self):
        # raw 컬럼은 2개지만 One-Hot 이후 3개이므로 Top-3을 허용해야 한다.
        frame = self._frame()[["sensor_a", "target"]].assign(machine=["A"] * 6 + ["B"] * 6)
        config = replace(self.config, dataset=replace(self.config.dataset, categorical_feature_columns=("machine",)),
                         top_k_feature_counts=(3,), candidate_models=("logistic_regression_l1",))
        self.assertEqual(len(compare_candidates(frame, config)), 2)
        result = compare_features(frame, config)
        self.assertEqual(result.loc[result.experiment == "lightgbm_all", "selected_feature_count_mean"].item(), 3)

    def test_removes_entirely_missing_numeric_columns(self):
        frame = self._frame().assign(sensor_c=np.nan)
        pipeline = build_experiments(self.config)["lightgbm_all"]
        pipeline.fit(frame.drop(columns="target"), frame.target)
        self.assertEqual(fitted_feature_count(pipeline), 2)
        result = compare_features(frame, replace(self.config, top_k_feature_counts=(2,)))
        self.assertEqual(result.loc[result.experiment == "lightgbm_top_2", "selected_feature_count_mean"].item(), 2)

    def test_top_k_checks_encoded_fold_width(self):
        frame = self._frame()[["sensor_a", "target"]].assign(machine="only_category")
        config = replace(self.config, dataset=replace(self.config.dataset, categorical_feature_columns=("machine",)),
                         top_k_feature_counts=(3,))
        with self.assertRaisesRegex(ValueError, "변환 후 feature 수보다 큽니다"):
            compare_features(frame, config)

    def test_oof_counts_and_original_row_ids(self):
        frame = self._frame().assign(**{
            SOURCE_ROW_ID: np.arange(100, 112), SPLIT_ROLE: "train", PROTOCOL_ID: "toy",
        })
        config = replace(self.config, experiment=replace(self.config.experiment,
                                                        cv=CrossValidationConfig(2, 2, 42)))
        scores, _ = generate_oof_scores(frame, config, score_method="repeated_mean")
        self.assertEqual(scores.oof_prediction_count.tolist(), [2] * len(frame))
        self.assertEqual(scores.source_row_id.tolist(), list(range(100, 112)))
        self.assertTrue(scores.oof_positive_score.between(0, 1).all())
        self.assertEqual(set(scores.threshold_use), {"analysis_only"})

    def test_default_oof_uses_one_prediction_per_row(self):
        # 후보 비교 설정이 반복 CV여도 threshold OOF는 단일 K-fold를 사용한다.
        config = replace(self.config, experiment=replace(self.config.experiment,
                                                        cv=CrossValidationConfig(2, 2, 42)))
        scores, _ = generate_oof_scores(self._frame(), config)
        self.assertEqual(scores.oof_prediction_count.tolist(), [1] * len(scores))
        self.assertEqual(set(scores.oof_score_method), {"single"})
        self.assertEqual(set(scores.threshold_use), {"candidate"})

    def test_cv_results_record_fold_removal_counts(self):
        # 결과의 각 모델 행에 학습 fold별 실제 제거 개수와 센서명이 저장되는지 확인한다.
        frame = self._frame().assign(sensor_c=np.nan)
        config = replace(self.config, top_k_feature_counts=(2,),
                         candidate_models=("logistic_regression_l1",))
        for results in (compare_candidates(frame, config), compare_features(frame, config)):
            for log in results.quality_filter_log:
                records = json.loads(log)
                self.assertEqual(len(records), 2)
                self.assertEqual([record["fold"] for record in records], [1, 2])
                for record in records:
                    self.assertEqual(record["high_missing_features"], ["sensor_c"])
                    self.assertEqual(record["removed_feature_count"], 1)
                    self.assertEqual(record["retained_feature_count"], 2)

    def test_oof_writes_fold_quality_log(self):
        # 샘플별 점수 파일 옆의 로그에서 fold별 제거 개수와 JSON 센서명을 확인한다.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            self._frame().assign(sensor_c=np.nan).to_csv(train_path, index=False)
            with patch("src.sensor_ml.experiments.step8_threshold_oof.load_modeling_config", return_value=self.config):
                run_threshold_oof_from_file(root / "config.json", train_path,
                                            root / "oof.csv", root / "threshold.csv")
            written = pd.read_csv(root / "oof_quality_filter.csv")
            self.assertEqual(written.fold.tolist(), [1, 2])
            self.assertEqual(written.removed_feature_count.tolist(), [1, 1])
            self.assertEqual(written.retained_feature_count.tolist(), [2, 2])
            self.assertEqual(written.high_missing_features.map(json.loads).tolist(),
                             [["sensor_c"], ["sensor_c"]])


if __name__ == "__main__":
    unittest.main()
