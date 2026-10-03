# ==========================================
# Train/Validation Baseline 테스트
# - 실제 SECOM split 없이 작은 가상 split으로 Pipeline, metric, 결과 저장을 검증
# - Test 데이터가 Baseline 입력에 필요하지 않음을 확인
# - 수치형 전용 및 Profile에 선언된 범주형 feature 재사용 경로를 모두 확인
# ==========================================

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.modeling_config import DatasetSpec, MODELING_CONFIG
from src.step4_baseline import run_baseline, split_frame_to_xy, write_baseline_result


class BaselineTest(unittest.TestCase):
    # 가상 데이터셋도 Profile만 바꾸면 같은 Baseline 코드를 재사용할 수 있도록 구성한다.
    numeric_dataset = DatasetSpec(
        dataset_id="toy_numeric",
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

    # 현재 공통 실험 설정은 유지하고, 테스트용 Dataset Profile만 교체한다.
    numeric_config = replace(MODELING_CONFIG, dataset=numeric_dataset)

    # 결측 수치형 feature를 포함한 Train/Validation split으로 기본 Pipeline을 검증한다.
    def test_runs_numeric_baseline_and_writes_one_row_result(self) -> None:
        train_frame = pd.DataFrame(
            {
                "sensor_a": [0.0, 0.2, 0.1, 0.9, 1.0, 0.8],
                "sensor_b": [0.1, None, 0.2, 0.8, 0.9, 1.0],
                "target": ["pass", "pass", "pass", "fail", "fail", "fail"],
            }
        )
        validation_frame = pd.DataFrame(
            {
                "sensor_a": [0.05, 0.95, 0.75, 0.15],
                "sensor_b": [0.1, 0.95, 0.7, 0.2],
                "target": ["pass", "fail", "fail", "pass"],
            }
        )

        result = run_baseline(train_frame, validation_frame, self.numeric_config)

        self.assertEqual(result.dataset_id, "toy_numeric")
        self.assertEqual(result.model_name, "logistic_regression")
        self.assertEqual(result.train_samples, 6)
        self.assertEqual(result.validation_samples, 4)
        self.assertEqual(result.feature_count, 2)
        self.assertEqual(result.metrics.positive_support, 2)
        self.assertGreaterEqual(result.training_seconds, 0.0)

        # 결과 CSV는 임시 위치에 한 행만 기록해 실제 logs/를 오염시키지 않는다.
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "baseline_result.csv"
            write_baseline_result(result, output_path)
            written_result = pd.read_csv(output_path)

        self.assertEqual(len(written_result), 1)
        self.assertIn("average_precision", written_result.columns)
        self.assertIn("training_seconds", written_result.columns)

    # 선언된 범주형 feature는 unseen category가 Validation에 있어도 one-hot encoder가 처리해야 한다.
    def test_runs_profile_defined_categorical_feature(self) -> None:
        categorical_dataset = replace(
            self.numeric_dataset,
            dataset_id="toy_categorical",
            categorical_feature_columns=("machine",),
        )
        categorical_config = replace(MODELING_CONFIG, dataset=categorical_dataset)
        train_frame = pd.DataFrame(
            {
                "temperature": [20.0, 20.5, 21.0, 25.0, 25.5, 26.0],
                "machine": ["A", "A", "B", "B", "B", "A"],
                "target": ["pass", "pass", "pass", "fail", "fail", "fail"],
            }
        )
        validation_frame = pd.DataFrame(
            {
                "machine": ["A", "C", "B", "C"],
                "temperature": [20.1, 25.2, 24.8, 20.3],
                "target": ["pass", "fail", "fail", "pass"],
            }
        )

        result = run_baseline(train_frame, validation_frame, categorical_config)

        self.assertEqual(result.feature_count, 2)
        self.assertEqual(result.metrics.support, 4)

    # Train과 Validation의 feature 집합이 다르면 자동 보정하지 않고 split 오류로 중단해야 한다.
    def test_rejects_mismatched_train_validation_features(self) -> None:
        train_frame = pd.DataFrame(
            {
                "sensor_a": [0.0, 1.0],
                "sensor_b": [0.0, 1.0],
                "target": ["pass", "fail"],
            }
        )
        validation_frame = pd.DataFrame(
            {"sensor_a": [0.2, 0.8], "target": ["pass", "fail"]}
        )

        train_x, _, feature_columns = split_frame_to_xy(train_frame, self.numeric_dataset)
        self.assertEqual(tuple(train_x.columns), feature_columns)
        with self.assertRaisesRegex(ValueError, "feature column 구성이 일치하지"):
            split_frame_to_xy(
                validation_frame,
                self.numeric_dataset,
                expected_feature_columns=feature_columns,
            )
