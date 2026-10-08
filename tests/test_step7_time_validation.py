# ==========================================
# Time Validation 후보 비교 테스트
# - Test split을 사용하지 않고 두 후보 결과가 생성되는지 확인
# - 잘못된 후보 이름과 빈 후보 목록을 방어하는지 확인
# ==========================================

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.modeling_config import load_modeling_config
from src.experiments.step7_time_validation import compare_time_validation, compare_time_validation_from_files


class TimeValidationComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_config(Path("config.json"))

        # 작지만 두 class가 모두 있는 수치형 SECOM 형태의 split을 만든다.
        cls.train_frame = pd.DataFrame(
            {
                "sensor_0": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9, 0.3, 0.7],
                "sensor_1": [1.0, 0.8, 0.2, 0.0, 0.9, 0.1, 0.7, 0.3],
                "label": [-1, -1, 1, 1, -1, 1, -1, 1],
                "timestamp": ["01/01/2008 00:00:00"] * 8,
            }
        )
        cls.validation_frame = pd.DataFrame(
            {
                "sensor_0": [0.15, 0.85, 0.25, 0.75],
                "sensor_1": [0.85, 0.15, 0.75, 0.25],
                "label": [-1, 1, -1, 1],
                "timestamp": ["02/01/2008 00:00:00"] * 4,
            }
        )

    def test_compares_the_preselected_models(self) -> None:
        result = compare_time_validation(
            self.train_frame,
            self.validation_frame,
            self.config,
            n_jobs=1,
        )

        self.assertEqual(set(result["experiment"]), {"lightgbm_all", "l1_balanced_l1_select"})
        self.assertTrue((result["split_strategy"] == "time_validation").all())
        self.assertTrue((result["threshold"] == 0.5).all())
        self.assertTrue((result["validation_samples"] == 4).all())

    def test_rejects_invalid_experiment_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "정의되지 않은 후보 모델"):
            compare_time_validation(
                self.train_frame,
                self.validation_frame,
                self.config,
                experiment_names=("unknown_model",),
            )

    def test_uses_preselected_threshold_without_retuning(self) -> None:
        result = compare_time_validation(
            self.train_frame,
            self.validation_frame,
            self.config,
            experiment_names=("lightgbm_all",),
            threshold=0.25,
            n_jobs=1,
        )

        self.assertEqual(result["threshold"].tolist(), [0.25])

    def test_rejects_out_of_range_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "0.0 이상 1.0 이하"):
            compare_time_validation(
                self.train_frame,
                self.validation_frame,
                self.config,
                threshold=1.1,
            )

    def test_writes_a_reproducible_result_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "time_train.csv"
            validation_path = root / "time_valid.csv"
            output_path = root / "result.csv"
            self.train_frame.to_csv(train_path, index=False)
            self.validation_frame.to_csv(validation_path, index=False)

            compare_time_validation_from_files(
                "config.json", train_path, validation_path, output_path
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(pd.read_csv(output_path)), 2)


if __name__ == "__main__":
    unittest.main()
