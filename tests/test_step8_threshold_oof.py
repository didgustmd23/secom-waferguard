# ==========================================
# OOF threshold 비교 테스트
# - threshold 표가 FN/FP를 포함해 생성되는지 확인
# - 잘못된 threshold 입력을 한국어 오류로 차단하는지 확인
# ==========================================

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling_config import load_modeling_config
from src.step8_threshold_oof import compare_thresholds


class ThresholdComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_config(Path("config.json"))
        cls.oof_frame = pd.DataFrame(
            {
                "label": [-1, -1, 1, 1],
                "oof_positive_score": [0.1, 0.4, 0.6, 0.9],
            }
        )

    def test_compares_each_threshold_with_error_counts(self) -> None:
        result = compare_thresholds(
            self.oof_frame, self.config, thresholds=np.array([0.2, 0.8])
        )

        self.assertEqual(result["threshold"].tolist(), [0.2, 0.5, 0.8])
        self.assertIn("false_negative", result.columns)
        self.assertIn("false_positive", result.columns)
        self.assertEqual(result.loc[result["threshold"] == 0.8, "false_negative"].item(), 1)

    def test_rejects_out_of_range_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "0.0 이상 1.0 이하"):
            compare_thresholds(
                self.oof_frame, self.config, thresholds=np.array([1.1])
            )


if __name__ == "__main__":
    unittest.main()
