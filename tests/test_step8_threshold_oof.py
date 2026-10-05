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
from src.step8_threshold_oof import compare_thresholds, validate_threshold_report
from src.split_contract import PROTOCOL_ID


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

    def test_rejects_empty_oof(self):
        with self.assertRaisesRegex(ValueError, "비어 있으면"):
            compare_thresholds(self.oof_frame.iloc[:0], self.config)

    def test_rejects_mixed_oof_methods(self):
        frame = self.oof_frame.assign(oof_score_method=["single", "single", "repeated_mean", "repeated_mean"])
        with self.assertRaisesRegex(ValueError, "OOF 실행 결과가 섞여"):
            compare_thresholds(frame, self.config)

    def test_preserves_analysis_only_status(self):
        frame = self.oof_frame.assign(oof_score_method="repeated_mean", threshold_use="analysis_only")
        result = compare_thresholds(frame, self.config)
        self.assertEqual(set(result.threshold_use), {"analysis_only"})

    def _report(self):
        # 같은 시간 학습 계약에서 생성한 단일 OOF 비교표를 준비한다.
        frame = self.oof_frame.assign(training_protocol_id="time:toy", oof_score_method="single", threshold_use="candidate")
        return compare_thresholds(frame, self.config, thresholds=np.array([0.2]))

    def test_accepts_matching_threshold_source(self):
        validate_threshold_report(self._report(), pd.DataFrame({PROTOCOL_ID: ["time:toy"]}),
                                  self.config, "lightgbm_all", 0.2)

    def test_rejects_threshold_from_other_train(self):
        with self.assertRaisesRegex(ValueError, "training_protocol_id"):
            validate_threshold_report(self._report(), pd.DataFrame({PROTOCOL_ID: ["time:other"]}),
                                      self.config, "lightgbm_all", 0.2)

    def test_rejects_repeated_mean_for_single_model(self):
        with self.assertRaisesRegex(ValueError, "oof_score_method"):
            validate_threshold_report(self._report().assign(oof_score_method="repeated_mean"),
                                      pd.DataFrame({PROTOCOL_ID: ["time:toy"]}), self.config, "lightgbm_all", 0.2)

    def test_rejects_threshold_not_in_report(self):
        with self.assertRaisesRegex(ValueError, "비교표에 없습니다"):
            validate_threshold_report(self._report(), pd.DataFrame({PROTOCOL_ID: ["time:toy"]}),
                                      self.config, "lightgbm_all", 0.123)


if __name__ == "__main__":
    unittest.main()
