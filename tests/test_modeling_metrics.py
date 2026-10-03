# ==========================================
# 범용 이진 분류 평가 함수 테스트
# - Dataset Profile에서 전달되는 정상·불량 Label을 명시적으로 사용하는지 확인
# - 허용 외 Label과 결측 Label이 성능 계산 전에 차단되는지 확인
# ==========================================

from __future__ import annotations

import unittest

import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.modeling_metrics import evaluate_binary_scores


class ModelingMetricsTest(unittest.TestCase):
    # SECOM의 -1/1 Label과 0.50 threshold에서 confusion matrix를 확인
    def test_evaluates_secom_labels(self) -> None:
        result = evaluate_binary_scores(
            [-1, 1, -1, 1],
            [0.1, 0.9, 0.7, 0.3],
            positive_label=1,
            negative_label=-1,
            threshold=MODELING_CONFIG.experiment.default_threshold,
        )

        self.assertEqual(result.true_negative, 1)
        self.assertEqual(result.false_positive, 1)
        self.assertEqual(result.false_negative, 1)
        self.assertEqual(result.true_positive, 1)

    # 다른 데이터셋의 0/1 Label도 호출부가 Label을 전달하면 평가할 수 있는지 확인
    def test_evaluates_custom_binary_labels(self) -> None:
        result = evaluate_binary_scores(
            [0, 1],
            [0.1, 0.9],
            positive_label=1,
            negative_label=0,
            threshold=MODELING_CONFIG.experiment.default_threshold,
        )

        self.assertEqual(result.positive_support, 1)
        self.assertEqual(result.recall, 1.0)

    # 허용하지 않은 Label은 정상으로 묵시 처리하지 않고 오류를 발생시키는지 확인
    def test_rejects_unsupported_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside negative_label"):
            evaluate_binary_scores(
                [-1, 1, 999],
                [0.1, 0.9, 0.2],
                positive_label=1,
                negative_label=-1,
                threshold=MODELING_CONFIG.experiment.default_threshold,
            )

    # None, NaN, pd.NA를 모두 결측 Label로 차단하는지 확인
    def test_rejects_missing_labels(self) -> None:
        for missing_label in (None, float("nan"), pd.NA):
            with self.subTest(missing_label=missing_label):
                with self.assertRaisesRegex(ValueError, "missing labels"):
                    evaluate_binary_scores(
                        [-1, 1, missing_label],
                        [0.1, 0.9, 0.2],
                        positive_label=1,
                        negative_label=-1,
                        threshold=MODELING_CONFIG.experiment.default_threshold,
                    )
