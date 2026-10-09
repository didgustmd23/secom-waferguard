# ==========================================
# 저장 Test 그림의 입력 검증 테스트
# - 합성 예측만 사용하며 모델 학습·실제 Test 평가는 수행하지 않음
# - 정상 기록과 다른 문턱·확률 기록을 구분
# ==========================================
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.modeling_metrics import evaluate_binary_scores
from src.sensor_ml.diagnostics.plot_v2_test import read_evaluation, validate_pair
from src.sensor_ml.diagnostics.plot_sensor_test_compare import read_comparison


class PlotV2Test(unittest.TestCase):
    def setUp(self):
        """작은 합성 예측과 일치하는 평가 기록을 준비한다."""
        self.frame = pd.DataFrame({
            "source_row_id": [3, 1, 2], "label": [-1, 1, -1],
            "positive_score": [0.1, 0.8, 0.6],
            "predicted_positive": [False, True, True], "predicted_label": [-1, 1, 1],
        })
        metrics = asdict(evaluate_binary_scores(
            self.frame.label, self.frame.positive_score,
            positive_label=1, negative_label=-1, threshold=0.5,
        ))
        self.record = {"status": "completed", "refit": False,
                       "evaluation_scope": "existing_v1_test_comparison",
                       "target_recall": 0.9, "threshold": 0.5,
                       "metrics": {**metrics, "reinspection_ratio": 2 / 3}}

    def read_mock(self):
        """파일 읽기만 대체하고 실제 입력 검증 함수를 실행한다."""
        with patch("pathlib.Path.read_text", return_value=json.dumps(self.record)), patch(
            "src.sensor_ml.diagnostics.plot_v2_test.pd.read_csv", return_value=self.frame.copy()
        ):
            return read_evaluation("합성기록", 0.9)

    def test_matching_record(self):
        """일치하는 지표는 허용하며 행 ID 기준으로 정렬한다."""
        frame, _ = self.read_mock()
        self.assertEqual(frame.source_row_id.tolist(), [1, 2, 3])
        validate_pair(frame, frame.copy())

    def test_inconsistent_records_rejected(self):
        """잘못된 지표와 저장 문턱의 판정 불일치를 거부한다."""
        self.record["metrics"]["true_positive"] = 0
        with self.assertRaisesRegex(ValueError, "지표가 다릅니다"):
            self.read_mock()
        self.record["threshold"] = 0.9
        with self.assertRaisesRegex(ValueError, "판정이"):
            self.read_mock()

    def test_different_probabilities_rejected(self):
        """두 시나리오의 점수가 다르면 공통 PR 곡선을 만들지 않는다."""
        other = self.frame.copy()
        other.loc[0, "positive_score"] = 0.2
        with self.assertRaisesRegex(ValueError, "공통 PR 곡선"):
            validate_pair(self.frame, other)

    def test_cross_model_scores_and_rows(self):
        """모델 간 확률 차이는 허용하지만 평가 행이 다르면 비교를 거부한다."""
        other = self.frame.copy()
        other["positive_score"] = [0.2, 0.7, 0.5]
        metadata = {"model_fit_rows": 3, "training_protocol_id": "합성계약", "sensors": ["sensor_1"]}
        def results():
            # 실제 파일 읽기 검증은 다른 테스트가 담당하며 모델 간 비교만 격리한다.
            return [(table.copy(), metadata.copy()) for table in (self.frame, self.frame, other, other)]
        from pathlib import Path
        with patch("src.sensor_ml.diagnostics.plot_sensor_test_compare.read_evaluation", side_effect=results()):
            read_comparison(Path("전체"), Path("20_80"), Path("20_90"))
        other.loc[0, "source_row_id"] = 99
        with patch("src.sensor_ml.diagnostics.plot_sensor_test_compare.read_evaluation", side_effect=results()):
            with self.assertRaisesRegex(ValueError, "모델 간 평가 행"):
                read_comparison(Path("전체"), Path("20_80"), Path("20_90"))


if __name__ == "__main__":
    unittest.main()
