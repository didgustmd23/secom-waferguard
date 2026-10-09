# ==========================================
# 합성 로그로 문턱 선택과 이후 정답 분리를 검증
# - 실제 실험 로그·Test·저장 모델은 읽지 않음
# ==========================================
import unittest
import pandas as pd
from src.sensor_ml.experiments.team_gain_recall_compare import compare_recall_targets


class RecallTargetComparisonTest(unittest.TestCase):
    def setUp(self):
        """작은 과거 문턱표와 이후 두 행의 예측을 만든다."""
        self.table = pd.DataFrame({"model_name": ["toy"] * 3, "outer_fold": [1] * 3,
            "mode": ["temporal_oof"] * 3, "threshold": [.01, .1, .05],
            "recall": [1., .8, 1.], "precision": [.5, 4/6, 5/8],
            "true_positive": [5, 4, 5], "false_positive": [5, 2, 3], "support": [10] * 3})
        self.pred = pd.DataFrame({"model_name": ["toy"] * 2, "outer_fold": [1, 1],
            "mode": ["default"] * 2, "source_row_id": [11, 12],
            "label": [-1, 1], "positive_score": [.08, .11]})

    def test_selection_uses_only_oof(self):
        """이후 정답이 바뀌어도 과거 OOF에서 선택한 문턱은 변하지 않는다."""
        result = compare_recall_targets(self.table, self.pred, positive_label=1, negative_label=-1)
        self.assertEqual(result.threshold.tolist(), [.1, .05])
        self.assertEqual(result.false_positive.tolist(), [0, 1])
        changed = compare_recall_targets(self.table, self.pred.assign(label=[1, -1]),
                                         positive_label=1, negative_label=-1)
        self.assertEqual(result.threshold.tolist(), changed.threshold.tolist())

    def test_bad_logs_are_rejected(self):
        """중복 행과 모델·fold 불일치를 거부한다."""
        for pred in (self.pred.assign(source_row_id=[11, 11]), self.pred.assign(outer_fold=2)):
            with self.assertRaises(ValueError):
                compare_recall_targets(self.table, pred, positive_label=1, negative_label=-1)


if __name__ == "__main__":
    unittest.main()
