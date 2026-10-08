# ==========================================
# OOF 정책 선택의 경계값·동률·미충족·오류 입력 검증
# - 미래 평가 결과 없이 후보 표의 지표만으로 선택
# ==========================================

import unittest
from dataclasses import replace

import pandas as pd

from src.modeling_config import ThresholdPolicy, MODELING_CONFIG
from src.threshold_policy import policy_candidates, select_policy_threshold


class ThresholdPolicyTest(unittest.TestCase):
    def test_current_policy_accepts_eighty_forty_scenario(self):
        # 작은 후보 표로 설정 변경을 검증하며 기존 실험 로그를 수정하지 않는다.
        table = pd.DataFrame({"threshold": [0.01, 0.03, 0.08],
                              "recall": [71 / 78, 63 / 78, 44 / 78],
                              "support": [1096] * 3, "positive_support": [78] * 3,
                              "true_positive": [71, 63, 44], "false_positive": [595, 344, 174]})
        policy = MODELING_CONFIG.threshold_policy
        self.assertEqual(policy.min_recall, 0.8)
        self.assertEqual(policy.max_reinspection_ratio, 0.4)
        candidates = policy_candidates(table, policy)
        self.assertEqual(candidates.policy_feasible.tolist(), [False, True, False])
        self.assertEqual(select_policy_threshold(table, policy), 0.03)

    def setUp(self):
        # 동일한 100건 중 불량 10건에 대해 계산한 후보 표를 사용한다.
        self.table = pd.DataFrame({"threshold": [0.1, 0.2, 0.3, 0.4],
                                   "recall": [0.9, 0.8, 0.8, 0.7],
                                   "support": [100] * 4, "positive_support": [10] * 4,
                                   "true_positive": [9, 8, 8, 7],
                                   "false_positive": [11, 2, 2, 3]})

    def test_minimum_reinspection_and_deterministic_ties(self):
        # 비율 10% 동률에서 Recall 최대, 다시 동률이면 threshold 최대를 택한다.
        self.assertEqual(select_policy_threshold(self.table, ThresholdPolicy()), 0.3)

    def test_inclusive_boundaries_and_infeasible(self):
        policy = ThresholdPolicy(min_recall=0.9, max_reinspection_ratio=0.2)
        self.assertEqual(select_policy_threshold(self.table, policy), 0.1)
        self.assertIsNone(select_policy_threshold(self.table, replace(policy, min_recall=1)))

    def test_ratio_is_not_false_positive_rate(self):
        candidates = policy_candidates(self.table, ThresholdPolicy())
        self.assertEqual(candidates.iloc[0].reinspection_ratio, 0.2)

    def test_rejects_invalid_candidate_data(self):
        for key, value in (("support", 0), ("support", float("nan")),
                           ("positive_support", 0), ("false_positive", -1),
                           ("true_positive", 11), ("recall", 0.1), ("threshold", 2)):
            table = self.table.copy()
            table.loc[0, key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                policy_candidates(table, ThresholdPolicy())
        with self.assertRaises(ValueError):
            policy_candidates(pd.DataFrame(), ThresholdPolicy())


if __name__ == "__main__":
    unittest.main()
