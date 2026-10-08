# ==========================================
# 센서 축소 AP 판정의 경계값과 정의 불가 입력 확인
# - 목표/허용 기준은 불량 Recall·threshold 정책과 분리
# - 기준 모델 누락·fold 불일치·AP 0을 성공으로 처리하지 않음
# ==========================================
import unittest

import pandas as pd

from src.feature_reduction import assess_reduction
from src.modeling_config import FeatureReductionPolicy


class FeatureReductionTest(unittest.TestCase):
    policy = FeatureReductionPolicy(20, 0.2, 0.3, (("all", "top"),))

    @staticmethod
    def folds(ap=0.15, count=20, raw=True, baseline=0.2):
        # 학습 없이 같은 fold의 전체·축소 점수만으로 판정 로직을 확인한다.
        return pd.DataFrame([
            {"experiment": "all", "fold": 1, "ap": baseline, "sensor_count": 444, "is_raw_sensor": True},
            {"experiment": "top", "fold": 1, "ap": ap, "sensor_count": count, "is_raw_sensor": raw},
        ])

    def test_loss_boundaries_and_sensor_limit(self):
        # 경계값을 포함하고, 성능 개선은 음수 하락률 그대로 기록한다.
        for ap, status in ((0.16, "목표 달성"), (0.15, "허용 범위·목표 미달"),
                           (0.14, "허용 범위·목표 미달"), (0.13, "허용 하락률 초과"),
                           (0.22, "목표 달성")):
            with self.subTest(ap=ap):
                row = assess_reduction(self.folds(ap), self.policy).iloc[0]
                self.assertEqual(row.ap_status, status)
                self.assertAlmostEqual(row.ap_loss_ratio, (0.2 - ap) / 0.2)
        self.assertEqual(assess_reduction(self.folds(count=21), self.policy).iloc[0].sensor_status, "센서 수 초과")
        self.assertEqual(assess_reduction(self.folds(raw=False), self.policy).iloc[0].sensor_status, "판정 불가")

    def test_undefined_or_unmatched_comparisons(self):
        # 기준 AP가 없거나 비교 fold가 다르면 비율을 만들지 않는다.
        for baseline in (0, float("nan")):
            self.assertEqual(assess_reduction(self.folds(baseline=baseline), self.policy).iloc[0].ap_status, "판정 불가")
        folds = self.folds()
        folds.loc[folds.experiment.eq("top"), "fold"] = 2
        self.assertEqual(assess_reduction(folds, self.policy).iloc[0].ap_status, "판정 불가")
        self.assertEqual(assess_reduction(folds.iloc[1:], self.policy).iloc[0].ap_status, "판정 불가")
        self.assertTrue(assess_reduction(folds, None).empty)


if __name__ == "__main__":
    unittest.main()
