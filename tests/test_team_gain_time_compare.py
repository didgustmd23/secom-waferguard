# ==========================================
# 시간순 후보 비교 설정 검증 — 실제 SECOM 데이터는 읽지 않음
# - 기존 V2와 팀원 후보의 분류기 조건 및 프리셋 위치 확인
# ==========================================
import json
import unittest
from src.modeling_config import load_modeling_config
from src.sensor_ml.experiments.team_gain_time_compare import DEFAULT_PRESET, build_candidates


class TeamTimeComparisonTest(unittest.TestCase):
    def test_candidate_conditions(self):
        """전체 기준은 유지하고 두 Top-20의 최종 분류기 설정을 동일하게 맞춘다."""
        preset = json.loads(DEFAULT_PRESET.read_text(encoding="utf-8"))
        pipes = build_candidates(load_modeling_config(), preset, n_jobs=2)
        self.assertEqual(set(pipes), {"v2_all", "v2_stable_gain_top20", "team_lgbm_gain_top20"})
        self.assertEqual(pipes["v2_all"].named_steps["model"].max_depth, 3)
        first = pipes["v2_stable_gain_top20"].named_steps["model"].get_params()
        second = pipes["team_lgbm_gain_top20"].named_steps["model"].get_params()
        self.assertEqual(first, second)
        self.assertEqual(second["max_depth"], 2)
        self.assertEqual(second["class_weight_mode"], "ratio")
        selector = pipes["team_lgbm_gain_top20"].named_steps["selector"]
        self.assertEqual(selector.estimator.n_estimators, 200)
        self.assertFalse(hasattr(selector, "estimator_"))


if __name__ == "__main__":
    unittest.main()
