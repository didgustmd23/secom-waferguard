# ==========================================
# V2 90% 모델 저장 준비와 80% 문턱 분기의 정합성 검증
# - 합성 데이터와 트리 3개만 사용
# - 축소 추론 확률 일치·독립 복사·80% 분기에서 재학습 없음
# - OOF 문턱의 전체 정밀도 유지와 잘못된 센서 목록 거부
# ==========================================
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.sensor_ml.inference.save_v2_scenario import train_inference, retarget_inference, scenario_source
from src.modeling_config import MODELING_CONFIG
from tests.test_fixed_sensor_compare import synthetic_frame


class V2ScenarioSaveTest(unittest.TestCase):
    def test_shared_model_and_no_refit(self):
        """80%는 90%의 센서·학습 통계·분류기를 보존하고 문턱만 변경한다."""
        frame = synthetic_frame()
        names = [f"sensor_{index}" for index in range(20)]
        base, samples, proof, checks = train_inference(
            frame, MODELING_CONFIG, names, 0.1, n_estimators=3, n_jobs=1)
        self.assertTrue(all(row["passed"] for row in checks))
        self.assertEqual(len(proof), 12)
        sample = samples.iloc[:8].copy()
        sample.iloc[0, 0] = np.nan
        payload = {"inference": base, "verification_input": sample,
                   "verification_expected": base.predict(sample)}
        with patch("src.sensor_ml.inference.save_v2_scenario.fit_pipeline", side_effect=AssertionError("재학습 금지")):
            other, _ = retarget_inference(payload, names, 0.5)
        self.assertEqual(base.threshold, 0.1)
        self.assertEqual(other.threshold, 0.5)
        np.testing.assert_array_equal(base._medians, other._medians)
        first = base.predict(sample); second = other.predict(sample)
        np.testing.assert_array_equal(first.positive_score, second.positive_score)
        self.assertTrue((~second.predicted_positive | first.predicted_positive).all())
        self.assertIsNot(base._model, other._model)
        with self.assertRaisesRegex(ValueError, "센서 목록"):
            retarget_inference(payload, list(reversed(names)), 0.5)

    def test_exact_threshold_source_and_guard(self):
        """문턱을 반올림하지 않고 과거 OOF 선택 규칙과 맞는지 확인한다."""
        threshold = 0.002947123456789012
        execution = {"oof_scope": "after_selection_only", "test_used": False}
        tables = [pd.DataFrame({"target_recall": [0.9], "threshold": [threshold]}),
                  pd.DataFrame({"recall": [1.0, 0.875], "alarm_ratio": [0.95, 0.55],
                                "precision": [0.03, 0.05], "threshold": [threshold, 0.03]}),
                  pd.DataFrame({"sensor_order": [0, 1], "feature": ["a", "b"]})]
        with patch.object(Path, "read_text", return_value=json.dumps(execution)), \
             patch("src.sensor_ml.inference.save_v2_scenario.pd.read_csv", side_effect=tables):
            value, _, _, row = scenario_source(Path("oof"), 0.9)
        self.assertEqual(value, threshold)
        self.assertEqual(row["threshold"], threshold)
        tables[0].loc[0, "threshold"] = 0.002947
        with patch.object(Path, "read_text", return_value=json.dumps(execution)), \
             patch("src.sensor_ml.inference.save_v2_scenario.pd.read_csv", side_effect=tables):
            with self.assertRaisesRegex(ValueError, "선택 기준"):
                scenario_source(Path("oof"), 0.9)


if __name__ == "__main__":
    unittest.main()
