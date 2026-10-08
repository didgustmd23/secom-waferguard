# ==========================================
# 전체 센서 중요도 진단의 합성 테스트
# - 학습 센서의 중요도 0과 제거 센서의 미학습 상태를 구분
# - V2 설정 재사용·Train 역할 방어 및 그림 저장을 검증
# - 실제 SECOM 데이터나 외부 평가 입력을 읽지 않음
# ==========================================

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.modeling_preprocessing import SensorQualityFilter
from src.sensor_importance import collect_importances, draw_importances, fit_importance_candidate
from src.split_contract import PROTOCOL_ID, SOURCE_ROW_ID, SPLIT_ROLE
from src.time_weight_compare import DEFAULT_PRESET


class SensorImportanceTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"sensor_a": np.arange(12, dtype=float),
                                   "sensor_b": np.arange(12, dtype=float)[::-1],
                                   "sensor_constant": 1.0, "sensor_empty": np.nan,
                                   "label": [-1] * 8 + [1] * 4,
                                   "timestamp": "01/01/2008 00:00:00",
                                   SOURCE_ROW_ID: np.arange(12), SPLIT_ROLE: "train",
                                   PROTOCOL_ID: "time:importance-toy"})

    def test_all_sensors_distinguish_removed_and_zero(self):
        """제거된 센서는 결측 중요도로 기록하고 사용된 0 중요도는 보존한다."""
        quality = SensorQualityFilter().fit(self.frame.filter(like="sensor_"))
        classifier = SimpleNamespace(importance_type="gain", feature_importances_=np.array([1., 0.]))
        pipeline = SimpleNamespace(named_steps={"quality_filter": quality, "model": classifier})
        table = collect_importances(pipeline).set_index("feature")
        self.assertEqual(len(table), 4)
        self.assertEqual(table.loc["sensor_b", "gain_importance"], 0.)
        self.assertEqual(table.loc["sensor_b", "status"], "used")
        self.assertTrue(table.loc[["sensor_empty", "sensor_constant"], "gain_importance"].isna().all())
        self.assertEqual(table.loc["sensor_empty", "status"], "removed_high_missing")
        self.assertEqual(table.loc["sensor_constant", "status"], "removed_constant")
        pipeline.named_steps["selector"] = object()
        with self.assertRaises(ValueError):
            collect_importances(pipeline)

    def test_candidate_uses_train_only_and_preserves_frame(self):
        """선택 후보의 동적 가중치와 전체 원본 센서 기록을 확인한다."""
        preset = json.loads(DEFAULT_PRESET.read_text(encoding="utf-8"))
        preset["base_parameters"]["n_estimators"] = 2
        original = self.frame.copy(deep=True)
        table, model = fit_importance_candidate(self.frame, MODELING_CONFIG, preset, "v2_m0_ratio")
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(len(table), 4)
        self.assertEqual(model.named_steps["model"].effective_scale_pos_weight_, 2.)
        with self.assertRaises(ValueError):
            fit_importance_candidate(self.frame.assign(**{SPLIT_ROLE: "test"}), MODELING_CONFIG, preset, "v2_m0_ratio")
        with self.assertRaises(ValueError):
            fit_importance_candidate(self.frame, MODELING_CONFIG, preset, "unknown")

    def test_all_and_top20_figures_are_saved(self):
        """제거 센서를 포함하는 전체 그림과 학습 센서 상위 그림을 저장한다."""
        table = pd.DataFrame({"feature": ["sensor_a", "sensor_b", "sensor_empty"],
                              "status": ["used", "used", "removed_high_missing"],
                              "gain_importance": [1., 0., np.nan]})
        with TemporaryDirectory() as temporary:
            paths = draw_importances(table, Path(temporary), model_name="toy")
            self.assertEqual(set(paths), {"all", "top20"})
            for path in paths.values():
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
