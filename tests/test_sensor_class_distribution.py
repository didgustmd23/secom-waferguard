# ==========================================
# 정상·불량 분포 진단의 합성 데이터 검증
# - 구간 중복 방지·결측 보존·센서 및 Train 역할 검증
# - 실제 SECOM 데이터나 외부 평가 데이터는 읽지 않음
# ==========================================

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.sensor_ml.diagnostics.sensor_class_distribution import prepare_distribution, draw_distribution
from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID


class SensorClassDistributionTest(unittest.TestCase):
    def setUp(self):
        """관측값·결측·정상·불량이 포함된 시간순 Train을 구성한다."""
        self.frame = pd.DataFrame({
            "sensor_0": [1., np.nan, 3., 4., 5., 6., 7., 8.],
            "label": [-1, 1] * 4,
            "timestamp": pd.date_range("2008-01-01", periods=8).strftime("%d/%m/%Y %H:%M:%S"),
            SOURCE_ROW_ID: np.arange(8), SPLIT_ROLE: "train", PROTOCOL_ID: "time:toy",
        })

    def test_periods_cover_once_and_missing_is_preserved(self):
        """구간을 중복 없이 나누며 결측 대치 없이 클래스별 통계를 기록한다."""
        original = self.frame.copy(deep=True)
        groups, periods, summary = prepare_distribution(self.frame, MODELING_CONFIG.dataset, ["sensor_0"], 3)
        combined = pd.concat(groups.values())
        self.assertEqual(len(combined), 8)
        self.assertTrue(combined.index.is_unique)
        self.assertEqual(periods.row_count.sum(), 8)
        whole_fail = summary.loc[summary.period.eq("전체 Train") & summary["class"].eq("불량")].iloc[0]
        self.assertEqual(whole_fail.missing_ratio, 0.25)
        pd.testing.assert_frame_equal(original, self.frame)

    def test_invalid_input_is_rejected(self):
        """외부 Test 역할·라벨 컬럼·중복 센서·없는 센서를 거부한다."""
        for features in (["label"], ["sensor_0", "sensor_0"], ["unknown"]):
            with self.assertRaises(ValueError):
                prepare_distribution(self.frame, MODELING_CONFIG.dataset, features, 3)
        with self.assertRaises(ValueError):
            prepare_distribution(self.frame.assign(**{SPLIT_ROLE: "test"}), MODELING_CONFIG.dataset, ["sensor_0"], 3)

    def test_plots_handle_missing_observations(self):
        """어떤 구간의 불량 관측값이 없어도 그림 두 종류를 생성한다."""
        groups, _, summary = prepare_distribution(self.frame, MODELING_CONFIG.dataset, ["sensor_0"], 3)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            draw_distribution(self.frame, groups, summary, MODELING_CONFIG.dataset, ["sensor_0"], directory)
            self.assertGreater((directory / "class_distribution.png").stat().st_size, 0)
            self.assertGreater((directory / "class_distribution_by_period.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
