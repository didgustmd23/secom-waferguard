# ==========================================
# 저장된 센서 목록의 분포 진단 테스트
# - 작은 가상 split만 사용하고 모델은 학습하지 않음
# - 원본 결측률·월별 관측 수·목록 오류·Test 오용 방어를 확인
# ==========================================

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID
from src.selected_sensor_drift import diagnose_selected_sensors, write_sensor_diagnosis
from src.sensor_detail import validate_detail_features, write_sensor_details


class SelectedSensorDriftTest(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"sensor_0": [0., 1., 2., 3.], "sensor_1": [1., 2., 3., 4.],
                                   "label": [-1, 1, -1, 1], "timestamp": ["01/01/2008 00:00:00"] * 4,
                                   SOURCE_ROW_ID: range(4), SPLIT_ROLE: "train", PROTOCOL_ID: "time:toy"})
        self.validation = pd.DataFrame({"sensor_0": [10., 11., None, None], "sensor_1": [1., 2., 3., 4.],
                                        "label": [-1, 1, -1, 1], "timestamp": ["01/02/2008 00:00:00"] * 4,
                                        SOURCE_ROW_ID: range(4, 8), SPLIT_ROLE: "validation", PROTOCOL_ID: "time:toy"})
        self.selected = pd.DataFrame({"feature": ["sensor_0"]})

    def test_summarizes_only_saved_sensors_without_mutation(self):
        original = self.validation.copy(deep=True)
        all_features, chosen, monthly = diagnose_selected_sensors(
            self.train, self.validation, self.selected, MODELING_CONFIG.dataset)
        self.assertEqual(len(all_features), 2)
        self.assertEqual(chosen.feature.tolist(), ["sensor_0"])
        self.assertEqual(chosen.missing_ratio_delta.tolist(), [0.5])
        self.assertEqual(monthly.observed_count.tolist(), [4, 2])
        self.assertEqual(monthly.month.tolist(), ["2008-01", "2008-02"])
        self.assertEqual(monthly.failure_count.tolist(), [2, 2])
        self.assertEqual(monthly.failure_ratio.tolist(), [0.5, 0.5])
        pd.testing.assert_frame_equal(self.validation, original)

    def test_rejects_invalid_sensor_lists_and_test_split(self):
        for selected in (pd.DataFrame({"feature": []}), pd.DataFrame({"feature": ["sensor_0"] * 2}),
                         pd.DataFrame({"feature": ["sensor_99"]})):
            with self.assertRaises(ValueError):
                diagnose_selected_sensors(self.train, self.validation, selected, MODELING_CONFIG.dataset)
        with self.assertRaises(ValueError):
            diagnose_selected_sensors(self.train, self.validation.assign(**{SPLIT_ROLE: "test"}),
                                      self.selected, MODELING_CONFIG.dataset)

    def test_writes_diagnostic_files(self):
        tables = diagnose_selected_sensors(self.train, self.validation, self.selected, MODELING_CONFIG.dataset)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sensor_diagnosis(*tables, root / "logs", root / "figures")
            self.assertTrue((root / "figures/selected_sensor_drift.png").is_file())
            self.assertIn("../figures/selected_sensor_drift.png",
                          (root / "logs/sensor_diagnostics.md").read_text(encoding="utf-8"))

    # 상세 그래프도 저장된 선택 센서만 사용하며 관측값을 대치하지 않는다.
    def test_detail_features_and_report(self):
        validate_detail_features(["sensor_0"], self.selected, MODELING_CONFIG.dataset)
        for names in ([], ["sensor_0", "sensor_0"], ["sensor_1"]):
            with self.assertRaises(ValueError):
                validate_detail_features(names, self.selected, MODELING_CONFIG.dataset)
        _, _, monthly = diagnose_selected_sensors(self.train, self.validation, self.selected, MODELING_CONFIG.dataset)
        original = self.validation.copy(deep=True)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sensor_details(self.train, self.validation, ["sensor_0"], monthly, root, root / "figures")
            self.assertTrue((root / "figures/sensor_detail.png").is_file())
            self.assertIn("failure_ratio", (root / "sensor_detail.md").read_text(encoding="utf-8"))
            values = pd.read_csv(root / "detail_value_summary.csv")
            self.assertEqual(values.observed_count.tolist(), [4, 2])
            self.assertEqual(values.missing_ratio.tolist(), [0.0, 0.5])
            with self.assertRaisesRegex(ValueError, "이미 있습니다"):
                write_sensor_details(self.train, self.validation, ["sensor_0"], monthly, root, root / "figures")
        pd.testing.assert_frame_equal(self.validation, original)


if __name__ == "__main__":
    unittest.main()
