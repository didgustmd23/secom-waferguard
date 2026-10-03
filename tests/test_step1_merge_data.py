# ==========================================
# Dataset Profile 기반 원본 병합 테스트
# - SECOM whitespace pair adapter가 metadata와 sensor 열을 올바르게 결합하는지 확인
# - 행 수가 다른 원본 파일은 병합 전에 차단하는지 확인
# ==========================================

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.modeling_config import DatasetSpec, IngestionSpec
from src.step1_merge_data import merge_secom_whitespace_pair


class Step1MergeDataTest(unittest.TestCase):
    @staticmethod
    def _dataset() -> DatasetSpec:
        return DatasetSpec(
            dataset_id="example",
            input_path=Path("unused.csv"),
            label_column="target",
            positive_label="fail",
            negative_label="pass",
            timestamp_column="event_time",
            timestamp_format="%Y-%m-%d",
            feature_selection_mode="prefix",
            feature_column_prefix="sensor_",
            missing_ratio_threshold=0.5,
            drop_zero_variance=True,
        )

    def test_merges_profile_defined_metadata_and_sensor_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            sensor_path = temporary_path / "sensor.data"
            metadata_path = temporary_path / "metadata.data"
            sensor_path.write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")
            metadata_path.write_text(
                "pass 2024-01-01\nfail 2024-01-02\n", encoding="utf-8"
            )
            ingestion = IngestionSpec(
                adapter="secom_whitespace_pair",
                sensor_path=sensor_path,
                metadata_path=metadata_path,
                separator=r"\s+",
                metadata_columns=("target", "event_time"),
            )

            merged_frame = merge_secom_whitespace_pair(self._dataset(), ingestion)

        self.assertEqual(
            list(merged_frame.columns),
            ["target", "event_time", "sensor_0", "sensor_1"],
        )
        self.assertEqual(merged_frame["target"].tolist(), ["pass", "fail"])

    def test_rejects_mismatched_raw_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            sensor_path = temporary_path / "sensor.data"
            metadata_path = temporary_path / "metadata.data"
            sensor_path.write_text("1.0\n2.0\n", encoding="utf-8")
            metadata_path.write_text("pass 2024-01-01\n", encoding="utf-8")
            ingestion = IngestionSpec(
                adapter="secom_whitespace_pair",
                sensor_path=sensor_path,
                metadata_path=metadata_path,
                separator=r"\s+",
                metadata_columns=("target", "event_time"),
            )

            with self.assertRaisesRegex(ValueError, "row counts must match"):
                merge_secom_whitespace_pair(self._dataset(), ingestion)
