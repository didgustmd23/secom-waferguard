# ==========================================
# Dataset Profile 기반 원본 병합 테스트
# - feature_metadata_pair adapter가 metadata와 feature 열을 올바르게 결합하는지 확인
# - 행 수가 다른 원본 파일은 병합 전에 차단하는지 확인
# ==========================================

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.modeling_config import DatasetSpec, IngestionSourceSpec, IngestionSpec
from src.step1_merge_data import merge_feature_metadata_pair


class Step1MergeDataTest(unittest.TestCase):
    @staticmethod
    def _dataset() -> DatasetSpec:
        # 실제 SECOM과 다른 label·timestamp 이름을 사용해 하드코딩 여부를 확인한다.
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

    def test_merges_profile_metadata_and_sensor_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            # 테스트 전용 원본 파일을 만들어 실제 파일 I/O 병합 경로를 검증한다.
            temporary_path = Path(temporary_directory)
            sensor_path = temporary_path / "sensor.data"
            metadata_path = temporary_path / "metadata.data"
            sensor_path.write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")
            metadata_path.write_text(
                "pass 2024-01-01\nfail 2024-01-02\n", encoding="utf-8"
            )
            ingestion = IngestionSpec(
                adapter="feature_metadata_pair",
                sources=(
                    IngestionSourceSpec(
                        name="features",
                        path=sensor_path,
                        read_csv_options={"sep": r"\s+", "header": None},
                    ),
                    IngestionSourceSpec(
                        name="metadata",
                        path=metadata_path,
                        read_csv_options={"sep": r"\s+", "header": None},
                    ),
                ),
                adapter_options={
                    "feature_source": "features",
                    "metadata_source": "metadata",
                    "metadata_columns": ["target", "event_time"],
                },
            )

            # source 이름과 read_csv 옵션을 Profile에서 읽어 canonical table을 만든다.
            merged_frame = merge_feature_metadata_pair(self._dataset(), ingestion)

        # metadata가 먼저, Profile prefix를 적용한 feature가 뒤에 오는지 확인한다.
        self.assertEqual(
            list(merged_frame.columns),
            ["target", "event_time", "sensor_0", "sensor_1"],
        )
        self.assertEqual(merged_frame["target"].tolist(), ["pass", "fail"])

    def test_rejects_mismatched_raw_row_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            # feature와 metadata 행 수가 다른 원본 파일을 의도적으로 구성한다.
            temporary_path = Path(temporary_directory)
            sensor_path = temporary_path / "sensor.data"
            metadata_path = temporary_path / "metadata.data"
            sensor_path.write_text("1.0\n2.0\n", encoding="utf-8")
            metadata_path.write_text("pass 2024-01-01\n", encoding="utf-8")
            ingestion = IngestionSpec(
                adapter="feature_metadata_pair",
                sources=(
                    IngestionSourceSpec(
                        name="features",
                        path=sensor_path,
                        read_csv_options={"sep": r"\s+", "header": None},
                    ),
                    IngestionSourceSpec(
                        name="metadata",
                        path=metadata_path,
                        read_csv_options={"sep": r"\s+", "header": None},
                    ),
                ),
                adapter_options={
                    "feature_source": "features",
                    "metadata_source": "metadata",
                    "metadata_columns": ["target", "event_time"],
                },
            )

            # 행 정렬을 보장할 수 없으므로 concat 전에 오류가 발생해야 한다.
            with self.assertRaisesRegex(ValueError, "행 수가 일치해야"):
                merge_feature_metadata_pair(self._dataset(), ingestion)
