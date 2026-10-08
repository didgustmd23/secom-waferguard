# ==========================================
# Dataset Profile 기반 품질 진단 테스트
# - 전체 데이터 품질 결과가 EDA 후보 정보로만 기록되는지 확인
# - 결측률, 상수 feature, timestamp 파싱 오류를 Profile 기준으로 확인
# ==========================================

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.modeling_config import DatasetSpec
from src.data_pipeline.step2_data_check import build_quality_log


class Step2DataCheckTest(unittest.TestCase):
    dataset = DatasetSpec(
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

    def test_records_eda_candidates_without_cleaned_data(self) -> None:
        # 결측률 초과·상수 feature·잘못된 timestamp를 하나의 작은 입력에 구성한다.
        dataframe = pd.DataFrame(
            {
                "target": ["pass", "fail", "fail"],
                "event_time": ["2024-01-01", "not-a-date", None],
                "sensor_sparse": [1.0, None, None],
                "sensor_constant": [7.0, 7.0, 7.0],
            }
        )

        # 전체 데이터 점검은 로그만 반환하며 cleaned feature DataFrame을 만들지 않는다.
        quality_log = build_quality_log(dataframe, self.dataset)

        # 제거 후보라는 사실은 로그 action으로만 확인한다.
        self.assertIn("eda_only", quality_log["action"].tolist())
        high_missing_candidate = quality_log.loc[
            (quality_log["category"] == "feature_missing_ratio")
            & (quality_log["item"] == "sensor_sparse"),
            "action",
        ].iloc[0]
        self.assertEqual(high_missing_candidate, "eda_candidate_drop")
        # Profile timestamp 형식에 맞지 않는 값 하나를 파싱 실패로 기록해야 한다.
        parse_failure = quality_log.loc[
            (quality_log["category"] == "timestamp")
            & (quality_log["item"] == "parse_failure_count"),
            "value",
        ].iloc[0]
        self.assertEqual(parse_failure, 1)
        # 두 feature가 각각 후보가 되어 EDA 기준 잔여 수는 0이 된다.
        remaining_feature_count = quality_log.loc[
            (quality_log["category"] == "feature_summary")
            & (quality_log["item"] == "remaining_feature_candidate_count"),
            "value",
        ].iloc[0]
        self.assertEqual(remaining_feature_count, 0)
