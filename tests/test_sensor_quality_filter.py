# ==========================================
# 센서 품질 기준에 따른 실제 제거 요구사항 검증
# - 결측률 기준 초과 및 상수 센서를 모델 전처리에서 제거하는지 확인
# - 정확히 50%인 결측률의 경계값과 Profile 설정 반영 여부를 확인
# - Validation에는 Train에서 결정한 센서 목록을 그대로 적용하는지 확인
# - 전체 데이터 EDA 후보 수는 실제 Pipeline 제거 결과와 별도로 검증
# - 제거 단계가 미구현된 상태에서는 관련 테스트가 실패하여 누락을 드러냄
# ==========================================

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling_config import MODELING_CONFIG
from src.modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
import json
from src.step2_data_check import build_quality_log


class SensorQualityFilterTest(unittest.TestCase):
    dataset = replace(
        MODELING_CONFIG.dataset,
        label_column="target",
        timestamp_column=None,
        timestamp_format=None,
        categorical_feature_columns=(),
        missing_ratio_threshold=0.5,
        drop_zero_variance=True,
    )

    # ==========================================
    # 실제 모델들이 공유하는 전처리 단계로 검증용 Pipeline 생성
    # - 컬럼명과 행 인덱스를 확인하기 위해 pandas 출력을 사용
    # ==========================================
    def _pipeline(self, dataset=None):
        return Pipeline(preprocessing_steps(
            self.dataset if dataset is None else dataset,
            pandas_output=True,
        ))

    def test_drops_high_missing_and_empty_sensors(self):
        # 결측률 75%와 100% 센서는 대치 전에 제거하고 정상 센서는 유지한다.
        frame = pd.DataFrame({
            "sensor_sparse": [1.0, np.nan, np.nan, np.nan],
            "sensor_empty": [np.nan] * 4,
            "sensor_valid": [1.0, 2.0, 3.0, 4.0],
        }, index=[10, 20, 30, 40])
        transformed = self._pipeline().fit_transform(frame)

        self.assertEqual(transformed.columns.tolist(), ["sensor_valid"])
        self.assertEqual(transformed.index.tolist(), frame.index.tolist())
        self.assertEqual(frame.shape[1] - transformed.shape[1], 2)

    def test_keeps_exactly_half_missing_sensor(self):
        # '50% 초과'이므로 정확히 50%이며 관측값이 서로 다른 센서는 유지한다.
        frame = pd.DataFrame({"sensor_half": [1.0, np.nan, 3.0, np.nan]})
        transformed = self._pipeline().fit_transform(frame)

        self.assertEqual(transformed.columns.tolist(), ["sensor_half"])
        self.assertEqual(transformed.sensor_half.tolist(), [1.0, 2.0, 3.0, 2.0])

    def test_drops_constants_before_imputation(self):
        # 결측을 제외한 값이 하나뿐인 센서도 완전한 상수 센서와 함께 제거한다.
        frame = pd.DataFrame({
            "sensor_constant": [7.0] * 4,
            "sensor_constant_missing": [9.0, 9.0, np.nan, 9.0],
            "sensor_valid": [1.0, 2.0, 3.0, 4.0],
        })
        transformed = self._pipeline().fit_transform(frame)

        self.assertEqual(transformed.columns.tolist(), ["sensor_valid"])
        self.assertEqual(frame.shape[1] - transformed.shape[1], 2)

    def test_respects_profile_missing_threshold(self):
        # 다른 Profile의 25% 기준을 적용하고 상수 제거는 비활성화한다.
        dataset = replace(self.dataset, missing_ratio_threshold=0.25,
                          drop_zero_variance=False)
        frame = pd.DataFrame({
            "sensor_half": [1.0, np.nan, 3.0, np.nan],
            "sensor_constant": [7.0] * 4,
        })
        transformed = self._pipeline(dataset).fit_transform(frame)

        self.assertEqual(transformed.columns.tolist(), ["sensor_constant"])

    def test_can_disable_quality_removal(self):
        # 결측률 기준 100%와 상수 제거 비활성화 설정에서는 모든 센서를 유지한다.
        dataset = replace(self.dataset, missing_ratio_threshold=1.0,
                          drop_zero_variance=False)
        frame = pd.DataFrame({
            "sensor_empty": [np.nan] * 4,
            "sensor_constant": [7.0] * 4,
        })
        transformed = self._pipeline(dataset).fit_transform(frame)

        self.assertEqual(transformed.columns.tolist(), frame.columns.tolist())

    def test_validation_uses_train_sensor_selection(self):
        # Train에서는 sparse/constant를 제거하고 half/valid는 유지해야 한다.
        train = pd.DataFrame({
            "sensor_sparse": [1.0, np.nan, np.nan, np.nan],
            "sensor_constant": [7.0] * 4,
            "sensor_half": [1.0, np.nan, 3.0, np.nan],
            "sensor_valid": [2.0, 4.0, 6.0, 8.0],
        })
        # Validation에서 제거 대상이 정상화되고 유지 대상이 결측/상수가 되어도
        # 센서 선택은 바뀌지 않으며 Train median으로 결측을 채워야 한다.
        validation = pd.DataFrame({
            "sensor_sparse": [10.0, 20.0],
            "sensor_constant": [11.0, 12.0],
            "sensor_half": [np.nan, np.nan],
            "sensor_valid": [100.0, 100.0],
        }, index=[100, 101])
        pipeline = self._pipeline().fit(train)
        transformed = pipeline.transform(validation)

        expected = pd.DataFrame({"sensor_half": [2.0, 2.0],
                                 "sensor_valid": [100.0, 100.0]}, index=[100, 101])
        pd.testing.assert_frame_equal(transformed, expected)
        self.assertEqual(pipeline.transform(train).columns.tolist(), expected.columns.tolist())

    def test_eda_counts_sequential_drop_candidates(self):
        # sparse는 상수이기도 하지만 결측률 단계에서 집계해 중복 차감하지 않는다.
        frame = pd.DataFrame({
            "sensor_sparse": [1.0, np.nan, np.nan, np.nan],
            "sensor_constant": [7.0, 7.0, np.nan, 7.0],
            "sensor_half": [1.0, np.nan, 3.0, np.nan],
            "sensor_valid": [1.0, 2.0, 3.0, 4.0],
            "target": [-1, -1, 1, 1],
        })
        quality_log = build_quality_log(frame, self.dataset)
        counts = quality_log.loc[quality_log.category == "feature_summary"].set_index("item")["value"]

        self.assertEqual(counts["feature_count"], 4)
        self.assertEqual(counts["high_missing_candidate_count"], 1)
        self.assertEqual(counts["constant_candidate_count"], 1)
        self.assertEqual(counts["remaining_feature_candidate_count"], 2)

    def test_records_actual_removal_counts_and_names(self):
        # 결측률 초과이면서 상수인 센서는 결측 기준에서 한 번만 집계한다.
        frame = pd.DataFrame({
            "sensor_sparse": [1.0, np.nan, np.nan, np.nan],
            "sensor_constant": [7.0] * 4,
            "sensor_valid": [1.0, 2.0, 3.0, 4.0],
        })
        pipeline = self._pipeline().fit(frame)
        record = json.loads(quality_filter_json([quality_filter_record(pipeline, fold=2)]))[0]
        self.assertEqual(record["fold"], 2)
        self.assertEqual(record["high_missing_removed_count"], 1)
        self.assertEqual(record["constant_removed_count"], 1)
        self.assertEqual(record["removed_feature_count"], 2)
        self.assertEqual(record["retained_feature_count"], 1)
        self.assertEqual(record["high_missing_features"], ["sensor_sparse"])
        self.assertEqual(record["constant_features"], ["sensor_constant"])

    def test_removes_declared_categorical_sensor(self):
        # 선언된 범주형 센서가 제거되어도 남은 수치형 전처리가 정상 동작해야 한다.
        dataset = replace(self.dataset, categorical_feature_columns=("sensor_machine",))
        frame = pd.DataFrame({"sensor_machine": ["A"] * 4,
                              "sensor_valid": [1.0, 2.0, 3.0, 4.0]})
        transformed = self._pipeline(dataset).fit_transform(frame)
        self.assertEqual(transformed.shape, (4, 1))

    def test_rejects_removal_of_all_sensors(self):
        # 남는 센서가 없으면 모델 학습 전에 원인을 명시하고 중단한다.
        with self.assertRaisesRegex(ValueError, "남은 feature가 없습니다"):
            self._pipeline().fit(pd.DataFrame({"sensor_constant": [7.0] * 4}))


if __name__ == "__main__":
    unittest.main()
