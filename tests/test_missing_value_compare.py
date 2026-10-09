# ==========================================
# 결측 처리 후보의 합성 학습·전처리 검증
# - 기존 모델 설정 유지와 대치/indicator/NaN 분기를 확인
# - 실제 SECOM 데이터나 외부 평가 데이터는 사용하지 않음
# ==========================================

from dataclasses import replace
import json
import unittest

import numpy as np
import pandas as pd

from src.sensor_ml.experiments.missing_value_compare import build_missing_candidates, compare_missing
from src.sensor_ml.experiments.time_weight_compare import DEFAULT_PRESET
from src.modeling_config import MODELING_CONFIG
from src.modeling_models import fit_pipeline
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID


class MissingValueCompareTest(unittest.TestCase):
    def setUp(self):
        """빠른 합성 검증을 위해 트리 수만 2개로 줄인다."""
        self.preset = json.loads(DEFAULT_PRESET.read_text(encoding="utf-8"))
        self.preset["base_parameters"]["n_estimators"] = 2
        self.features = pd.DataFrame({"sensor_a": [1., np.nan, 3., 4., 5., 6.],
                                      "sensor_b": [2., 3., 4., 5., 6., 7.]})
        self.labels = [-1, -1, -1, -1, 1, 1]

    def test_only_missing_step_differs(self):
        """모델과 품질 필터 설정은 같고 결측 처리만 변경된다."""
        candidates = build_missing_candidates(MODELING_CONFIG, self.preset, "v2_m0_ratio",
                                               ["median", "median_indicator", "native_nan"])
        parameters = [pipeline.named_steps["model"].get_params() for pipeline in candidates.values()]
        self.assertTrue(all(params == parameters[0] for params in parameters))
        self.assertEqual(candidates["P2_v2_m0_ratio"].named_steps["imputer"], "passthrough")
        self.assertTrue(candidates["P1_v2_m0_ratio"].named_steps["imputer"].add_indicator)

    def test_nan_and_indicator_are_preserved_as_intended(self):
        """P0/P1은 대치하고 P1은 특징을 추가하며 P2는 NaN을 모델에 전달한다."""
        original = self.features.copy(deep=True)
        candidates = build_missing_candidates(MODELING_CONFIG, self.preset, "v2_m0_ratio",
                                               ["median", "median_indicator", "native_nan"])
        counts = {}
        for name, pipeline in candidates.items():
            model = fit_pipeline(pipeline, self.features, self.labels, MODELING_CONFIG, name)
            transformed = np.asarray(model[:-1].transform(self.features))
            counts[name] = transformed.shape[1]
            self.assertEqual(np.isnan(transformed).any(), name.startswith("P2"))
            self.assertEqual(model.named_steps["model"].effective_scale_pos_weight_, 2.)
            self.assertEqual(model.predict_proba(self.features).shape, (6, 2))
        self.assertEqual(list(counts.values()), [2, 3, 2])
        pd.testing.assert_frame_equal(original, self.features)

    def test_invalid_modes_and_categorical_profile_are_rejected(self):
        """중복·알 수 없는 처리 방식과 지원하지 않는 범주형 경로를 거부한다."""
        for modes in ([], ["median", "median"], ["unknown"]):
            with self.assertRaises(ValueError):
                build_missing_candidates(MODELING_CONFIG, self.preset, "v2_m0_ratio", modes)
        categorical = replace(MODELING_CONFIG, dataset=replace(
            MODELING_CONFIG.dataset, categorical_feature_columns=("sensor_a",)))
        with self.assertRaises(ValueError):
            build_missing_candidates(categorical, self.preset, "v2_m0_ratio", ["median"])

    def test_temporal_comparison_runs_all_three_modes(self):
        """시간순 코어가 세 후보의 내부·외부 fit을 기록하고 입력을 보존한다."""
        rows = []
        for day, timestamp in enumerate(pd.date_range("2008-01-01", periods=16)):
            for position in range(3):
                rows.append({"sensor_value": float(day + position) if position else np.nan,
                             "label": 1 if position == 2 else -1,
                             "timestamp": timestamp.strftime("%d/%m/%Y %H:%M:%S"),
                             SOURCE_ROW_ID: len(rows), SPLIT_ROLE: "train", PROTOCOL_ID: "time:missing-toy"})
        frame = pd.DataFrame(rows)
        original = frame.copy(deep=True)
        tables = compare_missing(frame, MODELING_CONFIG, self.preset, "v2_m0_ratio",
                                 ["median", "median_indicator", "native_nan"])
        pd.testing.assert_frame_equal(original, frame)
        self.assertEqual(len(tables["fit_preprocessing"]), 27)
        for name, records in tables["fit_preprocessing"].groupby("model_name"):
            self.assertTrue(records.retained_sensor_count.eq(1).all())
            self.assertTrue(records.model_input_feature_count.eq(2 if name.startswith("P1") else 1).all())


if __name__ == "__main__":
    unittest.main()
