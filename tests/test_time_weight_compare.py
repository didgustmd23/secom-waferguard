# ==========================================
# 전체 센서 시간순 가중치 실험의 합성 테스트
# - 여섯 후보의 구조와 설정 차이 확인
# - 학습 구간별 실제 가중치 계산과 clone 호환성 검증
# - 외부 평가 입력 차단 및 시간순 OOF 기록 확인
# - 실제 SECOM 데이터나 저장 모델은 사용하지 않음
# ==========================================

import json
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.modeling_config import MODELING_CONFIG
from src.modeling_models import XGBoostClassifierAdapter
from src.split_contract import PROTOCOL_ID, SOURCE_ROW_ID, SPLIT_ROLE
from src.time_weight_compare import DEFAULT_PRESET, build_weight_candidates, compare_time_weights


class TimeWeightCompareTest(unittest.TestCase):
    def setUp(self):
        # 트리 수만 줄여 실행 시간을 제한한다. 비교 구조는 실제 프리셋을 유지한다.
        self.preset = json.loads(DEFAULT_PRESET.read_text(encoding="utf-8"))
        self.preset["base_parameters"]["n_estimators"] = 2

    @staticmethod
    def _frame():
        # 날짜마다 양성 한 건과 서로 다른 정상 수를 넣어 구간별 가중치 차이를 만든다.
        rows = []
        for day, timestamp in enumerate(pd.date_range("2008-01-01", periods=16)):
            for position in range(2 + day % 3):
                rows.append({"sensor_value": float(day + position),
                             "sensor_empty": np.nan,
                             "label": 1 if position == 0 else -1,
                             "timestamp": timestamp.strftime("%d/%m/%Y %H:%M:%S"),
                             SOURCE_ROW_ID: len(rows), SPLIT_ROLE: "train",
                             PROTOCOL_ID: "time:weight-toy"})
        return pd.DataFrame(rows)

    def test_six_candidates_use_all_features(self):
        """특징 선택기 없이 M0/M3와 세 가중치 방식만 교차한다."""
        pipelines = build_weight_candidates(MODELING_CONFIG, self.preset)
        self.assertEqual(len(pipelines), 6)
        for variant in ("m0", "m3"):
            for mode in ("none", "sqrt_ratio", "ratio"):
                pipeline = pipelines[f"v2_{variant}_{mode}"]
                self.assertNotIn("selector", pipeline.named_steps)
                classifier = pipeline.named_steps["model"]
                self.assertEqual(classifier.class_weight_mode, mode)
                self.assertEqual(classifier.max_depth, 3 if variant == "m0" else 2)
                self.assertEqual(classifier.reg_lambda, 1.0 if variant == "m0" else 5.0)

    def test_weights_are_recomputed_and_cloneable(self):
        """가중치는 현재 fit의 label에서 계산되며 생성자 설정은 보존된다."""
        features = np.arange(20, dtype=float).reshape(10, 2)
        for mode, expected in (("none", 1.0), ("sqrt_ratio", 2.0), ("ratio", 4.0)):
            model = XGBoostClassifierAdapter(n_estimators=2, class_weight_mode=mode)
            fitted = clone(model).fit(features, [-1] * 8 + [1] * 2)
            self.assertAlmostEqual(fitted.effective_scale_pos_weight_, expected)
            self.assertEqual(fitted.estimator_.get_params()["scale_pos_weight"], expected)
            fitted.fit(features, [-1] * 5 + [1] * 5)
            self.assertAlmostEqual(fitted.effective_scale_pos_weight_, 1.0)
            self.assertFalse(hasattr(model, "estimator_"))

    def test_temporal_comparison_records_actual_weights(self):
        """내부·외부 fit 기록과 OOF/미래 평가 분리 및 품질 제거를 확인한다."""
        frame = self._frame()
        original = frame.copy(deep=True)
        tables = compare_time_weights(frame, MODELING_CONFIG, self.preset)
        pd.testing.assert_frame_equal(frame, original)
        weights = tables["fit_weights"]
        self.assertEqual(len(weights), 6 * 3 * 3)
        for _, row in weights.iterrows():
            ratio = row.train_negative / row.train_positive
            expected = {"none": 1.0, "sqrt_ratio": np.sqrt(ratio), "ratio": ratio}[row.class_weight_mode]
            self.assertAlmostEqual(row.effective_scale_pos_weight, expected)
        self.assertGreater(weights.loc[weights.class_weight_mode == "ratio", "effective_scale_pos_weight"].nunique(), 1)
        self.assertTrue(tables["quality_filter"].retained_feature_count.eq(1).all())
        self.assertTrue(tables["quality_filter"].high_missing_removed_count.eq(1).all())
        for fold in (1, 2, 3):
            oof = tables["oof_predictions"].query("outer_fold == @fold")
            future = tables["predictions"].query("outer_fold == @fold")
            self.assertFalse(set(oof.source_row_id) & set(future.source_row_id))

    def test_rejects_external_roles_and_invalid_modes(self):
        """Test 역할이나 잘못된 가중치 방식은 학습 전에 거부한다."""
        with self.assertRaises(ValueError):
            compare_time_weights(self._frame().assign(**{SPLIT_ROLE: "test"}), MODELING_CONFIG, self.preset)
        invalid = {**self.preset, "weight_modes": ["none", "ratio"]}
        with self.assertRaises(ValueError):
            build_weight_candidates(MODELING_CONFIG, invalid)
        with self.assertRaises(ValueError):
            XGBoostClassifierAdapter(class_weight_mode="invalid").fit(np.ones((4, 1)), [-1, 1, -1, 1])


if __name__ == "__main__":
    unittest.main()
