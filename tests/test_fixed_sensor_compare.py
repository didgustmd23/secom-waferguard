# ==========================================
# 고정 센서 검증의 선택 출처·동일 입력·미래 정보 차단 확인
# - 실제 SECOM 대신 작은 합성 데이터와 트리 3개 사용
# - 최초 과거 선택 이후 모든 평가 구간의 센서 목록을 유지
# - 미래 label 변경이 초기 목록과 과거 모델에 영향을 주지 않음
# ==========================================
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.sensor_ml.experiments.fixed_sensor_compare import compare_fixed_sensors, parse_args, execution_json, main
from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID, temporal_folds


def synthetic_frame():
    """센서 60개·균형 label·중복 timestamp를 가진 작은 학습 계약을 만든다."""
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(48, 60)),
                         columns=[f"sensor_{index}" for index in range(60)])
    frame["sensor_empty"] = np.nan
    frame["label"] = [-1, 1] * 24
    frame["timestamp"] = [date.strftime("%d/%m/%Y %H:%M:%S")
                          for date in pd.date_range("2008-01-01", periods=24) for _ in range(2)]
    frame[SOURCE_ROW_ID] = np.arange(48)
    frame[SPLIT_ROLE] = "train"
    frame[PROTOCOL_ID] = "time:fixed-sensor-toy"
    return frame


class FixedSensorComparisonTest(unittest.TestCase):
    def test_later_selection_only_evaluates_future(self):
        """두 번째 과거에서 선택해 세 번째 구간만 평가하며 미래 변경 영향을 차단한다."""
        frame = synthetic_frame()
        options = dict(n_estimators=3, repeats=2, selection_fold=2, evaluate_from_fold=3)
        result = compare_fixed_sensors(frame, MODELING_CONFIG, **options)
        folds = list(temporal_folds(frame, MODELING_CONFIG.dataset, 3))
        self.assertEqual(result["selection_rows"].source_row_index.tolist(),
                         frame.iloc[folds[1][0]].index.tolist())
        self.assertEqual(set(result["fold_results"].outer_fold), {3})
        self.assertTrue(result["summary"].evaluated_folds.eq(1).all())
        self.assertFalse(set(result["selection_rows"].source_row_id)
                         & set(result["predictions"].source_row_id))
        changed = frame.copy()
        changed.loc[folds[2][1], "label"] *= -1
        changed.loc[folds[2][1], "sensor_0"] += 100
        other = compare_fixed_sensors(changed, MODELING_CONFIG, **options)
        pd.testing.assert_frame_equal(result["fixed_sensors"], other["fixed_sensors"])
        pd.testing.assert_frame_equal(result["selected_features"], other["selected_features"])
        for kwargs in ({"selection_fold": 0}, {"selection_fold": True},
                       {"selection_fold": 4}, {"selection_fold": 2, "evaluate_from_fold": 1},
                       {"evaluate_from_fold": 4}):
            with self.assertRaises(ValueError):
                compare_fixed_sensors(frame, MODELING_CONFIG, n_estimators=3, repeats=2, **kwargs)
        args = parse_args(["--train", "example.csv", "--output-dir", "logs/fixed_sensor_cli_test",
                           "--selection-fold", "2", "--evaluate-from-fold", "3"])
        record = json.loads(execution_json(args, MODELING_CONFIG))
        self.assertEqual(record["selection_fold"], 2)
        self.assertEqual(record["evaluate_from_fold"], 3)

    def test_execution_paths_and_early_validation(self):
        """실제 설정의 중첩 Path를 변환하고 저장 형식 오류는 학습 전에 차단한다."""
        args = parse_args(["--train", "example.csv", "--output-dir", "logs/fixed_sensor_cli_test"])
        record = json.loads(execution_json(args, MODELING_CONFIG))
        self.assertEqual(record["config"]["dataset"]["input_path"],
                         str(MODELING_CONFIG.dataset.input_path))
        self.assertEqual(record["train_path"], str(Path("example.csv").resolve()))
        self.assertFalse(record["test_used"])
        self.assertEqual(record["repeats"], 5)
        # 실제 데이터 읽기나 모델 학습 없이 직렬화 실패의 순서만 검증한다.
        with patch("src.sensor_ml.experiments.fixed_sensor_compare.parse_args", return_value=args), \
             patch("src.sensor_ml.experiments.fixed_sensor_compare.load_modeling_config", return_value=MODELING_CONFIG), \
             patch("src.sensor_ml.experiments.fixed_sensor_compare.execution_json", side_effect=TypeError("형식 오류")), \
             patch("src.sensor_ml.experiments.fixed_sensor_compare.compare_fixed_sensors") as compare:
            with self.assertRaises(TypeError):
                main()
            compare.assert_not_called()

    def test_fixed_names_and_paired_rows(self):
        """고정 순서·같은 미래 행·동적 가중치와 선택 출처를 검증한다."""
        frame = synthetic_frame()
        result = compare_fixed_sensors(frame, MODELING_CONFIG, n_estimators=3, repeats=2)
        names = result["fixed_sensors"].feature.tolist()
        self.assertEqual(len(names), 20)
        selected = result["selected_features"]
        for _, group in selected[selected.model_name.eq("fixed_topk")].groupby("outer_fold"):
            self.assertEqual(group.feature.tolist(), names)
        preds = result["predictions"]
        for _, group in preds.groupby("outer_fold"):
            adaptive = group[group.model_name.eq("adaptive_topk")].reset_index(drop=True)
            fixed = group[group.model_name.eq("fixed_topk")].reset_index(drop=True)
            pd.testing.assert_frame_equal(adaptive[["source_row_id", "label"]],
                                          fixed[["source_row_id", "label"]])
            # 첫 구간은 같은 초기 목록·학습 데이터이므로 두 경로의 확률도 같아야 한다.
            if int(group.outer_fold.iloc[0]) == 1:
                np.testing.assert_allclose(adaptive.positive_score, fixed.positive_score,
                                           rtol=0, atol=1e-12)
        first = list(temporal_folds(frame, MODELING_CONFIG.dataset, 3))[0][0]
        self.assertEqual(result["selection_rows"].source_row_index.tolist(), frame.iloc[first].index.tolist())
        self.assertFalse(set(result["selection_rows"].source_row_id) & set(preds.source_row_id))
        weights = result["fit_weights"]
        np.testing.assert_allclose(weights.effective_scale_pos_weight,
                                   weights.train_negative / weights.train_positive)
        # 기본 문턱만 사용하며 다음 단계의 OOF 정책 선택과 섞지 않는다.
        self.assertTrue(result["fold_results"].threshold.eq(MODELING_CONFIG.experiment.default_threshold).all())
        self.assertTrue(result["fold_results"].classifier_max_depth.eq(2).all())
        self.assertTrue(result["fold_results"].classifier_reg_lambda.eq(1).all())
        self.assertEqual(len(result["selection_bootstrap"]), 2)

    def test_future_changes_do_not_change_selection(self):
        """평가 미래의 값·label 변경이 고정 목록과 이전 예측을 바꾸지 않는다."""
        frame = synthetic_frame()
        baseline = compare_fixed_sensors(frame, MODELING_CONFIG, n_estimators=3, repeats=2)
        changed = frame.copy()
        last = list(temporal_folds(frame, MODELING_CONFIG.dataset, 3))[-1][1]
        changed.loc[last, "label"] *= -1
        changed.loc[last, "sensor_0"] += 1000
        result = compare_fixed_sensors(changed, MODELING_CONFIG, n_estimators=3, repeats=2)
        pd.testing.assert_frame_equal(baseline["fixed_sensors"], result["fixed_sensors"])
        pd.testing.assert_frame_equal(baseline["selected_features"], result["selected_features"])
        for table in ("predictions", "fold_results"):
            a = baseline[table]; b = result[table]
            pd.testing.assert_frame_equal(a[a.outer_fold.lt(3)].reset_index(drop=True),
                                          b[b.outer_fold.lt(3)].reset_index(drop=True))

    def test_invalid_inputs_and_cli(self):
        """Test 역할·잘못된 반복 수와 트리 수를 차단하고 CLI를 확인한다."""
        frame = synthetic_frame()
        for kwargs in ({"repeats": 1}, {"n_estimators": 0}, {"n_estimators": True}):
            with self.assertRaises(ValueError):
                compare_fixed_sensors(frame, MODELING_CONFIG, **kwargs)
        frame[SPLIT_ROLE] = "test"
        with self.assertRaisesRegex(ValueError, "train 역할"):
            compare_fixed_sensors(frame, MODELING_CONFIG, n_estimators=3, repeats=2)
        args = parse_args(["--train", "example.csv", "--output-dir", "logs/fixed_sensor_cli_test"])
        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.n_estimators, 300)


if __name__ == "__main__":
    unittest.main()
