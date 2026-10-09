# ==========================================
# 동결 평가의 합성 데이터 방어 검증
# - 실제 Test나 후보 묶음을 읽지 않고 역할·시간·ID 오류를 검사
# - 평가에 fit이나 문턱 변경이 없고 고정 확률로 지표를 계산하는지 확인
# - 다른 출력 경로를 지정해도 같은 묶음의 반복 실행은 차단
# ==========================================

from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID
from src.sensor_ml.evaluation.evaluate_frozen_model import validate_final_pair, evaluate_saved_predictions, reserve_evaluation
from src.sensor_ml.evaluation.evaluate_frozen_model import validate_frozen_context, save_evaluation_results, main


class FrozenEvaluationTest(unittest.TestCase):
    def setUp(self):
        """서로 다른 시점과 원본 ID의 작은 합성 split을 만든다."""
        self.train = pd.DataFrame({"sensor_a": [1., 2.], "sensor_b": [3., 4.],
                                   "label": [-1, 1], "timestamp": ["01/01/2008 00:00:00"] * 2,
                                   SOURCE_ROW_ID: [0, 1], SPLIT_ROLE: ["train"] * 2,
                                   PROTOCOL_ID: ["time:toy"] * 2})
        self.test = self.train.copy()
        self.test["timestamp"] = "02/01/2008 00:00:00"
        self.test[SOURCE_ROW_ID] = [2, 3]
        self.test[SPLIT_ROLE] = "test"

    def test_split_guards(self):
        """정상 split은 허용하되 역할·시간·계약·원본 중복은 차단한다."""
        features, labels = validate_final_pair(self.train, self.test, MODELING_CONFIG.dataset, "time:toy")
        self.assertEqual(features.columns.tolist(), ["sensor_a", "sensor_b"])
        self.assertEqual(labels.tolist(), [-1, 1])
        self.assertTrue(self.test[SPLIT_ROLE].eq("test").all())
        for key, value in ((SPLIT_ROLE, "validation"), (SOURCE_ROW_ID, [0, 3]),
                           (PROTOCOL_ID, "time:other"), ("timestamp", "01/01/2008 00:00:00")):
            changed = self.test.copy()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_final_pair(self.train, changed, MODELING_CONFIG.dataset, "time:toy")

    def test_fixed_scores_without_fit(self):
        """저장 문턱과 고정 확률만 사용하여 지표를 계산한다."""
        inference = Mock(sensors=("sensor_a",), positive_label=1, _negative_label=-1, threshold=0.4)
        inference.predict.return_value = pd.DataFrame({"positive_score": [0.3, 0.8]})
        metrics, _ = evaluate_saved_predictions(inference, self.test, self.test.label)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["threshold"], 0.4)
        self.assertEqual(metrics["reinspection_ratio"], 0.5)
        inference.fit.assert_not_called()

    def test_repeat_is_blocked_across_output_paths(self):
        """출력 폴더만 바꿔 같은 묶음을 반복 평가하지 못한다."""
        with TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            marker = reserve_evaluation(bundle, Path(temporary) / "first", {"status": "started"})
            self.assertTrue(marker.exists())
            with self.assertRaisesRegex(ValueError, "이미"):
                reserve_evaluation(bundle, Path(temporary) / "second", {"status": "started"})

    def test_context_reports_each_mismatch(self):
        """기존 검사 대상을 유지하고 일곱 항목의 불일치 원인을 구분한다."""
        provenance = {"model_parameters": {"max_depth": 2},
                      "rf_selector_parameters": {"max_depth": None},
                      "training_protocol_id": "time:toy", "selected_sensors": ["sensor_a"]}
        manifest = {"provenance": provenance, "threshold": 0.4,
                    "positive_label": 1, "sensors": ["sensor_a"]}
        reference = {**deepcopy(provenance), "threshold": 0.4,
                     "config": {"dataset": json.loads(json.dumps(asdict(MODELING_CONFIG.dataset), default=str))}}
        validate_frozen_context(MODELING_CONFIG, manifest, reference)
        # 기준 실행 기록 또는 manifest 한 항목만 바꾸어 해당 메시지를 확인한다.
        for key, message in (("dataset", "데이터 정의"), ("model_parameters", "모델 설정"),
                             ("rf_selector_parameters", "RF 선택기 설정"),
                             ("training_protocol_id", "학습 생성 계약"), ("threshold", "문턱"),
                             ("positive_label", "양성 label"), ("sensors", "선택 센서")):
            changed_manifest, changed_reference = deepcopy(manifest), deepcopy(reference)
            if key == "dataset":
                changed_reference["config"][key] = {}
            elif key in ("positive_label", "sensors"):
                changed_manifest[key] = "different"
            else:
                changed_reference[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                validate_frozen_context(MODELING_CONFIG, changed_manifest, changed_reference)

    def test_input_failure_precedes_reservation(self):
        """입력 검증에서 실패하면 실행 표식·평가·저장에 진입하지 않는다."""
        with patch("src.sensor_ml.evaluation.evaluate_frozen_model.parse_args"), \
                patch("src.sensor_ml.evaluation.evaluate_frozen_model.load_evaluation_context", return_value=(Mock(), {}, {}, MODELING_CONFIG)), \
                patch("src.sensor_ml.evaluation.evaluate_frozen_model.prepare_evaluation_input", side_effect=ValueError("입력 오류")), \
                patch("src.sensor_ml.evaluation.evaluate_frozen_model.reserve_evaluation") as reserve, \
                patch("src.sensor_ml.evaluation.evaluate_frozen_model.evaluate_saved_predictions") as evaluate, \
                patch("src.sensor_ml.evaluation.evaluate_frozen_model.save_evaluation_results") as save:
            with self.assertRaisesRegex(ValueError, "입력 오류"):
                main()
            reserve.assert_not_called()
            evaluate.assert_not_called()
            save.assert_not_called()

    def test_outputs_keep_schema_and_completed_record(self):
        """정상 저장의 파일 이름·열 순서와 완료 기록 일치를 확인한다."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            marker = directory / "evaluation_started.json"
            marker.write_text('{"status": "started"}', encoding="utf-8")
            metrics = {"recall": 1.0, "reinspection_ratio": 0.5, "policy_feasible": True}
            predictions = pd.DataFrame({"positive_score": [0.3, 0.8]})
            policy = {"min_recall": 0.8, "max_reinspection_ratio": 0.5}
            save_evaluation_results(directory, marker, {"status": "started"}, metrics,
                                    predictions, self.test, self.test.label, policy)
            self.assertEqual(pd.read_csv(directory / "test_predictions.csv").columns.tolist(),
                             ["source_row_id", "label", "positive_score"])
            self.assertEqual(pd.read_csv(directory / "test_result.csv").recall.iloc[0], 1.0)
            completed = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["threshold_policy"], policy)
            self.assertEqual(completed, json.loads((directory / "evaluation.json").read_text(encoding="utf-8")))
            self.assertIn("재학습·센서 재선택·threshold 탐색을 하지 않았습니다.",
                          (directory / "evaluation.md").read_text(encoding="utf-8"))

    def test_save_failure_keeps_started_marker(self):
        """CSV 저장이 실패하면 완료 표식으로 바꾸거나 시작 표식을 지우지 않는다."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            marker = directory / "evaluation_started.json"
            marker.write_text('{"status": "started"}', encoding="utf-8")
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("저장 실패")):
                with self.assertRaisesRegex(OSError, "저장 실패"):
                    save_evaluation_results(directory, marker, {"status": "started"}, {},
                                            pd.DataFrame(), self.test, self.test.label, {})
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["status"], "started")


if __name__ == "__main__":
    unittest.main()
