# ==========================================
# 변경 범위에 따른 실행 의존 관계 검증
# - 전처리·모델·데이터·split 변경은 OOF 생성 후 시간 검증 실행
# - threshold 변경은 성공한 기존 OOF를 사용해 시간 검증만 실행
# - 선행 실패 후 남은 CSV가 후속 평가에 사용되지 않는지 확인
# ==========================================

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from dataclasses import replace

import pandas as pd

from src.evaluation import run_evaluation as runner
from src.modeling_config import MODELING_CONFIG


class EvaluationRunnerTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # 프로젝트 운영 정책이 달라져도 단위 테스트의 조건은 70·20으로 유지한다.
        self.config = replace(MODELING_CONFIG, threshold_policy=replace(
            MODELING_CONFIG.threshold_policy, min_recall=0.7, max_reinspection_ratio=0.2))
        self.table = pd.DataFrame({"threshold": [0.2, 0.4], "f1": [0.3, 0.5],
                                  "support": [100, 100], "positive_support": [10, 10],
                                  "true_positive": [9, 7], "false_positive": [11, 3],
                                  "recall": [0.9, 0.7]})
        self.oof = Mock(side_effect=self._write_oof)
        self.validation = Mock(return_value=pd.DataFrame({"recall": [0.0]}))
        self.order = Mock()
        self.order.attach_mock(self.oof, "oof")
        self.order.attach_mock(self.validation, "validation")
        for name, mock in (("load_modeling_config", Mock(return_value=self.config)),
                           ("run_threshold_oof_from_file", self.oof),
                           ("compare_time_validation_from_files", self.validation)):
            patcher = patch.object(runner, name, mock)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_oof(self, config, train, oof_path, threshold_path, **kwargs):
        # 실패 시 과거 CSV가 남는 상황을 검증하기 위해 실제 파일로 저장한다.
        self.table.to_csv(threshold_path, index=False)
        return pd.DataFrame(), self.table.copy(), 0.0

    def _run(self, change, threshold=None):
        return runner.run_evaluation("config.json", "train.csv", "valid.csv", self.root,
                                     change=change, threshold=threshold)

    def _state(self):
        return json.loads((self.root / "execution.json").read_text(encoding="utf-8"))

    def test_changed_inputs_regenerate_oof_before_validation(self):
        for change in ("preprocessing", "model", "data", "split"):
            with self.subTest(change=change):
                self.order.reset_mock()
                self._run(change)
                self.assertEqual([call[0] for call in self.order.mock_calls], ["oof", "validation"])
                self.assertEqual(self.validation.call_args.kwargs["threshold"], 0.4)
                self.assertEqual(self._state()["status"], "completed")
                self.assertEqual(self._state()["completed_steps"], ["oof", "time_validation"])

    def test_threshold_change_reuses_successful_oof(self):
        self._run("model")
        self.order.reset_mock()
        self._run("threshold", 0.2)
        self.oof.assert_not_called()
        self.validation.assert_called_once()
        self.assertEqual(self.validation.call_args.kwargs["threshold"], 0.2)
        self.assertTrue(self._state()["oof_ready"])
        # threshold를 연속 변경해도 성공한 같은 OOF 결과를 재사용한다.
        self._run("threshold", 0.4)
        self.oof.assert_not_called()

    def test_failed_oof_blocks_validation_and_stale_log_reuse(self):
        self._run("model")
        self.order.reset_mock()
        self.oof.side_effect = RuntimeError("OOF 생성 실패")
        with self.assertRaisesRegex(RuntimeError, "OOF 생성 실패"):
            self._run("preprocessing")
        self.validation.assert_not_called()
        self.assertTrue((self.root / "threshold_compare.csv").exists())
        self.assertEqual(self._state()["status"], "failed")
        with self.assertRaisesRegex(ValueError, "완료"):
            self._run("threshold", 0.2)
        self.validation.assert_not_called()

    def test_threshold_change_requires_value_and_successful_run(self):
        with self.assertRaisesRegex(ValueError, "threshold 값을 명시"):
            self._run("threshold")
        with self.assertRaisesRegex(ValueError, "선행 OOF 실행이 없습니다"):
            self._run("threshold", 0.2)

    def test_validation_failure_is_recorded(self):
        self.validation.side_effect = ValueError("학습 계약 불일치")
        with self.assertRaisesRegex(ValueError, "학습 계약 불일치"):
            self._run("model")
        self.assertEqual(self._state()["status"], "failed")
        self.assertEqual(self._state()["completed_steps"], ["oof"])

    def test_policy_change_reselects_without_regenerating_oof(self):
        self._run("model")
        self.order.reset_mock()
        # 정책값만 바꾸면 기존 점수에서 후보를 다시 선택한다.
        config = replace(self.config, threshold_policy=replace(
            self.config.threshold_policy, min_recall=0.9))
        with patch.object(runner, "load_modeling_config", return_value=config):
            self._run("policy")
        self.oof.assert_not_called()
        self.assertEqual(self.validation.call_args.kwargs["threshold"], 0.2)
        self.assertEqual(self._state()["policy_status"], "feasible")

    def test_infeasible_policy_skips_validation_and_can_be_retried(self):
        config = replace(self.config, threshold_policy=replace(
            self.config.threshold_policy, min_recall=1.0))
        with patch.object(runner, "load_modeling_config", return_value=config):
            result = self._run("model")
        self.assertEqual(result.iloc[0].policy_status, "infeasible")
        self.validation.assert_not_called()
        self.assertIsNone(self._state()["threshold"])
        self.assertEqual(self._state()["time_validation_status"], "skipped_policy_infeasible")
        self.order.reset_mock()
        self._run("policy")
        self.oof.assert_not_called()
        self.validation.assert_called_once()

    def test_reuse_rejects_changed_training_config(self):
        self._run("model")
        self.order.reset_mock()
        config = replace(self.config, experiment=replace(
            self.config.experiment, default_threshold=0.3))
        with patch.object(runner, "load_modeling_config", return_value=config):
            with self.assertRaisesRegex(ValueError, "OOF부터"):
                self._run("policy")
        self.validation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
