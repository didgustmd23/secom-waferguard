# ==========================================
# 시간순 중요도 안정성 집계·학습 범위·저장 검증
# - 합성 표와 학습 함수 대체 객체를 사용
# - 실제 SECOM 학습과 외부 평가 데이터 접근은 하지 않음
# ==========================================

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.sensor_ml.diagnostics.sensor_stability import compare_stability, save_results, summarize_stability


class SensorStabilityTest(unittest.TestCase):
    def records(self):
        """학습된 0 중요도와 제거 센서를 구분하는 작은 Fold 기록을 만든다."""
        return pd.DataFrame({
            "feature": ["a", "b", "c"] * 2,
            "fold": [1] * 3 + [2] * 3,
            "importance_rank": pd.array([1, 2, None, 2, 1, 3], dtype="Int64"),
            "in_topk": [True, False, False, False, True, False],
        })

    def test_summary_preserves_missing_and_overlap(self):
        """제거된 Fold의 순위를 제외해 평균을 계산하고 공통 센서를 집계한다."""
        stability, overlap = summarize_stability(self.records())
        self.assertEqual(stability.loc["c", "used_fold_count"], 1)
        self.assertEqual(stability.loc["c", "mean_rank_when_used"], 3)
        self.assertTrue(pd.isna(stability.loc["c", "fold_1_rank"]))
        self.assertEqual(stability.loc["a", "topk_count"], 1)
        self.assertEqual(overlap.iloc[0].common_sensor_count, 0)
        self.assertEqual(overlap.iloc[0].jaccard_similarity, 0)

    def test_only_past_training_rows_are_fitted(self):
        """미래 평가 인덱스는 학습 함수에 전달하지 않고 Top-K를 정확히 제한한다."""
        frame = pd.DataFrame({"timestamp": pd.date_range("2008-01-01", periods=6)})
        config = SimpleNamespace(dataset=SimpleNamespace(
            timestamp_column="timestamp", timestamp_format=None))
        preset = {"outer_splits": 2}
        table = pd.DataFrame({"feature": ["a", "b", "c"],
                              "status": ["used", "used", "removed_constant"],
                              "importance_rank": pd.array([1, 1, None], dtype="Int64")})
        with patch("src.sensor_ml.diagnostics.sensor_stability.validate_train_role"), patch(
            "src.sensor_ml.diagnostics.sensor_stability.temporal_folds", return_value=[([0, 1], [2, 3]), ([0, 1, 2, 3], [4, 5])]
        ), patch("src.sensor_ml.diagnostics.sensor_stability.fit_importance_candidate", return_value=(table, None)) as fit:
            result = compare_stability(frame, config, preset, "toy", top_k=1)
        self.assertEqual(fit.call_args_list[0].args[0].index.tolist(), [0, 1])
        self.assertEqual(fit.call_args_list[1].args[0].index.tolist(), [0, 1, 2, 3])
        self.assertEqual(result["fold_importances"].groupby("fold").in_topk.sum().tolist(), [1, 1])
        self.assertEqual(result["overlap"].iloc[0].common_sensor_count, 1)

    def test_save_records_diagnostic_scope(self):
        """집계 파일과 진단용 실행 범위가 저장되는지 확인한다."""
        stability, overlap = summarize_stability(self.records())
        tables = {"stability": stability.reset_index(), "overlap": overlap,
                  "fold_importances": self.records(), "fold_summary": pd.DataFrame({"fold": [1, 2]})}
        config = SimpleNamespace(dataset=SimpleNamespace(dataset_id="toy"))
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            args = SimpleNamespace(output_dir=Path(temporary), train=Path("toy.csv"),
                                   config=Path("config.json"), model="toy", top_k=1, n_jobs=1)
            save_results(args, config, {"outer_splits": 2}, tables)
            record = json.loads((args.output_dir / "execution.json").read_text(encoding="utf-8"))
            self.assertFalse(record["test_used"])
            self.assertFalse(record["final_sensor_selection"])
            for name in tables:
                self.assertTrue((args.output_dir / f"{name}.csv").is_file())
            self.assertTrue((args.output_dir / "stability.md").is_file())


if __name__ == "__main__":
    unittest.main()
