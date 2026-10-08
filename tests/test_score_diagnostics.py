# ==========================================
# 확률 진단 요약·센서 교집합·문서 생성 테스트
# - 합성 예측만 사용하며 모델 학습과 SECOM 평가는 수행하지 않음
# - 클래스 분모와 고정 문턱 적용, 상대 그림 링크를 확인
# ==========================================

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.modeling_config import MODELING_CONFIG
from src.score_diagnostics import summarize_scores, sensor_overlap, write_score_diagnostics


class ScoreDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.oof = pd.DataFrame({"label": [-1, -1, 1, 1], "oof_positive_score": [0.1, 0.2, 0.8, 0.9]})
        self.validation = pd.DataFrame({"label": [-1, -1, 1, 1], "positive_score": [0.1, 0.2, 0.3, 0.8]})
        self.folds = pd.DataFrame({"cv_fold": [1, 1, 2, 2], "feature": ["a", "b", "b", "c"]})
        self.selected = pd.DataFrame({"feature": ["a", "b"]})

    def test_class_distribution_and_fixed_threshold(self):
        metrics, distributions = summarize_scores(self.oof, self.validation, MODELING_CONFIG, 0.5)
        self.assertEqual(metrics.recall.tolist(), [1.0, 0.5])
        self.assertEqual(metrics.reinspection_ratio.tolist(), [0.5, 0.25])
        failures = distributions[distributions["class"].eq("불량")]
        self.assertEqual(failures.above_threshold_ratio.tolist(), [1.0, 0.5])
        self.assertEqual(failures["count"].tolist(), [2, 2])
        self.assertEqual(sensor_overlap(self.folds, self.selected).common_count.tolist(), [2, 1])

    def test_invalid_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_scores(self.oof, self.validation.assign(positive_score=2), MODELING_CONFIG, 0.5)

    def test_writes_report_and_protects_existing_figure(self):
        # 임시 산출물은 작업공간 내에 생성하고 테스트가 끝나면 정리한다.
        with TemporaryDirectory(dir=".") as directory:
            output = Path(directory) / "logs"
            figures = Path(directory) / "figures"
            output.mkdir()
            write_score_diagnostics(self.oof, self.validation, self.folds, self.selected,
                                    MODELING_CONFIG, 0.5, output, figures)
            report = (output / "diagnostics.md").read_text(encoding="utf-8")
            self.assertIn("../figures/score_distribution.png", report)
            self.assertTrue((figures / "score_distribution.png").is_file())
            self.assertTrue((output / "sensor_overlap.csv").is_file())
            with self.assertRaisesRegex(ValueError, "이미 있습니다"):
                write_score_diagnostics(self.oof, self.validation, self.folds, self.selected,
                                        MODELING_CONFIG, 0.5, output, figures)


if __name__ == "__main__":
    unittest.main()
