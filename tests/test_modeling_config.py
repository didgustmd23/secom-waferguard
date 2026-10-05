# ==========================================
# Dataset Profile + 공통 실험 설정 로더 테스트
# - 기본 SECOM Profile이 config.json을 통해 정상 로드되는지 확인
# - 필수 Label 설정 누락과 범위 오류를 파일 단위로 검증
# ==========================================

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.modeling_config import MODELING_CONFIG, load_dataset_spec, load_modeling_config


class ModelingConfigTest(unittest.TestCase):
    def _load_variant(self, section, key, value):
        # 실제 설정은 보존하고 경계값을 바꾼 임시 설정만 로드한다.
        raw = json.loads(Path("config.json").read_text(encoding="utf-8"))
        raw["dataset_profile"] = str(Path("configs/datasets/secom.json").resolve())
        raw[section][key] = value
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            return load_modeling_config(path)

    def test_rejects_unknown_candidate(self):
        with self.assertRaisesRegex(ValueError, "지원하지 않는 모델"):
            self._load_variant("models", "candidates", ["lightgbm", "unknown"])

    def test_rejects_pca_ratio_one(self):
        with self.assertRaisesRegex(ValueError, "1 미만"):
            self._load_variant("experiment", "pca_explained_variance", 1.0)

    def test_rejects_negative_seed(self):
        with self.assertRaisesRegex(ValueError, "random_state"):
            self._load_variant("experiment", "random_state", -1)

    def test_rejects_unknown_baseline(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            self._load_variant("models", "baseline", "unknown")
    # 기본 config.json이 SECOM의 데이터 구조와 실험 조건을 분리해 로드하는지 확인
    def test_default_config_loads_secom_profile(self) -> None:
        self.assertEqual(MODELING_CONFIG.dataset.dataset_id, "secom")
        self.assertEqual(MODELING_CONFIG.dataset.positive_label, 1)
        self.assertEqual(MODELING_CONFIG.dataset.negative_label, -1)
        self.assertTrue(MODELING_CONFIG.dataset.input_path.is_absolute())
        self.assertEqual(MODELING_CONFIG.experiment.cv.n_splits, 5)
        self.assertIsNotNone(MODELING_CONFIG.dataset.ingestion)
        self.assertEqual(
            MODELING_CONFIG.dataset.ingestion.adapter, "feature_metadata_pair"
        )
        self.assertEqual(len(MODELING_CONFIG.dataset.ingestion.sources), 2)
        self.assertTrue(
            MODELING_CONFIG.dataset.ingestion.get_source("features").path.is_absolute()
        )

    # Dataset Profile에서 필수 positive Label이 빠지면 즉시 오류가 나는지 확인
    def test_profile_rejects_missing_positive_label(self) -> None:
        profile_path = Path("configs/datasets/secom.json")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        del profile["labels"]["positive"]

        # 실제 Profile 파일을 수정하지 않도록 임시 파일에서 오류 조건을 생성
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_profile_path = Path(temporary_directory) / "invalid_profile.json"
            invalid_profile_path.write_text(
                json.dumps(profile), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "필수 key가 없습니다: positive"):
                load_dataset_spec(invalid_profile_path)

    # 결측률 기준이 0~1 범위를 벗어나면 데이터 품질 정책으로 사용하지 못하게 차단
    def test_dataset_profile_rejects_invalid_missing_ratio(self) -> None:
        profile_path = Path("configs/datasets/secom.json")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["quality_rules"]["missing_ratio_threshold"] = 1.1

        # 실제 Profile 파일을 수정하지 않도록 임시 파일에서 오류 조건을 생성
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_profile_path = Path(temporary_directory) / "invalid_profile.json"
            invalid_profile_path.write_text(
                json.dumps(profile), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "missing_ratio_threshold"):
                load_dataset_spec(invalid_profile_path)
