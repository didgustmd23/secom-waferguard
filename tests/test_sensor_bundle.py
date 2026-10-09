# ==========================================
# 후보 추론 묶음의 저장·별도 프로세스 복원 검증
# - 작은 합성 데이터만 사용하며 실제 SECOM 파일은 읽지 않음
# - 새 Python 프로세스에서 센서 입력과 예측이 유지되는지 확인
# - 신뢰 확인·버전 불일치·계약 불일치·덮어쓰기 거부를 검증
# ==========================================

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.modeling_preprocessing import SensorQualityFilter
from src.sensor_inference import build_sensor_inference
from src.sensor_ml.inference.sensor_bundle import load_sensor_bundle, save_sensor_bundle, verify_sensor_bundle


class SensorBundleTest(unittest.TestCase):
    def setUp(self):
        """일반적인 수치형 선택 경로를 합성 데이터로 학습한다."""
        rng = np.random.RandomState(42)
        self.frame = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])
        model = Pipeline([
            ("quality_filter", SensorQualityFilter()),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("selector", SelectFromModel(RandomForestClassifier(n_estimators=2, random_state=42),
                                         threshold=-np.inf, max_features=2)),
            ("model", RandomForestClassifier(n_estimators=2, random_state=43)),
        ]).fit(self.frame, np.where(self.frame.a > 0, "fail", "pass"))
        self.inference = build_sensor_inference(model, positive_label="fail",
                                                threshold=0.3, max_sensors=2)

    def test_separate_process_restore(self):
        """새 프로세스가 현재 설정 없이 저장된 상태만으로 같은 예측을 반환한다."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            manifest = save_sensor_bundle(self.inference, self.frame, directory, provenance={"synthetic": True})
            self.assertFalse(manifest["is_final_model"])
            self.assertTrue(manifest["contains_local_sensor_samples"])
            self.assertEqual(manifest["verification_rows"], 21)
            result = subprocess.run(
                [sys.executable, "-m", "src.sensor_ml.inference.sensor_bundle", "--bundle-dir", str(directory),
                 "--trusted-local-bundle"], cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "passed")
            # 실제 새 프로세스에서 데모 CSV 내보내기와 예측까지 연결한다.
            demo, predictions = Path(temporary) / "demo.csv", Path(temporary) / "predictions.csv"
            for options in (["--export-demo-input", str(demo)],
                            ["--input", str(demo), "--output", str(predictions)]):
                completed = subprocess.run(
                    [sys.executable, "-m", "src.sensor_ml.inference.predict_cli", "--bundle-dir", str(directory),
                     "--trusted-local-bundle", *options], cwd=Path(__file__).resolve().parents[1],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            saved = pd.read_csv(predictions)
            self.assertEqual(len(saved), 21)
            self.assertEqual(saved.input_row_index.tolist(), list(range(21)))
            with self.assertRaisesRegex(ValueError, "비어 있지"):
                save_sensor_bundle(self.inference, self.frame, directory, provenance={})

    def test_trust_and_manifest_guards(self):
        """신뢰 확인과 버전·문턱 계약 검증으로 잘못 연결된 묶음을 거부한다."""
        with self.assertRaisesRegex(ValueError, "코드 실행 위험"):
            load_sensor_bundle("not-read", trusted=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            original = save_sensor_bundle(self.inference, self.frame, directory, provenance={})
            path = directory / "manifest.json"
            for key, value in (("versions", {}), ("threshold", 0.9)):
                manifest = dict(original)
                manifest[key] = value
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_sensor_bundle(directory, trusted=True)

    def test_prediction_mismatch_is_not_passed(self):
        """복원 검증의 기준 확률이 바뀌었으면 통과로 기록하지 않는다."""
        selected = self.frame.loc[:, list(self.inference.sensors)]
        expected = self.inference.predict(selected)
        expected["positive_score"] = 0.123
        result = verify_sensor_bundle({"inference": self.inference,
                                       "verification_input": selected,
                                       "verification_expected": expected})
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
