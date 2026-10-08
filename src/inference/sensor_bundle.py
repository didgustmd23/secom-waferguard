# ==========================================
# 후보 센서 추론 묶음 저장·복원 및 별도 프로세스 확인
# - 센서 순서·학습 통계·분류기·label·후보 문턱을 함께 저장
# - 현재 설정 파일이 아니라 저장된 입력 계약으로만 추론
# - joblib 복원은 임의 코드 실행 위험이 있어 신뢰 확인을 필수로 요구
# - 검증용 센서 값은 로컬 파일에 포함되며 외부로 전송하지 않음
# ==========================================

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.sensor_inference import SensorInference


def runtime_versions():
    """저장된 객체의 복원에 관련된 런타임 버전을 기록한다."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib"):
        versions[name] = version(name)
    return versions


def save_sensor_bundle(inference, features, output_dir, *, provenance):
    """학습하지 않고 후보 추론 객체와 로컬 검증 입력을 새 폴더에 저장한다."""
    if not isinstance(inference, SensorInference):
        raise ValueError("저장할 객체는 SensorInference여야 합니다.")
    # 모듈 실행 경로를 고정하여 별도 프로세스의 역직렬화에서 클래스를 찾게 한다.
    if inference.__class__.__module__ != "src.sensor_inference":
        raise ValueError("저장 기능은 python -m src.verification.check_sensor_inference로 실행하세요.")
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("모델 묶음 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    selected = features.loc[:, list(inference.sensors)]
    selected = selected.loc[~selected.isna().all(axis=1)].iloc[:32].copy()
    if selected.empty:
        raise ValueError("저장·복원 확인용 유효 입력이 없습니다.")
    # 실제 입력 일부와 NaN 시험 행을 보존하여 복원 후 변환 통계도 함께 확인한다.
    if len(inference.sensors) > 1:
        partial = selected.iloc[[0]].copy().fillna(0.0)
        partial.iloc[0, 0] = np.nan
        selected = pd.concat([selected, partial], ignore_index=True)
    expected = inference.predict(selected)
    payload = {"inference": inference, "verification_input": selected,
               "verification_expected": expected}
    manifest = {"format_version": 1, "is_final_model": False,
                "sensors": list(inference.sensors), "threshold": inference.threshold,
                "positive_label": inference.positive_label,
                "versions": runtime_versions(), "provenance": provenance,
                "contains_local_sensor_samples": True,
                "verification_rows": len(selected)}
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, directory / "candidate.joblib")
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def load_sensor_bundle(directory, *, trusted=False):
    """사용자가 신뢰한 묶음만 동일 버전 환경에서 복원하고 계약을 확인한다."""
    if not trusted:
        raise ValueError("joblib 파일은 코드 실행 위험이 있습니다. 신뢰한 로컬 파일만 복원하세요.")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("versions") != runtime_versions():
        raise ValueError("모델 묶음 형식 또는 런타임 버전이 다릅니다. 저장한 환경에서 복원하세요.")
    # manifest 검증은 보안 서명이나 안전한 역직렬화를 의미하지 않는다.
    payload = joblib.load(directory / "candidate.joblib")
    inference = payload["inference"]
    if not isinstance(inference, SensorInference):
        raise ValueError("모델 묶음에 올바른 추론 객체가 없습니다.")
    inference._validate_contract()
    if (list(inference.sensors) != manifest.get("sensors")
            or inference.threshold != manifest.get("threshold")
            or inference.positive_label != manifest.get("positive_label")):
        raise ValueError("저장된 센서·문턱·label 계약이 manifest와 다릅니다.")
    return payload


def verify_sensor_bundle(payload):
    """복원된 객체가 저장 당시 확률·판정·label을 그대로 반환하는지 확인한다."""
    expected = payload["verification_expected"]
    frame = payload["verification_input"]
    inference = payload["inference"]
    actual = inference.predict(frame.loc[:, list(reversed(inference.sensors))])
    if len(actual) != len(expected) or actual.empty:
        raise ValueError("복원 검증 입력과 저장된 예측 행 수가 다릅니다.")
    probability_match = bool(np.isfinite(expected.positive_score).all()
                             and np.isfinite(actual.positive_score).all()
                             and np.allclose(expected.positive_score, actual.positive_score,
                                             rtol=0, atol=1e-12))
    decision_match = bool(np.array_equal(expected.predicted_positive, actual.predicted_positive))
    label_match = bool(np.array_equal(expected.predicted_label, actual.predicted_label))
    return {"status": "passed" if probability_match and decision_match and label_match else "failed",
            "rows": len(actual), "selected_sensor_count": len(inference.sensors),
            "max_probability_difference": float(np.max(np.abs(expected.positive_score - actual.positive_score))),
            "probability_match": probability_match, "decision_match": decision_match,
            "label_match": label_match, "is_performance_evaluation": False, "test_used": False}


def main():
    """새 Python 프로세스에서 후보 묶음의 복원 정합성을 확인한다."""
    parser = argparse.ArgumentParser(description="신뢰한 후보 센서 모델의 저장·복원 확인")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--trusted-local-bundle", action="store_true",
                        help="직접 생성한 신뢰한 로컬 파일임을 확인합니다.")
    args = parser.parse_args()
    payload = load_sensor_bundle(args.bundle_dir, trusted=args.trusted_local_bundle)
    result = verify_sensor_bundle(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise ValueError("복원 후 예측이 저장 당시 결과와 다릅니다.")


if __name__ == "__main__":
    main()
