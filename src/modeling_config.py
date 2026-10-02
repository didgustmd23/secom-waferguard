# ==========================================
# config.json 기반 모델링 실험 설정 로더
# - 루트 config.json에서 모델링 고정값을 읽어 실험 설정 객체로 변환
# - Python 코드에 Random Seed, Threshold, CV 횟수 등의 값을 중복하지 않음
# - 설정 파일 형식과 범위를 검증해 잘못된 실험 조건을 조기에 차단
# - Test 데이터는 읽거나 참조하지 않음
#
# config.json 주요 항목:
# - experiment: Label, Threshold, Missing 기준, PCA, Cross Validation 설정
# - metrics: 핵심·보조 평가 지표
# - feature_selection: Top-K 비교 대상
# - models: Baseline과 후보 모델 목록
# ==========================================

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


# src/ 기준 상위 프로젝트 루트의 config.json 경로
DEFAULT_CONFIG_PATH: Final = Path(__file__).resolve().parents[1] / "config.json"


# ==========================================
# 후보 모델 비교용 Repeated Stratified Cross Validation 설정
# - 불량(Fail) 클래스 비율을 각 Fold에 유지
# - 반복 실행으로 단일 Split 결과에 대한 의존도를 줄임
# - random_state를 고정해 동일 조건의 실험을 재현
# ==========================================
@dataclass(frozen=True)
class CrossValidationConfig:
    n_splits: int
    n_repeats: int
    random_state: int


# ==========================================
# 전체 모델 실험 공통 Protocol
# - config.json의 experiment 설정을 코드에서 안전하게 사용하기 위한 객체
# - 실제 Imputer, Scaler, Model은 단계별 Script에서 생성
# - 값 변경은 Python 코드가 아닌 config.json에서 수행
# ==========================================
@dataclass(frozen=True)
class ExperimentProtocol:
    positive_label: int
    default_threshold: float
    pca_explained_variance: float
    missing_ratio_threshold: float
    cv: CrossValidationConfig


# ==========================================
# 모델링 실험 전체 설정 객체
# - Experiment Protocol, 평가 지표, Feature Selection, 모델 목록을 함께 관리
# - 모든 Day 2~4 Script는 같은 객체를 읽어 비교 조건을 통일
# ==========================================
@dataclass(frozen=True)
class ModelingConfig:
    experiment: ExperimentProtocol
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    top_k_feature_counts: tuple[int, ...]
    baseline_model: str
    candidate_models: tuple[str, ...]


# ==========================================
# JSON 설정에서 필수 Key를 안전하게 읽기
# - 누락된 Key 또는 기대와 다른 자료형을 명확한 오류로 변환
# - 이후 설정 값 검증 전에 잘못된 JSON 구조를 차단
# ==========================================
def _require_mapping(source: dict[str, Any], key: str) -> dict[str, Any]:
    # 지정한 Key가 존재하는지 확인
    if key not in source:
        raise ValueError(f"config.json is missing required key: {key}")

    # JSON Object가 아닌 값은 하위 설정을 가질 수 없으므로 거부
    value = source[key]
    if not isinstance(value, dict):
        raise ValueError(f"config.json key '{key}' must be an object")

    return value


# ==========================================
# JSON 설정에서 필수 값을 안전하게 읽기
# - Scalar와 List 모두 지정 자료형인지 확인
# - Python의 bool은 int를 상속하므로 정수 설정에서는 별도 제외
# ==========================================
def _require_value(
    source: dict[str, Any], key: str, expected_type: type | tuple[type, ...]
) -> Any:
    # 지정한 Key가 존재하는지 확인
    if key not in source:
        raise ValueError(f"config.json is missing required key: {key}")

    value = source[key]

    # bool은 CV 횟수, random seed, 비율 설정에 숫자 값으로 허용하지 않음
    integer_allowed = expected_type is int or (
        isinstance(expected_type, tuple) and int in expected_type
    )
    if integer_allowed and isinstance(value, bool):
        raise ValueError(f"config.json key '{key}' must be an integer")

    # 설정값의 자료형이 기대한 JSON 자료형과 같은지 확인
    if not isinstance(value, expected_type):
        expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        expected_name = " or ".join(item.__name__ for item in expected_types)
        raise ValueError(f"config.json key '{key}' must be {expected_name}")

    return value


# ==========================================
# config.json을 ModelingConfig 객체로 로드 및 검증
# - 파일 경로를 지정하면 테스트·다른 데이터셋 설정에도 재사용 가능
# - Cross Validation, Threshold, PCA, Top-K의 허용 범위를 확인
# - 검증이 끝난 설정만 불변 Dataclass 객체로 반환
# ==========================================
def load_modeling_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> ModelingConfig:
    # Path 또는 문자열 경로를 일관된 Path 객체로 변환
    path = Path(config_path)

    # UTF-8 JSON 파일을 읽고 JSON 문법 오류를 명확히 전달
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"config.json not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"config.json is not valid JSON: {path}") from error

    # 최상위 JSON 구조가 Object인지 확인
    if not isinstance(raw_config, dict):
        raise ValueError("config.json root must be an object")

    # 실험·평가·특징 선택·모델 설정 영역을 분리
    experiment = _require_mapping(raw_config, "experiment")
    metrics = _require_mapping(raw_config, "metrics")
    feature_selection = _require_mapping(raw_config, "feature_selection")
    models = _require_mapping(raw_config, "models")
    cv = _require_mapping(experiment, "cross_validation")

    # 숫자 기반 실험 설정을 읽음
    random_state = _require_value(experiment, "random_state", int)
    positive_label = _require_value(experiment, "positive_label", int)
    default_threshold = float(_require_value(experiment, "default_threshold", (int, float)))
    missing_ratio_threshold = float(
        _require_value(experiment, "missing_ratio_threshold", (int, float))
    )
    pca_explained_variance = float(
        _require_value(experiment, "pca_explained_variance", (int, float))
    )
    n_splits = _require_value(cv, "n_splits", int)
    n_repeats = _require_value(cv, "n_repeats", int)

    # List 기반 모델링 설정을 Tuple로 변환해 실행 중 수정 방지
    primary_metrics = tuple(_require_value(metrics, "primary", list))
    secondary_metrics = tuple(_require_value(metrics, "secondary", list))
    top_k_feature_counts = tuple(
        _require_value(feature_selection, "top_k_feature_counts", list)
    )
    baseline_model = _require_value(models, "baseline", str)
    candidate_models = tuple(_require_value(models, "candidates", list))

    # 확률·비율·교차 검증 횟수의 논리적 범위를 검증
    if not 0.0 < default_threshold < 1.0:
        raise ValueError("experiment.default_threshold must be between 0 and 1")
    if not 0.0 <= missing_ratio_threshold <= 1.0:
        raise ValueError("experiment.missing_ratio_threshold must be between 0 and 1")
    if not 0.0 < pca_explained_variance <= 1.0:
        raise ValueError("experiment.pca_explained_variance must be in (0, 1]")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("cross_validation requires n_splits >= 2 and n_repeats >= 1")

    # 지표·모델 이름·Top-K 값은 비어 있지 않고 기대한 자료형인지 확인
    if not primary_metrics or not all(isinstance(metric, str) for metric in primary_metrics):
        raise ValueError("metrics.primary must contain at least one metric name")
    if not secondary_metrics or not all(isinstance(metric, str) for metric in secondary_metrics):
        raise ValueError("metrics.secondary must contain metric names")
    if not candidate_models or not all(isinstance(model, str) for model in candidate_models):
        raise ValueError("models.candidates must contain at least one model name")
    if not top_k_feature_counts or not all(
        isinstance(count, int) and not isinstance(count, bool) and count > 0
        for count in top_k_feature_counts
    ):
        raise ValueError("feature_selection.top_k_feature_counts must contain positive integers")

    # 검증된 값을 Day 2~4에서 공통으로 사용할 불변 설정 객체로 반환
    return ModelingConfig(
        experiment=ExperimentProtocol(
            positive_label=positive_label,
            default_threshold=default_threshold,
            pca_explained_variance=pca_explained_variance,
            missing_ratio_threshold=missing_ratio_threshold,
            cv=CrossValidationConfig(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            ),
        ),
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics,
        top_k_feature_counts=top_k_feature_counts,
        baseline_model=baseline_model,
        candidate_models=candidate_models,
    )


# config.json을 한 번 로드해 다른 Script에서 공통 설정으로 import
MODELING_CONFIG: Final = load_modeling_config()
