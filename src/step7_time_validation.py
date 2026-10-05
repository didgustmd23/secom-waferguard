# ==========================================
# 시간 기반 Validation에서 사전 선택 후보를 비교
# - Random CV 결과로 미리 정한 후보만 Time Train으로 학습한다.
# - Time Validation은 시간 순서 일반화 확인용이며 threshold 탐색에는 사용하지 않는다.
# - Test split은 이 단계에서 읽거나 평가하지 않는다.
# ==========================================

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.base import clone

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.step4_baseline import split_frame_to_xy
    from src.step6_feature_compare import build_experiments
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from step4_baseline import split_frame_to_xy
    from step6_feature_compare import build_experiments


# Random CV 결과를 본 뒤, 시간 검증 전에 고정한 후보 목록이다.
# 이 목록을 시간 검증 결과에 따라 늘리면 Validation set에 과적합될 수 있다.
PRESELECTED_EXPERIMENTS = (
    "lightgbm_all",
    "l1_balanced_l1_select",
)


# ==========================================
# 학습된 Pipeline에서 실제 사용한 feature 개수를 계산
# - selector가 없으면 입력 feature 전체를 사용한 모델이다.
# - L1 selector는 Train 데이터에서 선택된 support 개수만 기록한다.
# ==========================================
def _selected_feature_count(model, input_feature_count: int) -> int:
    # selector가 없는 모델은 입력 feature 전체를 사용한 것으로 기록한다.
    if "selector" not in model.named_steps:
        return input_feature_count

    selector = model.named_steps["selector"]
    if not hasattr(selector, "get_support"):
        raise ValueError("선택된 후보의 feature selector 결과를 확인할 수 없습니다.")
    return int(selector.get_support().sum())


# ==========================================
# Time Train/Validation으로 후보 모델을 학습·평가
# - Train feature 구성을 기준으로 Validation feature 구성을 엄격히 확인한다.
# - 각 Pipeline의 imputation, scaling, feature selection은 Time Train에서만 fit한다.
# - 비교 threshold는 config.json의 고정값을 사용한다.
# ==========================================
def compare_time_validation(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    experiment_names: tuple[str, ...] = PRESELECTED_EXPERIMENTS,
    threshold: float | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    if not experiment_names:
        raise ValueError("시간 검증할 후보 모델을 하나 이상 지정해야 합니다.")

    # Train의 feature 집합과 순서를 기준으로 Validation을 검증한다.
    train_x, train_y, feature_columns = split_frame_to_xy(train_frame, config.dataset)
    validation_x, validation_y, _ = split_frame_to_xy(
        validation_frame,
        config.dataset,
        expected_feature_columns=feature_columns,
    )
    if train_y.nunique() < 2:
        raise ValueError("Time Train split에는 정상과 Fail label이 모두 있어야 합니다.")

    # 별도 threshold가 없으면 후보 모델 비교 때 사용한 config 기본값을 쓴다.
    # 최종 후보 threshold는 CLI 인자로만 주입해 config.json의 비교 기준을 바꾸지 않는다.
    evaluation_threshold = (
        config.experiment.default_threshold if threshold is None else threshold
    )
    if not 0.0 <= evaluation_threshold <= 1.0:
        raise ValueError("Time Validation threshold는 0.0 이상 1.0 이하여야 합니다.")

    # step6과 동일한 후보 정의를 재사용해 Random CV와의 모델 조건을 유지한다.
    available_experiments = build_experiments(config, n_jobs=n_jobs)
    unknown_names = sorted(set(experiment_names) - set(available_experiments))
    if unknown_names:
        raise ValueError(f"정의되지 않은 후보 모델입니다: {unknown_names}")

    rows: list[dict[str, object]] = []
    for experiment_name in experiment_names:
        pipeline = clone(available_experiments[experiment_name])

        # 모든 전처리와 모델은 과거 구간인 Time Train에서만 학습한다.
        started_at = perf_counter()
        pipeline.fit(train_x, train_y)
        fit_seconds = perf_counter() - started_at

        # predict_proba의 열 순서를 가정하지 않고 Profile의 Fail label 위치를 찾는다.
        fitted_model = pipeline.named_steps["model"]
        class_positions = [
            index
            for index, label in enumerate(fitted_model.classes_)
            if label == config.dataset.positive_label
        ]
        if len(class_positions) != 1:
            raise ValueError("학습된 후보 모델에서 Profile의 Fail label을 찾을 수 없습니다.")
        positive_scores = pipeline.predict_proba(validation_x)[:, class_positions[0]]

        # OOF에서 정한 threshold를 그대로 적용해 시간 구간의 일반화만 확인한다.
        # 이 함수는 Time Validation 결과로 threshold를 다시 탐색하지 않는다.
        metrics = evaluate_binary_scores(
            validation_y,
            positive_scores,
            positive_label=config.dataset.positive_label,
            negative_label=config.dataset.negative_label,
            threshold=evaluation_threshold,
        )
        rows.append(
            {
                "dataset_id": config.dataset.dataset_id,
                "split_strategy": "time_validation",
                "experiment": experiment_name,
                "train_samples": len(train_frame),
                "validation_samples": len(validation_frame),
                "selected_feature_count": _selected_feature_count(
                    pipeline, len(feature_columns)
                ),
                "fit_time_seconds": fit_seconds,
                **asdict(metrics),
            }
        )

    return pd.DataFrame(rows).sort_values("average_precision", ascending=False)


# ==========================================
# CSV 파일을 입력받아 Time Validation 결과를 기록
# - 실행 결과는 재생성 가능한 로그이므로 기존 파일은 최신 실행 결과로 교체한다.
# - Test 경로 인자는 제공하지 않아 최종 평가 데이터 사용을 구조적으로 막는다.
# ==========================================
def compare_time_validation_from_files(
    config_path: Path | str,
    train_path: Path | str,
    validation_path: Path | str,
    output_path: Path | str,
    *,
    experiment_names: tuple[str, ...] = PRESELECTED_EXPERIMENTS,
    threshold: float | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    # 파일을 읽는 경계에서만 I/O를 수행하고, 실제 평가는 순수 함수에 위임한다.
    config = load_modeling_config(config_path)
    result = compare_time_validation(
        pd.read_csv(train_path),
        pd.read_csv(validation_path),
        config,
        experiment_names=experiment_names,
        threshold=threshold,
        n_jobs=n_jobs,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result


def _parse_arguments() -> argparse.Namespace:
    # Test 경로 인자를 제공하지 않아 CLI 단계에서 Test 오용을 막는다.
    parser = argparse.ArgumentParser(description="사전 선택 모델의 Time Validation 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/time_validation_compare.csv")
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="LightGBM 내부 병렬 처리 수입니다. 로컬 실행은 -1을 사용할 수 있습니다.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "OOF에서 미리 정한 평가 threshold입니다. 생략하면 config.json의 "
            "후보 비교용 기본값을 사용합니다. Time Validation 결과로 이 값을 재탐색하지 않습니다."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=PRESELECTED_EXPERIMENTS,
        default=list(PRESELECTED_EXPERIMENTS),
        help="Time Validation에 평가할 사전 선택 후보 이름입니다.",
    )
    return parser.parse_args()


def main() -> None:
    # OOF에서 정한 threshold를 그대로 전달해 Time Validation 결과만 출력한다.
    arguments = _parse_arguments()
    result = compare_time_validation_from_files(
        arguments.config,
        arguments.train,
        arguments.validation,
        arguments.output,
        experiment_names=tuple(arguments.experiments),
        threshold=arguments.threshold,
        n_jobs=arguments.n_jobs,
    )
    print(result.to_string(index=False))
    print(f"결과 저장 경로: {arguments.output}")


if __name__ == "__main__":
    main()
