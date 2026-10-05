# ==========================================
# LightGBM 후보의 OOF 확률과 threshold별 지표 비교
# - Random Train 내부의 Repeated Stratified K-Fold로만 OOF 확률을 생성한다.
# - Time Validation과 Test split은 읽거나 threshold 결정에 사용하지 않는다.
# - 이 파일은 후보별 trade-off를 기록할 뿐, 최종 threshold를 자동 확정하지 않는다.
# ==========================================

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

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


EXPERIMENT_NAME = "lightgbm_all"


# ==========================================
# OOF 평균 확률 생성
# - 한 repeat에서 각 샘플은 정확히 한 번 validation fold에 속한다.
# - repeats만큼 얻은 validation 확률을 평균내어 안정적인 OOF 확률로 사용한다.
# ==========================================
def generate_oof_scores(
    train_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, float]:
    features, labels, _ = split_frame_to_xy(train_frame, config.dataset)
    if labels.nunique() < 2:
        raise ValueError("OOF를 생성하려면 Random Train에 정상과 Fail label이 모두 있어야 합니다.")

    experiments = build_experiments(config, n_jobs=n_jobs)
    if EXPERIMENT_NAME not in experiments:
        raise ValueError(f"OOF 대상 후보가 정의되어 있지 않습니다: {EXPERIMENT_NAME}")

    cv = RepeatedStratifiedKFold(
        n_splits=config.experiment.cv.n_splits,
        n_repeats=config.experiment.cv.n_repeats,
        random_state=config.experiment.cv.random_state,
    )
    score_sum = np.zeros(len(train_frame), dtype=float)
    prediction_count = np.zeros(len(train_frame), dtype=int)
    started_at = perf_counter()

    for fit_indices, validation_indices in cv.split(features, labels):
        # fold별 Pipeline을 새로 만들어 imputing과 model fit이 validation에 닿지 않게 한다.
        pipeline = clone(experiments[EXPERIMENT_NAME])
        pipeline.fit(features.iloc[fit_indices], labels.iloc[fit_indices])

        fitted_model = pipeline.named_steps["model"]
        class_positions = [
            index
            for index, label in enumerate(fitted_model.classes_)
            if label == config.dataset.positive_label
        ]
        if len(class_positions) != 1:
            raise ValueError("학습된 LightGBM에서 Profile의 Fail label을 찾을 수 없습니다.")

        fold_scores = pipeline.predict_proba(features.iloc[validation_indices])[:, class_positions[0]]
        score_sum[validation_indices] += fold_scores
        prediction_count[validation_indices] += 1

    elapsed_seconds = perf_counter() - started_at
    expected_count = config.experiment.cv.n_repeats
    if not np.all(prediction_count == expected_count):
        raise RuntimeError(
            "모든 Random Train 샘플의 OOF 예측 횟수가 반복 횟수와 일치하지 않습니다. "
            f"기대값={expected_count}, 실제 범위={prediction_count.min()}~{prediction_count.max()}"
        )

    # 원본 행 번호를 함께 남겨 이후 FN/FP 사례 분석 시 원본 split과 다시 연결할 수 있게 한다.
    oof_frame = pd.DataFrame(
        {
            "source_row_index": train_frame.index,
            "label": labels.to_numpy(),
            "oof_positive_score": score_sum / prediction_count,
            "oof_prediction_count": prediction_count,
        }
    )
    return oof_frame, elapsed_seconds


# ==========================================
# OOF 확률을 threshold 후보별 지표로 변환
# - threshold 범위는 0~1 내부에서만 허용한다.
# - Recall, Precision, FN, FP를 모두 저장해 운영 기준으로 사람이 선택한다.
# ==========================================
def compare_thresholds(
    oof_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    required_columns = {"label", "oof_positive_score"}
    missing_columns = sorted(required_columns - set(oof_frame.columns))
    if missing_columns:
        raise ValueError(f"OOF 결과에 필수 column이 없습니다: {missing_columns}")

    # 모델의 확률 보정 상태에 따라 유효 threshold 범위가 크게 달라질 수 있다.
    # 따라서 고정 격자 대신 OOF 확률의 분위수를 기본 후보로 사용한다.
    if thresholds is None:
        thresholds = np.quantile(
            oof_frame["oof_positive_score"], np.linspace(0.0, 1.0, 101)
        )
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("threshold 후보는 비어 있지 않은 1차원 배열이어야 합니다.")
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("모든 threshold 후보는 0.0 이상 1.0 이하의 유한한 값이어야 합니다.")

    rows: list[dict[str, object]] = []
    for threshold in np.unique(np.append(thresholds, config.experiment.default_threshold)):
        metrics = evaluate_binary_scores(
            oof_frame["label"],
            oof_frame["oof_positive_score"],
            positive_label=config.dataset.positive_label,
            negative_label=config.dataset.negative_label,
            threshold=float(threshold),
        )
        rows.append(
            {
                "dataset_id": config.dataset.dataset_id,
                "split_strategy": "random_train_oof",
                "experiment": EXPERIMENT_NAME,
                **asdict(metrics),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


# ==========================================
# Random Train 파일에서 OOF·threshold 비교 결과를 저장
# - OOF 결과와 threshold 표를 별도 파일로 저장해 재현성과 오류 분석을 지원한다.
# - Test 경로 인자는 제공하지 않아 봉인된 Test 사용을 막는다.
# ==========================================
def run_threshold_oof_from_file(
    config_path: Path | str,
    train_path: Path | str,
    oof_output_path: Path | str,
    threshold_output_path: Path | str,
    *,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    # OOF 원본과 threshold 비교표를 분리 저장해 이후 오류 사례 분석에 재사용한다.
    config = load_modeling_config(config_path)
    oof_frame, elapsed_seconds = generate_oof_scores(
        pd.read_csv(train_path), config, n_jobs=n_jobs
    )
    threshold_frame = compare_thresholds(oof_frame, config)

    for frame, output_path in (
        (oof_frame, oof_output_path),
        (threshold_frame, threshold_output_path),
    ):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    return oof_frame, threshold_frame, elapsed_seconds


def _parse_arguments() -> argparse.Namespace:
    # OOF 생성은 Random Train만 입력으로 받아 Test 사용을 구조적으로 막는다.
    parser = argparse.ArgumentParser(description="LightGBM OOF threshold 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, default=Path("logs/lightgbm_oof_predictions.csv"))
    parser.add_argument("--threshold-output", type=Path, default=Path("logs/threshold_compare.csv"))
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    _, threshold_frame, elapsed_seconds = run_threshold_oof_from_file(
        arguments.config,
        arguments.train,
        arguments.oof_output,
        arguments.threshold_output,
        n_jobs=arguments.n_jobs,
    )
    # 전체 비교표는 CSV에 저장하고, 콘솔에는 판단에 필요한 대표 후보만 표시한다.
    default_row = threshold_frame.loc[
        np.isclose(threshold_frame["threshold"], 0.5)
    ]
    console_summary = pd.concat(
        [
            threshold_frame.nlargest(5, "f1"),
            threshold_frame.nlargest(5, "recall"),
            default_row,
        ]
    ).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    print(console_summary.to_string(index=False))
    print(f"OOF 생성 시간(초): {elapsed_seconds:.3f}")
    print(f"OOF 저장 경로: {arguments.oof_output}")
    print(f"Threshold 비교 저장 경로: {arguments.threshold_output}")


if __name__ == "__main__":
    main()
