# ==========================================
# LightGBM 후보의 OOF 확률과 threshold별 지표 비교
# - 시간 holdout을 먼저 제외한 Time Train 내부에서만 OOF 확률을 생성한다.
# - Time Validation과 Test split은 읽거나 threshold 결정에 사용하지 않는다.
# - 이 파일은 후보별 trade-off를 기록할 뿐, 최종 threshold를 자동 확정하지 않는다.
# ==========================================

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.step4_baseline import split_frame_to_xy
    from src.step6_feature_compare import build_experiments
    from src.modeling_preprocessing import quality_filter_record, quality_filter_json
    from src.split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role, training_folds
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from step4_baseline import split_frame_to_xy
    from step6_feature_compare import build_experiments
    from modeling_preprocessing import quality_filter_record, quality_filter_json
    from split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role, training_folds


EXPERIMENT_NAME = "lightgbm_all"


# ==========================================
# 단일 모델 threshold 비교용 OOF 확률 생성
# - 한 repeat에서 각 샘플은 정확히 한 번 validation fold에 속한다.
# - 기본 single 모드는 반복 평균 없이 샘플당 한 번의 fold 모델 예측을 사용한다.
# - repeated_mean은 기존 방식의 분석용이며 최종 threshold 근거로 혼동하지 않는다.
# - 단일 OOF도 fold 학습과 전체 Train 재학습의 확률 척도 일치를 보장하지 않는다.
# ==========================================
def generate_oof_scores(
    train_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    n_jobs: int = 1,
    score_method: str = "single",
) -> tuple[pd.DataFrame, float]:
    validate_train_role(train_frame)
    if score_method not in {"single", "repeated_mean"}:
        raise ValueError("OOF score_method는 single 또는 repeated_mean이어야 합니다.")
    # 모델 비교의 5×5 CV 정책은 유지하고 threshold OOF에만 별도 반복 정책을 적용한다.
    oof_config = config if score_method == "repeated_mean" else replace(
        config, experiment=replace(config.experiment, cv=replace(config.experiment.cv, n_repeats=1))
    )
    features, labels, _ = split_frame_to_xy(train_frame, config.dataset)
    if labels.nunique() < 2:
        raise ValueError("OOF를 생성하려면 Train에 정상과 Fail label이 모두 있어야 합니다.")

    experiments = build_experiments(config, n_jobs=n_jobs)
    if EXPERIMENT_NAME not in experiments:
        raise ValueError(f"OOF 대상 후보가 정의되어 있지 않습니다: {EXPERIMENT_NAME}")

    score_sum = np.zeros(len(train_frame), dtype=float)
    prediction_count = np.zeros(len(train_frame), dtype=int)
    quality_records = []
    started_at = perf_counter()

    for fit_indices, validation_indices in training_folds(features, labels, train_frame, oof_config):
        # fold별 Pipeline을 새로 만들어 imputing과 model fit이 validation에 닿지 않게 한다.
        pipeline = clone(experiments[EXPERIMENT_NAME])
        pipeline.fit(features.iloc[fit_indices], labels.iloc[fit_indices])
        quality_records.append(quality_filter_record(pipeline, fold=len(quality_records) + 1))

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
    expected_count = oof_config.experiment.cv.n_repeats
    if not np.all(prediction_count == expected_count):
        raise RuntimeError(
            "모든 Train 샘플의 OOF 예측 횟수가 반복 횟수와 일치하지 않습니다. "
            f"기대값={expected_count}, 실제 범위={prediction_count.min()}~{prediction_count.max()}"
        )

    # split 파일의 행 위치와 원본 ID를 구분한다. 기존 CSV의 원본 ID는 알 수 없다.
    oof_frame = pd.DataFrame(
        {
            "source_row_index": train_frame.index,
            "source_row_id": train_frame[SOURCE_ROW_ID].to_numpy() if SOURCE_ROW_ID in train_frame else None,
            "label": labels.to_numpy(),
            "oof_positive_score": score_sum / prediction_count,
            "oof_prediction_count": prediction_count,
            "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train_frame else None,
            "oof_score_method": score_method,
            "threshold_use": "candidate" if score_method == "single" else "analysis_only",
        }
    )
    # fold별 제거 기록은 샘플별 점수와 별도로 보존해 같은 로그의 중복 저장을 방지한다.
    oof_frame.attrs["quality_filter_records"] = quality_records
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
    if oof_frame.empty:
        raise ValueError("OOF 결과는 비어 있으면 안 됩니다.")
    # 분위수 계산 전에도 공통 metric 방어 코드로 결측 label·확률 범위를 검사한다.
    evaluate_binary_scores(oof_frame["label"], oof_frame["oof_positive_score"],
                           positive_label=config.dataset.positive_label,
                           negative_label=config.dataset.negative_label,
                           threshold=config.experiment.default_threshold)
    for column in ("training_protocol_id", "oof_score_method", "threshold_use"):
        if column in oof_frame and oof_frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"서로 다른 OOF 실행 결과가 섞여 있습니다: {column}")

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
                "split_strategy": "time_train_oof" if "training_protocol_id" in oof_frame and str(oof_frame["training_protocol_id"].iloc[0]).startswith("time:") else "legacy_train_oof",
                "training_protocol_id": oof_frame["training_protocol_id"].iloc[0] if "training_protocol_id" in oof_frame else None,
                "experiment": EXPERIMENT_NAME,
                "oof_score_method": oof_frame["oof_score_method"].iloc[0] if "oof_score_method" in oof_frame else "unknown",
                "threshold_use": oof_frame["threshold_use"].iloc[0] if "threshold_use" in oof_frame else "analysis_only",
                **asdict(metrics),
                "reinspection_ratio": (metrics.true_positive + metrics.false_positive) / metrics.support,
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


# ==========================================
# 시간 검증에 적용할 threshold 비교표의 출처 확인
# - 동일한 Train 생성 계약·데이터셋·모델의 단일 OOF 후보만 허용
# - 반복 평균 분석 결과와 legacy 출처 불명 결과는 최종 threshold 근거로 사용하지 않음
# ==========================================
def validate_threshold_report(report, train_frame, config, experiment, threshold):
    required = {"dataset_id", "experiment", "training_protocol_id", "threshold",
                "oof_score_method", "threshold_use"}
    if report.empty or not required.issubset(report.columns):
        raise ValueError("threshold 비교표가 비어 있거나 출처 확인 컬럼이 없습니다.")
    if PROTOCOL_ID not in train_frame:
        raise ValueError("threshold 출처 확인에는 원본 split 생성 계약이 필요합니다.")
    expected = {"dataset_id": config.dataset.dataset_id, "experiment": experiment,
                "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0],
                "oof_score_method": "single", "threshold_use": "candidate"}
    for column, value in expected.items():
        if report[column].isna().any() or set(report[column]) != {value}:
            raise ValueError(f"threshold 비교표의 학습 출처 또는 방식이 일치하지 않습니다: {column}")
    try:
        values = report["threshold"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold 비교표의 threshold는 수치형이어야 합니다.") from error
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("threshold 비교표에는 0~1의 유한한 threshold만 허용됩니다.")
    if not np.isclose(values, threshold, rtol=1e-12, atol=0).any():
        raise ValueError("지정한 threshold가 비교표에 없습니다. 반올림하지 않은 값을 사용하세요.")


# ==========================================
# Train 파일에서 OOF·threshold 비교 결과를 저장
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
    score_method: str = "single",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    # OOF 원본과 threshold 비교표를 분리 저장해 이후 오류 사례 분석에 재사용한다.
    config = load_modeling_config(config_path)
    oof_frame, elapsed_seconds = generate_oof_scores(
        pd.read_csv(train_path), config, n_jobs=n_jobs, score_method=score_method
    )
    threshold_frame = compare_thresholds(oof_frame, config)

    for frame, output_path in (
        (oof_frame, oof_output_path),
        (threshold_frame, threshold_output_path),
    ):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    # OOF 점수 파일 옆에 fold별 센서 제거 개수와 JSON 센서 목록을 함께 저장한다.
    quality_log = pd.DataFrame(oof_frame.attrs["quality_filter_records"])
    for column in ("high_missing_features", "constant_features"):
        quality_log[column] = quality_log[column].map(quality_filter_json)
    quality_path = Path(oof_output_path).with_name(Path(oof_output_path).stem + "_quality_filter.csv")
    quality_log.to_csv(quality_path, index=False, encoding="utf-8-sig")

    return oof_frame, threshold_frame, elapsed_seconds


def _parse_arguments() -> argparse.Namespace:
    # 새 평가 경로에서는 integrated/time_train.csv만 입력한다.
    parser = argparse.ArgumentParser(description="LightGBM OOF threshold 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, default=Path("logs/lightgbm_oof_predictions.csv"))
    parser.add_argument("--threshold-output", type=Path, default=Path("logs/threshold_compare.csv"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--score-method", choices=("single", "repeated_mean"), default="single",
                        help="single은 threshold 후보용, repeated_mean은 반복 평균 분석용입니다.")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    _, threshold_frame, elapsed_seconds = run_threshold_oof_from_file(
        arguments.config,
        arguments.train,
        arguments.oof_output,
        arguments.threshold_output,
        n_jobs=arguments.n_jobs,
        score_method=arguments.score_method,
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
