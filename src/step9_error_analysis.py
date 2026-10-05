# ==========================================
# Time Validation FN/FP 사례와 feature·drift 비교
# - OOF에서 미리 정한 threshold를 그대로 적용한다.
# - Time Validation 결과로 threshold나 모델을 다시 탐색하지 않는다.
# - 오류 사례는 모델 개선 가설을 세우기 위한 진단 자료이며, Test 평가는 수행하지 않는다.
# ==========================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from src.step4_baseline import split_frame_to_xy
    from src.step6_feature_compare import build_experiments
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from step4_baseline import split_frame_to_xy
    from step6_feature_compare import build_experiments


EXPERIMENT_NAME = "lightgbm_all"
ERROR_GROUP_ORDER = ("TP", "FN", "FP", "TN")


# ==========================================
# 실제 label·예측 label을 오류 그룹으로 변환
# - TP/FN은 실제 Fail, FP/TN은 실제 Pass에서 발생한다.
# - threshold는 호출자가 OOF에서 미리 정한 값만 전달해야 한다.
# ==========================================
def assign_error_groups(
    labels: pd.Series,
    positive_scores: np.ndarray,
    *,
    positive_label: object,
    threshold: float,
) -> pd.Series:
    # threshold 검증을 먼저 수행해 잘못된 분류 그룹을 만들지 않는다.
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("오류 분석 threshold는 0.0 이상 1.0 이하여야 합니다.")
    if len(labels) != len(positive_scores):
        raise ValueError("label과 예측 확률의 길이가 일치해야 합니다.")

    actual_fail = labels.to_numpy() == positive_label
    predicted_fail = np.asarray(positive_scores) >= threshold
    groups = np.select(
        [
            actual_fail & predicted_fail,
            actual_fail & ~predicted_fail,
            ~actual_fail & predicted_fail,
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    return pd.Series(groups, index=labels.index, name="error_group")


# ==========================================
# 두 error group의 sensor 분포 차이를 계산
# - FN과 TN 비교는 놓친 불량이 정상과 어떻게 다른지 찾기 위한 기준이다.
# - FP는 표본 수가 작을 수 있으므로 절대값을 과도하게 해석하지 않는다.
# ==========================================
def _standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    # 결측값은 제외하고 두 그룹의 평균 차이를 공통 표준편차로 나눈다.
    left_values = left.dropna().to_numpy(dtype=float)
    right_values = right.dropna().to_numpy(dtype=float)
    if len(left_values) < 2 or len(right_values) < 2:
        return float("nan")

    pooled_std = np.sqrt((left_values.var(ddof=1) + right_values.var(ddof=1)) / 2)
    if pooled_std == 0.0 or not np.isfinite(pooled_std):
        return float("nan")
    return float((left_values.mean() - right_values.mean()) / pooled_std)


# ==========================================
# 오류 그룹별 건수·확률·행 단위 결측률을 요약
# - 각 행의 결측률은 모델 입력 feature에서 계산한다.
# - 그룹이 없는 경우에도 0건으로 남겨 오류 형태를 명확히 보여 준다.
# ==========================================
def summarize_error_groups(
    cases: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    # 사례와 feature 행이 어긋나면 결측률 요약이 잘못되므로 먼저 차단한다.
    if len(cases) != len(features):
        raise ValueError("오류 사례와 feature 행 수가 일치해야 합니다.")

    working = cases.copy()
    working["row_missing_ratio"] = features.isna().mean(axis=1).to_numpy()
    rows: list[dict[str, object]] = []
    for group in ERROR_GROUP_ORDER:
        group_rows = working[working["error_group"] == group]
        rows.append(
            {
                "error_group": group,
                "sample_count": len(group_rows),
                "score_mean": group_rows["positive_score"].mean(),
                "score_median": group_rows["positive_score"].median(),
                "row_missing_ratio_mean": group_rows["row_missing_ratio"].mean(),
                "row_missing_ratio_median": group_rows["row_missing_ratio"].median(),
            }
        )
    return pd.DataFrame(rows)


# ==========================================
# FN/TN·FP/TN의 sensor별 차이와 기존 drift 결과를 결합
# - 이 표는 중요 sensor 후보를 정렬할 뿐, feature 제거 결정을 자동으로 내리지 않는다.
# - drift report가 없으면 오류 사례 feature 차이만으로도 분석할 수 있다.
# ==========================================
def compare_error_features(
    features: pd.DataFrame,
    error_groups: pd.Series,
    drift_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if len(features) != len(error_groups):
        raise ValueError("feature와 오류 그룹의 행 수가 일치해야 합니다.")

    rows: list[dict[str, object]] = []
    for feature in features.columns:
        values = features[feature]
        fn_values = values[error_groups == "FN"]
        fp_values = values[error_groups == "FP"]
        tn_values = values[error_groups == "TN"]
        rows.append(
            {
                "feature": feature,
                "fn_missing_ratio": fn_values.isna().mean(),
                "tn_missing_ratio": tn_values.isna().mean(),
                "fn_vs_tn_missing_ratio_delta": fn_values.isna().mean() - tn_values.isna().mean(),
                "fn_median": fn_values.median(),
                "tn_median": tn_values.median(),
                "fn_vs_tn_standardized_mean_difference": _standardized_mean_difference(
                    fn_values, tn_values
                ),
                "fp_missing_ratio": fp_values.isna().mean(),
                "fp_vs_tn_standardized_mean_difference": _standardized_mean_difference(
                    fp_values, tn_values
                ),
            }
        )
    result = pd.DataFrame(rows)

    if drift_report is not None:
        required_columns = {"feature", "drift_priority", "drift_priority_score"}
        missing_columns = sorted(required_columns - set(drift_report.columns))
        if missing_columns:
            raise ValueError(f"drift report에 필수 column이 없습니다: {missing_columns}")
        result = result.merge(
            drift_report.loc[:, list(required_columns)], on="feature", how="left"
        )

    # 높은 drift와 큰 FN/TN 차이를 먼저 보되, 이 순서는 해석 우선순위일 뿐 선택 기준이 아니다.
    result["absolute_fn_vs_tn_smd"] = result[
        "fn_vs_tn_standardized_mean_difference"
    ].abs()
    sort_columns = ["absolute_fn_vs_tn_smd"]
    if "drift_priority_score" in result:
        sort_columns.insert(0, "drift_priority_score")
    return result.sort_values(sort_columns, ascending=False, na_position="last")


# ==========================================
# Time Train으로 LightGBM을 fit하고 Time Validation 오류 사례를 생성
# - 학습에 사용하지 않은 Time Validation만 예측한다.
# - 모델과 threshold는 호출 시점에 고정되어 있어야 한다.
# ==========================================
def analyze_time_validation_errors(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    threshold: float,
    drift_report: pd.DataFrame | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_x, train_y, feature_columns = split_frame_to_xy(train_frame, config.dataset)
    validation_x, validation_y, _ = split_frame_to_xy(
        validation_frame,
        config.dataset,
        expected_feature_columns=feature_columns,
    )
    if train_y.nunique() < 2:
        raise ValueError("Time Train split에는 정상과 Fail label이 모두 있어야 합니다.")

    pipeline = clone(build_experiments(config, n_jobs=n_jobs)[EXPERIMENT_NAME])
    pipeline.fit(train_x, train_y)
    model = pipeline.named_steps["model"]
    class_positions = [
        index
        for index, label in enumerate(model.classes_)
        if label == config.dataset.positive_label
    ]
    if len(class_positions) != 1:
        raise ValueError("학습된 LightGBM에서 Profile의 Fail label을 찾을 수 없습니다.")
    positive_scores = pipeline.predict_proba(validation_x)[:, class_positions[0]]
    error_groups = assign_error_groups(
        validation_y,
        positive_scores,
        positive_label=config.dataset.positive_label,
        threshold=threshold,
    )

    # 모델 입력에 쓰지 않은 metadata는 사례 확인에만 남긴다.
    cases = pd.DataFrame(
        {
            "source_row_index": validation_frame.index,
            "label": validation_y.to_numpy(),
            "positive_score": positive_scores,
            "threshold": threshold,
            "error_group": error_groups.to_numpy(),
        }
    )
    if config.dataset.timestamp_column is not None:
        cases.insert(1, "timestamp", validation_frame[config.dataset.timestamp_column].to_numpy())

    summary = summarize_error_groups(cases, validation_x)
    feature_comparison = compare_error_features(validation_x, error_groups, drift_report)
    return cases, summary, feature_comparison


# ==========================================
# 오류 분석 결과 CSV 세 개를 저장
# - case CSV는 사례 확인용, summary는 오류량 확인용, feature CSV는 해석 우선순위용이다.
# - 모든 파일은 재생성 가능한 로그이며 Test 데이터를 사용하지 않는다.
# ==========================================
def analyze_time_validation_errors_from_files(
    config_path: Path | str,
    train_path: Path | str,
    validation_path: Path | str,
    *,
    threshold: float,
    cases_output_path: Path | str,
    summary_output_path: Path | str,
    feature_output_path: Path | str,
    drift_report_path: Path | str | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # drift report는 선택 입력이며, 전달된 경우에만 feature 비교 결과와 결합한다.
    drift_report = None
    if drift_report_path is not None:
        path = Path(drift_report_path)
        if not path.exists():
            raise FileNotFoundError(f"drift report 파일을 찾을 수 없습니다: {path}")
        drift_report = pd.read_csv(path)

    cases, summary, feature_comparison = analyze_time_validation_errors(
        pd.read_csv(train_path),
        pd.read_csv(validation_path),
        load_modeling_config(config_path),
        threshold=threshold,
        drift_report=drift_report,
        n_jobs=n_jobs,
    )
    for frame, output_path in (
        (cases, cases_output_path),
        (summary, summary_output_path),
        (feature_comparison, feature_output_path),
    ):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return cases, summary, feature_comparison


def _parse_arguments() -> argparse.Namespace:
    # threshold는 필수 인자로 받아 Time Validation에서 임의 탐색하지 않게 한다.
    parser = argparse.ArgumentParser(description="Time Validation FN/FP 사례 분석")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--drift-report", type=Path, default=Path("logs/temporal_drift_report.csv")
    )
    parser.add_argument(
        "--cases-output", type=Path, default=Path("logs/time_validation_error_cases.csv")
    )
    parser.add_argument(
        "--summary-output", type=Path, default=Path("logs/time_validation_error_summary.csv")
    )
    parser.add_argument(
        "--feature-output", type=Path, default=Path("logs/time_validation_error_features.csv")
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    # 사례·요약·feature 비교 로그를 저장하고 콘솔에는 검토용 요약만 표시한다.
    arguments = _parse_arguments()
    _, summary, features = analyze_time_validation_errors_from_files(
        arguments.config,
        arguments.train,
        arguments.validation,
        threshold=arguments.threshold,
        cases_output_path=arguments.cases_output,
        summary_output_path=arguments.summary_output,
        feature_output_path=arguments.feature_output,
        drift_report_path=arguments.drift_report,
        n_jobs=arguments.n_jobs,
    )
    print("오류 그룹 요약")
    print(summary.to_string(index=False))
    print("FN/TN 차이와 drift 기준 상위 feature 10개")
    print(features.head(10).to_string(index=False))
    print(f"사례 저장 경로: {arguments.cases_output}")
    print(f"요약 저장 경로: {arguments.summary_output}")
    print(f"feature 비교 저장 경로: {arguments.feature_output}")


if __name__ == "__main__":
    main()
