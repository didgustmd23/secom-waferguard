# ============================================================
# 목적
# - Time-based Train / Validation의 입력 분포 변화 진단
# - 주요 센서의 시간 변화 및 drift 가능성 점검
#
# 분석 항목
# 1. 센서별 결측률 변화
# 2. 센서별 평균 / 중앙값 변화
# 3. Standardized Mean Difference (SMD)
# 4. Population Stability Index (PSI)
# 5. Drift Priority
# 6. Label 비율 변화
#
# 주의
# - 이 코드는 EDA / drift 진단용이다.
# - 센서를 자동으로 제거하지 않는다.
# - 모델 학습이나 threshold 선택에 사용하지 않는다.
# - Test 데이터는 사용하지 않는다.
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.dataset_schema import validate_dataset_frame
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH,
        DatasetSpec,
        load_modeling_config,
    )
except ModuleNotFoundError:
    from dataset_schema import validate_dataset_frame
    from modeling_config import (
        DEFAULT_CONFIG_PATH,
        DatasetSpec,
        load_modeling_config,
    )


# ============================================================
# 1. PSI 계산
# ============================================================
def calculate_population_stability_index(
    train_values: pd.Series,
    validation_values: pd.Series,
    *,
    bin_count: int = 10,
) -> float | None:
    """
    Train을 기준으로 구간을 만들고
    Train과 Validation의 분포 차이를 PSI로 계산한다.

    PSI가 클수록 Train과 Validation의 분포가
    더 많이 달라졌을 가능성이 있다.
    """

    # 숫자로 변환할 수 없는 값은 NaN 처리
    train_numeric = pd.to_numeric(
        train_values,
        errors="coerce",
    ).to_numpy(dtype=float)

    validation_numeric = pd.to_numeric(
        validation_values,
        errors="coerce",
    ).to_numpy(dtype=float)

    # NaN / inf 제거
    train_numeric = train_numeric[np.isfinite(train_numeric)]
    validation_numeric = validation_numeric[
        np.isfinite(validation_numeric)
    ]

    # 어느 한쪽이라도 값이 없으면 계산 불가능
    if train_numeric.size == 0 or validation_numeric.size == 0:
        return None

    # Train의 분위수 기준으로 구간 생성
    boundaries = np.unique(
        np.quantile(
            train_numeric,
            np.linspace(
                0.0,
                1.0,
                bin_count + 1,
            ),
        )
    )

    # 값이 거의 모두 같은 경우
    if boundaries.size < 3:
        return None

    # 양 끝을 무한대로 확장
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    # Train / Validation 구간별 개수
    train_counts, _ = np.histogram(
        train_numeric,
        bins=boundaries,
    )

    validation_counts, _ = np.histogram(
        validation_numeric,
        bins=boundaries,
    )

    # 0으로 나누거나 log(0)이 되는 문제 방지
    epsilon = 1e-6

    train_ratio = np.maximum(
        train_counts / train_counts.sum(),
        epsilon,
    )

    validation_ratio = np.maximum(
        validation_counts / validation_counts.sum(),
        epsilon,
    )

    psi = np.sum(
        (validation_ratio - train_ratio)
        * np.log(validation_ratio / train_ratio)
    )

    return float(psi)


# ============================================================
# 2. Train / Validation feature 구조 확인
# ============================================================
def align_split_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: DatasetSpec,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
]:
    """
    Train과 Validation의 feature 구성이 같은지 확인하고
    Train의 feature 순서에 맞춰 정렬한다.
    """

    train_features = validate_dataset_frame(
        train_frame,
        dataset,
    )

    validation_features = validate_dataset_frame(
        validation_frame,
        dataset,
    )

    # Feature 개수나 이름이 다르면 오류
    if set(train_features) != set(validation_features):
        raise ValueError(
            "Train과 Validation의 feature column 구성이 "
            "일치하지 않습니다.\n"
            f"Train={list(train_features)}\n"
            f"Validation={list(validation_features)}"
        )

    # Train 기준 feature 순서로 정렬
    train_aligned = train_frame.loc[
        :,
        list(train_features),
    ]

    validation_aligned = validation_frame.loc[
        :,
        list(train_features),
    ]

    return (
        train_aligned,
        validation_aligned,
        train_features,
    )


# ============================================================
# 3. 수치형 센서 Drift 분석
# ============================================================
def summarize_numeric_feature(
    feature_name: str,
    train_values: pd.Series,
    validation_values: pd.Series,
) -> dict[str, object]:
    """
    하나의 수치형 센서에 대해
    Train vs Validation 분포 변화를 계산한다.
    """

    # ------------------------------------------
    # 결측률
    # ------------------------------------------
    train_missing_ratio = float(
        train_values.isna().mean()
    )

    validation_missing_ratio = float(
        validation_values.isna().mean()
    )

    # ------------------------------------------
    # 숫자형 변환
    # ------------------------------------------
    train_numeric = pd.to_numeric(
        train_values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    validation_numeric = pd.to_numeric(
        validation_values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ------------------------------------------
    # 평균
    # ------------------------------------------
    train_mean = train_numeric.mean()
    validation_mean = validation_numeric.mean()

    # ------------------------------------------
    # 표준편차
    # ------------------------------------------
    train_std = train_numeric.std(ddof=0)
    validation_std = validation_numeric.std(ddof=0)

    # ------------------------------------------
    # SMD
    # ------------------------------------------
    pooled_std = float(
        np.sqrt(
            (
                train_std**2
                + validation_std**2
            )
            / 2
        )
    )

    if (
        pd.isna(train_mean)
        or pd.isna(validation_mean)
        or pooled_std == 0.0
    ):
        standardized_mean_difference = None
    else:
        standardized_mean_difference = float(
            (validation_mean - train_mean)
            / pooled_std
        )

    # ------------------------------------------
    # 결과
    # ------------------------------------------
    return {
        "feature": feature_name,
        "feature_type": "numeric",

        "train_missing_ratio": train_missing_ratio,
        "validation_missing_ratio": validation_missing_ratio,

        "missing_ratio_delta": (
            validation_missing_ratio
            - train_missing_ratio
        ),

        "train_mean": train_mean,
        "validation_mean": validation_mean,

        "train_median": train_numeric.median(),
        "validation_median": validation_numeric.median(),

        "standardized_mean_difference": (
            standardized_mean_difference
        ),

        "population_stability_index": (
            calculate_population_stability_index(
                train_numeric,
                validation_numeric,
            )
        ),

        "unseen_category_ratio": None,
        "category_distribution_tvd": None,
    }


# ============================================================
# 4. 범주형 feature Drift 분석
# ============================================================
def summarize_categorical_feature(
    feature_name: str,
    train_values: pd.Series,
    validation_values: pd.Series,
) -> dict[str, object]:
    """
    범주형 feature의 Train vs Validation 차이를 계산한다.
    """

    train_missing_ratio = float(
        train_values.isna().mean()
    )

    validation_missing_ratio = float(
        validation_values.isna().mean()
    )

    train_non_missing = train_values.dropna()
    validation_non_missing = validation_values.dropna()

    # ------------------------------------------
    # Validation에서 처음 등장한 범주 비율
    # ------------------------------------------
    if validation_non_missing.empty:
        unseen_category_ratio = None
    else:
        unseen_category_ratio = float(
            (
                ~validation_non_missing.isin(
                    train_non_missing.unique()
                )
            ).mean()
        )

    # ------------------------------------------
    # 범주 분포 차이
    # TVD = Total Variation Distance
    # ------------------------------------------
    if (
        train_non_missing.empty
        or validation_non_missing.empty
    ):
        category_distribution_tvd = None
    else:
        train_ratio = train_non_missing.value_counts(
            normalize=True
        )

        validation_ratio = (
            validation_non_missing.value_counts(
                normalize=True
            )
        )

        categories = train_ratio.index.union(
            validation_ratio.index
        )

        category_distribution_tvd = float(
            0.5
            * (
                train_ratio.reindex(
                    categories,
                    fill_value=0.0,
                )
                - validation_ratio.reindex(
                    categories,
                    fill_value=0.0,
                )
            )
            .abs()
            .sum()
        )

    return {
        "feature": feature_name,
        "feature_type": "categorical",

        "train_missing_ratio": train_missing_ratio,
        "validation_missing_ratio": (
            validation_missing_ratio
        ),

        "missing_ratio_delta": (
            validation_missing_ratio
            - train_missing_ratio
        ),

        "train_mean": None,
        "validation_mean": None,

        "train_median": None,
        "validation_median": None,

        "standardized_mean_difference": None,
        "population_stability_index": None,

        "unseen_category_ratio": (
            unseen_category_ratio
        ),

        "category_distribution_tvd": (
            category_distribution_tvd
        ),
    }


# ============================================================
# 5. 전체 센서 Drift 보고서 생성
# ============================================================
def build_temporal_drift_report(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: DatasetSpec,
) -> pd.DataFrame:
    """
    모든 feature에 대해 Train vs Validation
    drift 분석 결과를 만든다.
    """

    (
        train_features,
        validation_features,
        feature_columns,
    ) = align_split_features(
        train_frame,
        validation_frame,
        dataset,
    )

    categorical_columns = set(
        dataset.categorical_feature_columns
    )

    records: list[dict[str, object]] = []

    # ------------------------------------------
    # 센서 하나씩 분석
    # ------------------------------------------
    for column in feature_columns:

        if column in categorical_columns:

            record = summarize_categorical_feature(
                column,
                train_features[column],
                validation_features[column],
            )

        else:

            record = summarize_numeric_feature(
                column,
                train_features[column],
                validation_features[column],
            )

        records.append(record)

    report = pd.DataFrame(records)

    # ========================================================
    # 6. Drift 우선순위 계산
    # ========================================================

    report["absolute_missing_ratio_delta"] = (
        report["missing_ratio_delta"].abs()
    )

    report[
        "absolute_standardized_mean_difference"
    ] = report[
        "standardized_mean_difference"
    ].abs()

    # ------------------------------------------
    # 각 지표를 공통 점수로 변환
    # ------------------------------------------
    priority_components = pd.DataFrame(
        {
            # 결측률 10%p 변화
            "missing": (
                report["absolute_missing_ratio_delta"]
                / 0.10
            ),

            # SMD 0.5
            "standardized_mean": (
                report[
                    "absolute_standardized_mean_difference"
                ]
                / 0.50
            ),

            # PSI 0.25
            "psi": (
                report[
                    "population_stability_index"
                ]
                / 0.25
            ),

            # 새 범주 10%
            "unseen_category": (
                report[
                    "unseen_category_ratio"
                ]
                / 0.10
            ),

            # TVD 0.2
            "category_tvd": (
                report[
                    "category_distribution_tvd"
                ]
                / 0.20
            ),
        }
    )

    # 각 센서에서 가장 큰 변화 지표를 사용
    report["drift_priority_score"] = (
        priority_components.max(
            axis=1,
            skipna=True,
        )
    )

    # ------------------------------------------
    # 사람이 보기 쉬운 등급
    # ------------------------------------------
    report["drift_priority"] = np.select(
        [
            report["drift_priority_score"] >= 1.0,
            report["drift_priority_score"] >= 0.5,
        ],
        [
            "높음",
            "중간",
        ],
        default="낮음",
    )

    # ------------------------------------------
    # 큰 변화부터 정렬
    # ------------------------------------------
    report = report.sort_values(
        by=[
            "drift_priority_score",
            "population_stability_index",
            "absolute_standardized_mean_difference",
        ],
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    return report


# ============================================================
# 7. Label Drift 분석
# ============================================================
def build_label_drift_summary(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: DatasetSpec,
) -> pd.DataFrame:
    """
    Train과 Validation의 정상/불량 비율 차이를 계산한다.

    주의:
    이 값은 EDA용이며 feature 선택에는 사용하지 않는다.
    """

    train_labels = train_frame[
        dataset.label_column
    ]

    validation_labels = validation_frame[
        dataset.label_column
    ]

    records: list[dict[str, object]] = []

    for label in (
        dataset.negative_label,
        dataset.positive_label,
    ):

        train_count = int(
            (train_labels == label).sum()
        )

        validation_count = int(
            (validation_labels == label).sum()
        )

        train_ratio = float(
            (train_labels == label).mean()
        )

        validation_ratio = float(
            (validation_labels == label).mean()
        )

        records.append(
            {
                "label": label,

                "train_count": train_count,
                "validation_count": validation_count,

                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,

                "ratio_delta": (
                    validation_ratio
                    - train_ratio
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 8. 파일을 읽어서 Drift 분석 실행
# ============================================================
def run_temporal_drift_check(
    config_path: Path | str,
    train_path: Path | str,
    validation_path: Path | str,
    output_path: Path | str,
    label_output_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # ------------------------------------------
    # 설정 불러오기
    # ------------------------------------------
    config = load_modeling_config(
        config_path
    )

    # ------------------------------------------
    # Time Train / Validation 읽기
    # ------------------------------------------
    train_frame = pd.read_csv(
        train_path
    )

    validation_frame = pd.read_csv(
        validation_path
    )

    # ------------------------------------------
    # 센서 Drift 분석
    # ------------------------------------------
    report = build_temporal_drift_report(
        train_frame,
        validation_frame,
        config.dataset,
    )

    # ------------------------------------------
    # Label Drift 분석
    # ------------------------------------------
    label_summary = build_label_drift_summary(
        train_frame,
        validation_frame,
        config.dataset,
    )

    # ------------------------------------------
    # 출력 폴더 생성
    # ------------------------------------------
    destination = Path(output_path)
    label_destination = Path(
        label_output_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------
    # CSV 저장
    # ------------------------------------------
    report.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    label_summary.to_csv(
        label_destination,
        index=False,
        encoding="utf-8-sig",
    )

    return report, label_summary


# ============================================================
# 9. 실행 인자
# ============================================================
def _parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Time-based Train/Validation "
            "센서 Drift 분석"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/integrated/time_train.csv"),
        help="Time Train CSV",
    )

    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/splits/integrated/time_valid.csv"),
        help="Time Validation CSV",
)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/temporal_drift_report.csv"
        ),
    )

    parser.add_argument(
        "--label-output",
        type=Path,
        default=Path(
            "logs/temporal_drift_label_summary.csv"
        ),
    )

    return parser.parse_args()


# ============================================================
# 10. main
# ============================================================
def main() -> None:

    arguments = _parse_arguments()

    # ------------------------------------------
    # 설정 다시 불러오기
    # ------------------------------------------
    config = load_modeling_config(
        arguments.config
    )

    # ------------------------------------------
    # Drift 분석 실행
    # ------------------------------------------
    report, label_summary = (
        run_temporal_drift_check(
            arguments.config,
            arguments.train,
            arguments.validation,
            arguments.output,
            arguments.label_output,
        )
    )

    # ------------------------------------------
    # 기본 결과 출력
    # ------------------------------------------
    print()
    print("=" * 70)
    print("STEP 11 - TEMPORAL DRIFT CHECK")
    print("=" * 70)

    print()
    print(
        f"분석 feature 수: {len(report)}"
    )

    # ------------------------------------------
    # 불량 label 비율 변화
    # ------------------------------------------
    positive_rows = label_summary.loc[
        label_summary["label"]
        == config.dataset.positive_label
    ]

    if not positive_rows.empty:

        positive_ratio_delta = (
            positive_rows[
                "ratio_delta"
            ].iloc[0]
        )

        print(
            "불량 label 비율 변화: "
            f"{positive_ratio_delta:.4f}"
        )

    # ------------------------------------------
    # Drift 우선순위 개수
    # ------------------------------------------
    priority_counts = (
        report["drift_priority"]
        .value_counts()
    )

    print()
    print("Drift 우선순위:")

    for priority in [
        "높음",
        "중간",
        "낮음",
    ]:

        count = int(
            priority_counts.get(
                priority,
                0,
            )
        )

        print(
            f"  {priority}: {count}"
        )

    # ------------------------------------------
    # 상위 센서 출력
    # ------------------------------------------
    print()
    print(
        "Drift Priority 상위 센서 10개:"
    )

    display_columns = [
        "feature",
        "drift_priority",
        "drift_priority_score",
        "population_stability_index",
        "standardized_mean_difference",
        "missing_ratio_delta",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in report.columns
    ]

    print(
        report[
            available_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    # ------------------------------------------
    # 저장 경로
    # ------------------------------------------
    print()
    print(
        "분포 변화 보고서:"
    )
    print(
        f"  {arguments.output}"
    )

    print(
        "Label 변화 보고서:"
    )
    print(
        f"  {arguments.label_output}"
    )

    print()
    print("=" * 70)
    print("STEP 11 완료")
    print("=" * 70)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    main()