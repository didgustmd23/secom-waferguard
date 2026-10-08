# ============================================================
# STEP 12 - FINAL ANALYSIS VISUALIZATION
# ============================================================
# 목적
# - Day 4까지 생성된 분석 결과를 시각화
# - FN / FP
# - Random vs Time
# - Drift
#
# 주의
# - Test 데이터 사용 없음
# - 모델 재학습 없음
# - threshold 변경 없음
# - feature 제거 없음
# ============================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 입력 / 출력 경로
# ============================================================

LOG_DIR = Path("logs")

FIGURE_DIR = Path(
    "reports/figures/step12"
)


RANDOM_TIME_PATH = (
    LOG_DIR / "random_vs_time.csv"
)

ERROR_SUMMARY_PATH = (
    LOG_DIR / "time_validation_error_summary.csv"
)

ERROR_FEATURE_PATH = (
    LOG_DIR / "time_validation_error_features.csv"
)

DRIFT_PATH = (
    LOG_DIR / "temporal_drift_report.csv"
)


# ============================================================
# 1. Random vs Time 그래프
# ============================================================

def plot_random_vs_time() -> None:

    if not RANDOM_TIME_PATH.exists():
        print(
            f"[SKIP] 파일 없음: "
            f"{RANDOM_TIME_PATH}"
        )
        return

    result = pd.read_csv(
        RANDOM_TIME_PATH
    )


    metrics = [
        "AP",
        "Recall",
        "Precision",
        "F1",
        "ROC-AUC",
    ]

    random_row = result.loc[
    result["validation"].astype(str).str.strip().str.lower() == "random"
].iloc[0]

    time_row = result.loc[
    result["validation"].astype(str).str.strip().str.lower() == "time"
].iloc[0]

    random_values = [
        random_row[m]
        for m in metrics
    ]

    time_values = [
        time_row[m]
        for m in metrics
    ]

    x = range(
        len(metrics)
    )

    plt.figure(
        figsize=(10, 6)
    )

    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        random_values,
        width=width,
        label="Random",
    )

    plt.bar(
        [i + width / 2 for i in x],
        time_values,
        width=width,
        label="Time",
    )

    plt.xticks(
        list(x),
        [
            "AP",
            "Recall",
            "Precision",
            "F1",
            "ROC-AUC",
        ],
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1,
    )

    plt.title(
        "Random vs Time Validation"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "random_vs_time.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(
        f"[OK] {output}"
    )


# ============================================================
# 2. FN / FP 개수 그래프
# ============================================================

def plot_error_groups() -> None:

    if not ERROR_SUMMARY_PATH.exists():
        print(
            f"[SKIP] 파일 없음: "
            f"{ERROR_SUMMARY_PATH}"
        )
        return

    summary = pd.read_csv(
        ERROR_SUMMARY_PATH
    )

    plot_data = summary[
        summary["error_group"].isin(
            ["FN", "FP", "TP", "TN"]
        )
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        plot_data["error_group"],
        plot_data["sample_count"],
    )

    plt.xlabel(
        "Prediction Group"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Time Validation Error Groups"
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "error_groups.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(
        f"[OK] {output}"
    )


# ============================================================
# 3. Drift 상위 센서
# ============================================================

def plot_top_drift_sensors(
    top_n: int = 15,
) -> None:

    if not DRIFT_PATH.exists():
        print(
            f"[SKIP] 파일 없음: "
            f"{DRIFT_PATH}"
        )
        return

    drift = pd.read_csv(
        DRIFT_PATH
    )

    drift = drift.dropna(
        subset=[
            "drift_priority_score"
        ]
    )

    drift = drift.sort_values(
        "drift_priority_score",
        ascending=False,
    ).head(top_n)

    drift = drift.sort_values(
        "drift_priority_score"
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        drift["feature"],
        drift["drift_priority_score"],
    )

    plt.xlabel(
        "Drift Priority Score"
    )

    plt.ylabel(
        "Sensor"
    )

    plt.title(
        f"Top {top_n} Drift Sensors"
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "top_drift_sensors.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(
        f"[OK] {output}"
    )


# ============================================================
# 4. FN과 Drift가 동시에 큰 센서
# ============================================================

def print_fn_drift_candidates(
    top_n: int = 20,
) -> None:

    if not ERROR_FEATURE_PATH.exists():
        print(
            f"[SKIP] 파일 없음: "
            f"{ERROR_FEATURE_PATH}"
        )
        return

    features = pd.read_csv(
        ERROR_FEATURE_PATH
    )

    required = {
        "feature",
        "absolute_fn_vs_tn_smd",
    }

    if not required.issubset(
        features.columns
    ):
        print(
            "[SKIP] 필요한 column이 없습니다."
        )
        return

    columns = [
        "feature",
        "absolute_fn_vs_tn_smd",
    ]

    if "drift_priority_score" in features:
        columns.append(
            "drift_priority_score"
        )

    if "drift_priority" in features:
        columns.append(
            "drift_priority"
        )

    result = features[
        columns
    ].sort_values(
        "absolute_fn_vs_tn_smd",
        ascending=False,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "FN 분석 주요 센서"
    )
    print(
        "=" * 70
    )

    print(
        result.head(top_n).to_string(
            index=False
        )
    )


# ============================================================
# 5. 전체 실행
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 70
    )
    print(
        "STEP 12 - FINAL ANALYSIS"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "[1] Random vs Time"
    )

    plot_random_vs_time()

    print()
    print(
        "[2] FN / FP"
    )

    plot_error_groups()

    print()
    print(
        "[3] Drift"
    )

    plot_top_drift_sensors()

    print()
    print(
        "[4] FN + Drift 후보"
    )

    print_fn_drift_candidates()

    print()
    print(
        "=" * 70
    )
    print(
        "STEP 12 완료"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()