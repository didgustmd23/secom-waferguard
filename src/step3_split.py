from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Config import
# ============================================================

try:
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )
except ImportError:
    from modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "split"
LOG_DIR = PROJECT_ROOT / "logs"
FIGURE_DIR = LOG_DIR / "figures"


# ============================================================
# EDA
# ============================================================

def run_eda(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    timestamp_format=None,
) -> None:
    """
    클래스 분포, 결측률, Timestamp EDA 및 시각화
    """

    print("\n" + "=" * 70)
    print("EDA START")
    print("=" * 70)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. 기본 데이터 정보
    # --------------------------------------------------------

    print("\n[EDA] Dataset information")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    # --------------------------------------------------------
    # 2. Class distribution
    # --------------------------------------------------------

    print("\n[EDA] Class distribution")

    class_count = df[target_col].value_counts(dropna=False)
    class_ratio = df[target_col].value_counts(
        normalize=True,
        dropna=False,
    )

    class_distribution = pd.DataFrame(
        {
            "class": class_count.index.astype(str),
            "count": class_count.values,
            "ratio": class_ratio.values,
        }
    )

    print(class_distribution.to_string(index=False))

    class_distribution.to_csv(
        LOG_DIR / "class_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Class distribution visualization
    # --------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))

        plt.bar(
            class_distribution["class"],
            class_distribution["count"],
        )

        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "class_distribution.png",
            dpi=150,
        )

        plt.close()

    except ImportError:
        print(
            "[WARNING] matplotlib is not installed. "
            "Class distribution figure was not created."
        )

    # --------------------------------------------------------
    # 4. Missingness
    # --------------------------------------------------------

    print("\n[EDA] Missingness")

    missing_count = df.isna().sum()
    missing_rate = df.isna().mean()

    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_rate": missing_rate.values,
        }
    )

    missing_summary = missing_summary.sort_values(
        "missing_rate",
        ascending=False,
    )

    print(
        missing_summary[
            missing_summary["missing_count"] > 0
        ].to_string(index=False)
    )

    missing_summary.to_csv(
        LOG_DIR / "missing_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Missingness visualization
    # --------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        missing_plot = missing_summary[
            missing_summary["missing_rate"] > 0
        ].head(30)

        if not missing_plot.empty:
            plt.figure(figsize=(12, 6))

            plt.bar(
                missing_plot["column"].astype(str),
                missing_plot["missing_rate"],
            )

            plt.title("Missing Rate")
            plt.xlabel("Column")
            plt.ylabel("Missing Rate")
            plt.xticks(
                rotation=90,
                fontsize=7,
            )

            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR / "missing_rate.png",
                dpi=150,
            )

            plt.close()

        else:
            print(
                "[EDA] No missing values found. "
                "Missing-rate figure was not created."
            )

    except ImportError:
        print(
            "[WARNING] matplotlib is not installed. "
            "Missing-rate figure was not created."
        )

    # --------------------------------------------------------
    # 6. Timestamp EDA
    # --------------------------------------------------------

    print("\n[EDA] Timestamp")

    timestamp_series = pd.to_datetime(
        df[timestamp_col],
        format=timestamp_format,
        errors="coerce",
    )

    timestamp_invalid_count = timestamp_series.isna().sum()

    if timestamp_invalid_count > 0:
        raise ValueError(
            f"Timestamp parsing failed: "
            f"{timestamp_invalid_count:,} rows could not be parsed."
        )

    timestamp_summary = pd.DataFrame(
        {
            "timestamp_min": [timestamp_series.min()],
            "timestamp_max": [timestamp_series.max()],
            "unique_timestamp_count": [
                timestamp_series.nunique()
            ],
            "duplicate_timestamp_count": [
                len(timestamp_series)
                - timestamp_series.nunique()
            ],
        }
    )

    print(timestamp_summary.to_string(index=False))

    timestamp_summary.to_csv(
        LOG_DIR / "timestamp_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Timestamp visualization
    # --------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        timestamp_by_date = (
            timestamp_series
            .dt.floor("D")
            .value_counts()
            .sort_index()
        )

        plt.figure(figsize=(12, 5))

        plt.plot(
            timestamp_by_date.index,
            timestamp_by_date.values,
        )

        plt.title("Timestamp Distribution")
        plt.xlabel("Date")
        plt.ylabel("Row Count")
        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "timestamp_distribution.png",
            dpi=150,
        )

        plt.close()

    except ImportError:
        print(
            "[WARNING] matplotlib is not installed. "
            "Timestamp figure was not created."
        )

    print("\n[EDA] EDA completed.")


# ============================================================
# Split summary
# ============================================================

def make_split_summary(
    data: pd.DataFrame,
    split_name: str,
    target_col: str,
    all_classes,
) -> pd.DataFrame:
    """
    Split별 클래스 count / ratio 생성
    """

    counts = (
        data[target_col]
        .value_counts()
        .reindex(all_classes, fill_value=0)
    )

    total = len(data)

    if total == 0:
        ratios = counts.astype(float)
    else:
        ratios = counts / total

    return pd.DataFrame(
        {
            "split": split_name,
            "class": counts.index.astype(str),
            "count": counts.values,
            "ratio": ratios.values,
        }
    )


# ============================================================
# Random split
# ============================================================

def make_random_split(
    df: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
):
    """
    Random Train / Validation / Test
    = 70% / 15% / 15%
    stratify 적용
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df[target_col],
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[target_col],
    )

    return train_df, valid_df, test_df


# ============================================================
# Time-based split
# ============================================================

def make_time_split(
    df: pd.DataFrame,
    timestamp_col: str,
    timestamp_format=None,
):
    """
    Timestamp 기준 Time-based split

    목표:
        Train      70%
        Validation 15%
        Test       15%

    동일 Timestamp가 서로 다른 split에 들어가지 않도록
    경계를 다음 Timestamp까지 이동한다.
    """

    work = df.copy()

    work["_parsed_timestamp"] = pd.to_datetime(
        work[timestamp_col],
        format=timestamp_format,
        errors="coerce",
    )

    invalid_count = work["_parsed_timestamp"].isna().sum()

    if invalid_count > 0:
        raise ValueError(
            f"Timestamp parsing failed: "
            f"{invalid_count:,} rows could not be parsed."
        )

    work = work.sort_values(
        "_parsed_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    n = len(work)

    if n < 3:
        raise ValueError(
            "Time split requires at least 3 rows."
        )

    # 목표 위치
    train_target = int(n * 0.70)
    valid_target = int(n * 0.85)

    # 최소/최대 경계 보호
    train_target = max(1, min(train_target, n - 2))
    valid_target = max(train_target + 1, min(valid_target, n - 1))

    # --------------------------------------------------------
    # Train boundary
    # 동일 timestamp가 이어지면 그 timestamp 전체를
    # Train에 포함하도록 뒤로 이동
    # --------------------------------------------------------

    train_end = train_target

    while (
        train_end < n
        and train_end > 0
        and work.loc[
            train_end,
            "_parsed_timestamp"
        ]
        == work.loc[
            train_end - 1,
            "_parsed_timestamp"
        ]
    ):
        train_end += 1

    # --------------------------------------------------------
    # Validation boundary
    # --------------------------------------------------------

    valid_end = max(valid_target, train_end)

    while (
        valid_end < n
        and valid_end > 0
        and work.loc[
            valid_end,
            "_parsed_timestamp"
        ]
        == work.loc[
            valid_end - 1,
            "_parsed_timestamp"
        ]
    ):
        valid_end += 1

    # --------------------------------------------------------
    # 최종 경계 검증
    # --------------------------------------------------------

    if not (
        0 < train_end < valid_end < n
    ):
        raise ValueError(
            "Time split boundary is invalid after "
            "same-timestamp boundary adjustment. "
            f"n={n}, train_end={train_end}, "
            f"valid_end={valid_end}"
        )

    train_df = work.iloc[:train_end].copy()
    valid_df = work.iloc[train_end:valid_end].copy()
    test_df = work.iloc[valid_end:].copy()

    # 임시 컬럼 제거
    train_df = train_df.drop(
        columns=["_parsed_timestamp"]
    )
    valid_df = valid_df.drop(
        columns=["_parsed_timestamp"]
    )
    test_df = test_df.drop(
        columns=["_parsed_timestamp"]
    )

    print("\n[Time Split]")
    print(
        f"Train      : {len(train_df):,} "
        f"({len(train_df) / n:.4f})"
    )
    print(
        f"Validation : {len(valid_df):,} "
        f"({len(valid_df) / n:.4f})"
    )
    print(
        f"Test       : {len(test_df):,} "
        f"({len(test_df) / n:.4f})"
    )

    return train_df, valid_df, test_df


# ============================================================
# Duplicate leakage check
# ============================================================

def check_duplicate_leakage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    동일 row가 여러 split에 존재하는지 확인
    """

    common_columns = sorted(
        set(train_df.columns)
        & set(valid_df.columns)
        & set(test_df.columns)
    )

    if not common_columns:
        return [
            {
                "check": "duplicate_row_overlap",
                "status": "INFO",
                "detail": "No common columns available.",
            }
        ]

    def make_row_keys(data):
 
        return set(
            pd.util.hash_pandas_object(
                data[common_columns],
                index=False,
            )
        )



    train_keys = make_row_keys(train_df)
    valid_keys = make_row_keys(valid_df)
    test_keys = make_row_keys(test_df)

    train_valid = len(train_keys & valid_keys)
    train_test = len(train_keys & test_keys)
    valid_test = len(valid_keys & test_keys)

    total_overlap = (
        train_valid
        + train_test
        + valid_test
    )

    status = (
        "PASS"
        if total_overlap == 0
        else "FAIL"
    )

    return [
        {
            "check": "duplicate_row_overlap",
            "status": status,
            "detail": (
                f"train-valid={train_valid}, "
                f"train-test={train_test}, "
                f"valid-test={valid_test}"
            ),
        }
    ]


# ============================================================
# Timestamp overlap check
# ============================================================

def check_timestamp_overlap(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str,
    split_type: str,
):

    train_ts = set(
        pd.to_datetime(
            train_df[timestamp_col],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        ).dropna()
    )

    valid_ts = set(
        pd.to_datetime(
            valid_df[timestamp_col],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        ).dropna()
    )

    test_ts = set(
        pd.to_datetime(
            test_df[timestamp_col],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        ).dropna()
    )

    train_valid = len(train_ts & valid_ts)
    train_test = len(train_ts & test_ts)
    valid_test = len(valid_ts & test_ts)

    overlap_exists = (
        train_valid > 0
        or train_test > 0
        or valid_test > 0
    )

    if split_type == "random":
        status = "INFO"
    else:
        status = (
            "FAIL"
            if overlap_exists
            else "PASS"
        )

    return [
        {
            "check": "timestamp_overlap",
            "status": status,
            "detail": (
                f"train-valid={train_valid}, "
                f"train-test={train_test}, "
                f"valid-test={valid_test}"
            ),
        }
    ]


# ============================================================
# Leakage checks
# ============================================================

def run_leakage_checks(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str,
    split_type: str,
):
    """
    Duplicate row / Timestamp leakage 검사
    """

    results = []

    results.extend(
        check_duplicate_leakage(
            train_df,
            valid_df,
            test_df,
        )
    )

    results.extend(
        check_timestamp_overlap(
            train_df,
            valid_df,
            test_df,
            timestamp_col,
            split_type,
        )
    )

    for result in results:
        result["split_type"] = split_type

    return results


# ============================================================
# Time boundary check
# ============================================================

def check_time_boundaries(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str,
    timestamp_format=None,
):
    """
    Time split에서
        max(train timestamp)
        <
        min(valid timestamp)
        <
        max(valid timestamp)
        <
        min(test timestamp)

    관계를 확인
    """

    train_ts = pd.to_datetime(
        train_df[timestamp_col],
        format=timestamp_format,
        errors="coerce",
    )

    valid_ts = pd.to_datetime(
        valid_df[timestamp_col],
        format=timestamp_format,
        errors="coerce",
    )

    test_ts = pd.to_datetime(
        test_df[timestamp_col],
        format=timestamp_format,
        errors="coerce",
    )

    train_max = train_ts.max()
    valid_min = valid_ts.min()
    valid_max = valid_ts.max()
    test_min = test_ts.min()

    train_valid_ok = train_max < valid_min
    valid_test_ok = valid_max < test_min

    overall_ok = (
        train_valid_ok
        and valid_test_ok
    )

    return pd.DataFrame(
        [
            {
                "check": "train_max < valid_min",
                "status": (
                    "PASS"
                    if train_valid_ok
                    else "FAIL"
                ),
                "train_max": train_max,
                "valid_min": valid_min,
            },
            {
                "check": "valid_max < test_min",
                "status": (
                    "PASS"
                    if valid_test_ok
                    else "FAIL"
                ),
                "valid_max": valid_max,
                "test_min": test_min,
            },
            {
                "check": "overall",
                "status": (
                    "PASS"
                    if overall_ok
                    else "FAIL"
                ),
                "train_max": train_max,
                "valid_min": valid_min,
                "valid_max": valid_max,
                "test_min": test_min,
            },
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("STEP 3 - DATA SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = load_modeling_config(
        DEFAULT_CONFIG_PATH
    )

    dataset = config.dataset

    INPUT_FILE = dataset.input_path
    TARGET_COL = dataset.label_column
    TIMESTAMP_COL = dataset.timestamp_column

    # config에 timestamp format이 없을 수도 있으므로 안전하게 처리
    TIMESTAMP_FORMAT = getattr(
        dataset,
        "timestamp_format",
        None,
    )

    RANDOM_STATE = 42

    # --------------------------------------------------------
    # Path
    # --------------------------------------------------------

    input_path = Path(INPUT_FILE)

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n[1] Loading dataset")
    print(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    print(
        f"Loaded: {len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Target column not found: {TARGET_COL}"
        )

    if TIMESTAMP_COL not in df.columns:
        raise KeyError(
            f"Timestamp column not found: {TIMESTAMP_COL}"
        )

    # --------------------------------------------------------
    # Target missingness
    # --------------------------------------------------------

    target_missing = df[TARGET_COL].isna().sum()

    if target_missing > 0:
        raise ValueError(
            f"Target column '{TARGET_COL}' contains "
            f"{target_missing:,} missing values. "
            "Stratified split cannot be performed safely."
        )

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    print("\n[2] Running EDA")

    run_eda(
        df=df,
        target_col=TARGET_COL,
        timestamp_col=TIMESTAMP_COL,
        timestamp_format=TIMESTAMP_FORMAT,
    )

    # --------------------------------------------------------
    # Random split
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] Random split 70 / 15 / 15")
    print("=" * 70)

    random_train, random_valid, random_test = (
        make_random_split(
            df,
            TARGET_COL,
            RANDOM_STATE,
        )
    )

    random_train_path = (
        SPLIT_DIR / "random_train.csv"
    )

    random_valid_path = (
        SPLIT_DIR / "random_valid.csv"
    )

    random_test_path = (
        SPLIT_DIR / "random_test.csv"
    )

    random_train.to_csv(
        random_train_path,
        index=False,
    )

    random_valid.to_csv(
        random_valid_path,
        index=False,
    )

    random_test.to_csv(
        random_test_path,
        index=False,
    )

    print(
        f"Random Train : {len(random_train):,}"
    )
    print(
        f"Random Valid : {len(random_valid):,}"
    )
    print(
        f"Random Test  : {len(random_test):,}"
    )

    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] Time-based split")
    print("=" * 70)

    time_train, time_valid, time_test = (
        make_time_split(
            df,
            TIMESTAMP_COL,
            TIMESTAMP_FORMAT,
        )
    )

    time_train_path = (
        SPLIT_DIR / "time_train.csv"
    )

    time_valid_path = (
        SPLIT_DIR / "time_valid.csv"
    )

    time_test_path = (
        SPLIT_DIR / "time_test.csv"
    )

    time_train.to_csv(
        time_train_path,
        index=False,
    )

    time_valid.to_csv(
        time_valid_path,
        index=False,
    )

    time_test.to_csv(
        time_test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    print("\n[5] Creating split summary")

    all_classes = sorted(
        df[TARGET_COL].unique()
    )

    summary_list = []

    summary_list.append(
        make_split_summary(
            random_train,
            "random_train",
            TARGET_COL,
            all_classes,
        )
    )

    summary_list.append(
        make_split_summary(
            random_valid,
            "random_valid",
            TARGET_COL,
            all_classes,
        )
    )

    summary_list.append(
        make_split_summary(
            random_test,
            "random_test",
            TARGET_COL,
            all_classes,
        )
    )

    summary_list.append(
        make_split_summary(
            time_train,
            "time_train",
            TARGET_COL,
            all_classes,
        )
    )

    summary_list.append(
        make_split_summary(
            time_valid,
            "time_valid",
            TARGET_COL,
            all_classes,
        )
    )

    summary_list.append(
        make_split_summary(
            time_test,
            "time_test",
            TARGET_COL,
            all_classes,
        )
    )

    split_summary = pd.concat(
        summary_list,
        ignore_index=True,
    )

    split_summary_path = (
        LOG_DIR / "split_summary.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    print(
        f"Saved: {split_summary_path}"
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    print("\n[6] Running leakage checks")

    leakage_results = []

    leakage_results.extend(
        run_leakage_checks(
            random_train,
            random_valid,
            random_test,
            TIMESTAMP_COL,
            "random",
        )
    )

    leakage_results.extend(
        run_leakage_checks(
            time_train,
            time_valid,
            time_test,
            TIMESTAMP_COL,
            "time",
        )
    )

    leakage_df = pd.DataFrame(
        leakage_results
    )

    leakage_path = (
        LOG_DIR / "leakage_check.csv"
    )

    leakage_df.to_csv(
        leakage_path,
        index=False,
    )

    print(
        f"Saved: {leakage_path}"
    )

    # --------------------------------------------------------
    # Time boundary check
    # --------------------------------------------------------

    print("\n[7] Checking time boundaries")

    time_boundary_df = check_time_boundaries(
        time_train,
        time_valid,
        time_test,
        TIMESTAMP_COL,
        TIMESTAMP_FORMAT,
    )

    time_boundary_path = (
        LOG_DIR / "time_split_boundary_check.csv"
    )

    time_boundary_df.to_csv(
        time_boundary_path,
        index=False,
    )

    print(
        time_boundary_df.to_string(index=False)
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    has_failure = (
        leakage_df["status"]
        .eq("FAIL")
        .any()
        or
        time_boundary_df["status"]
        .eq("FAIL")
        .any()
    )

    print("\n" + "=" * 70)

    if has_failure:
        print(
            "[WARNING] Leakage or time-boundary "
            "check contains FAIL."
        )
    else:
        print(
            "[SUCCESS] All split checks passed."
        )

    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"  Random Train : {random_train_path}"
    )
    print(
        f"  Random Valid : {random_valid_path}"
    )
    print(
        f"  Random Test  : {random_test_path}"
    )

    print(
        f"  Time Train   : {time_train_path}"
    )
    print(
        f"  Time Valid   : {time_valid_path}"
    )
    print(
        f"  Time Test    : {time_test_path}"
    )

    print(
        f"  Split Summary: {split_summary_path}"
    )

    print(
        f"  Leakage      : {leakage_path}"
    )

    print(
        f"  Time Boundary: {time_boundary_path}"
    )

    print(
        f"  EDA Logs     : {LOG_DIR}"
    )

    print(
        f"  Figures      : {FIGURE_DIR}"
    )

    print("\nSTEP 3 completed.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
