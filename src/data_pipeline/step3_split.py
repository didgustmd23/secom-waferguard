import argparse
import hashlib
import json
from pathlib import Path

# ==========================================
# 팀원 구현 기반 데이터 분할 및 점검
# - 기존 Random stratify·Time 경계 이동·EDA·요약 함수를 유지
# - 공통 Profile 검증과 원본 ID·그룹 검사·저장 전 검증을 통합
# - 전체 데이터 Random split은 탐색용이며 시간 평가 학습은 Time Train만 사용
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# ============================================================
# Config import
# ============================================================

try:
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )
    from src.dataset_schema import validate_dataset_frame
    from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID, SPLIT_METADATA
except ImportError:
    from modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )
    from dataset_schema import validate_dataset_frame
    from split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID, SPLIT_METADATA


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
LOG_DIR = PROJECT_ROOT / "logs"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "step3"


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
    group_columns=(),
):
    """
    Random Train / Validation / Test
    = 70% / 15% / 15%
    stratify 적용
    """

    # 그룹 선언이 없을 때는 팀원의 기존 2단계 stratify 분할을 그대로 유지한다.
    if group_columns:
        if df[list(group_columns)].isna().any().any():
            raise ValueError("Random split 그룹 컬럼에 결측값이 있습니다.")
        groups, _ = pd.factorize(pd.MultiIndex.from_frame(df[list(group_columns)]))
        first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
        train_index, temp_index = next(first.split(df, groups=groups))
        temp = df.iloc[temp_index]
        second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
        valid_index, test_index = next(second.split(temp, groups=groups[temp_index]))
        # 그룹 분할은 정확한 행 비율·label 계층화를 보장하지 않으므로 요약에 실제 비율을 기록한다.
        return df.iloc[train_index].copy(), temp.iloc[valid_index].copy(), temp.iloc[test_index].copy()

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
    *,
    group_columns=(),
    id_columns=(),
    train_ratio=0.70,
    validation_ratio=0.15,
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

    # 비율은 목표값이며 시간·그룹 경계를 지키는 것이 우선이다.
    if not (np.isfinite(train_ratio) and np.isfinite(validation_ratio)
            and train_ratio > 0 and validation_ratio > 0
            and train_ratio + validation_ratio < 1):
        raise ValueError("Train·Validation 비율은 양수이며 합계가 1보다 작아야 합니다.")
    if "_parsed_timestamp" in df:
        raise ValueError("입력에 내부 작업용 컬럼 '_parsed_timestamp'가 있습니다.")
    work = df.copy()

    work["_parsed_timestamp"] = pd.to_datetime(
        work[timestamp_col],
        format=timestamp_format,
        errors="coerce",
    )

    invalid_count = work["_parsed_timestamp"].isna().sum()

    if invalid_count > 0:
        raise ValueError(
            f"timestamp 파싱 실패 또는 결측값이 {invalid_count:,}행 있습니다."
        )

    work = work.sort_values(
        "_parsed_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    n = len(work)

    if n < 3:
        raise ValueError(
            "Time split에는 최소 3행이 필요합니다."
        )

    # 목표 위치
    train_target = int(n * train_ratio)
    valid_target = int(n * (train_ratio + validation_ratio))

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
    # 그룹·복합 ID가 경계를 가로지르는 위치를 제외한다.
    # 기본 timestamp 이동 결과가 안전하면 그대로 사용하고, 불가능할 때만 다른 경계를 찾는다.
    # --------------------------------------------------------

    timestamp_values = work["_parsed_timestamp"].to_numpy()
    candidates = np.flatnonzero(timestamp_values[1:] != timestamp_values[:-1]) + 1
    blocked_delta = np.zeros(n + 1, dtype=int)
    for columns in (group_columns, id_columns):
        if not columns:
            continue
        if work[list(columns)].isna().any().any():
            raise ValueError("시간 split의 그룹·ID 컬럼에 결측값이 있습니다.")
        for positions in work.groupby(list(columns), sort=False, observed=True).indices.values():
            blocked_delta[int(np.min(positions)) + 1] += 1
            blocked_delta[int(np.max(positions)) + 1] -= 1
    blocked = np.cumsum(blocked_delta)
    safe_boundaries = [int(cut) for cut in candidates if blocked[cut] == 0]
    if len(safe_boundaries) < 2:
        raise ValueError("timestamp·그룹을 유지하면서 세 구간으로 나눌 수 없습니다.")
    if not (0 < train_end < valid_end < n
            and train_end in safe_boundaries and valid_end in safe_boundaries):
        train_end = min(safe_boundaries[:-1], key=lambda cut: (abs(cut - train_target), cut))
        valid_end = min((cut for cut in safe_boundaries if cut > train_end),
                        key=lambda cut: (abs(cut - valid_target), cut))

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

    column_sets = [set(frame.columns) for frame in (train_df, valid_df, test_df)]
    if not (column_sets[0] == column_sets[1] == column_sets[2]):
        return [{"check": "duplicate_row_overlap", "status": "FAIL",
                 "detail": "split 간 컬럼 구성이 다릅니다."}]
    common_columns = sorted(
        set(train_df.columns)
        & set(valid_df.columns)
        & set(test_df.columns)
        - set(SPLIT_METADATA)
    )

    if not common_columns:
        return [
            {
                "check": "duplicate_row_overlap",
                "status": "FAIL",
                "detail": "검사할 데이터 컬럼이 없습니다.",
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
    timestamp_format=None,
):

    # Profile 형식으로 파싱하며, 실패한 값을 버려 거짓 PASS가 나오는 것을 방지한다.
    parsed = [pd.to_datetime(frame[timestamp_col], format=timestamp_format, errors="coerce")
              for frame in (train_df, valid_df, test_df)]
    invalid_count = sum(int(values.isna().sum()) for values in parsed)
    if invalid_count:
        return [{"check": "timestamp_overlap", "status": "FAIL",
                 "detail": f"timestamp 파싱 실패 또는 결측값: {invalid_count}행"}]

    train_ts, valid_ts, test_ts = (set(values) for values in parsed)

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
    timestamp_format=None,
    group_columns=(),
    id_columns=(),
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
            timestamp_format,
        )
    )

    # 원본 행 ID와 Profile의 그룹·복합 ID가 split 간에 공유되는지도 별도로 검사한다.
    for columns in ((SOURCE_ROW_ID,), group_columns, id_columns):
        if not columns or not all(column in train_df for column in columns):
            continue
        frames = (train_df, valid_df, test_df)
        missing = any(frame[list(columns)].isna().any().any() for frame in frames)
        keys = [set(map(tuple, frame[list(columns)].to_numpy())) for frame in frames]
        overlaps = [len(keys[0] & keys[1]), len(keys[0] & keys[2]), len(keys[1] & keys[2])]
        results.append({"check": "metadata_overlap:" + ",".join(columns),
                        "status": "FAIL" if missing or any(overlaps) else "PASS",
                        "detail": f"결측={missing}, train-valid/train-test/valid-test={overlaps}"})

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

    parse_ok = not any(values.isna().any() or values.empty for values in (train_ts, valid_ts, test_ts))
    overall_ok = (
        train_valid_ok
        and valid_test_ok
        and parse_ok
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
                "check": "timestamp_parsing",
                "status": "PASS" if parse_ok else "FAIL",
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


# ==========================================
# 팀원의 분할 함수를 재사용하고 원본 행 추적 metadata를 추가
# - Random은 탐색용, Time Train은 후보 비교·OOF용으로 역할 구분
# - metadata는 분할 이후에 부여하여 EDA와 label 계층화 입력에 섞이지 않음
# ==========================================
def prepare_split_frames(df, dataset, *, random_state=42):
    validate_dataset_frame(df, dataset)
    if set(df.columns) & set(SPLIT_METADATA):
        raise ValueError("입력 원본에 예약된 split metadata 컬럼이 있습니다.")
    if dataset.timestamp_column is None:
        raise ValueError("시간 split에는 Profile의 timestamp 컬럼이 필요합니다.")
    work = df.copy()
    work[SOURCE_ROW_ID] = np.arange(len(work))
    # 시간 분할을 먼저 검증하고, 그 뒤 별도 탐색용 Random split을 만든다.
    time_frames = make_time_split(work, dataset.timestamp_column, dataset.timestamp_format,
                                  group_columns=dataset.group_columns, id_columns=dataset.id_columns)
    random_frames = make_random_split(work, dataset.label_column, random_state,
                                      group_columns=dataset.group_columns or dataset.id_columns)
    if random_frames[0][dataset.label_column].nunique() < 2 or time_frames[0][dataset.label_column].nunique() < 2:
        raise ValueError("Train split에 정상·Fail label이 모두 있어야 합니다.")
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    digest.update(repr((dataset, random_state, "step3_integrated_v1")).encode("utf-8"))
    splits = {}
    for strategy, frames in (("random", random_frames), ("time", time_frames)):
        for suffix, role, frame in zip(("train", "valid", "test"), ("train", "validation", "test"), frames):
            tagged = frame.copy()
            tagged[SPLIT_ROLE] = role
            tagged[PROTOCOL_ID] = strategy + ":" + digest.hexdigest()
            splits[f"{strategy}_{suffix}"] = tagged.reset_index(drop=True)
    return splits


# ==========================================
# 검증을 통과한 split을 저장하고 생성 계약을 기록
# - 기존 split 파일이 있으면 덮어쓰지 않음
# - 원본 내용 중복·metadata 중복·timestamp·시간 경계를 저장 전에 검사
# ==========================================
def write_split_files(splits, dataset, output_dir):
    output_dir = Path(output_dir)
    paths = [output_dir / f"{name}.csv" for name in splits] + [output_dir / "manifest.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("split 결과가 이미 있습니다. 새로운 --output-dir을 지정하세요.")
    for strategy in ("random", "time"):
        frames = [splits[f"{strategy}_{suffix}"] for suffix in ("train", "valid", "test")]
        # 저장 경계에서도 역할·계약·ID 정합성을 검사하여 수동 변경된 split을 차단한다.
        protocol_ids = set()
        for frame, role in zip(frames, ("train", "validation", "test")):
            if frame.empty or not set(SPLIT_METADATA).issubset(frame.columns):
                raise ValueError("저장할 split이 비어 있거나 필수 metadata가 없습니다.")
            if frame[list(SPLIT_METADATA)].isna().any().any() or set(frame[SPLIT_ROLE]) != {role}:
                raise ValueError("저장할 split의 역할 또는 metadata가 올바르지 않습니다.")
            if frame[SOURCE_ROW_ID].duplicated().any():
                raise ValueError("split 내부의 원본 행 ID가 중복됩니다.")
            protocol_ids.update(frame[PROTOCOL_ID])
        if len(protocol_ids) != 1:
            raise ValueError("서로 다른 생성 계약의 split은 함께 저장할 수 없습니다.")
        checks = run_leakage_checks(*frames, dataset.timestamp_column, strategy,
                                    dataset.timestamp_format, dataset.group_columns, dataset.id_columns)
        if any(row["status"] == "FAIL" for row in checks):
            raise ValueError(f"{strategy} split 누수 검사가 실패하여 저장하지 않았습니다: {checks}")
        if strategy == "time" and check_time_boundaries(*frames, dataset.timestamp_column,
                                                       dataset.timestamp_format)["status"].eq("FAIL").any():
            raise ValueError("시간 경계 검사가 실패하여 split을 저장하지 않았습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    manifest = {"dataset_id": dataset.dataset_id,
                "protocol_ids": {name: frame[PROTOCOL_ID].iloc[0] for name, frame in splits.items()},
                "training_file": "time_train.csv", "validation_file": "time_valid.csv", "test_file": "time_test.csv",
                "policy": "Random은 탐색용; 후보 비교·특징 선택·OOF는 동일한 Time Train만 사용"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_arguments():
    # 팀원 스크립트를 단일 진입점으로 사용하고 기존 산출물은 새 폴더로 보존한다.
    parser = argparse.ArgumentParser(description="Random 탐색 split 및 시간 holdout split 생성")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/splits/integrated")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs/step3_integrated")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "reports/figures/step3")
    parser.add_argument("--skip-eda", action="store_true", help="분할·누수 검사만 수행하고 EDA·그림 생성은 생략")
    return parser.parse_args()

def main():
    global SPLIT_DIR, LOG_DIR, FIGURE_DIR
    arguments = _parse_arguments()
    SPLIT_DIR, LOG_DIR, FIGURE_DIR = arguments.output_dir, arguments.log_dir, arguments.figures_dir

    print("\n" + "=" * 70)
    print("STEP 3 - DATA SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = load_modeling_config(
        arguments.config
    )

    dataset = config.dataset

    INPUT_FILE = arguments.input or dataset.input_path
    TARGET_COL = dataset.label_column
    TIMESTAMP_COL = dataset.timestamp_column

    # config에 timestamp format이 없을 수도 있으므로 안전하게 처리
    TIMESTAMP_FORMAT = getattr(
        dataset,
        "timestamp_format",
        None,
    )

    RANDOM_STATE = config.experiment.cv.random_state

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
            f"입력 파일을 찾을 수 없습니다: {input_path}"
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
            f"label 컬럼을 찾을 수 없습니다: {TARGET_COL}"
        )

    if TIMESTAMP_COL not in df.columns:
        raise KeyError(
            f"timestamp 컬럼을 찾을 수 없습니다: {TIMESTAMP_COL}"
        )

    # --------------------------------------------------------
    # Target missingness
    # --------------------------------------------------------

    target_missing = df[TARGET_COL].isna().sum()

    if target_missing > 0:
        raise ValueError(
            f"label 컬럼 '{TARGET_COL}'에 결측값이 {target_missing:,}행 있어 안전하게 분할할 수 없습니다."
        )

    # 공통 Profile 검증 및 기존 분할 함수를 통해 여섯 split을 먼저 준비한다.
    splits = prepare_split_frames(df, dataset, random_state=RANDOM_STATE)
    expected_paths = [SPLIT_DIR / f"{name}.csv" for name in splits] + [SPLIT_DIR / "manifest.json"]
    if any(path.exists() for path in expected_paths):
        raise FileExistsError("split 결과가 이미 있습니다. 새로운 --output-dir을 지정하세요.")

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    print("\n[2] Running EDA")

    if not arguments.skip_eda:
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
        splits["random_train"], splits["random_valid"], splits["random_test"]
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

    # 실제 저장은 모든 누수·시간 경계 검사를 통과한 뒤에 수행한다.

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
        splits["time_train"], splits["time_valid"], splits["time_test"]
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

    # 시간 split 역시 검사 전에는 CSV로 저장하지 않는다.

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
            TIMESTAMP_FORMAT,
            dataset.group_columns,
            dataset.id_columns,
        )
    )

    leakage_results.extend(
        run_leakage_checks(
            time_train,
            time_valid,
            time_test,
            TIMESTAMP_COL,
            "time",
            TIMESTAMP_FORMAT,
            dataset.group_columns,
            dataset.id_columns,
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
        raise ValueError("누수 또는 시간 경계 검사가 실패하여 split CSV를 저장하지 않았습니다.")
    else:
        write_split_files(splits, dataset, SPLIT_DIR)
        print(
            "[성공] 모든 split 검사를 통과하고 CSV를 저장했습니다."
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
