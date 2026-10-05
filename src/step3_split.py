from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )
except ModuleNotFoundError:
    from modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
LOG_DIR = PROJECT_ROOT / "logs"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Dataset Profile 불러오기
# ============================================================

config = load_modeling_config(DEFAULT_CONFIG_PATH)
dataset = config.dataset

INPUT_FILE = dataset.input_path
TARGET_COL = dataset.label_column
TIMESTAMP_COL = dataset.timestamp_column


# ============================================================
# 3. 데이터 불러오기
# ============================================================

print("=" * 60)
print("데이터 불러오기")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"데이터 크기: {df.shape}")
print()
print(df.head())


# ============================================================
# 4. Random Train / Validation / Test Split
#    70 / 15 / 15
# ============================================================

print("\n" + "=" * 60)
print("Random Split")
print("=" * 60)

# 먼저 70% Train / 30% 임시 데이터
train_random, temp_random = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df[TARGET_COL]
)

# 임시 데이터의 절반씩 Validation / Test
valid_random, test_random = train_test_split(
    temp_random,
    test_size=0.50,
    random_state=42,
    stratify=temp_random[TARGET_COL]
)

print(f"Train: {len(train_random)}")
print(f"Validation: {len(valid_random)}")
print(f"Test: {len(test_random)}")


# 저장
train_random.to_csv(
    SPLIT_DIR / "random_train.csv",
    index=False
)

valid_random.to_csv(
    SPLIT_DIR / "random_valid.csv",
    index=False
)

test_random.to_csv(
    SPLIT_DIR / "random_test.csv",
    index=False
)


# ============================================================
# 5. Time-based Split
# ============================================================

print("\n" + "=" * 60)
print("Time-based Split")
print("=" * 60)

# 원본 데이터는 그대로 유지하고
# Timestamp를 임시 정렬용 컬럼으로 변환
df_time = df.copy()

df_time["_timestamp_sort"] = pd.to_datetime(
    df_time[TIMESTAMP_COL],
    format=dataset.timestamp_format,
    errors="coerce"
)

# Timestamp 파싱 실패 확인
if df_time["_timestamp_sort"].isna().any():
    failed_count = df_time["_timestamp_sort"].isna().sum()

    raise ValueError(
        f"Timestamp 파싱 실패: {failed_count}건\n"
        f"사용한 format: {dataset.timestamp_format}"
    )

# Timestamp 기준 정렬
df_time = (
    df_time
    .sort_values("_timestamp_sort")
    .reset_index(drop=True)
    .drop(columns="_timestamp_sort")
)

n = len(df_time)

train_end = int(n * 0.70)
valid_end = int(n * 0.85)

time_train = df_time.iloc[:train_end].copy()
time_valid = df_time.iloc[train_end:valid_end].copy()
time_test = df_time.iloc[valid_end:].copy()

print(f"Train: {len(time_train)}")
print(f"Validation: {len(time_valid)}")
print(f"Test: {len(time_test)}")


# 저장
time_train.to_csv(
    SPLIT_DIR / "time_train.csv",
    index=False
)

time_valid.to_csv(
    SPLIT_DIR / "time_valid.csv",
    index=False
)

time_test.to_csv(
    SPLIT_DIR / "time_test.csv",
    index=False
)


# ============================================================
# 6. Split별 클래스 비율 요약
# ============================================================

def make_split_summary(data, split_name):
    counts = data[TARGET_COL].value_counts()
    ratios = data[TARGET_COL].value_counts(normalize=True) * 100

    rows = []

    for cls in sorted(df[TARGET_COL].dropna().unique()):
        rows.append({
            "split": split_name,
            "class": cls,
            "count": int(counts.get(cls, 0)),
            "ratio_percent": round(
                float(ratios.get(cls, 0)),
                2
            ),
            "total": len(data)
        })

    return rows


summary_rows = []

summary_rows.extend(
    make_split_summary(train_random, "random_train")
)

summary_rows.extend(
    make_split_summary(valid_random, "random_valid")
)

summary_rows.extend(
    make_split_summary(test_random, "random_test")
)

summary_rows.extend(
    make_split_summary(time_train, "time_train")
)

summary_rows.extend(
    make_split_summary(time_valid, "time_valid")
)

summary_rows.extend(
    make_split_summary(time_test, "time_test")
)

split_summary = pd.DataFrame(summary_rows)

split_summary.to_csv(
    LOG_DIR / "split_summary.csv",
    index=False
)


# ============================================================
# 7. 결과 출력
# ============================================================

print("\n" + "=" * 60)
print("Split Summary")
print("=" * 60)

print(split_summary.to_string(index=False))

print("\n" + "=" * 60)
print("완료")
print("=" * 60)

print(f"Split 파일 위치: {SPLIT_DIR}")
print(f"Summary 파일: {LOG_DIR / 'split_summary.csv'}")