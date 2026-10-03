import pandas as pd
from pathlib import Path


# 프로젝트 기본 경로
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"


# 1. 병합 데이터 불러오기
input_path = PROCESSED_DIR / "secom_merged.csv"

df = pd.read_csv(input_path)

print("데이터 불러오기 완료")
print("데이터 크기:", df.shape)

print("\n데이터 앞부분:")
print(df.head())

print("\n데이터 정보:")
df.info()


# --------------------------------------------------
# 2. 센서 컬럼 찾기
# --------------------------------------------------

sensor_cols = [
    col for col in df.columns
    if col.startswith("sensor_")
]

print("\n센서 개수:", len(sensor_cols))

X = df[sensor_cols]

print("센서 데이터 크기:", X.shape)


# --------------------------------------------------
# 3. 결측치 확인
# --------------------------------------------------

total_missing = X.isna().sum().sum()

print("\n전체 결측치 개수:", total_missing)


# 센서별 결측률 계산
missing_ratio = X.isna().mean()

print("\n결측률이 높은 센서:")
print(
    missing_ratio
    .sort_values(ascending=False)
    .head(20)
)


# --------------------------------------------------
# 4. 결측률 50% 초과 센서 제거
# --------------------------------------------------

high_missing = missing_ratio > 0.5

missing_removed = high_missing.sum()

print("\n결측률 50% 초과 센서:", missing_removed)


# 50% 이하 센서만 유지
keep_cols = missing_ratio <= 0.5

X_clean = X.loc[:, keep_cols]

print("원래 센서 수:", X.shape[1])
print("제거 후 센서 수:", X_clean.shape[1])
print("제거한 센서 수:", X.shape[1] - X_clean.shape[1])


# --------------------------------------------------
# 5. 분산 0 센서 찾기
# --------------------------------------------------

variance = X_clean.var()

zero_var_cols = variance[variance == 0].index

zero_variance_removed = len(zero_var_cols)

print("\n분산 0 센서:", zero_variance_removed)


# 분산 0 센서 제거
X_clean = X_clean.drop(columns=zero_var_cols)

print("최종 센서 수:", X_clean.shape[1])


# --------------------------------------------------
# 6. 센서 제거 결과 요약
# --------------------------------------------------

original_sensor_count = len(sensor_cols)

final_sensor_count = X_clean.shape[1]

summary = pd.DataFrame({
    "단계": [
        "원본",
        "결측률 50% 초과 제거",
        "분산 0 제거"
    ],
    "센서 수": [
        original_sensor_count,
        original_sensor_count - missing_removed,
        final_sensor_count
    ],
    "제거 센서 수": [
        0,
        missing_removed,
        zero_variance_removed
    ]
})


print("\n센서 제거 결과:")
print(summary)


# --------------------------------------------------
# 7. Label + Timestamp + 정제된 센서 다시 결합
# --------------------------------------------------

df_clean = pd.concat(
    [
        df[["label", "timestamp"]],
        X_clean
    ],
    axis=1
)

print("\n최종 데이터 크기:", df_clean.shape)

print("\n최종 데이터:")
print(df_clean.head())


# --------------------------------------------------
# 8. 데이터 품질 로그 저장
# --------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

log_path = LOG_DIR / "dataset_log.csv"

log_df = pd.DataFrame({
    "항목": [
        "원본 센서 수",
        "결측률 50% 초과 제거",
        "분산 0 제거",
        "최종 센서 수"
    ],
    "센서 수": [
        original_sensor_count,
        missing_removed,
        zero_variance_removed,
        final_sensor_count
    ],
    "기준": [
        "원본 센서 개수",
        "결측률 > 50%",
        "분산 = 0",
        "제거 후 최종 센서 개수"
    ]
})

log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

print("\n데이터 품질 로그 저장 완료!")
print("로그 위치:", log_path)