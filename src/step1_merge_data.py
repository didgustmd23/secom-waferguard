import pandas as pd
from pathlib import Path


# 프로젝트 기본 경로
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# 1. 파일 불러오기
df1 = pd.read_csv(
    RAW_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
)

df2 = pd.read_csv(
    RAW_DIR / "secom.data",
    sep=r"\s+",
    header=None
)


# 2. 열 방향 결합
df = pd.concat([df1, df2], axis=1)


# 3. 컬럼명 지정
df.columns = (
    ["label", "timestamp"]
    + [f"sensor_{i}" for i in range(df2.shape[1])]
)


# 4. processed 폴더 생성
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# 5. 병합 데이터 저장
output_path = PROCESSED_DIR / "secom_merged.csv"

df.to_csv(output_path, index=False)


# 6. 결과 확인
print("성공적으로 저장되었습니다!")
print("저장 위치:", output_path)
print("데이터 크기:", df.shape)

print("\n컬럼명:")
print(df.columns.tolist()[:10])
print("...")
print(df.columns.tolist()[-10:])

print("\n데이터 정보:")
df.info()