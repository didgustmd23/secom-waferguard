# ============================================================
# STEP 5 - DAY 3 ANALYSIS
# 1. PCA 누적 설명분산 시각화
# 2. Logistic Regression Feature Importance 시각화
# 3. 센서 수와 Average Precision(AP)의 관계 분석
# ============================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"

REPORT_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Config import
# ============================================================

try:
    from src.dataset_schema import validate_dataset_frame
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )
    from src.modeling_metrics import evaluate_binary_scores

except ModuleNotFoundError:
    from dataset_schema import validate_dataset_frame
    from modeling_config import (
        DEFAULT_CONFIG_PATH,
        load_modeling_config,
    )
    from modeling_metrics import evaluate_binary_scores


# ============================================================
# 3. 설정 불러오기
# ============================================================

config = load_modeling_config(DEFAULT_CONFIG_PATH)

dataset = config.dataset

TARGET_COL = dataset.label_column

TRAIN_FILE = SPLIT_DIR / "random_train.csv"
VALID_FILE = SPLIT_DIR / "random_valid.csv"

PCA_FIGURE = REPORT_DIR / "day3_pca_cumulative_variance.png"
FEATURE_IMPORTANCE_FIGURE = REPORT_DIR / "day3_feature_importance.png"
SENSOR_AP_FIGURE = REPORT_DIR / "day3_sensor_count_vs_ap.png"

SENSOR_AP_LOG = LOG_DIR / "day3_sensor_ap.csv"


# ============================================================
# 4. 시작
# ============================================================

print("=" * 70)
print("STEP 5 - DAY 3 ANALYSIS")
print("=" * 70)


# ============================================================
# 5. 데이터 불러오기
# ============================================================

print("\n[1] 데이터 불러오기")

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

print(f"Train shape      : {train_df.shape}")
print(f"Validation shape : {valid_df.shape}")


# ============================================================
# 6. Dataset Profile 기준 Feature 확인
# ============================================================

print("\n[2] Feature 확인")

train_features = validate_dataset_frame(
    train_df,
    dataset,
)

valid_features = validate_dataset_frame(
    valid_df,
    dataset,
)

if set(train_features) != set(valid_features):
    raise ValueError(
        "Train과 Validation의 feature 구성이 다릅니다."
    )

# Train의 feature 순서를 기준으로 통일
feature_columns = tuple(train_features)

print(f"Feature 수: {len(feature_columns)}")

print("\nFeature 예시:")
print(list(feature_columns[:10]))


# ============================================================
# 7. X / y 분리
# ============================================================

X_train = train_df.loc[:, list(feature_columns)]
y_train = train_df[TARGET_COL]

X_valid = valid_df.loc[:, list(feature_columns)]
y_valid = valid_df[TARGET_COL]

print(f"\nX_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"X_valid: {X_valid.shape}")
print(f"y_valid: {y_valid.shape}")


# ============================================================
# 8. PCA 누적 설명분산
# ============================================================
#
# PCA 전처리:
#
# Train
#   ↓
# Median Imputation
#   ↓
# Standard Scaling
#   ↓
# PCA
#
# 중요한 원칙:
# PCA 전처리는 Train에만 fit한다.
# Validation 정보는 PCA 학습에 사용하지 않는다.
# ============================================================

print("\n" + "=" * 70)
print("[3] PCA 누적 설명분산 분석")
print("=" * 70)


pca_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# Train에만 fit
X_train_scaled = pca_preprocessor.fit_transform(
    X_train
)

print(
    f"전처리 후 Train shape: "
    f"{X_train_scaled.shape}"
)


# 모든 주성분 계산
pca = PCA()

X_train_pca = pca.fit_transform(
    X_train_scaled
)

explained_variance_ratio = (
    pca.explained_variance_ratio_
)

cumulative_variance = (
    explained_variance_ratio.cumsum()
)


# ------------------------------------------------------------
# 주요 누적 설명분산 확인
# ------------------------------------------------------------

variance_targets = [
    0.80,
    0.90,
    0.95,
]

for target in variance_targets:

    component_indices = (
        cumulative_variance >= target
    )

    if component_indices.any():

        component_count = (
            component_indices.argmax() + 1
        )

        actual_variance = cumulative_variance[
            component_count - 1
        ]

        print(
            f"{target:.0%} 설명에 필요한 "
            f"주성분 수: {component_count} "
            f"(실제 누적 설명분산: "
            f"{actual_variance:.4f})"
        )


# ------------------------------------------------------------
# PCA 그래프
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(
        1,
        len(cumulative_variance) + 1,
    ),
    cumulative_variance,
)

plt.axhline(
    0.80,
    linestyle="--",
    label="80%",
)

plt.axhline(
    0.90,
    linestyle="--",
    label="90%",
)

plt.axhline(
    0.95,
    linestyle="--",
    label="95%",
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PCA_FIGURE,
    dpi=150,
)

plt.close()

print(
    f"\nPCA 그래프 저장: "
    f"{PCA_FIGURE}"
)


# ============================================================
# 9. Logistic Regression Feature Importance
# ============================================================
#
# Logistic Regression coefficient의 절댓값을
# feature 영향도처럼 사용한다.
#
# coefficient의 절댓값이 클수록
# 해당 feature가 모델의 결정에 더 큰 영향을 줄 가능성이 있다.
#
# 주의:
# Tree 모델의 feature_importances_와 동일한 개념은 아니다.
# ============================================================

print("\n" + "=" * 70)
print("[4] Logistic Regression Feature Importance")
print("=" * 70)


baseline_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=(
                    config.experiment.cv.random_state
                ),
            ),
        ),
    ]
)


# Train에만 fit
baseline_pipeline.fit(
    X_train,
    y_train,
)


model = baseline_pipeline.named_steps[
    "model"
]

coefficients = model.coef_[0]


feature_importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "coefficient": coefficients,
        "importance": abs(coefficients),
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\nTop 20 Feature Importance")

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Top 20 Feature Importance 그래프
# ------------------------------------------------------------

top_n = 20

top_features = (
    feature_importance
    .head(top_n)
    .sort_values(
        "importance",
        ascending=True,
    )
)


plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"],
    top_features["importance"],
)

plt.xlabel(
    "|Logistic Regression Coefficient|"
)

plt.ylabel("Sensor")

plt.title(
    "Top 20 Sensor Feature Importance"
)

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_FIGURE,
    dpi=150,
)

plt.close()

print(
    f"\nFeature Importance 그래프 저장: "
    f"{FEATURE_IMPORTANCE_FIGURE}"
)


# ============================================================
# 10. 센서 수와 AP 관계 분석
# ============================================================
#
# Feature Importance 순서대로 센서를 정렬한 뒤
# 상위 K개 센서만 사용하여 Logistic Regression을 학습한다.
#
# Train:
#   선택된 센서로 모델 학습
#
# Validation:
#   AP / Recall 평가
#
# Test:
#   사용하지 않음
# ============================================================

print("\n" + "=" * 70)
print("[5] 센서 수와 Average Precision(AP) 관계 분석")
print("=" * 70)


# ------------------------------------------------------------
# 센서 개수 후보
# ------------------------------------------------------------

sensor_counts = [
    5,
    10,
    20,
    50,
    100,
    200,
    300,
    len(feature_columns),
]

sensor_counts = sorted(
    set(
        count
        for count in sensor_counts
        if count <= len(feature_columns)
    )
)


sensor_results = []


# Feature Importance 순위
ranked_features = (
    feature_importance["feature"]
    .tolist()
)


# ------------------------------------------------------------
# 센서 수별 모델 학습
# ------------------------------------------------------------

for sensor_count in sensor_counts:

    print(
        f"\n센서 {sensor_count}개 모델 학습 중..."
    )

    selected_features = ranked_features[
        :sensor_count
    ]


    X_train_selected = X_train.loc[
        :,
        selected_features,
    ]

    X_valid_selected = X_valid.loc[
        :,
        selected_features,
    ]


    sensor_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=(
                        config.experiment.cv.random_state
                    ),
                ),
            ),
        ]
    )


    # Train에만 fit
    sensor_pipeline.fit(
        X_train_selected,
        y_train,
    )


    # Validation 확률 예측
    valid_scores = sensor_pipeline.predict_proba(
        X_valid_selected
    )[:, 1]


    # 공통 평가 함수 사용
    metrics = evaluate_binary_scores(
        y_valid,
        valid_scores,
        positive_label=dataset.positive_label,
        negative_label=dataset.negative_label,
        threshold=config.experiment.default_threshold,
    )


    ap = metrics.average_precision
    recall = metrics.recall


    sensor_results.append(
        {
            "sensor_count": sensor_count,
            "average_precision": ap,
            "recall": recall,
        }
    )


    print(
        f"센서 {sensor_count:>4}개"
        f" | AP = {ap:.4f}"
        f" | Recall = {recall:.4f}"
    )


# ============================================================
# 11. 센서 수별 결과 저장
# ============================================================

sensor_ap_df = pd.DataFrame(
    sensor_results
)


sensor_ap_df.to_csv(
    SENSOR_AP_LOG,
    index=False,
    encoding="utf-8-sig",
)


print(
    f"\n센서 수별 결과 저장: "
    f"{SENSOR_AP_LOG}"
)


# ============================================================
# 12. 센서 수와 AP 그래프
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    sensor_ap_df["sensor_count"],
    sensor_ap_df["average_precision"],
    marker="o",
)

plt.xlabel(
    "Number of Sensors"
)

plt.ylabel(
    "Average Precision (AP)"
)

plt.title(
    "Sensor Count vs Average Precision"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    SENSOR_AP_FIGURE,
    dpi=150,
)

plt.close()


print(
    f"센서 수 vs AP 그래프 저장: "
    f"{SENSOR_AP_FIGURE}"
)


# ============================================================
# 13. 가장 좋은 센서 수 확인
# ============================================================

best_row = sensor_ap_df.loc[
    sensor_ap_df["average_precision"].idxmax()
]


print("\n" + "=" * 70)
print("최종 분석 결과")
print("=" * 70)


print(
    f"가장 높은 Validation AP: "
    f"{best_row['average_precision']:.4f}"
)


print(
    f"해당 센서 수: "
    f"{int(best_row['sensor_count'])}"
)


print(
    f"해당 Recall: "
    f"{best_row['recall']:.4f}"
)


# ============================================================
# 14. 생성 파일 목록
# ============================================================

print("\n" + "=" * 70)
print("생성된 결과 파일")
print("=" * 70)


print(
    f"PCA 그래프:"
    f"\n  {PCA_FIGURE}"
)


print(
    f"\nFeature Importance 그래프:"
    f"\n  {FEATURE_IMPORTANCE_FIGURE}"
)


print(
    f"\nSensor Count vs AP 그래프:"
    f"\n  {SENSOR_AP_FIGURE}"
)


print(
    f"\nSensor/AP 결과 CSV:"
    f"\n  {SENSOR_AP_LOG}"
)


print("\n" + "=" * 70)
print("STEP 5 ANALYSIS 완료")
print("=" * 70)
