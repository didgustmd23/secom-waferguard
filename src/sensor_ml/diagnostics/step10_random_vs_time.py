# ============================================================
# STEP 10 - Random Validation vs Time Validation
#
# 목적
# 1. Random Split과 Time Split의 성능 차이 비교
# 2. 동일한 Logistic Regression Baseline 사용
# 3. Test set은 사용하지 않음
# 4. Train 데이터에서만 전처리/스케일링을 학습
# 5. AP, Recall, Precision, F1, ROC-AUC 비교
# ============================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LOG_DIR = PROJECT_ROOT / "logs/secom/exploratory/legacy"
FIGURE_DIR = PROJECT_ROOT / "reports/secom/figures/exploratory"

LOG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

RANDOM_TRAIN_PATH = SPLIT_DIR / "random_train.csv"
RANDOM_VALID_PATH = SPLIT_DIR / "random_valid.csv"

TIME_TRAIN_PATH = SPLIT_DIR / "time_train.csv"
TIME_VALID_PATH = SPLIT_DIR / "time_valid.csv"


random_train = pd.read_csv(RANDOM_TRAIN_PATH)
random_valid = pd.read_csv(RANDOM_VALID_PATH)

time_train = pd.read_csv(TIME_TRAIN_PATH)
time_valid = pd.read_csv(TIME_VALID_PATH)


# ============================================================
# 3. Feature / Target 분리
# ============================================================

TARGET_COL = "label"
TIMESTAMP_COL = "timestamp"


def prepare_xy(df: pd.DataFrame):
    """
    label과 timestamp를 제외하고 센서 Feature만 사용한다.
    """

    y = df[TARGET_COL].astype(int)

    X = df.drop(
        columns=[TARGET_COL, TIMESTAMP_COL],
        errors="ignore",
    )

    return X, y


X_random_train, y_random_train = prepare_xy(random_train)
X_random_valid, y_random_valid = prepare_xy(random_valid)

X_time_train, y_time_train = prepare_xy(time_train)
X_time_valid, y_time_valid = prepare_xy(time_valid)


# ============================================================
# 4. Logistic Regression Pipeline
# ============================================================

def make_model():
    """
    Train 데이터에 대해서만
    결측치 처리 → 표준화 → Logistic Regression을 수행한다.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# 5. 평가 함수
# ============================================================

def evaluate_model(
    model,
    X_valid,
    y_valid,
):
    """
    Validation 데이터에 대한 성능을 계산한다.
    """

    y_prob = model.predict_proba(X_valid)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "AP": average_precision_score(y_valid, y_prob),
        "Recall": recall_score(
            y_valid,
            y_pred,
            zero_division=0,
        ),
        "Precision": precision_score(
            y_valid,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_valid,
            y_pred,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_valid,
            y_prob,
        ),
    }


# ============================================================
# 6. Random Validation
# ============================================================

random_model = make_model()

random_model.fit(
    X_random_train,
    y_random_train,
)

random_result = evaluate_model(
    random_model,
    X_random_valid,
    y_random_valid,
)


# ============================================================
# 7. Time Validation
# ============================================================

time_model = make_model()

time_model.fit(
    X_time_train,
    y_time_train,
)

time_result = evaluate_model(
    time_model,
    X_time_valid,
    y_time_valid,
)


# ============================================================
# 8. 결과 정리
# ============================================================

results = pd.DataFrame(
    [
        {
            "validation": "Random",
            **random_result,
        },
        {
            "validation": "Time",
            **time_result,
        },
    ]
)

print()
print("=" * 70)
print("STEP 10 - RANDOM VS TIME VALIDATION")
print("=" * 70)

print()
print(results.to_string(index=False))


# ============================================================
# 9. Random vs Time 성능 차이 계산
# ============================================================

metric_columns = [
    "AP",
    "Recall",
    "Precision",
    "F1",
    "ROC-AUC",
]

comparison = results.set_index("validation")[metric_columns]

print()
print("=" * 70)
print("RANDOM - TIME DIFFERENCE")
print("=" * 70)

print()
print(
    (
        comparison.loc["Random"]
        - comparison.loc["Time"]
    ).to_string()
)


# ============================================================
# 10. 결과 저장
# ============================================================

output_csv = LOG_DIR / "random_vs_time.csv"

results.to_csv(
    output_csv,
    index=False,
)

print()
print(f"결과 저장: {output_csv}")


# ============================================================
# 11. 시각화
# ============================================================

plot_data = results.set_index("validation")[metric_columns]

ax = plot_data.T.plot(
    kind="bar",
    figsize=(10, 6),
)

ax.set_title(
    "Random Validation vs Time Validation"
)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")

ax.set_ylim(0, 1)

plt.xticks(rotation=0)
plt.legend(title="Validation")

plt.tight_layout()

figure_path = (
    FIGURE_DIR / "random_vs_time.png"
)

plt.savefig(
    figure_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(f"그래프 저장: {figure_path}")

print()
print("=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)
