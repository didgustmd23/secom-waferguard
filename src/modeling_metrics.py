# ==========================================
# 불량(Fail) 클래스 중심 공통 모델 평가 함수
# - 원본 Label(-1, 1 등)을 Fail 기준 이진 Label로 변환
# - Recall, Average Precision, Precision, F1, ROC-AUC 계산
# - 지정 Threshold 기준의 Confusion Matrix(TN, FP, FN, TP) 계산
# - 후보 모델 비교에는 Threshold 0.50을 사용
# - 최종 운영 Threshold는 Day 4 OOF Prediction으로 별도 결정
# - 이상 탐지의 원본 점수는 evaluate_anomaly_scores로 별도 평가
#
# positive_scores:
# - Fail Label에 대한 0~1 범위의 예측 확률을 전달
# - Decision Function처럼 범위가 정해지지 않은 점수는 사용하지 않음
# ==========================================

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ==========================================
# 이진 분류 평가 결과 객체
# - positive_label을 Fail 클래스로 간주한 평가 지표 저장
# - CSV, JSON 기록에 사용할 수 있도록 to_dict() 제공
# - Confusion Matrix 항목을 함께 기록해 FN/FP 분석에 재사용
# ==========================================
@dataclass(frozen=True)
class BinaryMetrics:

    threshold: float
    support: int
    positive_support: int
    recall: float
    precision: float
    f1: float
    average_precision: float
    roc_auc: float | None
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, float | int | None]:
        # Dataclass의 평가 결과를 CSV/JSON 저장용 Dictionary로 변환
        return asdict(self)


# ==========================================
# 예측 확률로 이진 분류 성능 평가
# - y_true를 positive_label 기준의 0/1 Label로 변환
# - positive_scores와 Threshold로 예측 Class 생성
# - Fail Recall, AP, Precision, F1, ROC-AUC 계산
# - TN, FP, FN, TP를 함께 반환해 오류 분석에 활용
#
# positive_label:
# - 원본 Label 중 불량(Fail)을 의미하는 값
# - Dataset Profile에서 읽어 호출부가 명시적으로 전달
#
# negative_label:
# - 원본 Label 중 정상(Pass)을 의미하는 값
# - Dataset Profile에서 읽어 호출부가 명시적으로 전달
#
# threshold:
# - positive_scores가 Fail로 분류되는 기준값
# - 호출부가 config.json의 후보 비교 기준값 또는 Day 4의 확정값을 명시적으로 전달
# ==========================================
def evaluate_binary_scores(
    y_true: Iterable[object],
    positive_scores: Iterable[float],
    *,
    positive_label: object,
    negative_label: object,
    threshold: float,
) -> BinaryMetrics:
    # 확률용 공개 함수의 0~1 입력 계약을 유지한다.
    return _evaluate_scores(y_true, positive_scores, positive_label=positive_label,
                            negative_label=negative_label, threshold=threshold, probability=True)


# ==========================================
# 확률이 아닌 이상 점수로 동일한 이진 분류 지표 계산
# - 점수·threshold는 유한한 실수이며 클수록 이상이어야 함
# - 확률로 변환하거나 평가 데이터에서 정규화하지 않음
# - Isolation Forest의 방향 반전·PCA 복원오차 계산은 호출자가 담당
# - 양성 판정은 점수 >= threshold이고 AP·ROC-AUC는 원본 점수의 순위로 계산
# - 평가 결과 구조는 확률 모델과 동일해 오류 건수를 같은 형식으로 비교 가능
# ==========================================
def evaluate_anomaly_scores(y_true, anomaly_scores, *, positive_label, negative_label,
                            threshold) -> BinaryMetrics:
    # 방어 검증과 지표 계산은 공유하고 확률 범위 제한만 적용하지 않는다.
    return _evaluate_scores(y_true, anomaly_scores, positive_label=positive_label,
                            negative_label=negative_label, threshold=threshold, probability=False)


# ==========================================
# 확률·이상 점수 공통 입력 검증과 Fail 중심 지표 계산
# - positive_scores는 내부 공통 이름이며 이상 점수도 전달받을 수 있음
# - probability=True에서만 점수·threshold의 0~1 범위를 검증
# - 결측 label·허용되지 않은 label·비유한 점수는 두 방식 모두 차단
# - 모델 학습이나 문턱 탐색을 하지 않고 전달받은 문턱을 평가만 함
# ==========================================
def _evaluate_scores(y_true, positive_scores, *, positive_label, negative_label,
                     threshold, probability):
    # 확률만 0~1을 강제하고 이상 점수의 문턱에는 유한한 실수를 허용한다.
    if not np.isfinite(threshold):
        raise ValueError("threshold는 유한한 값이어야 합니다")
    # 확률 모델의 threshold는 0~1, 이상 점수의 threshold는 범위를 제한하지 않는다.
    if probability and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold는 0.0 이상 1.0 이하여야 합니다")

    # 이진 분류의 정상·불량 Label은 서로 다른 값이어야 함
    if positive_label == negative_label:
        raise ValueError("positive_label과 negative_label은 서로 달라야 합니다")

    # Generator, List, Series 등 다양한 입력을 1차원 NumPy Array로 통일
    labels = np.asarray(list(y_true))
    scores = np.asarray(list(positive_scores), dtype=float)

    # Label과 예측 확률 또는 이상 점수는 샘플별로 1:1 대응해야 한다.
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true와 positive_scores는 1차원이어야 합니다")
    if labels.size == 0:
        raise ValueError("y_true와 positive_scores는 비어 있으면 안 됩니다")
    if labels.size != scores.size:
        raise ValueError("y_true와 positive_scores의 길이는 같아야 합니다")

    # 두 종류 모두 NaN/Infinity를 차단하고 확률일 때만 0~1 범위를 추가 검사한다.
    if not np.isfinite(scores).all():
        raise ValueError("positive_scores에는 유한한 값만 포함되어야 합니다")
    if probability and ((scores < 0.0).any() or (scores > 1.0).any()):
        raise ValueError("positive_scores는 0.0 이상 1.0 이하의 확률이어야 합니다")

    # None, NaN, pd.NA는 정상 Label로 묵시적으로 처리하지 않고 데이터 오류로 차단
    missing_labels = np.asarray(pd.isna(labels), dtype=bool)
    if missing_labels.any():
        raise ValueError("y_true에 결측 label이 있으면 안 됩니다")

    # Dataset Profile에 정의된 정상·불량 Label 외 값은 성능을 왜곡하므로 즉시 차단
    allowed_labels = (negative_label, positive_label)
    invalid_labels = [
        label for label in labels if not any(label == allowed_label for allowed_label in allowed_labels)
    ]
    if invalid_labels:
        raise ValueError(
            "y_true에 negative_label과 positive_label 이외의 label이 있습니다: "
            f"{invalid_labels}"
        )

    # 검증된 원본 Label에서 Fail Label만 1로 변환해 지표 기준을 통일
    y_binary = (labels == positive_label).astype(int)
    if y_binary.sum() == 0:
        raise ValueError("y_true에 positive_label이 없습니다")

    # 예측 확률 또는 이상 점수가 Threshold 이상이면 Fail로 분류한다.
    predictions = (scores >= threshold).astype(int)

    # labels=(0, 1)을 명시해 단일 예측 Class가 나와도 TN/FP/FN/TP 순서를 고정
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_binary, predictions, labels=(0, 1)
    ).ravel()

    # ROC-AUC는 정답 Label이 두 Class 모두 존재할 때만 계산 가능
    roc_auc: float | None = None
    if np.unique(y_binary).size == 2:
        roc_auc = float(roc_auc_score(y_binary, scores))

    # Fail 중심 지표와 오류 개수를 하나의 불변 결과 객체로 반환
    return BinaryMetrics(
        threshold=threshold,
        support=int(labels.size),
        positive_support=int(y_binary.sum()),
        recall=float(recall_score(y_binary, predictions, zero_division=0)),
        precision=float(precision_score(y_binary, predictions, zero_division=0)),
        f1=float(f1_score(y_binary, predictions, zero_division=0)),
        average_precision=float(average_precision_score(y_binary, scores)),
        roc_auc=roc_auc,
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
    )
