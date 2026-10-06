# ==========================================
# CV fold 내부 feature 선택 비교
# - Train만 사용하며 외부 Validation/Test는 읽지 않음
# - PCA·L1·Importance 선택은 공통 모델 모듈의 Pipeline 안에서 fit
# - 실험별 성능 평균·편차와 실제 특징 수·센서 제거 기록을 저장
# ==========================================
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.modeling_models import build_experiments, fit_pipeline, positive_scores
    from src.modeling_preprocessing import (
        fitted_feature_count, quality_filter_record, quality_filter_json,
        checked_top_k as _checked_top_k,
    )
    from src.split_contract import PROTOCOL_ID, validate_train_role, training_folds
except ModuleNotFoundError:
    from dataset_schema import split_frame_to_xy
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from modeling_models import build_experiments, fit_pipeline, positive_scores
    from modeling_preprocessing import (
        fitted_feature_count, quality_filter_record, quality_filter_json,
        checked_top_k as _checked_top_k,
    )
    from split_contract import PROTOCOL_ID, validate_train_role, training_folds


# build_experiments와 _checked_top_k는 기존 import 경로와의 호환을 위해 노출한다.
# 신규 호출자는 modeling_models / modeling_preprocessing에서 직접 가져온다.


def compare_features(frame, config, n_jobs: int = 1):
    """같은 Train CV fold에서 특징 선택 방식별 성능과 특징 수를 비교한다."""
    validate_train_role(frame)
    # Profile 검증을 통과한 특징과 label만 학습에 사용한다.
    features, labels, _ = split_frame_to_xy(frame, config.dataset)
    if labels.nunique() < 2:
        raise ValueError("특징 선택 비교를 위해 Train split에는 정상과 Fail label이 모두 있어야 합니다.")
    smallest_class_count = int(labels.value_counts().min())
    if smallest_class_count < config.experiment.cv.n_splits:
        raise ValueError(
            "가장 적은 class의 샘플 수가 CV fold 수보다 작습니다: "
            f"최소 class 샘플={smallest_class_count}, fold={config.experiment.cv.n_splits}"
        )
    # 범주형 입력은 One-Hot 이후 특징 수가 달라지므로 실제 fold 안에서 검사한다.
    too_large_top_k = (
        [count for count in config.top_k_feature_counts if count > features.shape[1]]
        if not config.dataset.categorical_feature_columns else []
    )
    if too_large_top_k:
        raise ValueError(
            "Top-K feature 수가 입력 feature 수보다 큽니다: "
            f"Top-K={too_large_top_k}, feature 수={features.shape[1]}"
        )

    rows = []
    for name, pipeline in build_experiments(config, n_jobs).items():
        fold_metrics, fit_times, feature_counts, quality_records = [], [], [], []
        for fit_indices, validation_indices in training_folds(features, labels, frame, config):
            # 매 fold마다 독립된 Pipeline을 현재 학습 데이터로만 fit한다.
            started_at = perf_counter()
            model = fit_pipeline(
                pipeline, features.iloc[fit_indices], labels.iloc[fit_indices],
                config, name, n_jobs=n_jobs,
            )
            fit_times.append(perf_counter() - started_at)
            feature_counts.append(fitted_feature_count(model))
            quality_records.append(quality_filter_record(model, fold=len(quality_records) + 1))

            # 설정된 양성 label의 확률을 공통 평가 함수에 전달한다.
            scores = positive_scores(model, features.iloc[validation_indices], config.dataset)
            fold_metrics.append(evaluate_binary_scores(
                labels.iloc[validation_indices], scores,
                positive_label=config.dataset.positive_label,
                negative_label=config.dataset.negative_label,
                threshold=config.experiment.default_threshold,
            ))

        protocol_id = frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in frame else None
        row = {
            "dataset_id": config.dataset.dataset_id,
            "split_strategy": "time_train_cv" if str(protocol_id).startswith("time:") else "random_cv",
            "experiment": name,
            "n_splits": config.experiment.cv.n_splits,
            "n_repeats": config.experiment.cv.n_repeats,
            "threshold": config.experiment.default_threshold,
            "selected_feature_count_mean": float(np.mean(feature_counts)),
            "fit_time_mean_seconds": float(np.mean(fit_times)),
            "training_protocol_id": protocol_id,
            "quality_filter_log": quality_filter_json(quality_records),
        }
        # ROC-AUC가 정의되지 않는 fold의 None은 기존과 같이 집계에서 제외한다.
        for key in ("average_precision", "recall", "precision", "f1", "roc_auc"):
            values = [getattr(metric, key) for metric in fold_metrics if getattr(metric, key) is not None]
            row[f"{key}_mean"] = float(np.mean(values)) if values else None
            row[f"{key}_std"] = float(np.std(values)) if values else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("average_precision_mean", ascending=False)


def main():
    """CLI에서 설정과 Train 파일을 받아 비교표를 CSV로 기록한다."""
    parser = argparse.ArgumentParser(description="CV 내부 feature 선택 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/feature_compare.csv"))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    result = compare_features(
        pd.read_csv(args.train), load_modeling_config(args.config), args.n_jobs,
    )
    # 최신 실행 결과를 저장하지만 Test와 외부 Validation은 입력받지 않는다.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.drop(columns="quality_filter_log").to_string(index=False))


if __name__ == "__main__":
    main()
