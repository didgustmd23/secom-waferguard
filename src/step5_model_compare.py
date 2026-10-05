# ==========================================
# Repeated CV 기반 후보 모델 비교
# - 지정된 Train split만 사용하며 Validation/Test 파일은 읽지 않음
# - 결측 처리와 scaling은 매 CV 학습 fold 내부 Pipeline에서 fit
# - AP와 Fail Recall을 중심으로 후보 모델의 평균·표준편차·학습 시간을 기록
# ==========================================

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.step4_baseline import split_frame_to_xy
    from src.modeling_preprocessing import preprocessing_steps
    from src.split_contract import PROTOCOL_ID, validate_train_role, training_folds
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from step4_baseline import split_frame_to_xy
    from modeling_preprocessing import preprocessing_steps
    from split_contract import PROTOCOL_ID, validate_train_role, training_folds


# ==========================================
# 후보별 Pipeline 생성
# - 선형 모델과 SVM에는 scaling을 적용하고 tree 모델에는 적용하지 않음
# - 모든 Pipeline은 CV fold의 train index에서만 fit됨
# ==========================================
def build_candidate_pipelines(config: ModelingConfig) -> dict[str, Pipeline]:
    # 모든 후보에 같은 난수 시드를 사용해 성능 차이가 모델 조건에서만 나도록 한다.
    seed = config.experiment.cv.random_state
    return {
        # L1은 희소 feature 선택 효과와 선형 기준 성능을 함께 확인한다.
        "logistic_regression_l1": Pipeline([
            *preprocessing_steps(config.dataset, scale=True),
            ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=seed)),
        ]),
        "logistic_regression_l1_balanced": Pipeline([
            *preprocessing_steps(config.dataset, scale=True),
            ("model", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)),
        ]),
        "rbf_svm": Pipeline([
            *preprocessing_steps(config.dataset, scale=True),
            ("model", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            *preprocessing_steps(config.dataset),
            ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
        ]),
        "lightgbm": Pipeline([
            *preprocessing_steps(config.dataset),
            ("model", LGBMClassifier(n_estimators=300, random_state=seed, n_jobs=1, verbosity=-1)),
        ]),
        "lightgbm_scale_pos_weight": Pipeline([
            *preprocessing_steps(config.dataset),
            ("model", LGBMClassifier(n_estimators=300, random_state=seed, n_jobs=1, verbosity=-1)),
        ]),
    }


# ==========================================
# Train split 하나에서 후보 모델의 반복 계층 CV 실행
# - 후보 비교 threshold는 config.json의 0.50 고정값만 사용
# - Test 및 외부 Validation의 정보는 모델 선택 과정에 사용하지 않음
# ==========================================
def compare_candidates(train_frame: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    validate_train_role(train_frame)
    train_x, train_y, _ = split_frame_to_xy(train_frame, config.dataset)
    # 계층 CV와 모든 분류 모델은 정상·Fail label이 모두 있어야 학습할 수 있다.
    if train_y.nunique() < 2:
        raise ValueError("후보 모델 비교를 위해 Train split에는 정상과 Fail label이 모두 있어야 합니다.")
    # 각 fold validation에 두 class가 포함되도록 가장 적은 class 수를 먼저 확인한다.
    smallest_class_count = int(train_y.value_counts().min())
    if smallest_class_count < config.experiment.cv.n_splits:
        raise ValueError(
            "가장 적은 class의 샘플 수가 CV fold 수보다 작습니다: "
            f"최소 class 샘플={smallest_class_count}, fold={config.experiment.cv.n_splits}"
        )
    records: list[dict[str, object]] = []

    for model_name, pipeline in build_candidate_pipelines(config).items():
        base_model_name = model_name.removesuffix("_balanced").removesuffix("_scale_pos_weight")
        if base_model_name not in config.candidate_models:
            continue
        fold_metrics: list[dict[str, float]] = []
        fit_times: list[float] = []
        for fit_index, valid_index in training_folds(train_x, train_y, train_frame, config):
            model = clone(pipeline)
            # LightGBM의 불량 가중치는 현재 CV 학습 fold의 class 수로만 계산한다.
            if model_name == "lightgbm_scale_pos_weight":
                fit_labels = train_y.iloc[fit_index]
                positive_count = int((fit_labels == config.dataset.positive_label).sum())
                negative_count = int((fit_labels == config.dataset.negative_label).sum())
                model.set_params(model__scale_pos_weight=negative_count / positive_count)
            started_at = perf_counter()
            model.fit(train_x.iloc[fit_index], train_y.iloc[fit_index])
            fit_times.append(perf_counter() - started_at)
            positive_index = list(model.named_steps["model"].classes_).index(config.dataset.positive_label)
            scores = model.predict_proba(train_x.iloc[valid_index])[:, positive_index]
            metrics = evaluate_binary_scores(
                train_y.iloc[valid_index], scores,
                positive_label=config.dataset.positive_label,
                negative_label=config.dataset.negative_label,
                threshold=config.experiment.default_threshold,
            )
            fold_metrics.append({key: float(value) for key, value in metrics.to_dict().items()
                                 if key in {"average_precision", "recall", "precision", "f1", "roc_auc"} and value is not None})

        record: dict[str, object] = {
            "dataset_id": config.dataset.dataset_id,
            "split_strategy": "time_train_cv" if PROTOCOL_ID in train_frame and str(train_frame[PROTOCOL_ID].iloc[0]).startswith("time:") else "random_cv",
            "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train_frame else None,
            "train_samples": len(train_frame),
            "model_name": model_name,
            "n_splits": config.experiment.cv.n_splits,
            "n_repeats": config.experiment.cv.n_repeats,
            "threshold": config.experiment.default_threshold,
            "fit_time_mean_seconds": float(np.mean(fit_times)),
            "fit_time_std_seconds": float(np.std(fit_times)),
        }
        for metric_name in ("average_precision", "recall", "precision", "f1", "roc_auc"):
            values = [fold[metric_name] for fold in fold_metrics if metric_name in fold]
            record[f"{metric_name}_mean"] = float(np.mean(values)) if values else None
            record[f"{metric_name}_std"] = float(np.std(values)) if values else None
        records.append(record)
    if not records:
        raise ValueError("models.candidates에 지원하는 후보 모델이 없습니다.")
    return pd.DataFrame(records).sort_values("average_precision_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    # Train 파일만 받아 후보 비교 결과를 재생성 가능한 CSV로 기록한다.
    parser = argparse.ArgumentParser(description="반복 계층 CV 후보 모델 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/model_compare.csv"))
    arguments = parser.parse_args()
    config = load_modeling_config(arguments.config)
    results = compare_candidates(pd.read_csv(arguments.train), config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(arguments.output, index=False, encoding="utf-8-sig")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
