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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.step4_baseline import split_frame_to_xy
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from step4_baseline import split_frame_to_xy


# ==========================================
# 후보별 Pipeline 생성
# - 선형 모델과 SVM에는 scaling을 적용하고 tree 모델에는 적용하지 않음
# - 모든 Pipeline은 CV fold의 train index에서만 fit됨
# ==========================================
def build_candidate_pipelines(config: ModelingConfig) -> dict[str, Pipeline]:
    seed = config.experiment.cv.random_state
    return {
        "logistic_regression_l1": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=seed)),
        ]),
        "logistic_regression_l1_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)),
        ]),
        "rbf_svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
        ]),
        "lightgbm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(n_estimators=300, random_state=seed, n_jobs=1, verbosity=-1)),
        ]),
        "lightgbm_scale_pos_weight": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(n_estimators=300, random_state=seed, n_jobs=1, verbosity=-1)),
        ]),
    }


# ==========================================
# Train split 하나에서 후보 모델의 반복 계층 CV 실행
# - 후보 비교 threshold는 config.json의 0.50 고정값만 사용
# - Test 및 외부 Validation의 정보는 모델 선택 과정에 사용하지 않음
# ==========================================
def compare_candidates(train_frame: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    train_x, train_y, _ = split_frame_to_xy(train_frame, config.dataset)
    splitter = RepeatedStratifiedKFold(
        n_splits=config.experiment.cv.n_splits,
        n_repeats=config.experiment.cv.n_repeats,
        random_state=config.experiment.cv.random_state,
    )
    records: list[dict[str, object]] = []

    for model_name, pipeline in build_candidate_pipelines(config).items():
        base_model_name = model_name.removesuffix("_balanced").removesuffix("_scale_pos_weight")
        if base_model_name not in config.candidate_models:
            continue
        fold_metrics: list[dict[str, float]] = []
        fit_times: list[float] = []
        for fold_number, (fit_index, valid_index) in enumerate(splitter.split(train_x, train_y), start=1):
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
            "split_strategy": "random_cv",
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
    return pd.DataFrame(records).sort_values("average_precision_mean", ascending=False).reset_index(drop=True)


def main() -> None:
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
