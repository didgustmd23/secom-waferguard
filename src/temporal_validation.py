# ==========================================
# Train 내부 확장형 시간 검증 및 OOF threshold 전이 비교
# - 동일 timestamp를 하나의 단위로 나눠 과거 학습·이후 평가를 분리
# - 기본 threshold, 학습 내부 계층 OOF, 시간순 OOF를 같은 모델로 비교
# - 시간순 OOF 초기 미예측 구간은 제외하고 예측·제거·기간 기록 저장
# - 입력은 Train만 받으며 외부 Validation과 Test는 사용하지 않음
# ==========================================

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import lightgbm
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.threshold_policy import select_policy_threshold
    from src.modeling_preprocessing import quality_filter_record, quality_filter_json
    from src.step4_baseline import split_frame_to_xy
    from src.step5_model_compare import build_candidate_pipelines
    from src.split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID, validate_train_role, validate_split_pair, training_folds
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from threshold_policy import select_policy_threshold
    from modeling_preprocessing import quality_filter_record, quality_filter_json
    from step4_baseline import split_frame_to_xy
    from step5_model_compare import build_candidate_pipelines
    from split_contract import SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID, validate_train_role, validate_split_pair, training_folds


DEFAULT_MODELS = ("lightgbm", "lightgbm_scale_pos_weight", "random_forest",
                  "logistic_regression_l1_balanced")


# ==========================================
# timestamp 단위의 확장형 학습·평가 인덱스 생성
# - 같은 시각과 선언된 그룹·복합 ID가 경계를 넘으면 허용하지 않음
# - 입력 행 순서와 관계없이 시간 범위가 겹치지 않는지 검증
# ==========================================
def temporal_folds(frame, dataset, n_splits):
    if dataset.timestamp_column is None:
        raise ValueError("시간 검증에는 Dataset Profile의 timestamp 컬럼이 필요합니다.")
    times = pd.to_datetime(frame[dataset.timestamp_column],
                           format=dataset.timestamp_format, errors="coerce")
    if times.isna().any():
        raise ValueError("시간 검증 입력에 timestamp 파싱 실패 또는 결측값이 있습니다.")
    unique_times = np.sort(times.unique())
    if n_splits < 2 or len(unique_times) <= n_splits:
        raise ValueError("시간 fold 수는 2 이상이며 고유 timestamp 수보다 작아야 합니다.")
    folds = []
    for fit_times, eval_times in TimeSeriesSplit(n_splits=n_splits).split(unique_times):
        fit_i = np.flatnonzero(times.isin(unique_times[fit_times]).to_numpy())
        eval_i = np.flatnonzero(times.isin(unique_times[eval_times]).to_numpy())
        train = frame.iloc[fit_i]
        evaluation = frame.iloc[eval_i].copy()
        # 원본 Train의 부분집합을 내부 평가로 사용하므로 복사본의 역할만 변경한다.
        if SPLIT_ROLE in evaluation:
            evaluation[SPLIT_ROLE] = "validation"
        validate_split_pair(train, evaluation, dataset, temporal=True)
        if dataset.id_columns:
            ids = list(dataset.id_columns)
            if train[ids].isna().any().any() or evaluation[ids].isna().any().any():
                raise ValueError("시간 검증의 ID 컬럼에 결측값이 있습니다.")
            if set(map(tuple, train[ids].to_numpy())) & set(map(tuple, evaluation[ids].to_numpy())):
                raise ValueError("시간 검증 학습·평가 구간에 동일한 복합 ID가 포함됩니다.")
        folds.append((fit_i, eval_i))
    return folds


def _fit_model(pipeline, features, labels, config, model_name, n_jobs):
    # 매번 복제해 이전 fold의 전처리 상태가 다음 학습에 섞이지 않도록 한다.
    model = clone(pipeline)
    if model_name.startswith(("lightgbm", "random_forest")):
        model.set_params(model__n_jobs=n_jobs)
    if labels.nunique() != 2:
        raise ValueError("시간 검증 학습 구간에는 정상·Fail label이 모두 필요합니다.")
    if model_name == "lightgbm_scale_pos_weight":
        positive = int(labels.eq(config.dataset.positive_label).sum())
        negative = int(labels.eq(config.dataset.negative_label).sum())
        model.set_params(model__scale_pos_weight=negative / positive)
    model.fit(features, labels)
    return model


def _scores(model, features, dataset):
    # class 열 순서를 가정하지 않고 Profile의 Fail 확률을 선택한다.
    index = list(model.named_steps["model"].classes_).index(dataset.positive_label)
    return model.predict_proba(features)[:, index]


def _period(frame, dataset):
    times = pd.to_datetime(frame[dataset.timestamp_column], format=dataset.timestamp_format)
    return str(times.min()), str(times.max())


def _quality_record(model, train, evaluation, config, outer_fold, model_name, mode, inner_fold=None):
    # 내부 OOF를 포함한 모든 학습의 기간·표본 수·센서 제거 사유를 기록한다.
    record = quality_filter_record(model, fold=inner_fold)
    record.update(outer_fold=outer_fold, model_name=model_name, mode=mode,
                  train_samples=len(train), evaluation_samples=len(evaluation),
                  train_fail=int(train[config.dataset.label_column].eq(config.dataset.positive_label).sum()),
                  evaluation_fail=int(evaluation[config.dataset.label_column].eq(config.dataset.positive_label).sum()),
                  model_parameters=quality_filter_json(model.named_steps["model"].get_params(deep=False)),
                  train_start=_period(train, config.dataset)[0],
                  train_end=_period(train, config.dataset)[1],
                  evaluation_start=_period(evaluation, config.dataset)[0],
                  evaluation_end=_period(evaluation, config.dataset)[1])
    return record


# ==========================================
# 과거 학습 구간 내부에서만 OOF 생성
# - 시간순 방식의 초기 미예측 행에는 값을 만들지 않음
# - 반복 계층 CV는 1회로 제한하여 샘플별 단일 OOF 유지
# ==========================================
def _inner_oof(train, config, pipeline, model_name, mode, outer_fold, inner_splits, n_jobs,
               fit_observer=None):
    features, labels, _ = split_frame_to_xy(train, config.dataset)
    if mode == "temporal_oof":
        folds = temporal_folds(train, config.dataset, inner_splits)
    else:
        single = replace(config, experiment=replace(config.experiment,
                         cv=replace(config.experiment.cv, n_repeats=1)))
        folds = training_folds(features, labels, train, single)
    predictions, records = [], []
    for inner_fold, (fit_i, eval_i) in enumerate(folds, 1):
        model = _fit_model(pipeline, features.iloc[fit_i], labels.iloc[fit_i],
                           config, model_name, n_jobs)
        scores = _scores(model, features.iloc[eval_i], config.dataset)
        evaluation = train.iloc[eval_i]
        predictions.append(pd.DataFrame({
            "outer_fold": outer_fold, "model_name": model_name, "mode": mode,
            "inner_fold": inner_fold, "source_row_index": evaluation.index,
            "source_row_id": evaluation[SOURCE_ROW_ID].to_numpy() if SOURCE_ROW_ID in evaluation else None,
            "label": labels.iloc[eval_i].to_numpy(), "positive_score": scores,
        }))
        records.append(_quality_record(model, train.iloc[fit_i], evaluation, config,
                                       outer_fold, model_name, mode, inner_fold))
        # 선택 센서 기록 등 부가 작업은 fit된 모델을 전달받아 수행한다.
        if fit_observer is not None:
            fit_observer(model, train.iloc[fit_i], evaluation, outer_fold, model_name, mode, inner_fold)
    oof = pd.concat(predictions, ignore_index=True)
    if oof.source_row_index.duplicated().any():
        raise RuntimeError("학습 내부 OOF에서 동일한 행이 중복 예측됐습니다.")
    return oof, records


def _threshold_table(oof, config, metadata):
    # 동일한 101개 분위수와 기본값 후보를 사용해 OOF F1만으로 선택한다.
    candidates = np.unique(np.append(np.quantile(oof.positive_score, np.linspace(0, 1, 101)),
                                     config.experiment.default_threshold))
    rows = []
    for threshold in candidates:
        metrics = evaluate_binary_scores(oof.label, oof.positive_score,
                    positive_label=config.dataset.positive_label,
                    negative_label=config.dataset.negative_label, threshold=float(threshold))
        rows.append({**metadata, **asdict(metrics)})
    return pd.DataFrame(rows)


# ==========================================
# 세 시간 구간에서 기본 threshold와 OOF 후보의 전이 비교
# - 동일한 외부 fit과 예측 점수를 여러 threshold에 적용
# - AP는 구간별 평균으로, 오류 건수는 구간 합계로 별도 요약
# ==========================================
def compare_temporal(frame, config, *, outer_splits=3, inner_splits=2,
                     model_names=DEFAULT_MODELS, n_jobs=1, pipelines=None,
                     oof_modes=("stratified_oof", "temporal_oof"), fit_observer=None):
    validate_train_role(frame)
    split_frame_to_xy(frame, config.dataset)
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("시간 검증 모델 목록은 중복 없이 하나 이상 지정해야 합니다.")
    # 기존 모델 비교와 특징 선택 실험이 같은 시간 분할·평가 처리를 공유한다.
    pipelines = build_candidate_pipelines(config) if pipelines is None else pipelines
    if not oof_modes or len(set(oof_modes)) != len(oof_modes) or not set(oof_modes).issubset({"stratified_oof", "temporal_oof"}):
        raise ValueError("내부 OOF 방식은 중복 없이 stratified_oof 또는 temporal_oof를 지정하세요.")
    unknown = set(model_names) - set(pipelines)
    if unknown:
        raise ValueError(f"지원하지 않는 시간 검증 모델입니다: {sorted(unknown)}")
    if config.dataset.timestamp_column is None:
        raise ValueError("시간 검증에는 timestamp 컬럼이 필요합니다.")
    # 과거 실험과 같은 시간순 행 순서를 사용하고 원본 CSV 행 위치는 index에 유지한다.
    times = pd.to_datetime(frame[config.dataset.timestamp_column],
                           format=config.dataset.timestamp_format, errors="coerce")
    frame = frame.iloc[np.argsort(times.to_numpy(), kind="stable")].copy()
    results, predictions, oofs, thresholds, quality, periods, coverage = [], [], [], [], [], [], []
    policy_results = []
    for outer_fold, (fit_i, eval_i) in enumerate(temporal_folds(frame, config.dataset, outer_splits), 1):
        train, evaluation = frame.iloc[fit_i], frame.iloc[eval_i]
        train_x, train_y, columns = split_frame_to_xy(train, config.dataset)
        eval_x, eval_y, _ = split_frame_to_xy(evaluation, config.dataset, expected_feature_columns=columns)
        periods.append({"outer_fold": outer_fold, "train_samples": len(train),
                        "train_fail": int(train_y.eq(config.dataset.positive_label).sum()),
                        "evaluation_samples": len(evaluation),
                        "evaluation_fail": int(eval_y.eq(config.dataset.positive_label).sum()),
                        "train_start": _period(train, config.dataset)[0],
                        "train_end": _period(train, config.dataset)[1],
                        "evaluation_start": _period(evaluation, config.dataset)[0],
                        "evaluation_end": _period(evaluation, config.dataset)[1]})
        for model_name in model_names:
            started = perf_counter()
            model = _fit_model(pipelines[model_name], train_x, train_y, config, model_name, n_jobs)
            fit_seconds = perf_counter() - started
            scores = _scores(model, eval_x, config.dataset)
            quality.append(_quality_record(model, train, evaluation, config,
                                           outer_fold, model_name, "outer_fit"))
            if fit_observer is not None:
                fit_observer(model, train, evaluation, outer_fold, model_name, "outer_fit", None)
            selected = [("default", config.experiment.default_threshold)]
            for mode in oof_modes:
                oof, logs = _inner_oof(train, config, pipelines[model_name], model_name,
                                       mode, outer_fold, inner_splits, n_jobs, fit_observer)
                quality.extend(logs)
                oofs.append(oof)
                metadata = {"outer_fold": outer_fold, "model_name": model_name, "mode": mode}
                table = _threshold_table(oof, config, metadata)
                thresholds.append(table)
                selected.append((mode, float(table.loc[table.f1.idxmax(), "threshold"])))
                # 기존 F1 진단은 보존하고 정책 선택·미충족 결과를 별도 표에 기록한다.
                policy_threshold = select_policy_threshold(table, config.threshold_policy)
                policy_row = {**metadata, **asdict(config.threshold_policy),
                              "policy_status": "feasible" if policy_threshold is not None else "infeasible",
                              "threshold": policy_threshold}
                if policy_threshold is not None:
                    # 미래 지표는 선택이 끝난 뒤에만 계산하며 선택 조건으로 사용하지 않는다.
                    policy_metrics = evaluate_binary_scores(eval_y, scores,
                        positive_label=config.dataset.positive_label,
                        negative_label=config.dataset.negative_label, threshold=policy_threshold)
                    policy_row.update(asdict(policy_metrics))
                    policy_row["reinspection_ratio"] = (
                        policy_metrics.true_positive + policy_metrics.false_positive) / policy_metrics.support
                    policy_row["evaluation_meets_policy"] = (
                        policy_metrics.recall >= config.threshold_policy.min_recall and
                        policy_row["reinspection_ratio"] <= config.threshold_policy.max_reinspection_ratio)
                policy_results.append(policy_row)
                coverage.append({**metadata, "train_samples": len(train),
                                 "oof_samples": len(oof), "excluded_initial_samples": len(train) - len(oof),
                                 "oof_fail": int(oof.label.eq(config.dataset.positive_label).sum())})
            for mode, threshold in selected:
                metrics = evaluate_binary_scores(eval_y, scores,
                            positive_label=config.dataset.positive_label,
                            negative_label=config.dataset.negative_label, threshold=threshold)
                results.append({"dataset_id": config.dataset.dataset_id, "outer_fold": outer_fold,
                                "model_name": model_name, "mode": mode,
                                "fit_seconds": fit_seconds, **asdict(metrics)})
                predictions.append(pd.DataFrame({
                    "outer_fold": outer_fold, "model_name": model_name, "mode": mode,
                    "source_row_index": evaluation.index,
                    "source_row_id": evaluation[SOURCE_ROW_ID].to_numpy() if SOURCE_ROW_ID in evaluation else None,
                    "label": eval_y.to_numpy(), "positive_score": scores, "threshold": threshold,
                    "predicted_fail": scores >= threshold,
                }))
    result_frame = pd.DataFrame(results)
    summaries = []
    for (name, mode), rows in result_frame.groupby(["model_name", "mode"], sort=False):
        counts = rows[["true_positive", "false_positive", "false_negative", "true_negative"]].sum()
        tp, fp, fn, tn = (int(counts[key]) for key in counts.index)
        summaries.append({"model_name": name, "mode": mode,
                          "ap_mean": rows.average_precision.mean(), "ap_std": rows.average_precision.std(ddof=0),
                          "roc_auc_mean": rows.roc_auc.mean(), "recall_mean": rows.recall.mean(),
                          "pooled_recall": tp / (tp + fn),
                          "pooled_precision": tp / (tp + fp) if tp + fp else 0.0,
                          "alarm_ratio": (tp + fp) / (tp + fp + fn + tn), **counts.to_dict()})
    return {"folds": pd.DataFrame(periods), "fold_results": result_frame,
            "summary": pd.DataFrame(summaries), "predictions": pd.concat(predictions, ignore_index=True),
            "oof_predictions": pd.concat(oofs, ignore_index=True),
            "oof_coverage": pd.DataFrame(coverage),
            "threshold_compare": pd.concat(thresholds, ignore_index=True),
            "policy_selection": pd.DataFrame(policy_results),
            "quality_filter": pd.DataFrame(quality)}


def main():
    parser = argparse.ArgumentParser(description="Train 내부 시간순 OOF·threshold 전이 검증")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 기존 결과를 보존할 새 폴더를 지정하세요.")
    config = load_modeling_config(args.config)
    train_frame = pd.read_csv(args.train)
    results = compare_temporal(train_frame, config, outer_splits=args.outer_splits,
                               inner_splits=args.inner_splits, model_names=tuple(args.models), n_jobs=args.n_jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        if name == "quality_filter":
            for column in ("high_missing_features", "constant_features"):
                table[column] = table[column].map(quality_filter_json)
        table.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    record = {"config": asdict(config), "train": str(args.train.resolve()),
              "outer_splits": args.outer_splits, "inner_temporal_splits": args.inner_splits,
              "inner_stratified_splits": config.experiment.cv.n_splits,
              "models": args.models, "n_jobs": args.n_jobs,
              "python": sys.version.split()[0], "sklearn": sklearn.__version__,
              "lightgbm": lightgbm.__version__,
              "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train_frame else None}
    (args.output_dir / "temporal_run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(results["summary"].to_string(index=False))
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()
