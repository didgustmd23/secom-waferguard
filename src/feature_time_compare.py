# ==========================================
# 전체·Top-50·Top-20의 시간순 성능과 센서 선택 안정성 비교
# - LightGBM 분류기는 동일하게 유지하고 선택 중요도 모델만 비교
# - LightGBM 중요도와 원본 과제의 RF 중요도 선택을 모두 실험
# - 선택기는 각 과거 학습 fold의 Pipeline 안에서만 fit
# - 최종 센서가 아닌 후보 목록을 전체 Time Train에서 별도로 준비
# - 외부 Validation·최종 Test를 읽지 않음
#
# 처리 순서:
# - 1. 전체·두 종류의 중요도 Top-50/20 Pipeline을 생성
# - 2. 기존 시간 검증 코어로 다음 구간 성능과 내부 OOF 정책 평가
# - 3. fit 직후 선택 센서를 기록하고 외부·내부 fit 빈도를 따로 계산
# - 4. 전체 Time Train에서 연구용 후보 센서 목록만 준비
# - 5. 성능·센서·설정 기록을 저장하며 최종 후보는 자동 확정하지 않음
# ==========================================

import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from src.step4_baseline import split_frame_to_xy
    from src.step6_feature_compare import _checked_top_k
    from src.temporal_validation import compare_temporal
    from src.split_contract import PROTOCOL_ID
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from step4_baseline import split_frame_to_xy
    from step6_feature_compare import _checked_top_k
    from temporal_validation import compare_temporal
    from split_contract import PROTOCOL_ID


# ==========================================
# 같은 분류 모델에서 센서 수와 선택 방법만 비교하는 Pipeline 생성
# - RF selector + LightGBM 분류기로 선택 방법의 차이를 분리
# - 불균형 가중치와 scaling 등 다른 조건을 동시에 바꾸지 않음
# ==========================================
def build_feature_pipelines(config, *, n_jobs=1, n_estimators=300):
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("트리 수는 양의 정수여야 합니다.")
    seed = config.experiment.cv.random_state
    # 모든 최종 분류기의 설정이 같아야 센서 축소에 따른 차이를 해석할 수 있다.
    classifier = LGBMClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs, verbosity=-1)
    steps = preprocessing_steps(config.dataset, pandas_output=True)
    pipelines = {"lightgbm_all": Pipeline(steps + [("model", clone(classifier))])}
    for method in ("lightgbm", "rf_importance"):
        for count in (50, 20):
            # 매 Pipeline마다 독립된 전처리기·선택기·분류기를 사용한다.
            importance = clone(classifier) if method == "lightgbm" else RandomForestClassifier(
                n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
            selector = SelectFromModel(importance, threshold=-np.inf,
                max_features=partial(_checked_top_k, count=count))
            # -inf는 중요도 자체의 최소값 제한을 없애고 max_features로 정확히 K개 선택한다.
            # _checked_top_k는 대치·One-Hot 후 해당 fold의 실제 특징 수를 검사한다.
            name = f"lightgbm_top_{count}" if method == "lightgbm" else f"lightgbm_rf_top_{count}"
            pipelines[name] = Pipeline(preprocessing_steps(config.dataset, pandas_output=True) +
                                        [("selector", selector), ("model", clone(classifier))])
    return pipelines


# ==========================================
# 학습된 Pipeline에서 실제 선택 특징명·중요도·순위 추출
# - One-Hot을 쓴 데이터에서는 원본 센서가 아니라 변환 특징일 수 있음
# - 최종 분류기의 중요도가 아니라 선택에 사용한 모델의 중요도 기록
# - 선택기의 get_support 순서는 모델 입력 순서이며 중요도 순위와 다름
# ==========================================
def selection_rows(pipeline, metadata):
    quality = pipeline.named_steps["quality_filter"]
    if "selector" in pipeline.named_steps:
        # 선택기 전까지의 특징명에 get_support 마스크를 적용해 실제 입력 특징을 찾는다.
        names = pipeline[:-2].get_feature_names_out()
        selector = pipeline.named_steps["selector"]
        mask = selector.get_support()
        selected = np.asarray(names)[mask]
        importance = selector.estimator_.feature_importances_[mask]
        method = "rf_importance" if isinstance(selector.estimator_, RandomForestClassifier) else "lightgbm_importance"
    else:
        # 전체 모델에는 중요도 순위 선택이 없으므로 순위는 입력 순서이고 중요도는 결측이다.
        selected = pipeline[:-1].get_feature_names_out()
        importance = np.full(len(selected), np.nan)
        method = "all_after_quality_filter"
    # 중요도가 같으면 원래 특징 순서를 유지해 같은 결과를 재현한다.
    order = np.argsort(-importance, kind="stable") if method != "all_after_quality_filter" else np.arange(len(selected))
    return [{**metadata, "feature": str(selected[index]), "selection_method": method,
             "selection_rank": rank, "selection_importance": float(importance[index]),
             "feature_space": "raw_sensor" if str(selected[index]) in quality.retained_features_ else "transformed_feature"}
            for rank, index in enumerate(order, 1)]


# ==========================================
# 학습별 선택 목록에서 특징별 선택 빈도 계산
# - 빈도 = 특징을 선택한 fit 수 / 해당 모델·역할의 전체 fit 수
# - outer_fit의 분모는 3, inner_oof의 분모는 6
# - 학습 구간이 겹치므로 빈도는 안정성 참고값이며 독립 검정 결과가 아님
# ==========================================
def feature_frequency(selected):
    # 외부 fit·내부 OOF fit은 분모를 구분하며 겹치는 학습을 독립 실험으로 보지 않는다.
    counts = selected.groupby(["model_name", "fit_role"])["fit_id"].nunique()
    frequency = selected.groupby(["model_name", "fit_role", "feature", "feature_space"], as_index=False).agg(
        selected_fit_count=("fit_id", "nunique"), mean_selection_rank=("selection_rank", "mean"))
    frequency["total_fit_count"] = [counts.loc[(name, role)] for name, role in zip(frequency.model_name, frequency.fit_role)]
    frequency["selection_frequency"] = frequency.selected_fit_count / frequency.total_fit_count
    return frequency.sort_values(["model_name", "fit_role", "selection_frequency", "mean_selection_rank", "feature"],
                                 ascending=[True, True, False, True, True]).reset_index(drop=True)


# ==========================================
# 같은 시간순 코어로 평가하고 각 fit의 선택 센서 기록
# - 기록용 hook은 fit 결과를 조회할 뿐 예측 계산을 바꾸지 않음
# - 외부 fit 3회·내부 OOF fit 6회의 센서 목록과 빈도를 별도로 저장
# ==========================================
def compare_feature_time(frame, config, *, n_jobs=1, n_estimators=300):
    pipelines = build_feature_pipelines(config, n_jobs=n_jobs, n_estimators=n_estimators)
    selected = []
    # 시간 검증 코어에 기록용 hook을 전달해 같은 학습을 중복 실행하지 않는다.
    def record_selection(model, train, evaluation, outer, name, mode, inner):
        # outer:mode:inner 조합으로 fit을 구분해 같은 특징이 중복 집계되지 않게 한다.
        # 평가 데이터는 hook에서 읽지 않아 선택 목록에 미래 정보가 섞이지 않는다.
        selected.extend(selection_rows(model, {"outer_fold": outer, "model_name": name,
            "fit_role": "outer_fit" if inner is None else "inner_oof", "inner_fold": inner,
            "fit_id": f"{outer}:{mode}:{inner}", "train_samples": len(train)}))
    results = compare_temporal(frame, config, model_names=tuple(pipelines), pipelines=pipelines,
                              oof_modes=("temporal_oof",), n_jobs=n_jobs, fit_observer=record_selection)
    results["selected_features"] = pd.DataFrame(selected)
    results["feature_frequency"] = feature_frequency(results["selected_features"])
    # 전체 Train 후보 fit은 평가 fold가 아니다. 이 목록으로 과거 점수를 재계산하지 않는다.
    features, labels, _ = split_frame_to_xy(frame, config.dataset)
    candidates, quality = [], []
    for name, pipeline in pipelines.items():
        # clone으로 CV의 학습 상태를 버린 뒤 전체 Time Train에서 새로 fit한다.
        # 최종 모델·학습 범위를 아직 합의하지 않았으므로 is_final=False로 저장한다.
        model = clone(pipeline).fit(features, labels)
        candidates.extend(selection_rows(model, {"model_name": name, "fit_role": "candidate_full_train",
                           "train_samples": len(frame), "is_final": False}))
        quality.append({"model_name": name, **quality_filter_record(model)})
    results["candidate_features"] = pd.DataFrame(candidates)
    results["candidate_quality_filter"] = pd.DataFrame(quality)
    return results


# ==========================================
# 명시한 Train에서 실험 실행 후 새 폴더에 로그 저장
# - CSV는 노트북에서 읽고 feature_run.json은 재현 조건을 보관
# - 후보 센서 목록을 최종 모델·threshold로 자동 연결하지 않음
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="전체·Top-50·Top-20 시간순 성능 및 센서 선택 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    config = load_modeling_config(args.config)
    train = pd.read_csv(args.train)
    results = compare_feature_time(train, config, n_jobs=args.n_jobs, n_estimators=args.n_estimators)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        # 센서 목록의 JSON 변환은 저장용 복사본에만 적용한다.
        if name in {"quality_filter", "candidate_quality_filter"}:
            table = table.copy()
            for column in ("high_missing_features", "constant_features"):
                table[column] = table[column].map(quality_filter_json)
        table.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    record = {"config": asdict(config), "train_path": str(args.train.resolve()), "n_estimators": args.n_estimators,
              "n_jobs": args.n_jobs, "outer_splits": 3, "inner_temporal_splits": 2,
              "training_protocol_id": train[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train else None,
              "sklearn": sklearn.__version__, "candidate_lists_are_final": False}
    (args.output_dir / "feature_run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(results["summary"].to_string(index=False))
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()
