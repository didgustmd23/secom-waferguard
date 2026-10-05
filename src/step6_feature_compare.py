# ==========================================
# CV fold 내부 feature 선택 비교
# - 시간 holdout을 제외한 time_train만 사용하고 Test/Time Validation은 읽지 않음
# - PCA, L1, LightGBM Importance 선택기를 Pipeline 안에서 fit해 누수를 방지
# ==========================================
from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
from functools import partial
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.step4_baseline import split_frame_to_xy
    from src.modeling_preprocessing import preprocessing_steps, fitted_feature_count
    from src.split_contract import PROTOCOL_ID, validate_train_role, training_folds
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from step4_baseline import split_frame_to_xy
    from modeling_preprocessing import preprocessing_steps, fitted_feature_count
    from split_contract import PROTOCOL_ID, validate_train_role, training_folds


def _checked_top_k(features, *, count):
    # One-Hot Encoding 이후의 실제 fold 입력 크기로 Top-K를 검사한다.
    if count > features.shape[1]:
        raise ValueError(f"Top-K feature 수가 변환 후 feature 수보다 큽니다: {count} > {features.shape[1]}")
    return count

def build_experiments(config, n_jobs: int = 1):
    # selector가 변환 후 feature 이름을 유지하도록 공통 전처리를 재사용한다.
    seed=config.experiment.cv.random_state
    l1=LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced",max_iter=2000,random_state=seed)
    light=LGBMClassifier(n_estimators=300,random_state=seed,n_jobs=n_jobs,verbosity=-1)
    # 각 selector는 이후 CV 학습 fold에서만 fit되는 Pipeline 내부 단계로 둔다.
    result={
      "l1_balanced_all":Pipeline(preprocessing_steps(config.dataset, scale=True, pandas_output=True)+[("model",l1)]),
      "l1_balanced_pca90":Pipeline(preprocessing_steps(config.dataset, scale=True, pandas_output=True)+[("selector",PCA(n_components=config.experiment.pca_explained_variance,random_state=seed)),("model",clone(l1))]),
      "l1_balanced_l1_select":Pipeline(preprocessing_steps(config.dataset, scale=True, pandas_output=True)+[("selector",SelectFromModel(clone(l1))),("model",clone(l1))]),
      "lightgbm_all":Pipeline(preprocessing_steps(config.dataset, pandas_output=True)+[("model",light)]),
    }
    for k in config.top_k_feature_counts:
      # -inf threshold와 max_features를 함께 사용해 importance 상위 K개를 선택한다.
      selector=SelectFromModel(LGBMClassifier(n_estimators=300,random_state=seed,n_jobs=n_jobs,verbosity=-1),threshold=-np.inf,max_features=partial(_checked_top_k,count=k))
      result[f"lightgbm_top_{k}"]=Pipeline(preprocessing_steps(config.dataset, pandas_output=True)+[("selector",selector),("model",clone(light))])
    return result

def compare_features(frame,config,n_jobs: int = 1):
    validate_train_role(frame)
    # Profile 검증을 통과한 feature와 label만 CV 입력으로 사용한다.
    x,y,_=split_frame_to_xy(frame,config.dataset)
    # 계층 CV와 feature selector가 정상적으로 학습되려면 두 label이 모두 필요하다.
    if y.nunique()<2:
      raise ValueError("특징 선택 비교를 위해 Train split에는 정상과 Fail label이 모두 있어야 합니다.")
    smallest_class_count=int(y.value_counts().min())
    if smallest_class_count<config.experiment.cv.n_splits:
      raise ValueError("가장 적은 class의 샘플 수가 CV fold 수보다 작습니다: "+f"최소 class 샘플={smallest_class_count}, fold={config.experiment.cv.n_splits}")
    # Top-K는 실제 입력 feature 수보다 클 수 없으며, 설정 오류를 모델 fit 전에 알려 준다.
    too_large_top_k=[k for k in config.top_k_feature_counts if k>x.shape[1]] if not config.dataset.categorical_feature_columns else []
    if too_large_top_k:
      raise ValueError("Top-K feature 수가 입력 feature 수보다 큽니다: "+f"Top-K={too_large_top_k}, feature 수={x.shape[1]}")
    # 후보별 fold 결과를 평균·표준편차로 집계하기 위해 동일한 반복 계층 CV를 만든다.
    rows=[]
    for name,pipeline in build_experiments(config,n_jobs).items():
      values=[]; times=[]; counts=[]
      for fit_i,val_i in training_folds(x,y,frame,config):
        # 매 fold마다 Pipeline을 복제해 selector와 model이 validation 정보를 보지 못하게 한다.
        model=clone(pipeline); started=perf_counter(); model.fit(x.iloc[fit_i],y.iloc[fit_i]); times.append(perf_counter()-started)
        # One-Hot 확장·선택 이후 최종 모델에 전달된 실제 특징 수를 기록한다.
        counts.append(fitted_feature_count(model))
        pos=list(model.named_steps["model"].classes_).index(config.dataset.positive_label); score=model.predict_proba(x.iloc[val_i])[:,pos]
        metric=evaluate_binary_scores(y.iloc[val_i],score,positive_label=config.dataset.positive_label,negative_label=config.dataset.negative_label,threshold=config.experiment.default_threshold)
        values.append(metric)
      row={"dataset_id":config.dataset.dataset_id,"split_strategy":"random_cv","experiment":name,"n_splits":config.experiment.cv.n_splits,"n_repeats":config.experiment.cv.n_repeats,"threshold":config.experiment.default_threshold,"selected_feature_count_mean":float(np.mean(counts)),"fit_time_mean_seconds":float(np.mean(times))}
      row["split_strategy"]="time_train_cv" if PROTOCOL_ID in frame and str(frame[PROTOCOL_ID].iloc[0]).startswith("time:") else "random_cv"
      row["training_protocol_id"]=frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in frame else None
      for key in ("average_precision","recall","precision","f1","roc_auc"):
        metric_values=[getattr(v,key) for v in values if getattr(v,key) is not None]; row[f"{key}_mean"]=float(np.mean(metric_values)); row[f"{key}_std"]=float(np.std(metric_values))
      rows.append(row)
    return pd.DataFrame(rows).sort_values("average_precision_mean",ascending=False)

def main():
    # 결과 CSV는 최신 실행 결과로 교체하며 Test·Time Validation 파일은 받지 않는다.
    parser=argparse.ArgumentParser(description="CV 내부 feature 선택 비교"); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG_PATH); parser.add_argument("--train",type=Path,required=True); parser.add_argument("--output",type=Path,default=Path("logs/feature_compare.csv")); parser.add_argument("--n-jobs",type=int,default=1); args=parser.parse_args()
    result=compare_features(pd.read_csv(args.train),load_modeling_config(args.config),args.n_jobs); args.output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(args.output,index=False,encoding="utf-8-sig"); print(result.to_string(index=False))
if __name__=="__main__": main()
