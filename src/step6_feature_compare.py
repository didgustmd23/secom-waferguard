# ==========================================
# CV fold 내부 feature 선택 비교
# - random_train만 사용하고 Test/Time Validation은 읽지 않음
# - PCA, L1, LightGBM Importance 선택기를 Pipeline 안에서 fit해 누수를 방지
# ==========================================
from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.step4_baseline import split_frame_to_xy
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from step4_baseline import split_frame_to_xy

def build_experiments(config, n_jobs: int = 1):
    seed=config.experiment.cv.random_state
    imputer=SimpleImputer(strategy="median").set_output(transform="pandas")
    scaler=StandardScaler().set_output(transform="pandas")
    l1=LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced",max_iter=2000,random_state=seed)
    light=LGBMClassifier(n_estimators=300,random_state=seed,n_jobs=n_jobs,verbosity=-1)
    result={
      "l1_balanced_all":Pipeline([("imputer",imputer),("scaler",scaler),("model",l1)]),
      "l1_balanced_pca90":Pipeline([("imputer",imputer),("scaler",scaler),("selector",PCA(n_components=config.experiment.pca_explained_variance,random_state=seed)),("model",clone(l1))]),
      "l1_balanced_l1_select":Pipeline([("imputer",imputer),("scaler",scaler),("selector",SelectFromModel(clone(l1))),("model",clone(l1))]),
      "lightgbm_all":Pipeline([("imputer",imputer),("model",light)]),
    }
    for k in config.top_k_feature_counts:
      selector=SelectFromModel(LGBMClassifier(n_estimators=300,random_state=seed,n_jobs=n_jobs,verbosity=-1),threshold=-np.inf,max_features=k)
      result[f"lightgbm_top_{k}"]=Pipeline([("imputer",imputer),("selector",selector),("model",clone(light))])
    return result

def compare_features(frame,config,n_jobs: int = 1):
    x,y,_=split_frame_to_xy(frame,config.dataset); cv=RepeatedStratifiedKFold(n_splits=config.experiment.cv.n_splits,n_repeats=config.experiment.cv.n_repeats,random_state=config.experiment.cv.random_state); rows=[]
    for name,pipeline in build_experiments(config,n_jobs).items():
      values=[]; times=[]; counts=[]
      for fit_i,val_i in cv.split(x,y):
        model=clone(pipeline); started=perf_counter(); model.fit(x.iloc[fit_i],y.iloc[fit_i]); times.append(perf_counter()-started)
        if "selector" in model.named_steps:
          selector=model.named_steps["selector"]
          counts.append(int(selector.n_components_) if hasattr(selector,"n_components_") else int(np.sum(selector.get_support())))
        else: counts.append(x.shape[1])
        pos=list(model.named_steps["model"].classes_).index(config.dataset.positive_label); score=model.predict_proba(x.iloc[val_i])[:,pos]
        metric=evaluate_binary_scores(y.iloc[val_i],score,positive_label=config.dataset.positive_label,negative_label=config.dataset.negative_label,threshold=config.experiment.default_threshold)
        values.append(metric)
      row={"dataset_id":config.dataset.dataset_id,"split_strategy":"random_cv","experiment":name,"n_splits":config.experiment.cv.n_splits,"n_repeats":config.experiment.cv.n_repeats,"threshold":config.experiment.default_threshold,"selected_feature_count_mean":float(np.mean(counts)),"fit_time_mean_seconds":float(np.mean(times))}
      for key in ("average_precision","recall","precision","f1","roc_auc"):
        metric_values=[getattr(v,key) for v in values if getattr(v,key) is not None]; row[f"{key}_mean"]=float(np.mean(metric_values)); row[f"{key}_std"]=float(np.std(metric_values))
      rows.append(row)
    return pd.DataFrame(rows).sort_values("average_precision_mean",ascending=False)

def main():
    parser=argparse.ArgumentParser(description="CV 내부 feature 선택 비교"); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG_PATH); parser.add_argument("--train",type=Path,required=True); parser.add_argument("--output",type=Path,default=Path("logs/feature_compare.csv")); parser.add_argument("--n-jobs",type=int,default=1); args=parser.parse_args()
    result=compare_features(pd.read_csv(args.train),load_modeling_config(args.config),args.n_jobs); args.output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(args.output,index=False,encoding="utf-8-sig"); print(result.to_string(index=False))
if __name__=="__main__": main()
