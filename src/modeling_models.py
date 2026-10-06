# ==========================================
# 모델 정의와 공통 학습·확률 예측
# - Step별 모델 설정 중복을 없애고 동일한 실험 이름은 동일한 Pipeline으로 생성
# - 전처리·특징 선택은 Pipeline 내부에서 현재 학습 구간에만 fit
# - 불균형 가중치는 호출자가 전달한 학습 label에서만 계산
# - 외부 Validation/Test와 threshold 선택은 이 모듈에서 처리하지 않음
# ==========================================

from functools import partial

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

try:
    from src.modeling_preprocessing import preprocessing_steps, checked_top_k
except ModuleNotFoundError:
    from modeling_preprocessing import preprocessing_steps, checked_top_k


class XGBoostClassifierAdapter(ClassifierMixin, BaseEstimator):
    """문자열 또는 -1/1 label을 XGBoost의 0/1 입력으로 변환하는 분류기.

    scikit-learn의 Pipeline과 clone 호환성을 유지하고, 학습 후 ``classes_``를
    원래 label 값으로 노출하여 공통 평가 코드가 양성 class를 찾게 한다.
    XGBoost는 실제 fit 시점에만 지연 import해 미설치 오류를 명확히 전달한다.
    """

    def __init__(self, positive_label=1, negative_label=-1,
                 n_estimators=300, max_depth=3, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 scale_pos_weight=1.0, random_state=42, n_jobs=1):
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """원래 label을 인코딩하고 XGBoost 분류기를 학습한다."""
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "XGBoost 후보를 사용하려면 requirements.txt의 xgboost 의존성을 설치하세요."
            ) from error

        if self.positive_label == self.negative_label:
            raise ValueError("XGBoost의 양성·음성 label은 서로 달라야 합니다.")
        if pd.Series(y).isna().any():
            raise ValueError("XGBoost 학습 label에 결측값이 있습니다.")
        observed_labels = set(pd.Series(y).unique())
        if observed_labels != {self.negative_label, self.positive_label}:
            raise ValueError("XGBoost 후보는 이진 분류 label 두 종류가 필요합니다.")
        # SECOM처럼 -1/1을 쓰거나 다른 문자열 label을 써도 불량은 항상 1로 학습한다.
        encoded_y = np.asarray(pd.Series(y).eq(self.positive_label), dtype=np.int32)
        self.classes_ = np.asarray([self.negative_label, self.positive_label])
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.estimator_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X, encoded_y)
        return self

    def predict(self, X):
        """내부 0/1 예측을 원래 label 값으로 되돌려 반환한다."""
        check_is_fitted(self, "estimator_")
        encoded = self.estimator_.predict(X).astype(int)
        return np.where(encoded == 1, self.positive_label, self.negative_label)

    def predict_proba(self, X):
        """XGBoost의 두 클래스 확률을 반환한다."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        """특징 선택기가 사용할 수 있도록 내부 모델의 중요도를 노출한다."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_


# 실제 후보 이름과 기존 특징 실험 이름을 유지하여 CSV·CLI 호환성을 보존한다.
CANDIDATE_VARIANTS = (
    "logistic_regression_l1", "logistic_regression_l1_balanced", "rbf_svm",
    "random_forest", "lightgbm", "lightgbm_scale_pos_weight",
    "xgboost", "xgboost_scale_pos_weight",
)
EXPERIMENT_ALIASES = {
    "l1_balanced_all": "logistic_regression_l1_balanced",
    "l1_balanced_pca90": "logistic_regression_l1_balanced",
    "l1_balanced_l1_select": "logistic_regression_l1_balanced",
    "lightgbm_all": "lightgbm",
}
WEIGHTED_VARIANTS = {"lightgbm_scale_pos_weight", "xgboost_scale_pos_weight"}


def candidate_base_name(name):
    """가중치 변형 이름을 config.json의 기본 후보 이름으로 변환한다."""
    return name.removesuffix("_balanced").removesuffix("_scale_pos_weight")


# ==========================================
# 분류기 설정의 단일 정의
# - 특징 선택용 분류기도 동일한 생성 함수를 사용
# - 가중치 비율은 여기서 계산하지 않고 학습할 때만 결정
# ==========================================
def build_classifier(config, name, *, n_jobs=1, n_estimators=300):
    """실험의 분류기를 만들되 아직 학습하지 않는다."""
    seed = config.experiment.cv.random_state
    if name in {"logistic_regression_l1", "logistic_regression_l1_balanced"}:
        return LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000, random_state=seed,
            class_weight="balanced" if name.endswith("_balanced") else None,
        )
    if name == "rbf_svm":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
        )
    if name in {"lightgbm", "lightgbm_scale_pos_weight"}:
        return LGBMClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs, verbosity=-1,
        )
    if name in {"xgboost", "xgboost_scale_pos_weight"}:
        return XGBoostClassifierAdapter(
            positive_label=config.dataset.positive_label,
            negative_label=config.dataset.negative_label,
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
        )
    raise ValueError(f"정의되지 않은 후보 모델입니다: {name}")


def build_pipeline(config, name, *, n_jobs=1):
    """모델·특징 선택 실험 이름 하나를 독립된 Pipeline으로 변환한다."""
    # 기존 lightgbm_top_K 이름의 K는 설정에 선언된 실험만 허용한다.
    top_k_names = {f"lightgbm_top_{count}": count for count in config.top_k_feature_counts}
    model_name = EXPERIMENT_ALIASES.get(name, "lightgbm" if name in top_k_names else name)
    classifier = build_classifier(config, model_name, n_jobs=n_jobs)
    scale = model_name.startswith("logistic_regression") or model_name == "rbf_svm"
    feature_experiment = name in EXPERIMENT_ALIASES or name in top_k_names
    steps = preprocessing_steps(config.dataset, scale=scale, pandas_output=feature_experiment)

    # 선택기를 전처리 뒤에 넣어 해당 학습 fold의 변환된 특징만 보게 한다.
    if name == "l1_balanced_pca90":
        steps.append(("selector", PCA(
            n_components=config.experiment.pca_explained_variance,
            random_state=config.experiment.cv.random_state,
        )))
    elif name == "l1_balanced_l1_select":
        steps.append(("selector", SelectFromModel(clone(classifier))))
    elif name in top_k_names:
        steps.append(("selector", SelectFromModel(
            clone(classifier), threshold=-np.inf,
            max_features=partial(checked_top_k, count=top_k_names[name]),
        )))
    return Pipeline(steps + [("model", classifier)])


def build_candidate_pipelines(config, n_jobs=1):
    """기존 Step 5 후보와 가중치 변형을 같은 정의로 생성한다."""
    return {name: build_pipeline(config, name, n_jobs=n_jobs) for name in CANDIDATE_VARIANTS}


def build_experiments(config, n_jobs=1):
    """기존 Step 6의 전체·PCA·L1·Top-K 실험을 생성한다."""
    names = ("l1_balanced_all", "l1_balanced_pca90", "l1_balanced_l1_select", "lightgbm_all")
    names += tuple(f"lightgbm_top_{count}" for count in config.top_k_feature_counts)
    return {name: build_pipeline(config, name, n_jobs=n_jobs) for name in names}


# ==========================================
# 학습 fold에만 의존하는 공통 fit
# - clone으로 이전 fold의 학습 상태를 재사용하지 않음
# - 병렬 수는 분류기·선택기에서 실제 지원하는 파라미터에만 적용
# - 가중치 변형은 현재 학습 label의 음성/양성 비율을 사용
# ==========================================
def fit_pipeline(pipeline, features, labels, config, experiment_name, *, n_jobs=1):
    """새 Pipeline을 학습하고 원본 미학습 Pipeline은 변경하지 않는다."""
    label_series = pd.Series(labels)
    expected = {config.dataset.negative_label, config.dataset.positive_label}
    if label_series.isna().any() or set(label_series.unique()) != expected:
        raise ValueError("학습 구간에는 결측 없이 설정된 정상·Fail label이 모두 필요합니다.")
    model = clone(pipeline)
    # 이름 접두어 대신 지원 파라미터를 확인해 사용자 정의 Pipeline도 처리한다.
    parallel_params = {
        key: n_jobs for key in model.get_params(deep=True)
        if key.endswith("__n_jobs")
    }
    if parallel_params:
        model.set_params(**parallel_params)
    if experiment_name in WEIGHTED_VARIANTS:
        positive_count = int(label_series.eq(config.dataset.positive_label).sum())
        negative_count = int(label_series.eq(config.dataset.negative_label).sum())
        model.set_params(model__scale_pos_weight=negative_count / positive_count)
    model.fit(features, labels)
    return model


def positive_scores(pipeline, features, dataset):
    """class 순서와 무관하게 설정된 양성 label의 확률만 반환한다."""
    # 문자열 label에서는 양성 class가 첫 번째일 수도 있으므로 [:, 1]을 가정하지 않는다.
    classes = pipeline.named_steps["model"].classes_
    positions = [index for index, label in enumerate(classes) if label == dataset.positive_label]
    if len(positions) != 1:
        raise ValueError("학습된 후보 모델에서 Profile의 Fail label을 찾을 수 없습니다.")
    return pipeline.predict_proba(features)[:, positions[0]]
