# ==========================================
# Dataset Profile 기반 공통 모델 전처리
# - 수치형은 median 대치, 범주형은 최빈값 대치와 One-Hot Encoding 사용
# - 전부 결측인 컬럼도 보존하여 fold마다 입력 차원이 사라지는 문제 방지
# - 전처리 학습은 호출자가 생성한 Pipeline의 학습 fold에서만 수행
# ==========================================

from functools import partial

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _numeric_columns(frame, *, categorical_columns):
    # 숫자로 저장된 범주형 컬럼도 Profile 선언을 우선하여 수치형에서 제외한다.
    return [column for column in frame.columns if column not in categorical_columns]


def preprocessing_steps(dataset, *, scale=False, pandas_output=False):
    # 수치형 데이터의 기존 단계 이름은 유지해 기존 실행 경로와 호환한다.
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    if not dataset.categorical_feature_columns:
        if pandas_output:
            for _, transformer in numeric_steps:
                transformer.set_output(transform="pandas")
        return numeric_steps

    # 범주형은 dense 출력으로 통일하여 PCA와 LightGBM도 같은 입력을 사용할 수 있게 한다.
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), partial(
            _numeric_columns, categorical_columns=dataset.categorical_feature_columns
        )),
        ("categorical", categorical, list(dataset.categorical_feature_columns)),
    ], sparse_threshold=0)
    if pandas_output:
        preprocessor.set_output(transform="pandas")
    return [("preprocessor", preprocessor)]


def fitted_feature_count(pipeline):
    # 선택기 또는 최종 모델이 실제로 받은 변환 후 특징 수를 기록한다.
    if "selector" in pipeline.named_steps:
        selector = pipeline.named_steps["selector"]
        if hasattr(selector, "n_components_"):
            return int(selector.n_components_)
        return int(selector.get_support().sum())
    return int(pipeline.named_steps["model"].n_features_in_)
