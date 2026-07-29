import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def prepare_features(
    df: pd.DataFrame,
    numeric_cols: list,
    binary_cols: list,
    categorical_cols: list,
    target_col: str = "esi",
) -> tuple:
    """
    Select features and target; drop rows with a missing target.
    Filters each column group to names that actually exist in df
    so the function is robust to dataset version differences.

    Parameters
    ----------
    df               : pd.DataFrame — output of load_dataset()
    numeric_cols     : list of str  — from config['features']['numeric']
    binary_cols      : list of str  — from config['features']['binary']
    categorical_cols : list of str  — from config['features']['categorical']
    target_col       : str, default 'esi'

    Returns
    -------
    tuple of (X, y, numeric, binary, categorical)
        X           : pd.DataFrame — feature matrix
        y           : pd.Series   — integer ESI labels (1–5)
        numeric     : list of str — confirmed-present numeric features
        binary      : list of str — confirmed-present binary features
        categorical : list of str — confirmed-present categorical features
    """
    numeric     = [c for c in numeric_cols     if c in df.columns]
    binary      = [c for c in binary_cols      if c in df.columns]
    categorical = [c for c in categorical_cols if c in df.columns]

    all_features = numeric + binary + categorical

    df_model = (
        df[all_features + [target_col]]
        .dropna(subset=[target_col])
        .copy()
    )
    df_model[target_col] = df_model[target_col].astype(int)

    X = df_model[all_features]
    y = df_model[target_col]

    return X, y, numeric, binary, categorical


def build_preprocessor(
    numeric_features: list,
    binary_features: list,
    categorical_features: list,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer applying group-appropriate preprocessing:

    - Numeric     : median imputation → standard scaling.
      Median is robust to the vital sign outliers identified in
      Week 5 profiling. Standard scaling is required for models
      sensitive to feature magnitude.

    - Binary      : mode imputation only. The most frequent value
      for a binary complaint column is 0 (absent), which is
      clinically correct. Scaling a 0/1 variable is not meaningful.

    - Categorical : mode imputation → one-hot encoding.
      handle_unknown='ignore' prevents unseen category values at
      test time from crashing the pipeline.

    Parameters
    ----------
    numeric_features     : list of str
    binary_features      : list of str
    categorical_features : list of str

    Returns
    -------
    sklearn.compose.ColumnTransformer (unfitted)
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_features),
            ("bin", binary_transformer,      binary_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
