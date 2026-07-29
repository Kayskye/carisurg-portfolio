import pandas as pd
from sklearn.model_selection import train_test_split

# Columns that may be imported as object dtype due to mixed entries
# (e.g. triage_glucose with 'HIGH' / 'LOW' string values).
_NUMERIC_COLS_TO_FIX = [
    "age", "esi",
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp",
    "triage_vital_rr", "triage_vital_o2",
    "triage_vital_temp", "triage_glucose",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the Yale Emergency Department triage CSV and correct
    dtype anomalies identified during Week 5 profiling.

    Parameters
    ----------
    filepath : str
        Path to yaleemmlc_admissionprediction_triage.csv.
        Taken from config['data']['filepath'] — not hard-coded.

    Returns
    -------
    pd.DataFrame
        Raw data with numeric columns correctly typed.
    """
    df = pd.read_csv(filepath)

    for col in _NUMERIC_COLS_TO_FIX:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> tuple:
    """
    Stratified 80/20 train/test split.

    Parameters
    ----------
    X           : pd.DataFrame — feature matrix
    y           : pd.Series   — target labels
    test_size   : float, default 0.20 — from config['data']['test_size']
    random_seed : int, default 42   — from config['data']['random_seed']
        Using the same seed and data reproduces the identical split
        used across Weeks 6, 7, and 8.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
