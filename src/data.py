import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def load_dataset(filepath):
    """
    Load the Yale Emergency Department dataset.
    """
    df = pd.read_csv(filepath)

    # Convert important numeric columns
    for col in [
        "age",
        "esi",
        "triage_vital_hr",
        "triage_vital_sbp",
        "triage_vital_dbp",
        "triage_vital_rr",
        "triage_vital_o2",
        "triage_vital_temp",
        "triage_glucose"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_features(df):

    NUMERIC_FEATURES = [
        c for c in [
            "triage_vital_o2",
            "triage_vital_hr",
            "triage_vital_sbp",
            "triage_vital_rr",
            "triage_vital_temp",
            "age",
            "triage_glucose"
        ] if c in df.columns
    ]

    BINARY_FEATURES = [
        c for c in [
            "gender",
            "arrivalmode"
        ] if c in df.columns
    ]

    CATEGORICAL_FEATURES = [
        c for c in [
            "race",
            "ethnicity",
            "insurance_status",
            "arrivalday",
            "arrivalmonth",
            "arrivalhour_bin",
            "previousdispo"
        ] if c in df.columns
    ]

    X = df.drop(columns=["esi"])
    y = df["esi"]

    return (
        X,
        y,
        NUMERIC_FEATURES,
        BINARY_FEATURES,
        CATEGORICAL_FEATURES
    )


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y
    )
