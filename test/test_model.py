"""
tests/test_model.py
-------------------
Sanity checks for src/model.py and src/features.py.

(b) Training smoke test — runs the full pipeline on ~50 synthetic
    rows and verifies it completes without crashing, predictions
    are valid ESI values, and evaluation returns all required keys.

Run from the repository root with:
    pytest tests/
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data     import split_data
from src.features import prepare_features, build_preprocessor
from src.model    import (
    build_pipeline,
    train_timed,
    infer_timed,
    evaluate_model,
)

# ── Feature groups (mirroring config.yaml) ────────────────────────
NUMERIC_COLS     = [
    "triage_vital_o2", "triage_vital_hr", "triage_vital_sbp",
    "triage_vital_rr", "triage_vital_temp", "age", "triage_glucose",
]
BINARY_COLS      = ["cc_shortnessofbreath"]
CATEGORICAL_COLS = ["arrivalmode", "previousdispo"]


# ── Shared 50-row synthetic dataset ──────────────────────────────
@pytest.fixture
def small_df():
    """
    50-row synthetic DataFrame that mimics the Yale EMMLC schema.
    Small enough to train in milliseconds; large enough to produce
    a stratified 80/20 split with at least one example per class.
    All five ESI classes are represented to allow stratified splitting.
    """
    np.random.seed(42)
    n = 50

    # Guarantee at least one row per ESI class for stratification
    esi_values = [1, 2, 3, 4, 5] * 10  # 10 of each = 50 rows

    return pd.DataFrame({
        "esi":                  esi_values,
        "age":                  np.random.randint(18, 90, n).astype(float),
        "triage_vital_hr":      np.random.normal(80, 20, n),
        "triage_vital_sbp":     np.random.normal(120, 20, n),
        "triage_vital_dbp":     np.random.normal(80, 15, n),
        "triage_vital_rr":      np.random.normal(16, 4, n),
        "triage_vital_o2":      np.clip(np.random.normal(97, 3, n), 0, 100),
        "triage_vital_temp":    np.random.normal(37, 0.5, n),
        "triage_glucose":       np.random.normal(6, 2, n),
        "arrivalmode":          np.random.choice(
                                    ["Ambulance", "Walk-in", "Other"], n),
        "previousdispo":        np.random.choice(
                                    ["Discharge", "Admitted"], n),
        "cc_shortnessofbreath": np.random.choice([0, 1], n),
        "gender":               np.random.choice(["M", "F"], n),
        "race":                 np.random.choice(["White", "Black"], n),
        "ethnicity":            np.random.choice(
                                    ["Hispanic", "Non-Hispanic"], n),
        "insurance_status":     np.random.choice(
                                    ["Private", "Medicaid"], n),
        "arrivalday":           np.random.choice(
                                    ["Monday", "Tuesday", "Wednesday"], n),
        "arrivalmonth":         np.random.randint(1, 13, n),
        "arrivalhour_bin":      np.random.choice(
                                    ["Morning", "Night"], n),
    })


# ── Shared split fixture ──────────────────────────────────────────
@pytest.fixture
def prepared_split(small_df):
    """Prepare features and return a train/test split for reuse."""
    X, y, numeric, binary, categorical = prepare_features(
        small_df,
        numeric_cols=NUMERIC_COLS,
        binary_cols=BINARY_COLS,
        categorical_cols=CATEGORICAL_COLS,
    )
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.20, random_seed=42
    )
    preprocessor = build_preprocessor(numeric, binary, categorical)
    return X_train, X_test, y_train, y_test, preprocessor, y


# ── Smoke tests ───────────────────────────────────────────────────

def test_logistic_regression_smoke(prepared_split):
    """
    Training smoke test — LogisticRegression pipeline must fit and
    predict on ~50 rows without raising any exception.
    """
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test, preprocessor, y = prepared_split

    clf      = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = build_pipeline(preprocessor, clf)

    # Must fit without error
    pipeline.fit(X_train, y_train)

    # Must predict correct number of outputs
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test), (
        f"Expected {len(X_test)} predictions, got {len(predictions)}"
    )


def test_predictions_are_valid_esi_values(prepared_split):
    """
    All predictions must be valid ESI levels (i.e. integers in the
    set of classes present in y_train). No out-of-range values allowed.
    """
    from sklearn.dummy import DummyClassifier

    X_train, X_test, y_train, y_test, preprocessor, y = prepared_split

    clf      = DummyClassifier(strategy="stratified", random_state=42)
    pipeline = build_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    valid_classes = set(y.unique())

    for pred in predictions:
        assert pred in valid_classes, (
            f"Prediction '{pred}' is not a valid ESI class. "
            f"Valid classes: {sorted(valid_classes)}"
        )


def test_evaluate_model_required_keys(prepared_split):
    """
    evaluate_model() must return a dict containing all four required
    headline metric keys plus report and confusion_matrix.
    """
    from sklearn.dummy import DummyClassifier

    X_train, X_test, y_train, y_test, preprocessor, _ = prepared_split

    clf      = DummyClassifier(strategy="most_frequent", random_state=42)
    pipeline = build_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)

    results = evaluate_model(pipeline, X_test, y_test)

    REQUIRED_KEYS = [
        "accuracy", "precision", "recall", "f1",
        "report", "confusion_matrix",
    ]
    for key in REQUIRED_KEYS:
        assert key in results, (
            f"evaluate_model() is missing required key: '{key}'"
        )


def test_metric_values_in_range(prepared_split):
    """
    Accuracy, precision, recall, and F1 must all be between 0 and 1.
    """
    from sklearn.dummy import DummyClassifier

    X_train, X_test, y_train, y_test, preprocessor, _ = prepared_split

    clf      = DummyClassifier(strategy="most_frequent", random_state=42)
    pipeline = build_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)

    results = evaluate_model(pipeline, X_test, y_test)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        val = results[metric]
        assert 0.0 <= val <= 1.0, (
            f"Metric '{metric}' = {val} is outside the valid range [0.0, 1.0]"
        )


def test_train_timed_returns_correct_types(prepared_split):
    """
    train_timed() must return (fitted_pipeline, float).
    The float must be non-negative and the pipeline must expose .predict().
    """
    from sklearn.dummy import DummyClassifier

    X_train, X_test, y_train, y_test, preprocessor, _ = prepared_split

    clf      = DummyClassifier(random_state=42)
    pipeline = build_pipeline(preprocessor, clf)

    fitted, t = train_timed(pipeline, X_train, y_train)

    assert isinstance(t, float), (
        f"train_timed() should return a float timing value, got {type(t)}"
    )
    assert t >= 0, "Training time must be non-negative."
    assert hasattr(fitted, "predict"), (
        "train_timed() returned object does not have a .predict() method."
    )


def test_prepare_features_returns_correct_structure(small_df):
    """
    prepare_features() must return (X, y, numeric, binary, categorical)
    with X and y of equal length and non-empty feature lists.
    """
    X, y, numeric, binary, categorical = prepare_features(
        small_df,
        numeric_cols=NUMERIC_COLS,
        binary_cols=BINARY_COLS,
        categorical_cols=CATEGORICAL_COLS,
    )

    assert len(X) == len(y), "X and y must have the same number of rows."
    assert len(numeric) > 0,     "Numeric feature list must not be empty."
    assert len(X.columns) == len(numeric) + len(binary) + len(categorical), (
        "X column count must equal sum of all feature groups."
    )
