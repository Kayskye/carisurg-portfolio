"""
tests/test_data.py
------------------
Sanity checks for src/data.py.

(a) Data-loading test — verifies the expected schema (column names
    and count) after load_dataset() runs.

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

from src.data import load_dataset, split_data


# ── Columns that must be present after loading ────────────────────
REQUIRED_COLUMNS = [
    "esi",
    "age",
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
    "triage_glucose",
    "arrivalmode",
    "previousdispo",
]


# ── Shared synthetic dataset fixture ─────────────────────────────
@pytest.fixture
def synthetic_csv(tmp_path):
    """
    Write a minimal synthetic CSV that mirrors the Yale EMMLC schema.
    Used so the test suite runs without the real dataset being present
    (the real dataset is excluded from the repository via .gitignore).
    """
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "esi":                  np.random.choice([1, 2, 3, 4, 5], n),
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
                                    ["Discharge", "Admitted", "Transfer"], n),
        "gender":               np.random.choice(["M", "F"], n),
        "race":                 np.random.choice(
                                    ["White", "Black", "Hispanic"], n),
        "ethnicity":            np.random.choice(
                                    ["Hispanic", "Non-Hispanic"], n),
        "insurance_status":     np.random.choice(
                                    ["Private", "Medicaid", "Medicare"], n),
        "arrivalday":           np.random.choice(
                                    ["Monday", "Wednesday", "Friday"], n),
        "arrivalmonth":         np.random.randint(1, 13, n),
        "arrivalhour_bin":      np.random.choice(
                                    ["Morning", "Afternoon", "Night"], n),
        "cc_shortnessofbreath": np.random.choice([0, 1], n),
    })

    filepath = tmp_path / "test_dataset.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


# ── Tests ────────────────────────────────────────────────────────

def test_required_columns_present(synthetic_csv):
    """
    Schema check — all required clinical columns must exist after
    load_dataset() runs on the CSV.
    """
    df = load_dataset(synthetic_csv)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing, (
        f"Required columns missing from loaded dataset: {missing}"
    )


def test_column_count_meets_minimum(synthetic_csv):
    """
    The loaded DataFrame must contain at least 10 columns.
    Guards against accidentally loading an empty or truncated file.
    """
    df = load_dataset(synthetic_csv)
    assert df.shape[1] >= 10, (
        f"Expected at least 10 columns, got {df.shape[1]}"
    )


def test_row_count_preserved(synthetic_csv):
    """Loading must not silently drop rows."""
    df = load_dataset(synthetic_csv)
    assert len(df) == 100, (
        f"Expected 100 rows after loading, got {len(df)}"
    )


def test_esi_column_is_numeric_after_load(synthetic_csv):
    """
    load_dataset() applies pd.to_numeric to ESI.
    The column must not remain as object dtype.
    """
    df = load_dataset(synthetic_csv)
    assert pd.api.types.is_numeric_dtype(df["esi"]), (
        f"ESI column dtype after load_dataset() is '{df['esi'].dtype}' "
        f"— expected a numeric type."
    )


def test_esi_values_in_valid_range(synthetic_csv):
    """All non-null ESI values must be between 1 and 5 inclusive."""
    df = load_dataset(synthetic_csv)
    esi_vals = df["esi"].dropna().astype(int).unique()
    out_of_range = [v for v in esi_vals if v < 1 or v > 5]
    assert not out_of_range, (
        f"ESI values outside 1–5 range found: {out_of_range}"
    )


def test_split_data_proportions(synthetic_csv):
    """
    split_data() with test_size=0.20 must produce a test set of
    approximately 20% of the data (±2 rows tolerance for small n).
    """
    df = load_dataset(synthetic_csv)
    y  = df["esi"].dropna().astype(int)
    X  = df.loc[y.index].drop(columns=["esi"])

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.20, random_seed=42
    )

    total   = len(X_train) + len(X_test)
    expected_test = int(total * 0.20)

    assert abs(len(X_test) - expected_test) <= 2, (
        f"Test set size {len(X_test)} deviates from expected "
        f"~{expected_test} by more than 2 rows."
    )


def test_split_is_reproducible(synthetic_csv):
    """
    Calling split_data() twice with the same seed must return
    identical splits — required for like-for-like model comparison.
    """
    df = load_dataset(synthetic_csv)
    y  = df["esi"].dropna().astype(int)
    X  = df.loc[y.index].drop(columns=["esi"])

    _, X_test_a, _, _ = split_data(X, y, test_size=0.20, random_seed=42)
    _, X_test_b, _, _ = split_data(X, y, test_size=0.20, random_seed=42)

    assert list(X_test_a.index) == list(X_test_b.index), (
        "split_data() with the same random_seed produced different splits — "
        "not reproducible."
    )
