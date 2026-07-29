import time

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ── Training and inference timing ──────────────────────────────

def train_timed(
    pipeline: Pipeline,
    X_train,
    y_train,
) -> tuple:
    """
    Fit a pipeline and return (fitted_pipeline, train_seconds).

    Parameters
    ----------
    pipeline : sklearn Pipeline (unfitted)
    X_train  : array-like
    y_train  : array-like

    Returns
    -------
    tuple of (fitted Pipeline, float)
    """
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = round(time.perf_counter() - start, 3)
    return pipeline, training_time


def infer_timed(
    pipeline: Pipeline,
    X_test,
    repeats: int = 3,
) -> float:
    """
    Run predict repeats times; return mean ms per prediction.
    Repeating reduces timing noise on fast models.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test   : array-like
    repeats  : int, default 3

    Returns
    -------
    float — ms per prediction (4 decimal places)
    """
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.predict(X_test)
        timings.append(time.perf_counter() - start)
    return round((sum(timings) / repeats / len(X_test)) * 1000, 4)


# ── Evaluation ─────────────────────────────────────────────────

def evaluate_model(
    model: Pipeline,
    X_test,
    y_test,
) -> dict:
    """
    Compute evaluation metrics for a fitted pipeline.

    Returns accuracy, macro precision, macro recall, macro F1,
    a full classification report string, and the confusion matrix.
    Macro averaging gives equal weight to all ESI classes —
    performance on rare ESI 1 is not diluted by majority-class results.

    Parameters
    ----------
    model  : fitted sklearn Pipeline
    X_test : array-like
    y_test : array-like

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1,
        report (str), confusion_matrix (np.ndarray)
    """
    predictions = model.predict(X_test)

    return {
        "accuracy":  round(accuracy_score(y_test, predictions), 4),

        "precision": round(precision_score(
            y_test, predictions,
            average="macro", zero_division=0,
        ), 4),

        "recall":    round(recall_score(
            y_test, predictions,
            average="macro", zero_division=0,
        ), 4),

        "f1":        round(f1_score(
            y_test, predictions,
            average="macro", zero_division=0,
        ), 4),

        "report":    classification_report(
            y_test, predictions,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            y_test, predictions,
        ),
    }


# ── Pipeline builders (used by scripts/train.py) ───────────────

def build_pipeline(
    preprocessor: ColumnTransformer,
    classifier,
) -> Pipeline:
    """
    Wrap a preprocessor and classifier into a single sklearn Pipeline.

    Parameters
    ----------
    preprocessor : fitted or unfitted ColumnTransformer
    classifier   : any sklearn-compatible classifier

    Returns
    -------
    sklearn.pipeline.Pipeline (unfitted)
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",        classifier),
    ])
