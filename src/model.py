import time

from sklearn.pipeline import Pipeline

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
    confusion_matrix
)

RANDOM_SEED = 42

def train_timed(pipeline, X_train, y_train):

    start = time.perf_counter()

    pipeline.fit(X_train, y_train)

    training_time = round(
        time.perf_counter() - start,
        3
    )

    return pipeline, training_time


def infer_timed(pipeline, X_test, repeats=3):

    timings = []

    for _ in range(repeats):

        start = time.perf_counter()

        pipeline.predict(X_test)

        timings.append(
            time.perf_counter() - start
        )

    return round(
        (sum(timings) / repeats / len(X_test)) * 1000,
        4
    )


def train_logistic(preprocessor, X_train, y_train):

    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        ("model",
         LogisticRegression(
             class_weight="balanced",
             max_iter=1000,
             random_state=RANDOM_SEED
         ))
    ])

    return train_timed(
        pipeline,
        X_train,
        y_train
    )


def train_decision_tree(preprocessor, X_train, y_train):

    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        ("model",
         DecisionTreeClassifier(
             max_depth=10,
             class_weight="balanced",
             random_state=RANDOM_SEED
         ))
    ])

    return train_timed(
        pipeline,
        X_train,
        y_train
    )

def train_random_forest(preprocessor, X_train, y_train):

    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        ("model",
         RandomForestClassifier(
             n_estimators=100,
             max_depth=15,
             min_samples_leaf=5,
             class_weight="balanced",
             random_state=RANDOM_SEED,
             n_jobs=-1
         ))
    ])

    return train_timed(
        pipeline,
        X_train,
        y_train
    )

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {

        "accuracy":
            accuracy_score(y_test, predictions),

        "precision":
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ),

        "f1":
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ),

        "report":
            classification_report(
                y_test,
                predictions
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions
            )
    }
