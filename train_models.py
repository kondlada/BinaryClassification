"""Train and evaluate the five classifiers for BITS Pilani ML Assignment 2.

Run in the repository root:
    python train_models.py

Source: UCI Breast Cancer Wisconsin (Diagnostic), available through
scikit-learn's load_breast_cancer dataset helper.  The target is recoded
as 1 = malignant (positive class), 0 = benign.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_dataset():
    """Return full UCI dataset with a documented binary target encoding."""
    raw = load_breast_cancer()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    # Original scikit-learn dataset: 0=malignant, 1=benign.
    # Recode so precision, recall, F1, and AUC target malignant (1).
    y = pd.Series((raw.target == 0).astype(int), name=TARGET_COLUMN)
    return X, y


def make_models():
    """Pipelines avoid test-set leakage by fitting scalers only on training data."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_model(model, X, y):
    """Compute six requested metrics; positive class is malignant (1)."""
    preds = model.predict(X)
    positive_index = list(model.classes_).index(1)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, positive_index]
    else:
        scores = model.decision_function(X)

    return {
        "Accuracy": accuracy_score(y, preds),
        "AUC": roc_auc_score(y, scores) if len(np.unique(y)) == 2 else float("nan"),
        "Precision": precision_score(y, preds, pos_label=1, zero_division=0),
        "Recall": recall_score(y, preds, pos_label=1, zero_division=0),
        "F1": f1_score(y, preds, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, preds),
    }


def run_training():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    test = X_test.copy()
    test[TARGET_COLUMN] = y_test.to_numpy()
    test.to_csv(TEST_CSV, index=False, float_format="%.12g")

    metrics = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        metrics.append({"ML Model Name": name, **evaluate_model(model, X_test, y_test)})

    results = pd.DataFrame(metrics)
    print(f"Dataset: {len(X)} rows, {X.shape[1]} features; 1=malignant, 0=benign")
    print(f"Train: {len(X_train)} rows | Held-out test: {len(X_test)} rows")
    print("\nHeld-out test-set results (positive class: malignant):")
    print(results.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
    print(f"\nSaved 5 trained pipelines under {MODEL_DIR}")
    print(f"Saved held-out test CSV to {TEST_CSV}")
    return results


if __name__ == "__main__":
    run_training()
