"""BITS Pilani ML Assignment 2: Streamlit evaluation of five saved models.

No training occurs in this web app. All predictions use the five saved
joblib pipelines and the user-supplied or bundled held-out evaluation CSV.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "diagnosis"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="ML Classification Model Comparison", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading trained classifiers...")
def load_saved_models():
    """Load pretrained models once; never fit/retrain on test uploads."""
    loaded = {}
    for name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing required model file: model/{filename}")
        loaded[name] = joblib.load(model_path)
    return loaded


def validate_test_data(df, required_features):
    """Validate column names, numerical values, and the binary target."""
    if df.empty:
        raise ValueError("The CSV is empty. Please upload a nonempty CSV.")
    if df.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")

    missing = [col for col in [*required_features, TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))

    features = df.loc[:, required_features].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features must contain valid, finite numeric values with no missing entries.")

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("The 'diagnosis' target must contain only 0 (benign) or 1 (malignant).")

    return features, y.astype(int)


def evaluate(model, X, y):
    predictions = model.predict(X)
    positive_idx = list(model.classes_).index(1)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, positive_idx]
    else:
        scores = model.decision_function(X)

    results = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, scores) if y.nunique() == 2 else float("nan"),
        "Precision": precision_score(y, predictions, pos_label=1, zero_division=0),
        "Recall": recall_score(y, predictions, pos_label=1, zero_division=0),
        "F1": f1_score(y, predictions, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return results, predictions


st.title("Machine Learning — Assignment 2")
st.caption("BITS Pilani | Five classification models | Evaluation using saved models only")
st.markdown(
    "**Dataset:** Breast Cancer Wisconsin (Diagnostic) · **30 features** · "
    "**Binary target:** `diagnosis` (`0 = benign`, `1 = malignant`). "
    "The five models were trained on the same training split."
)

try:
    models = load_saved_models()
except Exception as exc:
    st.error(f"Cannot load saved models: {exc}")
    st.stop()

reference_model = next(iter(models.values()))
required_features = list(reference_model.feature_names_in_)

st.subheader("1. Choose evaluation dataset")
data_source = st.radio(
    "Test dataset source",
    ["Bundled held-out test data", "Upload a CSV"],
    horizontal=True,
    help="Use the provided test_data.csv or upload a CSV with the same 30 feature columns and diagnosis.",
)

if data_source == "Upload a CSV":
    uploaded_file = st.file_uploader("Upload test CSV (including the diagnosis target column)", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to evaluate all five models. A sample file is available below.")
        st.download_button(
            "Download sample test_data.csv",
            data=(ROOT / "test_data.csv").read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
        st.stop()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Cannot read uploaded CSV: {exc}")
        st.stop()
else:
    try:
        df = pd.read_csv(ROOT / "test_data.csv")
    except Exception as exc:
        st.error(f"Cannot read the included test_data.csv: {exc}")
        st.stop()

try:
    X_test, y_test = validate_test_data(df, required_features)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

extra_cols = [col for col in df.columns if col not in [*required_features, TARGET_COLUMN]]
if extra_cols:
    st.info("Unneeded extra columns will be ignored: " + ", ".join(extra_cols))

left, middle, right = st.columns(3)
left.metric("Evaluation rows", f"{len(df):,}")
middle.metric("Input features", str(len(required_features)))
right.metric("Malignant cases (1)", f"{int(y_test.sum()):,}")

if y_test.nunique() < 2:
    st.warning("Only one target class is present. AUC is undefined and will be shown as N/A.")

st.subheader("2. Model comparison on the selected test dataset")
comparison_rows = []
prediction_map = {}
for name, model in models.items():
    scores, preds = evaluate(model, X_test, y_test)
    comparison_rows.append({"ML Model Name": name, **scores})
    prediction_map[name] = preds

comparison = pd.DataFrame(comparison_rows)
st.dataframe(
    comparison.style.format({metric: "{:.4f}" for metric in METRICS}, na_rep="N/A"),
    hide_index=True,
    use_container_width=True,
)
st.download_button(
    "Download model comparison (CSV)",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv",
)

st.subheader("3. Detailed model evaluation")
model_choice = st.selectbox("Select a classifier", options=list(MODEL_FILES))
selected_row = comparison.loc[comparison["ML Model Name"] == model_choice].iloc[0]
cols = st.columns(3)
for col, metric in zip(cols, METRICS[:3]):
    col.metric(metric, "N/A" if pd.isna(selected_row[metric]) else f"{selected_row[metric]:.4f}")
cols = st.columns(3)
for col, metric in zip(cols, METRICS[3:]):
    col.metric(metric, f"{selected_row[metric]:.4f}")

predicted = prediction_map[model_choice]
chart_col, report_col = st.columns(2)
with chart_col:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_test, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign (0)", "Malignant (1)"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(model_choice)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with report_col:
    st.markdown("**Classification report**")
    report = classification_report(
        y_test, predicted, labels=[0, 1],
        target_names=["Benign (0)", "Malignant (1)"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("Preview evaluation data and expected columns"):
    st.dataframe(df.head(10), hide_index=True, use_container_width=True)
    st.write(f"Required target: `{TARGET_COLUMN}`. Feature columns must match the bundled CSV.")

st.caption(
    "Metrics are computed on the uploaded or bundled evaluation data only. "
    "All pipelines were fit before deployment; uploading data never retrains any model. "
    "For educational use only, not for clinical diagnosis."
)
