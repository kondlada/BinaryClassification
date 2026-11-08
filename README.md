# BITS Pilani — M.Tech (AIML) — Machine Learning Assignment 2

**Topic:** Comparison of five binary classification models with an interactive Streamlit application  
**Submission:** GitHub repository + deployed Streamlit link + one genuine BITS Virtual Lab execution screenshot + README content in a single submitted PDF  
**Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository

> **Add your real links after publishing:**  
> GitHub repository: **[PASTE YOUR GITHUB REPOSITORY URL HERE]**  
> Live Streamlit app: **[PASTE YOUR DEPLOYED STREAMLIT URL HERE]**

## A. Problem statement

The objective is to classify breast-cancer tumor measurements as **malignant** or **benign** by training and evaluating five supervised machine-learning classifiers on the **same dataset and the same stratified training/test split**. The exercise demonstrates data preparation, selection of classifiers, quantitative model comparison, and deployment of previously trained models in a Streamlit application. The deployed application accepts a CSV file containing the required features and the ground-truth target, loads each saved model without retraining, and displays six evaluation metrics, a confusion matrix, and a classification report.

This is an educational machine-learning demonstration, **not** a medical or clinical diagnostic system.

## B. Dataset description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source (UCI):** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Dataset access for reproducible code:** `sklearn.datasets.load_breast_cancer()` (scikit-learn's packaged copy of the UCI dataset)
- **Number of observations:** **569** (meets the minimum of 500)
- **Number of numerical input features:** **30** (meets the minimum of 12)
- **Task:** Binary classification
- **Target column in this project:** `diagnosis`
- **Target encoding:** `1 = malignant` (**positive class**); `0 = benign` (**negative class**)
- **Class distribution:** 212 malignant, 357 benign
- **Data split:** Stratified 80% training / 20% testing (`random_state=42`)
- **Training observations:** 455
- **Held-out testing observations:** 114
- **Test data file:** `test_data.csv`, with 30 feature columns and the `diagnosis` column
- **Features include:** mean radius, mean texture, mean perimeter, mean area, mean smoothness, and corresponding error and worst-value measurements (30 total).

**Preprocessing:** Logistic Regression, KNN, Naive Bayes, and SVM use `StandardScaler` in a scikit-learn `Pipeline`. Decision Tree uses the raw numeric features. In all applicable models, scaling parameters are estimated on **training data only**. The separate held-out test data is never used for model fitting or feature-scaler fitting.

**Note:** scikit-learn's original binary coding is `0 = malignant` and `1 = benign`; this project deliberately reverses those values so the positive-class precision, recall, and F1 concern the medically important **malignant** class. The CSV and app use the recoded target consistently.

## C. GitHub repository link

**GitHub repository URL:** [PASTE YOUR REAL GITHUB REPOSITORY URL HERE]

The repository includes the full Streamlit code, model-training source code, all five saved trained pipelines (`.joblib`), requirements, README, and held-out test CSV. An optional notebook is provided for running and documenting the work on BITS Virtual Lab.

### Repository structure

```text
ML_Assignment_2/
├── app.py
├── train_models.py
├── BITS_Lab_Execution.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

The joblib files contain fitted `Pipeline` objects, **including their fitted scalers where required**. Separate `scaler.joblib` files are therefore unnecessary.

## D. Models used and their evaluation metrics

All five models were trained on the same 455 training observations and evaluated on the same 114 held-out observations.

1. **Logistic Regression:** `LogisticRegression(max_iter=2000, random_state=42)` with standardization.
2. **Decision Tree:** `DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42)`.
3. **K-Nearest Neighbors (KNN):** `KNeighborsClassifier(n_neighbors=7)` with standardization.
4. **Naive Bayes:** `GaussianNB()` with standardization.
5. **Support Vector Machine (SVM):** `SVC(kernel="rbf", probability=True, random_state=42)` with standardization.

### Evaluation metrics and definitions

- **Accuracy:** Fraction of all test examples predicted correctly.
- **AUC:** Area under the receiver operating characteristic curve (computed from predicted probabilities for malignant class 1).
- **Precision:** Of all predictions labeled malignant, the proportion that were truly malignant.
- **Recall:** Of all truly malignant examples, the proportion classified as malignant.
- **F1 score:** Harmonic mean of malignant-class precision and recall.
- **MCC:** Matthews correlation coefficient, a balanced measure reflecting all four confusion-matrix outcomes.

### Five-model comparison table

*Actual results from a reproducible local execution of `python train_models.py` using scikit-learn 1.8.0; metrics rounded to four decimals.*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | **0.9286** | 0.9512 | 0.9245 |
| Decision Tree | 0.8772 | 0.9654 | 0.9118 | 0.7381 | 0.8158 | 0.7343 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

The application's comparison table is recalculated on **whichever test CSV is selected or uploaded**, so its values may differ if you use a different labeled evaluation dataset. AUC requires both classes to be present in the test file; otherwise, the app displays N/A.

## E. Performance observations and overall winner

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | A strong linear baseline (accuracy **0.9649**, AUC **0.9960**). It produced the **highest AUC**, indicating particularly strong discrimination of benign versus malignant samples on the held-out data. Its malignant recall was **0.9286**. |
| **Decision Tree** | The lowest accuracy (**0.8772**) and malignant recall (**0.7381**) of the five on this specific split. Its individual splitting rules are interpretable, but this fitted tree missed more malignant examples than the other approaches. |
| **KNN** | Performed strongly after standardization (accuracy **0.9561**, F1 **0.9383**). Nearby examples in the standardized feature space support useful predictions, although its performance remains sensitive to feature scaling and choice of neighbors. |
| **Naive Bayes** | Achieved accuracy **0.9211** and AUC **0.9891**, providing a simple and fast probabilistic baseline. Its conditional-independence assumption can be restrictive for correlated tumor measurements. |
| **Support Vector Machine** | Achieved the **highest accuracy (0.9737), F1 (0.9630), and MCC (0.9442)**. It also had **perfect malignant-class precision (1.0000)** on this held-out split, while its recall matched Logistic Regression at **0.9286**. |
| **Overall winner for this dataset** | **Support Vector Machine (SVM)**, based on its highest accuracy, F1, and MCC. **Logistic Regression** has a slightly better AUC, so SVM is not the winner on every metric. Outcomes are specific to this dataset/split; the comparison is not evidence of clinical suitability. |

### Interpretation

SVM is preferred for the assignment's overall scorecard because it leads on three relevant metrics, including MCC, which is informative when class frequencies differ. However, the SVM and Logistic Regression models have equal malignant recall in this test, and Logistic Regression has a marginally higher AUC. In health-related classification, missed malignant cases may be particularly consequential. Therefore, choosing a real clinical model would require independent external validation, appropriate decision thresholds, calibration, and domain-specific safety assessment, none of which is established here.

## F. Reproduce training and run the application

### 1. Clone or download the repository

Use the GitHub repository's **Code → Download ZIP** option, then extract it. Open a terminal in the `ML_Assignment_2` folder.

### 2. Install packages

Python **3.11 or later** is recommended, with a compatible version supported by the pinned libraries.

```bash
python -m pip install -r requirements.txt
```

### 3. Reproduce model training (optional when the `.joblib` files are already present)

```bash
python train_models.py
```

This command generates/refreshes all five saved fitted models under `model/`, recreates `test_data.csv` deterministically, and prints the comparison metrics. The app **does not call this script** and **does not retrain** during test-data upload.

### 4. Run Streamlit locally

```bash
streamlit run app.py
```

The app opens in a browser. Select **Bundled held-out test data** to see results immediately, or select **Upload a CSV** and upload `test_data.csv`. Choose a model from the dropdown to inspect its six metrics, confusion matrix, and classification report. The full comparison of all five classifiers remains visible.

### 5. Upload your repository to GitHub and deploy

1. Create a new GitHub repository and upload **all project files**, preserving the `model/` directory and `.joblib` files.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository, select the `main` branch, and set the main file path to `app.py`.
4. Deploy and verify that all five saved models load, CSV upload works, the comparison table updates, and the model-specific confusion matrix and classification report appear.
5. Replace both placeholder URLs at the top of this README with your **actual** GitHub and live app links.

**Version compatibility:** The `.joblib` files in this repository were generated using `scikit-learn==1.8.0`. The requirements file pins that version to avoid loading models with an incompatible scikit-learn release. The application does not download data or models at runtime.

## G. Upload CSV format and limitations

The uploaded test CSV must contain all **30 exact feature names** found in `test_data.csv` and the `diagnosis` target column containing only `0` or `1`. The order of columns can vary. Extra columns are ignored. Missing, nonnumeric, or infinite feature values are rejected. AUC is undefined if a test CSV contains only one target class.

Use the bundled `test_data.csv` to verify functionality. Only uploaded or bundled **test data** are scored. Saved trained models are loaded and used for prediction; they are not refit to the uploaded observations.

## H. BITS Virtual Lab evidence and final submission PDF

The assignment requires the work to be performed on **BITS Virtual Lab** and a **genuine screenshot** of that execution included in the submitted PDF. You can use `BITS_Lab_Execution.ipynb` in the Virtual Lab to run the training script and show the resulting comparison table; alternatively, execute `python train_models.py` in a Virtual Lab terminal. Capture your actual on-screen execution for the screenshot. **No virtual-lab screenshot is included in this package because it must be taken from your real lab session.**

The assignment requires **one final PDF**, in this order:

1. **GitHub Repository Link** — working URL to the complete repository.
2. **Live Streamlit App Link** — working deployed app URL.
3. **BITS Virtual Lab Screenshot** — actual execution proof.
4. **Full README content** — including problem statement, dataset, GitHub URL, comparison table, individual model observations, and overall winner.

Verify that both URLs open, the Streamlit app is interactive, all five models are selectable, and the README is copied into the PDF. The deadline stated in the assignment brief is **8 November 2026, 11:59 PM**.
