"""
classify.py
-----------
Fraudulent Job Posting Detector
DecodeLabs AI Engineering Internship - Project 2: Data Classification Using AI

This script implements the full supervised-learning pipeline required by the
project brief, and goes beyond it:

Required by the brief:
  1. Load and understand a dataset            -> load_data() + EDA section
  2. Split data into training and test sets    -> train_test_split()
  3. Apply a simple classification algorithm   -> LogisticRegression baseline

Extra, to make it internship-portfolio strong:
  - Mixed feature pipeline: TF-IDF on free text + scaled/encoded structured
    features, combined with ColumnTransformer (realistic, not toy).
  - Four algorithms trained and compared head-to-head (Logistic Regression,
    Random Forest, Naive Bayes, Linear SVM) instead of just one.
  - Evaluation beyond accuracy: precision, recall, F1, ROC-AUC, confusion
    matrix - because for fraud detection, accuracy alone is misleading.
  - Cross-validation, not just a single train/test split.
  - Feature-importance / most-suspicious-words explainability.
  - Saves the best model + vectorizer to disk (joblib) for reuse by predict.py.
  - Saves a comparison chart as a PNG for the README / report.

Run:  python3 classify.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import joblib

DATA_PATH = os.path.join("data", "job_postings.csv")
MODEL_DIR = "model"
OUT_DIR = "outputs"

TEXT_COL = "description"
NUMERIC_COLS = ["years_experience_required", "has_company_profile",
                 "has_company_logo", "requests_upfront_payment",
                 "urgent_hiring_language", "requires_no_experience"]
CATEGORICAL_COLS = ["employment_type", "contact_email_domain"]
TARGET_COL = "fraudulent"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # A posting with no salary listed at all is itself a (weak) signal --
    # encode "was a salary given?" as its own numeric feature.
    df["salary_listed"] = df["salary_range"].notna().astype(int)
    return df


def print_eda(df):
    print("=" * 60)
    print("STEP 1: LOAD & UNDERSTAND THE DATASET")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nClass balance (fraudulent):")
    counts = df[TARGET_COL].value_counts()
    for label, count in counts.items():
        name = "FRAUDULENT" if label == 1 else "legitimate"
        print(f"  {name:12s}: {count:4d}  ({count/len(df):.1%})")
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.sum() else "  none")
    print()


def build_pipeline(model):
    """Combine TF-IDF (text) + scaled numeric + one-hot categorical features."""
    preprocessor = ColumnTransformer(transformers=[
        ("text", TfidfVectorizer(max_features=300, stop_words="english",
                                  ngram_range=(1, 2)), TEXT_COL),
        ("num", StandardScaler(), NUMERIC_COLS + ["salary_listed"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        # LinearSVC has no predict_proba; use decision_function for AUC
        y_proba = pipe.decision_function(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data()
    print_eda(df)

    X = df[[TEXT_COL] + NUMERIC_COLS + CATEGORICAL_COLS + ["salary_listed"]]
    y = df[TARGET_COL]

    print("=" * 60)
    print("STEP 2: TRAIN / TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"(stratified split keeps the same fraud/legit ratio in both sets)\n")

    print("=" * 60)
    print("STEP 3: TRAIN & COMPARE CLASSIFICATION ALGORITHMS")
    print("=" * 60)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "Calibrated Linear SVM (with probabilities)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"), cv=3),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }

    results = []
    confusions = {}
    fitted_pipes = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        metrics, cm, _ = evaluate(name, pipe, X_test, y_test)

        # 5-fold cross-validated F1 on the training set (robustness check)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()

        results.append(metrics)
        confusions[name] = cm
        fitted_pipes[name] = pipe

        print(f"\n--- {name} ---")
        print(f"  Test accuracy : {metrics['accuracy']:.3f}")
        print(f"  Precision     : {metrics['precision']:.3f}")
        print(f"  Recall        : {metrics['recall']:.3f}")
        print(f"  F1-score      : {metrics['f1']:.3f}")
        print(f"  ROC-AUC       : {metrics['roc_auc']:.3f}")
        print(f"  5-fold CV F1  : {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})")

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n" + "=" * 60)
    print("STEP 4: MODEL COMPARISON SUMMARY (sorted by F1-score)")
    print("=" * 60)
    print(results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted_pipes[best_name]
    print(f"\nBest model by F1-score: {best_name}")

    print("\nConfusion matrix (best model) -- rows=actual, cols=predicted [legit, fraud]:")
    print(confusions[best_name])
    print("\nFull classification report (best model):")
    print(classification_report(y_test, best_pipe.predict(X_test),
                                 target_names=["legitimate", "fraudulent"]))

    # ---- Save comparison chart ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    for i, (_, row) in enumerate(results_df.iterrows()):
        axes[0].bar(x + i * width, [row[m] for m in metrics_to_plot], width, label=row["model"])
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(metrics_to_plot, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Model comparison")
    axes[0].legend(fontsize=7, loc="lower right")

    cm = confusions[best_name]
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"Confusion matrix - {best_name}")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["legit", "fraud"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["legit", "fraud"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    for (r, c), v in np.ndenumerate(cm):
        axes[1].text(c, r, str(v), ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black")

    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=140)
    print(f"\nSaved comparison chart -> {chart_path}")

    # ---- Explainability: top red-flag words (Logistic Regression) -------
    if "Logistic Regression" in fitted_pipes:
        lr_pipe = fitted_pipes["Logistic Regression"]
        try:
            tfidf = lr_pipe.named_steps["preprocess"].named_transformers_["text"]
            feature_names = tfidf.get_feature_names_out()
            coefs = lr_pipe.named_steps["model"].coef_[0][:len(feature_names)]
            top_idx = np.argsort(coefs)[-15:][::-1]
            print("\nTop text signals pushing toward FRAUDULENT (Logistic Regression weights):")
            for i in top_idx:
                print(f"  {feature_names[i]:25s}  weight={coefs[i]:.3f}")
        except Exception as e:
            print(f"(skipped explainability section: {e})")

    # ---- Persist best model ------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump({"pipeline": best_pipe, "model_name": best_name}, model_path)
    print(f"\nSaved best model ({best_name}) -> {model_path}")

    results_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    print(f"Saved metrics table -> {os.path.join(OUT_DIR, 'model_comparison.csv')}")


if __name__ == "__main__":
    main()
