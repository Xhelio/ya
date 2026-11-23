#!/usr/bin/env python3
"""Reproducible skincare product-type classification experiment.

Run from the repository root:
    python -m pip install -r requirements.txt
    python train_model.py

The holdout split uses StratifiedGroupKFold so product variants sharing a URL
stay in the same fold. All learned preprocessing is fitted inside the sklearn
Pipeline on training data only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "skincare_products_clean.csv"
OUTPUT_DIR = ROOT / "experiment_output"
RANDOM_STATE = 42
N_SPLITS = 5


def parse_price(value: object) -> float:
    """Convert currency-formatted price (e.g. £5.20) to a numeric amount."""
    text = str(value).strip()
    cleaned = re.sub(r"[^0-9.\-]", "", text.replace(",", ""))
    if not cleaned:
        raise ValueError(f"Empty or invalid price: {value!r}")
    return float(cleaned)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = {"product_name", "product_url", "product_type", "clean_ingreds", "price"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    for col in ["product_name", "product_url", "product_type", "clean_ingreds", "price"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    if (df[["product_url", "product_type", "clean_ingreds", "price"]] == "").any().any():
        raise ValueError("Dataset contains empty URL, target, ingredient text, or price.")

    df["price_numeric"] = df["price"].map(parse_price)
    if (df["price_numeric"] < 0).any():
        raise ValueError("Negative product prices found.")

    # Keep URL as a grouping key, not as a model feature. Variants of the same
    # product page must not be split between training and testing.
    df["url_group"] = df["product_url"].str.lower().str.rstrip("/")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_dataset()

    target = df["product_type"]
    X = df[["clean_ingreds", "price_numeric"]]
    groups = df["url_group"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, target, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

    # TF-IDF handles text; StandardScaler handles the numeric price feature.
    # ColumnTransformer combines them while preserving sparse text features.
    preprocess = ColumnTransformer(
        transformers=[
            ("ingredients_tfidf", TfidfVectorizer(), "clean_ingreds"),
            ("price_scaler", StandardScaler(), ["price_numeric"]),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # A simple majority-class baseline gives context to the model's accuracy.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    labels = sorted(target.unique())
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    metrics = {
        "dataset": DATA_PATH.name,
        "rows": int(len(df)),
        "classes": int(target.nunique()),
        "class_distribution": {str(k): int(v) for k, v in target.value_counts().sort_index().items()},
        "unique_url_groups": int(groups.nunique()),
        "repeated_url_rows": int(groups.duplicated(keep=False).sum()),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "split_method": "first fold of StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "model": "TF-IDF + StandardScaler + LogisticRegression(lbfgs)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "majority_baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "sklearn_version": sklearn.__version__,
        "note": "Metrics are newly generated and must not be represented as the earlier 58.33% result.",
    }

    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "classification_report.csv", index_label="class")
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv", index_label="actual\\predicted")
    pd.DataFrame({
        "product_name": df.iloc[test_idx]["product_name"].values,
        "actual": y_test.values,
        "predicted": predictions,
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    joblib.dump(model, OUTPUT_DIR / "skincare_logistic_regression.joblib")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\\nReports and trained pipeline saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
