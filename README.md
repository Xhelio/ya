# Skincare Product Type Classification

This repository contains a cleaned skincare dataset and a reproducible
multiclass classification experiment using TF-IDF, StandardScaler, and
Logistic Regression.

## Dataset

Input file: `skincare_products_clean.csv`

Expected columns: `product_name`, `product_url`, `product_type` (target),
`clean_ingreds` (text feature), and `price` (GBP-formatted numeric feature).

## Run the experiment

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python train_model.py
```

The script validates the dataset, converts GBP price strings to numeric values,
and uses product URL as a grouping key (not as a model feature). It creates a
holdout fold using StratifiedGroupKFold so package-size variants sharing a
product page cannot be split across training and testing. TF-IDF,
StandardScaler, and Logistic Regression are fitted together in one sklearn
Pipeline, preventing learned preprocessing from being fitted on test data.

## Outputs

The script creates `experiment_output/` containing:

- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `test_predictions.csv`
- `skincare_logistic_regression.joblib`

The exact train/test row counts are determined by the grouped stratified fold;
they are not guaranteed to be exactly 80:20.

## Research integrity notes

- The earlier 58.33% accuracy in the research draft is not assumed or forced.
  It must be independently reproduced before being reported as this run's result.
- The 9 repeated product URLs represent multiple package sizes/variants.
  Grouping prevents variants from the same product page appearing in both
  training and test folds.
- Price remains in GBP numeric units. StandardScaler standardizes its scale;
  it does not convert GBP to Indonesian rupiah.
- For inference, the saved pipeline expects a dataframe with columns
  `clean_ingreds` and `price_numeric`.
