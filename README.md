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

Hyperparameter selection uses GridSearchCV with StratifiedGroupKFold on the
training partition only. The holdout test partition is not used for model
selection. The search optimizes macro F1 across C, class_weight, and TF-IDF
ngram_range settings.

## Latest experiment result

GitHub Actions run on 2026-11-23, using scikit-learn 1.9.1:

| Metric | Untuned | Tuned |
| --- | ---: | ---: |
| Accuracy | 54.82% | 57.89% |
| Macro F1 | 47.82% | 55.61% |
| Weighted F1 | not recorded for untuned comparison | 57.14% |

The tuned holdout result is based on 228 test rows from 1,138 records across
14 classes. The split used the first fold of 5-fold StratifiedGroupKFold
(random_state=42); there were 910 training rows and 228 test rows, with zero
product-URL group overlap. The majority-class baseline accuracy was 10.96%.

Best parameters selected using grouped cross-validation on training data:
`C=10.0`, `class_weight=balanced`, and `ngram_range=(1, 2)`.
The best cross-validation macro F1 was 56.76%.

Performance remains uneven across categories. In the holdout report,
Exfoliator recall was 16.67% and Peel recall was 33.33%; these limitations
should be discussed rather than hidden. The 57.89% accuracy is a new,
reproducible result and is not the same experiment as the earlier 58.33%
figure in the research draft.

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

- The 9 repeated product URLs represent multiple package sizes/variants.
  Grouping prevents variants from the same product page appearing in both
  training and test folds.
- Price remains in GBP numeric units. StandardScaler standardizes its scale;
  it does not convert GBP to Indonesian rupiah.
- For inference, the saved pipeline expects a dataframe with columns
  `clean_ingreds` and `price_numeric`.
