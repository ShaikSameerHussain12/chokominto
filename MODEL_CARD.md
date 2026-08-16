# Model Card — WaterGuard Fraud Classifiers

This is the document to point an interviewer to if they ask "how good is your model, really." It's written to survive that question, not to dodge it.

## 1. What the models are

Two supervised binary classifiers trained on the same 25 engineered features, predicting `fraud_class` (0/1) per customer:

- **SVM** — RBF kernel, `class_weight='balanced'`, `probability=True`
- **KNN** — k=5, distance-weighted, minkowski metric

Both are trained on a stratified 80/20 split with `StandardScaler` fit on train only. Neither model sees test data during fitting.

## 2. Where the labels come from — read this section first

There is no real fraud data behind this project. `fraud_class` is assigned by `scripts/generate_demo_data.py`, which:

1. Randomly picks 10% of 100 synthetic customer IDs to be "fraud."
2. For each, randomly assigns one of five fraud *patterns* (sudden consumption drop, high variance, billing mismatch, negative meter resets, zero consumption) starting at a random month.
3. Injects that pattern's signature directly into the synthetic consumption series.

The classifiers are then trained on features engineered from that same series — meaning they are, at best, learning to recognize the mathematical signature the generator script itself wrote in. This is closer to "does the pipeline correctly detect an injected rule" than "can this model find fraud it's never seen the shape of."

**What this does prove:** the feature engineering captures the right signal (trend, volatility, zero-reading ratios) and the training/evaluation pipeline is correctly wired end-to-end.

**What this does not prove:** that the model would catch real, unlabeled fraud in production. Real meter tampering doesn't announce itself with one of five clean synthetic signatures.

## 3. Evaluation results and why they need a caveat

Run on this repo's own generated dataset (100 customers, 6,000 readings, seed=42):

| Model | Accuracy | Precision | Recall | F1 | Confusion Matrix (tn/fp/fn/tp) |
|---|---|---|---|---|---|
| SVM | 0.95 | 1.00 | 0.50 | 0.667 | 18 / 0 / 1 / 1 |
| KNN | 1.00 | 1.00 | 1.00 | 1.00 | 18 / 0 / 0 / 2 |

The test set is **20 customers, 2 of them fraud**. At n=2 positive cases, "recall" can only take the values 0%, 50%, or 100% — there's no meaningful gradation. KNN's perfect score is one lucky split away from being a mediocre one; it is not evidence of a strong model. If asked to defend this number, the honest answer is: *"it's not defensible as a generalization claim, the sample is too small — what it shows is the pipeline runs correctly end-to-end."*

The README of the reference academic project (Yarmouk Water Company data, Jordan) reports ~74% SVM / ~70% KNN accuracy. That's a different dataset entirely (real utility billing records, larger scale) and isn't a fair comparison point for this project's numbers — it's included in the original README only as background on why SVM/KNN were chosen as the algorithms.

## 4. What a defensible version of this would need

If this were going into production or being defended as a real fraud-detection claim, it would need:

- Real historical inspection outcomes as labels (confirmed fraud cases from actual field audits), not synthetic injection
- A dataset several orders of magnitude larger — hundreds to thousands of confirmed cases, not 10
- K-fold cross-validation instead of a single 80/20 split, given the small-sample instability shown above
- Out-of-time validation (train on older data, test on newer) rather than random split, since fraud patterns likely drift
- Precision-recall tradeoff analysis at different score thresholds, since in this domain a false positive costs an inspection visit and a false negative costs ongoing revenue loss — those costs aren't symmetric

## 5. How to talk about this in an interview

Don't lead with the accuracy numbers. Lead with the pipeline: feature engineering choices, the preprocessing decisions (duplicate handling, negative-reading logic, outlier capping — see cleaning stats below), and the fact that you can articulate exactly why the evaluation numbers aren't trustworthy. That last part — knowing the limits of your own result — is the actual signal of ML maturity an interviewer is checking for.

Cleaning stats from the run used above, for reference:
```
initial_rows: 6063 → duplicates_removed: 63 → negative_readings_handled: 8
→ missing_readings_imputed: 128 → outliers_capped: 6 → final_rows: 6000 (100 customers)
```
