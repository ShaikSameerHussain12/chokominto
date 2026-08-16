# Architecture — WaterGuard

## 1. High-level shape

Standard Django MVT app, with the ML logic deliberately pulled out into its own package rather than living inside Django views. That separation is the one architectural decision worth explaining in an interview: `ml_engine/` has no Django imports and can be run, tested, or reused completely independently of the web layer (as done for this repo's own metric verification — see MODEL_CARD.md).

```
Browser
  │
  ▼
Django URLs (config/urls.py, core/urls.py)
  │
  ▼
Views (core/views.py) — admin portal + customer portal, session-based auth
  │
  ▼
Services (core/services.py) — orchestration layer: pulls DB rows into a DataFrame,
  │                             calls ml_engine, writes results back to DB
  ▼
ml_engine/ (framework-agnostic)
  ├── preprocessing.py   — clean_dataset(): dedup, negative handling, imputation, outlier capping
  ├── features.py        — engineer_features(): 25 features per customer
  ├── train.py            — prepare_training_data(), train_svm_model(), train_knn_model()
  ├── evaluate.py         — evaluate_predictions(), compare_models()
  └── visualization.py    — confusion matrix / model comparison charts (matplotlib)
  │
  ▼
models/*.joblib — persisted SVM, KNN, scaler, feature metadata
```

## 2. Data model (`core/models.py`)

| Model | Purpose |
|---|---|
| `UserProfile` | Customer metadata — type, location, meter type; linked 1:1 to Django `User` |
| `ConsumptionRecord` | One row per customer per month — the raw time series |
| `DatasetUpload` | Tracks CSV/Excel upload jobs (who uploaded, when, row count) |
| `Prediction` | Stores per-customer model output (score, risk level, model used) |
| `Investigation` | Physical audit tracking — status moves through a pipeline (e.g. Pending → Confirmed Fraud / Cleared) |
| `Feedback` | Customer support tickets |
| `ModelRun` | Historical log of each training run's metrics (accuracy, precision, recall, F1, ROC-AUC, training time) |
| `BlockedCustomer` | Accounts an admin has manually blocked pending investigation |

Note the feedback loop: `Investigation.status == 'Confirmed Fraud'` is checked first as the *real* label source in `run_model_training_pipeline()` (see `core/services.py`), and only falls back to the synthetic CSV labels when no confirmed investigations exist yet. That's the intended production path — the model is meant to eventually retrain on real confirmed outcomes, not stay on synthetic labels forever. In its current demo state, no investigations have been confirmed, so it's still running on synthetic labels.

## 3. Request flow: training a model (admin walkthrough)

1. Admin visits `/training/` → `views.model_training`
2. View calls `core.services.run_model_training_pipeline()`
3. Service pulls all `ConsumptionRecord` rows + `UserProfile` metadata into a DataFrame, joins in labels (confirmed investigations first, synthetic CSV as fallback)
4. `ml_engine.preprocessing.clean_dataset()` — cleaning
5. `ml_engine.features.engineer_features()` — 25 features per customer
6. `ml_engine.train.prepare_training_data()` — stratified split + scaling
7. Both models trained, evaluated, artifacts saved to `models/*.joblib`
8. Metrics written to `ModelRun` table for historical tracking
9. Confusion matrix + comparison charts rendered to `media/charts/*.png`
10. Admin sees results on the training page

## 4. Request flow: scoring customers

1. Admin visits `/predictions/`, selects a model → `views.predictions_log`
2. `core.services.execute_fraud_predictions(model_name)` loads the saved `.joblib` model + scaler
3. Same feature pipeline runs on current data (no retraining)
4. Each customer gets a `Prediction` record with a risk score
5. `/fraud-alerts/` surfaces the ranked, filterable results for inspection teams
6. Admin can open a customer's `/customers/<id>/` detail page, review their history graph, and open an `Investigation`

## 5. Why Django + server-rendered templates (not a SPA)

This is an internal tool for utility staff, not a consumer product — page-load latency and SEO don't matter, but session-based auth, server-side permission checks, and a fast build-out of CRUD-heavy admin screens do. Django's built-in auth, ORM, and template system cover all of that without extra tooling. Chart.js handles the interactive bits (consumption graphs) client-side without needing a full frontend framework.

## 6. Testing approach

`core/tests/test_all.py` runs through pytest-django, using fast (non-cryptographic) password hashing and lazy matplotlib imports specifically to keep the suite under 10 seconds — a deliberate tradeoff for a project-stage tool where iteration speed matters more than production-grade test hashing.
