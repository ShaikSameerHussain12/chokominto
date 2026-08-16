# Data Dictionary — WaterGuard

## 1. Raw input schema

One row per customer per month. This is the shape of `data/sample_water_consumption.csv` and of `ConsumptionRecord` in the Django DB.

| Field | Type | Description |
|---|---|---|
| `customer_id` | string | Unique account ID, format `C#####` |
| `date` | date | First of each billing month |
| `consumption` | float | Metered water volume for the month (units as billed) |
| `billing_amount` | float | Amount billed for that month |
| `customer_type` | categorical | `Residential` (77%) / `Commercial` (20%) / `Industrial` (3%) in the synthetic sample |
| `meter_type` | categorical | `Analog` (64%) / `Digital` (31%) / `Smart` (5%) in the synthetic sample |
| `location` | categorical | One of 5 synthetic zones (`Irbid_East/West/South/North`, `Center`) |
| `fraud_class` | int (0/1) | **Synthetic label** — 1 if this customer was chosen by the generator script to carry an injected fraud pattern. See MODEL_CARD.md. |

Sample size actually generated: 100 customers × ~60 months → 6,063 raw rows, 6,000 after cleaning.

## 2. Preprocessing (`ml_engine/preprocessing.py`)

Applied before feature engineering:
- Deduplication on `(customer_id, date)`
- Negative readings handled (flagged, not dropped — negative values are a real fraud signal: meter resets)
- Missing readings imputed
- Outliers capped (not removed) to avoid destroying genuine high-consumption signal
- Customers with fewer than `min_active_months` (default 3) filtered out entirely

Actual stats from the seeded dataset: 63 duplicates removed, 8 negative readings handled, 128 missing values imputed, 6 outliers capped, 0 customers filtered — 6,000 rows / 100 customers remained.

## 3. Engineered features (customer-level, one row per customer)

Computed in `ml_engine/features.py::engineer_features()`. 25 numeric features feed the model; 4 additional identity/category columns (`customer_id`, `customer_type`, `location`, `meter_type`) are kept for display but excluded from training.

### Distributional (consumption history shape)
| Feature | Definition |
|---|---|
| `n_readings` | Count of monthly readings for this customer |
| `mean_consumption` | Mean of all readings |
| `std_consumption`, `variance_consumption` | Spread of readings |
| `min_consumption`, `max_consumption`, `median_consumption` | Range and center |
| `coef_of_variation` | std / mean — normalized volatility, comparable across customers with different baseline usage |
| `skewness`, `kurtosis` | Distribution shape (asymmetry, tail weight) of the reading history |

### Recency-weighted trend
| Feature | Definition |
|---|---|
| `mean_3m`, `mean_6m`, `mean_12m` | Rolling mean over the most recent 3/6/12 readings |
| `trend_slope` | Slope of a linear fit across the full reading history — is usage trending up/down over time |
| `sudden_change_ratio` | (mean of last 3 readings) / (mean of prior readings) — catches abrupt drops, the classic tampering signature |

### Anomalous-reading counts
| Feature | Definition |
|---|---|
| `zero_consumption_count`, `zero_consumption_ratio` | How often the meter reads exactly zero while presumably active |
| `negative_consumption_count`, `negative_consumption_ratio` | Count/ratio of negative readings (meter resets, a tampering indicator) |
| `reading_interval_std` | Standard deviation of days between readings — irregular read cadence can indicate meter access issues |

### Comparative / peer-relative
| Feature | Definition |
|---|---|
| `location_deviation_ratio` | This customer's mean consumption ÷ the average for their location zone |
| `type_deviation_ratio` | This customer's mean consumption ÷ the average for their customer type (Residential/Commercial/Industrial) |

### Billing-consumption relationship
| Feature | Definition |
|---|---|
| `mean_billing` | Mean billed amount |
| `billing_to_consumption_mean` | Mean of (billing ÷ consumption) per reading — catches flat-rate billing that doesn't track actual usage |
| `billing_to_consumption_std` | Volatility of that ratio — inconsistent billing-to-usage relationship is a manipulation signal |

## 4. Target

`fraud_class` (int, 0/1) — customer-level label, taken as the max of any row-level fraud flag across that customer's history. **Synthetic, not observed.** See MODEL_CARD.md section 2.
