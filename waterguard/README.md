# Detecting Anomalies in Water Usage Patterns Using Data Mining Techniques
## Subtitle: Machine Learning Based Water Consumption Fraud Detection and Decision Support System
### Product Brand Name: **WaterGuard**

WaterGuard is a full-stack decision-support system designed to identify suspicious water consumption behavior, prioritize accounts for physical inspection, and mitigate Non-Technical Loss (NTL) in water utilities.

---

## 1. Problem Statement & Research Context

Water supply companies suffer substantial financial losses due to two main types of water losses:
1. **Technical Loss:** Caused by physical infrastructure failure, such as pipeline leaks, transmission drops, and network failures.
2. **Non-Technical Loss (NTL):** Caused by customer actions, such as meter tampering, meter bypassing, under-reporting consumption, and billing database manipulation.

The academic project this is based on proposes using historical water consumption data and machine learning classification algorithms (Support Vector Machine and K-Nearest Neighbors) to identify suspicious accounts.
* **Original Benchmarks:** The reference project reports an SVM accuracy of approximately **74%** and a KNN accuracy of approximately **70%** on historical Yarmouk Water Company (YWC) data in Jordan.
* **ML Honesty Rule:** Because original proprietary datasets are unavailable, this reconstructed application calculates and displays **actual metrics** dynamically based on currently trained datasets rather than showing hardcoded percentages.

---

## 2. CRISP-DM Methodology & Architecture

The system implements the six phases of the Cross-Industry Standard Process for Data Mining (CRISP-DM):
1. **Business Understanding:** Mitigating utility losses by directing site inspection crews to high-probability fraud accounts.
2. **Data Understanding:** Checking distributions, timeline lengths, missing values, duplicates, and class distributions.
3. **Data Preparation:** Data cleaning (deduplicating, interpolating NaNs, handling resets/negatives) and feature engineering (22+ attributes).
4. **Modeling:** Hyperparameter tuning of SVM (RBF kernel, balanced weights) and KNN (distance weight).
5. **Evaluation:** Calculation of Accuracy, Precision, Recall, F1, and Confusion Matrices.
6. **Deployment:** Exposing prediction engines via Django template views and MySQL databases.

---

## 3. Technology Stack
* **Backend:** Python 3.14.2, Django 6.1
* **Machine Learning:** Scikit-Learn 1.9.0, Pandas 3.0.5, NumPy 2.5.2, Joblib 1.5.3
* **Graphics/Visualization:** ChartJS (frontend), Matplotlib/Seaborn (static reports)
* **Database:** MySQL (Django ORM) with a default **SQLite** fallback out-of-the-box.
* **Testing:** Pytest, Pytest-Django

---

## 4. Project Structure

```text
waterguard/
├── config/                  # Django project settings and URLs
├── core/                    # Core Django App (Models, Views, Forms, Services, Tests)
│   ├── migrations/          # Django DB migration logs
│   ├── tests/               # Pytest verification suites
│   ├── models.py            # Relational database models
│   ├── views.py             # HTTP controllers (Admin and Customer views)
│   └── services.py          # ML pipeline orchestrations and DB seeders
├── ml_engine/               # Core machine learning package
│   ├── preprocessing.py     # Ingestion & preprocessing cleaning
│   ├── features.py          # Statistical & behavioral features extraction
│   ├── train.py             # SVM / KNN train configurations
│   ├── predict.py           # Class probabilities & risk level mapping
│   ├── evaluate.py          # Metric calculations & comparisons
│   └── visualization.py     # Matplotlib reporting chart generators
├── templates/               # Responsive HTML templates
├── static/                  # Style assets (Teal/Dark Navy design CSS)
├── data/                    # Ingested uploads and sample datasets
├── models/                  # Serialized .joblib model files & scalers
├── scripts/                 # Synthetic data generator script
├── docs/                    # Academic report outlines and Mermaid diagrams
├── pytest.ini               # Pytest settings
├── manage.py                # Django CLI entry point
└── requirements.txt         # Python dependencies manifest
```

---

## 5. Installation & Local Setup

### Step 1: Open the Project Directory
We recommend setting the active workspace directory to:
```text
C:\Users\shaik\.gemini\antigravity\scratch\waterguard
```

### Step 2: Install Python Dependencies
Run the command:
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
To connect to Microsoft SQL Server (SSMS):
1. Open SQL Server Management Studio (SSMS) and connect to your SQL Server instance (e.g. `localhost\SQLEXPRESS`).
2. Create an empty database: Right-click **Databases** -> **New Database...** -> name it `waterguard_db` -> click **OK**.
3. Edit the `.env` file in the project folder to set:
   ```env
   DB_ENGINE=mssql
   DB_NAME=waterguard_db
   DB_HOST=localhost\SQLEXPRESS  # Specify your server instance name
   
   # For Windows Integrated Authentication (Trusted Connection):
   DB_USER=
   DB_PASSWORD=
   
   # For SQL Server Authentication:
   # DB_USER=your_login_username
   # DB_PASSWORD=your_login_password
   
   DB_OPTIONS_DRIVER=ODBC Driver 17 for SQL Server  # Change to driver 18 if installed
   ```

### Step 4: Run Migrations
Run Django migrations to build the tables:
```bash
python manage.py makemigrations core
python manage.py migrate
```

### Step 5: Seed Demo Dataset and Credentials
Run the synthetic data generator to create a 5-year sample consumption timeline:
```bash
python scripts/generate_demo_data.py
```
This generates `data/sample_water_consumption.csv`.

Next, seed the database with admin and standard customer accounts:
* **Option A:** Start the local webserver and log in as admin, then click **Seed Demo Dataset** directly from the warning banner on the dashboard homepage.
* **Option B:** Run a python shell command to trigger it:
  ```bash
  python -c "import os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings'); django.setup(); from core.services import seed_demo_data; seed_demo_data()"
  ```

### Step 6: Launch Django Local Server
Run:
```bash
python manage.py runserver
```
Open your browser and navigate to: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 6. Development Demo Credentials
These accounts are created during database seeding.
> [!WARNING]
> **DEVELOPMENT ONLY:** Do not use these credentials in production environments.

* **Utility Administrator:**
  * **Email/Username:** `admin@example.com`
  * **Password:** `admin123`
* **Standard Customer Portal:**
  * **Email/Username:** `user@example.com`
  * **Password:** `user123` (Linked to Customer ID: `C10001`)

---

## 7. URL Map & Routing Directory

### Administrative Portal
* **Dashboard Home:** `/dashboard/` — Total customers, alert rate counts, top suspicious cases, feedback list.
* **Dataset Upload:** `/upload/` — Ingest CSV or Excel spreadsheets containing monthly reading lists.
* **Model Training:** `/training/` — Split datasets, train SVM + KNN, and evaluate comparison scores.
* **Scoring Predictions:** `/predictions/` — Select a classifier model and score all customers.
* **Fraud Alerts:** `/fraud-alerts/` — Filterable prioritizations table showing active alert risk flags.
* **Customer Directory:** `/customers/` — paginated search table of all registered accounts.
* **Customer Profile:** `/customers/<customer_id>/` — Line graph, indicators list, inspection form, block toggle actions.
* **Inspections Pipeline:** `/investigations/` — physical audits tracking states.
* **Customer Feedback Inbox:** `/feedback/` — List support requests and respond.
* **System Reports:** `/reports/` — Academic metrics logs and CSV data exporter.
* **Graph Analysis:** `/graphs/` — Analytics panel containing Chart.js distribution widgets.

### Customer Portal
* **My Profile:** `/profile/` — metadata stats and address/phone updater form.
* **Consumption Log:** `/my-consumption/` — History table of invoiced volume readings.
* **Usage Graphs:** `/my-graphs/` — Chronological consumption vs. billing line chart.
* **Submit Inquiry:** `/submit-feedback/` — Submit support tickets and trace ticket history.

---

## 8. Verification & Running Tests

Run the complete test suite containing unit, integration, and functional checks using pytest:
```bash
pytest
```
*Tests are optimized to execute in under 10 seconds by using fast MD5 password hashing and lazy Matplotlib loading during test executions.*

---

## 9. Known Limitations & Future Enhancements
* **Cold Start:** Supervised classifiers require labeled historical samples. In the absence of labels, anomaly scores or threshold-based alerts (e.g. Isolation Forest) should be integrated.
* **GIS Mapping:** Integrating physical location plotting (e.g. Mapbox / Google Maps) to group site inspections geographically.
* **Smart IoT Meters:** Upgrading batch monthly inputs to real-time hourly stream data monitoring.
