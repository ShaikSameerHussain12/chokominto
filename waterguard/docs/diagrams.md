# WaterGuard System Diagrams (Mermaid)

This document contains visual diagrams for the system architecture, database schema, data flows, use cases, activities, and sequence processes.

---

## 1. System Architecture

```mermaid
flowchart TD
    User([Customer]) -->|Web Browser| UI[HTML / CSS / JS Frontend]
    Admin([Utility Administrator]) -->|Web Browser| UI
    
    UI -->|HTTP Requests| Django[Django Backend Engine]
    Django -->|ORM Queries| DB[(MySQL Database)]
    
    Django -->|Dataframes| ML[ML Preprocessing & Scaling]
    ML -->|Features| Train[SVM & KNN Model Training]
    Train -->|Serialized Models| Disk[joblib model storage]
    
    Disk -->|Load models| Pred[Prediction & Risk Scoring Engine]
    Pred -->|Risk metrics| Django
```

---

## 2. Component Diagram

```mermaid
component
    [HTML5/CSS3/ChartJS Frontend] as FE
    [Django Views Controller] as Controller
    [Django ORM models] as ORM
    [MySQL Database] as MySQL
    [preprocessing.py] as Preprocess
    [features.py] as Features
    [train.py] as Train
    [predict.py] as Predict
    
    FE <--> Controller : HTTP / JSON
    Controller <--> ORM : ORM Queries
    ORM <--> MySQL : SQL
    
    Controller --> Preprocess : Raw Data
    Preprocess --> Features : Cleaned Time Series
    Features --> Train : Customer Features
    Features --> Predict : Scoring Features
    Predict --> Controller : Flags & Risk Probabilities
```

---

## 3. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    UserProfile {
        int id PK
        int user_id FK
        string customer_id UK
        string customer_type
        string location
        string meter_type
        string phone
        string address
        datetime created_at
    }
    
    ConsumptionRecord {
        int id PK
        string customer_id FK
        date date
        float consumption
        decimal billing_amount
        string payment_status
    }
    
    Prediction {
        int id PK
        string customer_id FK
        string model_name
        int predicted_class
        float probability
        string risk_level
        text key_indicators
        datetime created_at
    }
    
    Investigation {
        int id PK
        string customer_id FK
        string status
        text remarks
        boolean confirmed_fraud
        int investigated_by_id FK
        datetime investigated_at
    }
    
    Feedback {
        int id PK
        string customer_id FK
        string feedback_type
        text message
        text response
        string status
        datetime created_at
    }
    
    ModelRun {
        int id PK
        string model_name
        float accuracy
        float precision
        float recall
        float f1
        float roc_auc
        float training_time
        int dataset_size
        datetime created_at
    }
    
    BlockedCustomer {
        int id PK
        string customer_id FK
        text reason
        int blocked_by_id FK
        datetime blocked_at
        boolean active
    }

    UserProfile ||--o{ ConsumptionRecord : "associated monthly readings"
    UserProfile ||--o{ Prediction : "holds anomaly flags"
    UserProfile ||--o{ Investigation : "under physical review"
    UserProfile ||--o{ Feedback : "submits tickets"
    UserProfile ||--o| BlockedCustomer : "is suspended"
```

---

## 4. Use Case Diagrams

### Admin Actor Use Cases
```mermaid
leftToRightDirection
actor Admin
rectangle "WaterGuard Admin Dashboard" {
    usecase "Login & Authenticate" as UC1
    usecase "Seed / Upload Dataset" as UC2
    usecase "Clean & preprocess Data" as UC3
    usecase "Train SVM and KNN" as UC4
    usecase "Evaluate and Compare Classifiers" as UC5
    usecase "Batch Score Customer Fraud Risk" as UC6
    usecase "Inspect Suspicious Customer Details" as UC7
    usecase "Update physical Inspection Status" as UC8
    usecase "Respond to User Feedbacks" as UC9
    usecase "Block / Unblock Customer account" as UC10
    usecase "Download Reports & CSV Exports" as UC11
}
Admin --> UC1
Admin --> UC2
Admin --> UC3
Admin --> UC4
Admin --> UC5
Admin --> UC6
Admin --> UC7
Admin --> UC8
Admin --> UC9
Admin --> UC10
Admin --> UC11
```

### Customer Actor Use Cases
```mermaid
leftToRightDirection
actor Customer
rectangle "WaterGuard Customer Portal" {
    usecase "Login & Authenticate" as CU1
    usecase "View Profile Metadata" as CU2
    usecase "Edit Phone and Address" as CU3
    usecase "Review Consumption History" as CU4
    usecase "View Usage Line Chart" as CU5
    usecase "Submit feedback or disputes" as CU6
}
Customer --> CU1
Customer --> CU2
Customer --> CU3
Customer --> CU4
Customer --> CU5
Customer --> CU6
```

---

## 5. Sequence Diagrams

### Admin Training & Prediction Flow
```mermaid
sequenceDiagram
    actor Admin
    participant Dashboard as Django UI
    participant Service as services.py
    participant Pre as preprocessing.py
    participant Feat as features.py
    participant Train as train.py
    participant Pred as predict.py
    participant DB as MySQL DB

    Admin->>Dashboard: Click "Retrain Classifiers"
    Dashboard->>Service: run_model_training_pipeline()
    Service->>DB: Fetch all ConsumptionRecords
    DB-->>Service: return raw datasets
    Service->>Pre: clean_dataset(df_raw)
    Pre-->>Service: return cleaned_df & cleaning_stats
    Service->>Feat: engineer_features(cleaned_df)
    Feat-->>Service: return features_df & feature_cols
    Service->>Train: prepare_training_data() & train_svm_model() / train_knn_model()
    Train-->>Service: save scaler/models on disk & return evaluation metrics
    Service->>DB: Save ModelRun logs
    Service-->>Dashboard: Return metrics and comparison plot path
    Dashboard-->>Admin: Show training confirmation & comparison table

    Admin->>Dashboard: Click "Generate Predictions"
    Dashboard->>Service: execute_fraud_predictions(model_name)
    Service->>DB: Fetch records
    Service->>Pre: clean & preprocess
    Service->>Feat: extract features
    Service->>Pred: predict_anomalies(features_df, model_name)
    Pred-->>Service: return predictions df (probabilities, risk levels)
    Service->>DB: Save Prediction & auto-trigger Pending Investigations
    Service-->>Dashboard: Return prediction totals
    Dashboard-->>Admin: Update KPI statistics and refresh tables
```

---

## 6. Activity Diagrams

### Admin Investigation Workflow
```mermaid
stateDiagram-v2
    [*] --> Suspicious_Flagged: ML prediction flags customer (class = 1)
    Suspicious_Flagged --> Physical_Inspection_Scheduled: Admin marks status as "Under Review"
    Physical_Inspection_Scheduled --> On_Site_Check: Field crew audits water meter
    
    state Choice_Outcome <<choice>>
    On_Site_Check --> Choice_Outcome: Inspector verifies condition
    
    Choice_Outcome --> Confirmed_Fraud: Tampering / Bypass found
    Choice_Outcome --> Meter_Billing_Issue: Faulty meter / Admin clerical error
    Choice_Outcome --> False_Positive: Legitimate high consumption
    
    Confirmed_Fraud --> Blocked_Account: Admin suspends account and records reason
    Blocked_Account --> [*]
    
    Meter_Billing_Issue --> Resolved_Issue: Fix meter / Correct invoice
    False_Positive --> Resolved_Issue: Archive case
    
    Resolved_Issue --> [*]
```
