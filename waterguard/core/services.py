import os
import csv
import time
import json
import logging
import pandas as pd
from datetime import datetime
from django.conf import settings
from django.contrib.auth.models import User, Group
from core.models import (
    UserProfile, ConsumptionRecord, DatasetUpload, Prediction,
    Investigation, Feedback, ModelRun, BlockedCustomer
)
from ml_engine.preprocessing import clean_dataset
from ml_engine.features import engineer_features
from ml_engine.train import prepare_training_data, train_svm_model, train_knn_model, save_model_artifacts
from ml_engine.evaluate import evaluate_predictions, compare_models
from ml_engine.predict import predict_anomalies
from ml_engine.visualization import plot_confusion_matrix, plot_model_comparison

logger = logging.getLogger(__name__)

def seed_demo_data():
    """
    Seeds the database with demo accounts and consumption records.
    Creates:
    - Admin: admin@example.com (pass: admin123)
    - User: user@example.com (pass: user123, linked to customer C10001)
    - Loads transactions from data/sample_water_consumption.csv
    """
    # 1. Create Credentials
    # Admin
    admin_user, created = User.objects.get_or_create(
        username='admin@example.com',
        email='admin@example.com'
    )
    if created or not admin_user.has_usable_password():
        admin_user.set_password('admin123')
        admin_user.is_staff = True
        admin_user.is_superuser = True
        admin_user.save()
        logger.info("Admin user created (admin@example.com / admin123)")
        
    # Standard Customer User
    customer_user, created = User.objects.get_or_create(
        username='user@example.com',
        email='user@example.com'
    )
    if created or not customer_user.has_usable_password():
        customer_user.set_password('user123')
        customer_user.save()
        logger.info("Standard user created (user@example.com / user123)")
        
    # Check sample dataset
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'sample_water_consumption.csv')
    if not os.path.exists(csv_path):
        # Generate demo data if file doesn't exist
        from scripts.generate_demo_data import generate_synthetic_data
        generate_synthetic_data(num_customers=100, months=60, output_path=csv_path)
        
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # 2. Seed Customer Profiles in UserProfile for a few customers
    # Ensure C10001 maps to user@example.com
    c10001_data = df[df['customer_id'] == 'C10001']
    loc = c10001_data['location'].iloc[0] if len(c10001_data) > 0 else 'Center'
    cust_type = c10001_data['customer_type'].iloc[0] if len(c10001_data) > 0 else 'Residential'
    meter_type = c10001_data['meter_type'].iloc[0] if len(c10001_data) > 0 else 'Analog'
    
    profile, p_created = UserProfile.objects.get_or_create(
        user=customer_user,
        defaults={
            'customer_id': 'C10001',
            'customer_type': cust_type,
            'location': loc,
            'meter_type': meter_type,
            'phone': '+962-79-123-4567',
            'address': 'Qasabat Irbid, Jordan'
        }
    )
    if p_created:
        logger.info("UserProfile for C10001 linked to user@example.com")
        
    # Seed profiles for the rest of the customer IDs in the dataset as placeholder records without logins
    unique_cust_ids = df['customer_id'].unique()
    for cid in unique_cust_ids:
        if cid == 'C10001':
            continue
            
        # Create a dummy user for each customer so they can technically login or be represented
        user_email = f"{cid.lower()}@example.com"
        u, u_created = User.objects.get_or_create(
            username=user_email,
            email=user_email
        )
        if u_created:
            u.set_password('customer123')
            u.save()
            
        cust_subset = df[df['customer_id'] == cid]
        UserProfile.objects.get_or_create(
            user=u,
            defaults={
                'customer_id': cid,
                'customer_type': cust_subset['customer_type'].iloc[0] if len(cust_subset) > 0 else 'Residential',
                'location': cust_subset['location'].iloc[0] if len(cust_subset) > 0 else 'Center',
                'meter_type': cust_subset['meter_type'].iloc[0] if len(cust_subset) > 0 else 'Analog',
                'phone': f'+962-79-{hash(cid)%10000000:07d}',
                'address': 'Irbid Governorate, Jordan'
            }
        )
        
    # 3. Seed Consumption Records (Bulk load to save database performance)
    # Check if we already have records to avoid duplicate keys
    existing_count = ConsumptionRecord.objects.count()
    if existing_count > 0:
        logger.info(f"Database already contains {existing_count} consumption records. Skipping CSV load.")
        return
        
    records_to_create = []
    
    # We clean duplicate lines or sort by date
    df_unique = df.drop_duplicates(subset=['customer_id', 'date'])
    
    for _, row in df_unique.iterrows():
        # Handle nan values in columns
        cons = row['consumption']
        if pd.isna(cons):
            cons = 0.0
            
        bill = row['billing_amount']
        if pd.isna(bill):
            bill = 5.0 + 1.5 * cons
            
        records_to_create.append(
            ConsumptionRecord(
                customer_id=row['customer_id'],
                date=datetime.strptime(row['date'], '%Y-%m-%d').date(),
                consumption=float(cons),
                billing_amount=float(bill),
                payment_status='Paid' if hash(row['customer_id'] + row['date']) % 10 > 2 else 'Unpaid'
            )
        )
        
    # Batch save in chunks of 500
    ConsumptionRecord.objects.bulk_create(records_to_create, batch_size=500)
    logger.info(f"Seeded {len(records_to_create)} consumption records into database.")

def import_uploaded_dataset(upload_id):
    """
    Imports and validates consumption data from a DatasetUpload model.
    Runs validation, inserts new records, and sets status.
    """
    upload = DatasetUpload.objects.get(id=upload_id)
    upload.upload_status = 'Processing'
    upload.save()
    
    try:
        file_path = upload.file.path
        
        # Load CSV or XLSX
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel are supported.")
            
        # Clean columns: strip spaces and standard casing
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Ingestion validation
        # Map potential variations of fields
        col_mappings = {
            'customerid': 'customer_id',
            'cust_id': 'customer_id',
            'id': 'customer_id',
            'consumption_volume': 'consumption',
            'consumption_value': 'consumption',
            'bill': 'billing_amount',
            'billing': 'billing_amount',
            'amount': 'billing_amount',
            'label': 'fraud_class',
            'class': 'fraud_class',
            'is_fraud': 'fraud_class'
        }
        df = df.rename(columns=col_mappings)
        
        required = {'customer_id', 'date', 'consumption'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in dataset: {', '.join(missing)}")
            
        # Deduplicate uploaded file
        df = df.drop_duplicates(subset=['customer_id', 'date'])
        
        # Seed records into the database (insert or update)
        records_to_create = []
        updated_count = 0
        
        for _, row in df.iterrows():
            cid = str(row['customer_id']).strip()
            date_str = str(row['date']).split()[0] # YYYY-MM-DD
            try:
                date_val = pd.to_datetime(date_str).date()
            except Exception:
                continue # Skip invalid dates
                
            cons = pd.to_numeric(row['consumption'], errors='coerce')
            if pd.isna(cons):
                cons = 0.0
                
            bill = pd.to_numeric(row.get('billing_amount', 5.0 + 1.5 * cons), errors='coerce')
            if pd.isna(bill):
                bill = 5.0 + 1.5 * cons
                
            # If a record already exists, update it, otherwise stage for bulk create
            existing = ConsumptionRecord.objects.filter(customer_id=cid, date=date_val).first()
            if existing:
                existing.consumption = float(cons)
                existing.billing_amount = float(bill)
                existing.save()
                updated_count += 1
            else:
                records_to_create.append(
                    ConsumptionRecord(
                        customer_id=cid,
                        date=date_val,
                        consumption=float(cons),
                        billing_amount=float(bill)
                    )
                )
                
        if records_to_create:
            ConsumptionRecord.objects.bulk_create(records_to_create, batch_size=500)
            
        # Create corresponding UserProfiles if they don't exist
        for cid in df['customer_id'].unique():
            cid_str = str(cid).strip()
            if not UserProfile.objects.filter(customer_id=cid_str).exists():
                user_email = f"{cid_str.lower()}@example.com"
                u, created = User.objects.get_or_create(username=user_email, email=user_email)
                if created:
                    u.set_password('customer123')
                    u.save()
                
                cust_subset = df[df['customer_id'] == cid]
                UserProfile.objects.create(
                    user=u,
                    customer_id=cid_str,
                    customer_type=cust_subset.get('customer_type', pd.Series(['Residential'])).iloc[0] or 'Residential',
                    location=cust_subset.get('location', pd.Series(['Center'])).iloc[0] or 'Center',
                    meter_type=cust_subset.get('meter_type', pd.Series(['Analog'])).iloc[0] or 'Analog',
                    phone='',
                    address=''
                )
                
        upload.row_count = len(df)
        upload.upload_status = 'Completed'
        upload.save()
        logger.info(f"Successfully processed upload ID {upload_id}. Rows: {len(df)}")
        return True, len(df), updated_count
        
    except Exception as e:
        upload.upload_status = 'Failed'
        upload.save()
        logger.error(f"Failed to process upload ID {upload_id}: {str(e)}")
        raise e

def run_model_training_pipeline(test_size=0.20, random_state=42):
    """
    Executes the full pipeline:
    1. Fetches all ConsumptionRecords from DB.
    2. Runs Preprocessing (Duplicates, Negatives, Missing Values, Outliers).
    3. Runs Feature Engineering.
    4. Splits and Scales features.
    5. Trains SVM and KNN.
    6. Evaluates and compares.
    7. Stores metrics in ModelRun database.
    """
    # 1. Fetch records
    qs = ConsumptionRecord.objects.all()
    if not qs.exists():
        raise ValueError("No consumption records found in database. Please seed or upload a dataset first.")
        
    records = []
    # Join with UserProfile meta to enrich features if available
    profiles = {p.customer_id: p for p in UserProfile.objects.all()}
    
    for r in qs:
        prof = profiles.get(r.customer_id)
        records.append({
            'customer_id': r.customer_id,
            'date': r.date.strftime('%Y-%m-%d'),
            'consumption': r.consumption,
            'billing_amount': float(r.billing_amount),
            'customer_type': prof.customer_type if prof else 'Residential',
            'meter_type': prof.meter_type if prof else 'Analog',
            'location': prof.location if prof else 'Center',
            # Include ground-truth labels if available (e.g. if we have confirmed fraud labels)
            # Check if there is an active confirmed fraud investigation
            'fraud_class': 1 if Investigation.objects.filter(customer_id=r.customer_id, status='Confirmed Fraud').exists() else 0
        })
        
    df_raw = pd.DataFrame(records)
    
    # Check if a custom CSV label exists (if uploaded dataset contains fraud labels, use that instead of fallback zeros)
    # If the database doesn't have any confirmed frauds yet, check if there was a dataset upload with label columns.
    # To be extremely helpful, we will fallback to checking if any customer was labeled in the raw file or if we have
    # seed fraud_class values. Since we generated 10% fraud in scripts/generate_demo_data.py, let's load those labels!
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'sample_water_consumption.csv')
    if os.path.exists(csv_path):
        df_labels = pd.read_csv(csv_path, usecols=['customer_id', 'date', 'fraud_class']).drop_duplicates()
        # Merge labeled fraud_class into our database dataframe on customer_id
        df_labels_cust = df_labels.groupby('customer_id')['fraud_class'].max().to_dict()
        df_raw['fraud_class'] = df_raw['customer_id'].map(df_labels_cust).fillna(0).astype(int)
        
    # 2. Preprocess & Clean
    cleaned_df, cleaning_stats = clean_dataset(df_raw)
    
    # 3. Feature Engineering
    features_df, feature_cols = engineer_features(cleaned_df)
    
    # 4. Prepare splits and scaling
    split_data = prepare_training_data(features_df, feature_cols, test_size=test_size, random_state=random_state)
    
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    scaler = split_data['scaler']
    
    # 5. Train SVM
    svm_model, svm_time = train_svm_model(X_train, y_train)
    
    # 6. Train KNN
    knn_model, knn_time = train_knn_model(X_train, y_train)
    
    # 7. Save model artifacts
    save_model_artifacts(svm_model, knn_model, scaler, feature_cols, save_dir=os.path.join(settings.BASE_DIR, 'models'))
    
    # 8. Evaluate SVM
    svm_preds = svm_model.predict(X_test)
    svm_prob = svm_model.predict_proba(X_test)[:, 1] if hasattr(svm_model, 'predict_proba') else None
    svm_eval = evaluate_predictions(y_test, svm_preds, svm_prob)
    
    # 9. Evaluate KNN
    knn_preds = knn_model.predict(X_test)
    knn_prob = knn_model.predict_proba(X_test)[:, 1] if hasattr(knn_model, 'predict_proba') else None
    knn_eval = evaluate_predictions(y_test, knn_preds, knn_prob)
    
    # 10. Record ModelRuns in database
    run_svm = ModelRun.objects.create(
        model_name='SVM',
        accuracy=svm_eval['accuracy'],
        precision=svm_eval['precision'],
        recall=svm_eval['recall'],
        f1=svm_eval['f1'],
        roc_auc=svm_eval['roc_auc'],
        training_time=svm_time,
        dataset_size=len(features_df)
    )
    
    run_knn = ModelRun.objects.create(
        model_name='KNN',
        accuracy=knn_eval['accuracy'],
        precision=knn_eval['precision'],
        recall=knn_eval['recall'],
        f1=knn_eval['f1'],
        roc_auc=knn_eval['roc_auc'],
        training_time=knn_time,
        dataset_size=len(features_df)
    )
    
    # 11. Generate static visualization curves for dashboards
    # Save them in static/media/ (Skip if running tests to optimize performance)
    import sys
    is_testing = 'test' in sys.argv
    if not is_testing:
        static_media_dir = os.path.join(settings.BASE_DIR, 'media', 'charts')
        os.makedirs(static_media_dir, exist_ok=True)
        
        cm_path = os.path.join(static_media_dir, 'svm_confusion_matrix.png')
        plot_confusion_matrix(
            svm_eval['confusion_matrix']['tn'],
            svm_eval['confusion_matrix']['fp'],
            svm_eval['confusion_matrix']['fn'],
            svm_eval['confusion_matrix']['tp'],
            cm_path
        )
        
        comp_path = os.path.join(static_media_dir, 'model_comparison.png')
        plot_model_comparison(svm_eval, knn_eval, comp_path)
    
    comparison = compare_models(svm_eval, knn_eval)
    
    return {
        'success': True,
        'svm_metrics': svm_eval,
        'knn_metrics': knn_eval,
        'comparison': comparison,
        'features_used': feature_cols,
        'dataset_size': len(features_df)
    }

def execute_fraud_predictions(model_name='SVM'):
    """
    Uses trained model artifacts (scaler & model) to score all active customers.
    Stores results in the Prediction table and initiates active Investigation alerts.
    """
    qs = ConsumptionRecord.objects.all()
    if not qs.exists():
        raise ValueError("No consumption records available for scoring.")
        
    records = []
    profiles = {p.customer_id: p for p in UserProfile.objects.all()}
    
    for r in qs:
        prof = profiles.get(r.customer_id)
        records.append({
            'customer_id': r.customer_id,
            'date': r.date.strftime('%Y-%m-%d'),
            'consumption': r.consumption,
            'billing_amount': float(r.billing_amount),
            'customer_type': prof.customer_type if prof else 'Residential',
            'meter_type': prof.meter_type if prof else 'Analog',
            'location': prof.location if prof else 'Center',
        })
        
    df_raw = pd.DataFrame(records)
    
    # Clean & engineer features
    cleaned_df, _ = clean_dataset(df_raw)
    features_df, _ = engineer_features(cleaned_df)
    
    # Score anomalies
    model_dir = os.path.join(settings.BASE_DIR, 'models')
    pred_results = predict_anomalies(features_df, model_dir=model_dir, model_name=model_name)
    
    # Save/Update predictions in DB
    predictions_created = 0
    investigations_triggered = 0
    
    for _, row in pred_results.iterrows():
        cid = row['customer_id']
        pred_class = row['predicted_class']
        prob = row['probability']
        risk = row['risk_level']
        indicators = json.dumps(row['indicators'])
        
        # Create or update prediction entry
        Prediction.objects.create(
            customer_id=cid,
            model_name=model_name,
            predicted_class=pred_class,
            probability=prob,
            risk_level=risk,
            key_indicators=indicators
        )
        predictions_created += 1
        
        # If model flag is Suspicious/Fraud, register it for investigation if not already present
        if pred_class == 1:
            inv, created = Investigation.objects.get_or_create(
                customer_id=cid,
                defaults={'status': 'Pending'}
            )
            if created:
                investigations_triggered += 1
                
    logger.info(f"Generated {predictions_created} predictions using {model_name}. Triggered {investigations_triggered} new alerts.")
    return predictions_created, investigations_triggered
