import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, date

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.conf import settings

from core.models import UserProfile, ConsumptionRecord, Prediction, Investigation, ModelRun, BlockedCustomer
from core.forms import DatasetUploadForm, FeedbackForm, InvestigationForm
from core import services
from ml_engine.preprocessing import validate_dataset, clean_dataset
from ml_engine.features import engineer_features, calculate_slope
from ml_engine.predict import get_risk_level, identify_risk_indicators

class MachineLearningUnitTests(TestCase):
    """Unit tests for the standalone ML preprocessing, feature engineering, and predict logics."""

    def test_dataset_validation(self):
        # Good CSV structure
        df_good = pd.DataFrame({
            'customer_id': ['C101', 'C101'],
            'date': ['2018-01-01', '2018-02-01'],
            'consumption': [10.5, 12.0],
            'billing_amount': [20.75, 23.00],
            'fraud_class': [0, 0]
        })
        val_good = validate_dataset(df_good)
        self.assertTrue(val_good['valid'])
        self.assertEqual(val_good['num_customers'], 1)
        self.assertEqual(val_good['num_duplicates'], 0)

        # Missing required columns
        df_bad = pd.DataFrame({
            'cust_id': ['C101'],
            'consumption': [10.5]
        })
        val_bad = validate_dataset(df_bad)
        self.assertFalse(val_bad['valid'])
        self.assertIn('Missing required columns', val_bad['error'])

    def test_data_cleaning_pipeline(self):
        # Create dirty mock dataframe
        df_dirty = pd.DataFrame({
            'customer_id': ['C101', 'C101', 'C101', 'C101', 'C102', 'C102', 'C102'],
            'date': ['2018-01-01', '2018-01-01', '2018-02-01', '2018-03-01', '2018-01-01', '2018-02-01', '2018-03-01'], # duplicate date
            'consumption': [10.0, 10.0, np.nan, -5.0, 0.0, 0.0, 0.0], # duplicate, NaN, negative, all-zero customer
            'billing_amount': [15.0, 15.0, np.nan, 5.0, 5.0, 5.0, 5.0]
        })
        
        cleaned_df, stats = clean_dataset(df_dirty, min_active_months=2)
        
        # Deduplication check (initial 7 -> 6 after dedup)
        # C102 is excluded because total consumption sum is 0 (all-zero filter)
        # C101 has 3 valid records, negative consumption converted to positive (abs), NaN interpolated
        self.assertEqual(cleaned_df['customer_id'].nunique(), 1)
        self.assertNotIn('C102', cleaned_df['customer_id'].unique())
        
        # Check negative flagging
        # Index 2 of C101 is absolute value of -5.0 = 5.0
        c101_cons = cleaned_df[cleaned_df['customer_id'] == 'C101']['consumption'].values
        self.assertTrue(all(val >= 0 for val in c101_cons))
        self.assertEqual(stats['negative_readings_handled'], 1)
        self.assertEqual(stats['missing_readings_imputed'], 1)

    def test_slope_calculation(self):
        # Steady upward trend
        self.assertAlmostEqual(calculate_slope([1, 2, 3, 4, 5]), 1.0)
        # Steady downward trend
        self.assertAlmostEqual(calculate_slope([5, 4, 3, 2, 1]), -1.0)
        # Single element check
        self.assertEqual(calculate_slope([5]), 0.0)

    def test_feature_engineering_outputs(self):
        # 1 customer with 4 months timeline
        df_clean = pd.DataFrame({
            'customer_id': ['C101'] * 5,
            'date': pd.to_datetime(['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01']),
            'consumption': [10.0, 12.0, 11.0, 9.0, 8.0],
            'billing_amount': [20.0, 23.0, 22.0, 19.5, 17.0],
            'negative_flag': [0, 0, 0, 0, 0],
            'customer_type': ['Residential'] * 5,
            'location': ['Center'] * 5,
            'meter_type': ['Analog'] * 5,
            'fraud_class': [0] * 5
        })
        
        features, feature_cols = engineer_features(df_clean)
        self.assertEqual(len(features), 1)
        self.assertIn('mean_consumption', feature_cols)
        self.assertIn('coef_of_variation', feature_cols)
        self.assertIn('trend_slope', feature_cols)
        self.assertIn('mean_billing', features.columns)
        self.assertEqual(features['fraud_class'].iloc[0], 0)

    def test_risk_level_mapping_and_indicators(self):
        # High Risk mapping
        self.assertEqual(get_risk_level(0.85), 'HIGH')
        # Med Risk mapping
        self.assertEqual(get_risk_level(0.65), 'MEDIUM')
        # Low Risk mapping
        self.assertEqual(get_risk_level(0.20), 'LOW')

        # Test indicators check
        pop_features = pd.DataFrame({
            'coef_of_variation': [0.1, 0.2, 0.15],
            'billing_to_consumption_mean': [1.5, 1.6, 1.45]
        })
        
        # Tampered customer record (has negative reading and very high CV)
        tamper_cust = {
            'negative_consumption_count': 2,
            'coef_of_variation': 1.8, # population is around 0.15
            'sudden_change_ratio': 0.1,
            'mean_consumption': 10.0
        }
        
        indicators = identify_risk_indicators(tamper_cust, pop_features)
        self.assertTrue(any("Negative consumption" in ind for ind in indicators))
        self.assertTrue(any("Sudden consumption drop" in ind for ind in indicators))


class DjangoIntegrationTests(TestCase):
    """Integration tests verifying database transactions, seeding services, and ML execution runs."""

    def setUp(self):
        # Create directories if missing
        os.makedirs(os.path.join(settings.BASE_DIR, 'data'), exist_ok=True)
        os.makedirs(os.path.join(settings.BASE_DIR, 'models'), exist_ok=True)
        
    def test_database_seeding_service(self):
        # Trigger seeding
        services.seed_demo_data()
        
        # Verify default users are created
        self.assertTrue(User.objects.filter(username='admin@example.com').exists())
        self.assertTrue(User.objects.filter(username='user@example.com').exists())
        
        # Verify profiles and records
        self.assertTrue(UserProfile.objects.filter(customer_id='C10001').exists())
        self.assertGreater(ConsumptionRecord.objects.count(), 0)

    def test_model_training_and_scoring_services(self):
        # Seed first
        services.seed_demo_data()
        
        # Train classifiers
        train_results = services.run_model_training_pipeline(test_size=0.20, random_state=42)
        self.assertIn('accuracy', train_results['svm_metrics'])
        self.assertIn('accuracy', train_results['knn_metrics'])
        
        # Verify ModelRuns are written
        self.assertEqual(ModelRun.objects.filter(model_name='SVM').count(), 1)
        self.assertEqual(ModelRun.objects.filter(model_name='KNN').count(), 1)
        
        # Verify serialized files exist
        self.assertTrue(os.path.exists(os.path.join(settings.BASE_DIR, 'models', 'svm_model.joblib')))
        self.assertTrue(os.path.exists(os.path.join(settings.BASE_DIR, 'models', 'knn_model.joblib')))
        self.assertTrue(os.path.exists(os.path.join(settings.BASE_DIR, 'models', 'scaler.joblib')))
        
        # Score anomalies using SVM
        preds, alerts = services.execute_fraud_predictions(model_name='SVM')
        self.assertGreater(preds, 0)
        
        # Verify Predictions are stored
        self.assertGreater(Prediction.objects.filter(model_name='SVM').count(), 0)


class DjangoFunctionalViewsTests(TestCase):
    """Functional tests checking view authorizations, logins, dashboard counts, and customer blocking."""

    def setUp(self):
        # Clear database and seed demo credentials
        services.seed_demo_data()
        self.client = Client()

    def test_unauthenticated_redirection(self):
        # Unauthenticated users accessing dashboards should redirect to login
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login/', response.url)

    def test_role_based_login_redirection(self):
        # 1. Admin login redirects to admin dashboard
        login_success = self.client.login(username='admin@example.com', password='admin123')
        self.assertTrue(login_success)
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('dashboard'))
        self.client.logout()

        # 2. Customer login redirects to user profile
        login_success = self.client.login(username='user@example.com', password='user123')
        self.assertTrue(login_success)
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('user_profile'))
        self.client.logout()

    def test_block_and_unblock_actions(self):
        # Login as Admin
        self.client.login(username='admin@example.com', password='admin123')
        
        # Ensure customer C10001 is active initially
        self.assertFalse(BlockedCustomer.objects.filter(customer_id='C10001', active=True).exists())
        
        # Block Customer
        response = self.client.post(reverse('toggle_block', args=['C10001']), {'reason': 'Tamper confirmed'})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(BlockedCustomer.objects.filter(customer_id='C10001', active=True).exists())

        # Unblock Customer
        response = self.client.post(reverse('toggle_block', args=['C10001']))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(BlockedCustomer.objects.filter(customer_id='C10001', active=True).exists())

    def test_admin_views_http_status(self):
        """Verify all major admin portal pages return a 200 OK status code."""
        self.client.login(username='admin@example.com', password='admin123')
        
        admin_urls = [
            'dashboard',
            'upload_dataset',
            'model_training',
            'predictions',
            'fraud_alerts',
            'graphs',
            'admin_feedback',
            'investigations',
            'reports',
            'customers',
        ]
        
        for url_name in admin_urls:
            response = self.client.get(reverse(url_name))
            self.assertEqual(response.status_code, 200, f"URL name '{url_name}' returned status {response.status_code}")
            
        # Test customer detail view specifically
        response = self.client.get(reverse('customer_detail', args=['C10001']))
        self.assertEqual(response.status_code, 200)
