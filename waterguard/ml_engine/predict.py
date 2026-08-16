import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def load_ml_artifacts(model_dir='models', model_name='SVM'):
    """Loads scaler, specific model, and feature metadata from disk."""
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    meta_path = os.path.join(model_dir, 'feature_metadata.json')
    
    if model_name.upper() == 'SVM':
        model_path = os.path.join(model_dir, 'svm_model.joblib')
    else:
        model_path = os.path.join(model_dir, 'knn_model.joblib')
        
    if not (os.path.exists(scaler_path) and os.path.exists(model_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Model artifacts not found in {model_dir}. Please train models first.")
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    return model, scaler, metadata['feature_columns']

def get_risk_level(prob, high_threshold=0.80, med_threshold=0.50):
    """Maps probability to LOW, MEDIUM, HIGH risk levels."""
    if prob >= high_threshold:
        return 'HIGH'
    elif prob >= med_threshold:
        return 'MEDIUM'
    else:
        return 'LOW'

def identify_risk_indicators(customer_row, features_df):
    """
    Identifies which features triggered the anomaly flag by comparing the 
    customer's features against overall population distribution.
    """
    indicators = []
    
    # 1. Negative Readings (Direct tamper signal)
    if customer_row.get('negative_consumption_count', 0) > 0:
        indicators.append("Negative consumption readings detected (potential meter reset)")
        
    # 2. Sudden consumption drop
    sudden_ratio = customer_row.get('sudden_change_ratio', 1.0)
    if sudden_ratio < 0.25 and customer_row.get('mean_consumption', 0.0) > 2.0:
        indicators.append(f"Sudden consumption drop: recent average is {sudden_ratio:.1%} of prior baseline")
        
    # 3. High variability (coefficient of variation CV)
    cv = customer_row.get('coef_of_variation', 0.0)
    pop_cv_mean = features_df['coef_of_variation'].mean()
    pop_cv_std = features_df['coef_of_variation'].std()
    # If customer's CV is more than 2 std devs above population mean
    if cv > (pop_cv_mean + 1.5 * pop_cv_std):
        indicators.append("High consumption variability (irregular usage patterns)")
        
    # 4. Zero consumption ratio
    zero_ratio = customer_row.get('zero_consumption_ratio', 0.0)
    if zero_ratio > 0.40:
        indicators.append(f"Frequent zero consumption periods ({zero_ratio:.1%} of readings)")
        
    # 5. Billing to consumption relationship (if billing column exists)
    if 'billing_to_consumption_mean' in customer_row:
        ratio = customer_row['billing_to_consumption_mean']
        pop_ratio_mean = features_df['billing_to_consumption_mean'].mean()
        # If billing per unit is unusually low (more than 50% below population average)
        if ratio < (pop_ratio_mean * 0.5):
            indicators.append("Unusually low billing relative to consumption volume (potential billing manipulation)")
            
    # 6. Deviation from neighborhood/type average
    loc_dev = customer_row.get('location_deviation_ratio', 1.0)
    if loc_dev < 0.20:
        indicators.append(f"Consumption is 80%+ lower than neighborhood average ({loc_dev:.1%})")
        
    # Fallback if no specific indicators triggered but model flagged them
    if not indicators:
        indicators.append("Statistical deviation from normal consumption baseline")
        
    return indicators

def predict_anomalies(features_df, model_dir='models', model_name='SVM', high_threshold=0.80, med_threshold=0.50):
    """
    Loads saved scaler and model, scales features, runs predictions and probability calculations.
    Returns: DataFrame containing customer ID, predicted class, probability, risk level, and indicators list.
    """
    model, scaler, feature_cols = load_ml_artifacts(model_dir, model_name)
    
    # Extract IDs
    customer_ids = features_df['customer_id'].values
    
    # Scale features
    X = features_df[feature_cols].copy().fillna(0.0)
    X_scaled = scaler.transform(X)
    
    # Run predictions
    predictions = model.predict(X_scaled)
    
    # Probabilities
    # SVM requires probability=True during initialization; KNN supports predict_proba inherently.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        # Fallback if probability estimation isn't supported (e.g. SVM decision function)
        decision_scores = model.decision_function(X_scaled)
        # Min-max scale or sigmoid scale to get pseudo-probability
        probabilities = 1.0 / (1.0 + np.exp(-decision_scores))
        
    results = []
    
    for idx, customer_id in enumerate(customer_ids):
        pred_class = int(predictions[idx])
        prob = float(probabilities[idx])
        
        # Risk level based on probability
        risk_level = get_risk_level(prob, high_threshold, med_threshold)
        
        # Risk indicators
        row_feat = features_df.iloc[idx].to_dict()
        indicators = []
        if pred_class == 1 or risk_level in ['HIGH', 'MEDIUM']:
            indicators = identify_risk_indicators(row_feat, features_df)
            
        results.append({
            'customer_id': customer_id,
            'predicted_class': pred_class,
            'probability': prob,
            'risk_level': risk_level,
            'indicators': indicators,
            'model_name': model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    return pd.DataFrame(results)
