import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import logging

logger = logging.getLogger(__name__)

def calculate_slope(y):
    """Fits a simple linear regression to calculate the consumption trend slope."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    # Simple linear fit: y = mx + c. Returns m.
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except Exception:
        return 0.0

def engineer_features(df):
    """
    Groups clean transaction consumption data by customer_id and engineers features.
    Handles optional columns gracefully.
    Returns: (features_df, feature_cols)
    """
    df = df.copy().sort_values(by=['customer_id', 'date']).reset_index(drop=True)
    
    # Pre-calculate global or location-level statistics for comparative features
    has_location = 'location' in df.columns
    has_type = 'customer_type' in df.columns
    has_billing = 'billing_amount' in df.columns
    has_labels = 'fraud_class' in df.columns
    
    # Calculate group averages for comparison if columns are present
    location_avg = {}
    if has_location:
        location_avg = df.groupby('location')['consumption'].mean().to_dict()
        
    type_avg = {}
    if has_type:
        type_avg = df.groupby('customer_type')['consumption'].mean().to_dict()
        
    global_avg = df['consumption'].mean()
    if pd.isna(global_avg):
        global_avg = 0.0
        
    customer_features = []
    
    # Group by customer
    grouped = df.groupby('customer_id')
    
    for customer_id, group in grouped:
        consumption_series = group['consumption'].values
        
        # 1. Base Statistical Features
        n_readings = len(consumption_series)
        mean_val = float(np.mean(consumption_series))
        std_val = float(np.std(consumption_series))
        var_val = float(np.var(consumption_series))
        min_val = float(np.min(consumption_series))
        max_val = float(np.max(consumption_series))
        median_val = float(np.median(consumption_series))
        
        # Coefficient of variation (CV = std / mean)
        cv_val = std_val / mean_val if mean_val > 0.0 else 0.0
        
        # Skewness and Kurtosis
        skew_val = float(skew(consumption_series)) if len(consumption_series) >= 3 else 0.0
        if pd.isna(skew_val):
            skew_val = 0.0
        kurt_val = float(kurtosis(consumption_series)) if len(consumption_series) >= 3 else 0.0
        if pd.isna(kurt_val):
            kurt_val = 0.0
            
        # 2. Rolling/Window Averages (last 3, 6, 12 months)
        mean_3m = float(np.mean(consumption_series[-3:])) if n_readings >= 3 else mean_val
        mean_6m = float(np.mean(consumption_series[-6:])) if n_readings >= 6 else mean_val
        mean_12m = float(np.mean(consumption_series[-12:])) if n_readings >= 12 else mean_val
        
        # 3. Temporal Features
        # Trend (slope over the whole period)
        trend_slope = calculate_slope(consumption_series)
        
        # Sudden changes: compare recent 3 months with prior 9 months (or history)
        if n_readings >= 6:
            recent_mean = np.mean(consumption_series[-3:])
            prior_mean = np.mean(consumption_series[:-3])
            sudden_change_ratio = float(recent_mean / (prior_mean + 0.1))
        else:
            sudden_change_ratio = 1.0
            
        # Zero consumption count and ratio
        zero_count = int(np.sum(consumption_series == 0.0))
        zero_ratio = float(zero_count / n_readings) if n_readings > 0 else 0.0
        
        # Negative readings (meter resets) count from cleaning flags
        neg_count = int(group['negative_flag'].sum()) if 'negative_flag' in group.columns else 0
        neg_ratio = float(neg_count / n_readings) if n_readings > 0 else 0.0
        
        # Reading intervals (time delta consistency in days)
        dates_diff = group['date'].diff().dropna().dt.days
        interval_std = float(dates_diff.std()) if len(dates_diff) >= 2 else 0.0
        if pd.isna(interval_std):
            interval_std = 0.0
            
        # 4. Behavioral & Comparative Features
        cust_type = group['customer_type'].iloc[0] if has_type else 'Unknown'
        cust_loc = group['location'].iloc[0] if has_location else 'Unknown'
        cust_meter = group['meter_type'].iloc[0] if 'meter_type' in group.columns else 'Unknown'
        
        # Deviation from neighborhood / type average
        loc_mean = location_avg.get(cust_loc, global_avg)
        type_mean = type_avg.get(cust_type, global_avg)
        
        loc_deviation_ratio = mean_val / (loc_mean + 0.1)
        type_deviation_ratio = mean_val / (type_mean + 0.1)
        
        # 5. Billing & Payment Features (if billing exists)
        billing_mean = 0.0
        billing_to_consumption_mean = 0.0
        billing_to_consumption_std = 0.0
        
        if has_billing:
            billing_series = group['billing_amount'].values
            billing_mean = float(np.mean(billing_series))
            
            # Ratio of billing to consumption (detecting flat bills or under-billing anomalies)
            ratio_series = billing_series / (consumption_series + 0.1)
            billing_to_consumption_mean = float(np.mean(ratio_series))
            billing_to_consumption_std = float(np.std(ratio_series))
            if pd.isna(billing_to_consumption_std):
                billing_to_consumption_std = 0.0
                
        # 6. Target label extraction (class of customer: fraud vs normal)
        # Check if the customer has ANY fraud record in their history, or use the last record.
        # Since fraud is a customer label, we take the max (if any record is fraud, the customer is fraud).
        target_label = int(group['fraud_class'].max()) if has_labels else 0
        
        feat_dict = {
            'customer_id': customer_id,
            'customer_type': cust_type,
            'location': cust_loc,
            'meter_type': cust_meter,
            'n_readings': n_readings,
            'mean_consumption': mean_val,
            'std_consumption': std_val,
            'variance_consumption': var_val,
            'min_consumption': min_val,
            'max_consumption': max_val,
            'median_consumption': median_val,
            'coef_of_variation': cv_val,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'mean_3m': mean_3m,
            'mean_6m': mean_6m,
            'mean_12m': mean_12m,
            'trend_slope': trend_slope,
            'sudden_change_ratio': sudden_change_ratio,
            'zero_consumption_count': zero_count,
            'zero_consumption_ratio': zero_ratio,
            'negative_consumption_count': neg_count,
            'negative_consumption_ratio': neg_ratio,
            'reading_interval_std': interval_std,
            'location_deviation_ratio': loc_deviation_ratio,
            'type_deviation_ratio': type_deviation_ratio,
        }
        
        if has_billing:
            feat_dict['mean_billing'] = billing_mean
            feat_dict['billing_to_consumption_mean'] = billing_to_consumption_mean
            feat_dict['billing_to_consumption_std'] = billing_to_consumption_std
            
        if has_labels:
            feat_dict['fraud_class'] = target_label
            
        customer_features.append(feat_dict)
        
    features_df = pd.DataFrame(customer_features)
    
    # Determine the feature column list (exclude IDs, metadata, and target class)
    exclude_cols = {'customer_id', 'customer_type', 'location', 'meter_type', 'fraud_class'}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    return features_df, feature_cols
