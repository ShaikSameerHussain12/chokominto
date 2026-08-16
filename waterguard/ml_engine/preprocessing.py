import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_dataset(df):
    """
    Validates the uploaded dataset schema, columns, and types.
    Returns a dictionary of metadata and checking results.
    """
    required_cols = {'customer_id', 'date', 'consumption'}
    optional_cols = {'billing_amount', 'customer_type', 'meter_type', 'location', 'fraud_class'}
    
    current_cols = set(df.columns)
    missing_required = required_cols - current_cols
    
    if missing_required:
        return {
            'valid': False,
            'error': f"Missing required columns: {', '.join(missing_required)}"
        }
        
    # Standardize column types
    try:
        df_clean_check = df.copy()
        df_clean_check['date'] = pd.to_datetime(df_clean_check['date'])
        df_clean_check['consumption'] = pd.to_numeric(df_clean_check['consumption'], errors='coerce')
        if 'billing_amount' in df_clean_check.columns:
            df_clean_check['billing_amount'] = pd.to_numeric(df_clean_check['billing_amount'], errors='coerce')
    except Exception as e:
        return {
            'valid': False,
            'error': f"Data type conversion failed: {str(e)}"
        }
        
    num_rows = len(df)
    num_customers = df['customer_id'].nunique()
    missing_values = df[['customer_id', 'date', 'consumption']].isna().sum().to_dict()
    num_duplicates = df.duplicated(subset=['customer_id', 'date']).sum()
    
    # Class distribution if label exists
    class_distribution = {}
    if 'fraud_class' in df.columns:
        # Standardize fraud_class values
        # e.g., Yes/No, Fraud/Non-Fraud, 1/0
        df_clean_check['fraud_class'] = df_clean_check['fraud_class'].astype(str).str.upper().str.strip()
        df_clean_check['fraud_class'] = df_clean_check['fraud_class'].map({
            '1': 1, '1.0': 1, 'YES': 1, 'FRAUD': 1, 'TRUE': 1,
            '0': 0, '0.0': 0, 'NO': 0, 'NON-FRAUD': 0, 'FALSE': 0, 'NORMAL': 0
        }).fillna(0).astype(int)
        
        counts = df_clean_check['fraud_class'].value_counts().to_dict()
        class_distribution = {
            'normal': int(counts.get(0, 0)),
            'fraud': int(counts.get(1, 0))
        }
        
    min_date = df_clean_check['date'].min()
    max_date = df_clean_check['date'].max()
    
    # Statistics
    desc_stats = df_clean_check['consumption'].describe().to_dict()
    # Convert np types to native Python types for JSON serialization
    desc_stats = {k: float(v) if not pd.isna(v) else 0.0 for k, v in desc_stats.items()}
    
    return {
        'valid': True,
        'num_rows': num_rows,
        'num_customers': num_customers,
        'missing_values': missing_values,
        'num_duplicates': int(num_duplicates),
        'date_range': {
            'start': min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None,
            'end': max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None
        },
        'class_distribution': class_distribution,
        'statistics': desc_stats,
        'detected_columns': list(df.columns)
    }

def clean_dataset(df, min_active_months=3):
    """
    Cleans the raw water consumption dataframe:
    - Deduplicates records on (customer_id, date)
    - Standardizes column formats and dates
    - Flags and handles negative consumption (meter resets)
    - Imputes missing consumption values (interpolation & mean fallback)
    - Caps extreme keying error outliers
    - Excludes inactive or too-new customers (less than min_active_months history)
    Returns: (cleaned_df, cleaning_stats)
    """
    df = df.copy()
    
    # 0. Standardize names and dates
    df['date'] = pd.to_datetime(df['date'])
    df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
    if 'billing_amount' in df.columns:
        df['billing_amount'] = pd.to_numeric(df['billing_amount'], errors='coerce')
        
    # Standardize fraud_class if exists
    if 'fraud_class' in df.columns:
        df['fraud_class'] = df['fraud_class'].astype(str).str.upper().str.strip()
        df['fraud_class'] = df['fraud_class'].map({
            '1': 1, '1.0': 1, 'YES': 1, 'FRAUD': 1, 'TRUE': 1,
            '0': 0, '0.0': 0, 'NO': 0, 'NON-FRAUD': 0, 'FALSE': 0, 'NORMAL': 0
        }).fillna(0).astype(int)
        
    initial_rows = len(df)
    initial_duplicates = df.duplicated(subset=['customer_id', 'date']).sum()
    
    # 1. Remove exact duplicate periods
    # Keep the first reading or average them? Keeping first is standard.
    df = df.drop_duplicates(subset=['customer_id', 'date'], keep='first')
    rows_after_dedup = len(df)
    
    # 2. Flag and count negative consumption (meter resets indicators)
    # We create a column to track this, then absolute value it for numerical training
    df['negative_flag'] = (df['consumption'] < 0).astype(int)
    
    # Store information on how many negative consumption records exist
    total_negative_readings = df['negative_flag'].sum()
    # Replace negative values with their absolute value, since consumption magnitude matters
    # but the negative aspect is captured via the negative flag
    df['consumption'] = df['consumption'].abs()
    
    # 3. Handle Missing Values
    # Sort by customer and date first to enable interpolation
    df = df.sort_values(by=['customer_id', 'date']).reset_index(drop=True)
    
    total_missing_before = df['consumption'].isna().sum()
    
    # Group by customer_id and interpolate consumption
    # If a customer has all NaNs, interpolation does nothing, so we will fill with overall median as fallback.
    df['consumption'] = df.groupby('customer_id')['consumption'].transform(
        lambda group: group.interpolate(method='linear', limit_direction='both')
    )
    
    # Fallback fill with overall median if there are still NaNs
    remaining_nan = df['consumption'].isna().sum()
    if remaining_nan > 0:
        overall_median = df['consumption'].median()
        if pd.isna(overall_median):
            overall_median = 0.0
        df['consumption'] = df['consumption'].fillna(overall_median)
        
    # Same interpolation for billing_amount if it exists
    if 'billing_amount' in df.columns:
        df['billing_amount'] = df.groupby('customer_id')['billing_amount'].transform(
            lambda group: group.interpolate(method='linear', limit_direction='both')
        )
        # Fallback fill: base rate 5.0 + 1.5 * consumption
        df['billing_amount'] = df['billing_amount'].fillna(5.0 + 1.5 * df['consumption'])
        # Also ensure billing amount is non-negative
        df['billing_amount'] = df['billing_amount'].clip(lower=0.0)
        
    # 4. Outlier Handling: Cap extreme values
    # Check overall 99.9th percentile to catch obvious keying errors (e.g. multiplied by 10 or 100)
    # Don't eliminate normal high consumption, but cap anything above 99.9th percentile to 99th percentile of active consumption.
    q999 = df['consumption'].quantile(0.999)
    q99 = df['consumption'].quantile(0.99)
    
    outliers_capped = 0
    if pd.notna(q999) and pd.notna(q99):
        outlier_mask = df['consumption'] > q999
        outliers_capped = outlier_mask.sum()
        df.loc[outlier_mask, 'consumption'] = q99
        
    # 5. Customer Filtering (Minimum active history + Zero consumption filter)
    # Calculate history length (number of records) per customer
    history_counts = df['customer_id'].value_counts().to_dict()
    
    # Calculate sum of consumption per customer to identify completely inactive accounts
    inactive_customers = df.groupby('customer_id')['consumption'].sum()
    inactive_set = set(inactive_customers[inactive_customers == 0.0].index)
    
    # Filter customers
    def keep_customer(cust_id):
        # Must have sufficient history length AND not be completely inactive
        return history_counts.get(cust_id, 0) >= min_active_months and cust_id not in inactive_set
        
    keep_mask = df['customer_id'].apply(keep_customer)
    
    filtered_out_insufficient_history = len(df[~keep_mask]['customer_id'].unique())
    
    df_cleaned = df[keep_mask].copy().reset_index(drop=True)
    final_rows = len(df_cleaned)
    
    stats = {
        'initial_rows': int(initial_rows),
        'duplicates_removed': int(initial_duplicates),
        'rows_after_dedup': int(rows_after_dedup),
        'negative_readings_handled': int(total_negative_readings),
        'missing_readings_imputed': int(total_missing_before),
        'outliers_capped': int(outliers_capped),
        'customers_filtered_out': int(filtered_out_insufficient_history),
        'final_rows': int(final_rows),
        'final_customers': int(df_cleaned['customer_id'].nunique())
    }
    
    return df_cleaned, stats
