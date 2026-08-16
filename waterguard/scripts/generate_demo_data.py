import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_customers=100, months=60, output_path='data/sample_water_consumption.csv'):
    np.random.seed(42)
    random.seed(42)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_date = datetime(2018, 1, 1)
    date_list = [start_date + timedelta(days=30 * i) for i in range(months)]
    date_strings = [d.strftime('%Y-%m-%d') for d in date_list]
    
    locations = ['Irbid_East', 'Irbid_West', 'Irbid_South', 'Irbid_North', 'Center']
    customer_types = ['Residential', 'Commercial', 'Industrial']
    meter_types = ['Analog', 'Digital', 'Smart']
    
    records = []
    
    # Define customer labels: ~90% normal, ~10% fraud/suspicious
    fraud_pct = 0.10
    num_fraud = int(num_customers * fraud_pct)
    fraud_indices = set(random.sample(range(num_customers), num_fraud))
    
    for cust_idx in range(num_customers):
        cust_id = f"C{cust_idx + 10001:05d}"
        is_fraud = 1 if cust_idx in fraud_indices else 0
        
        # Metadata
        cust_type = np.random.choice(customer_types, p=[0.75, 0.20, 0.05])
        loc = np.random.choice(locations)
        meter = np.random.choice(meter_types, p=[0.60, 0.30, 0.10])
        
        # Base consumption depending on customer type
        if cust_type == 'Industrial':
            base_cons = np.random.uniform(150, 300)
        elif cust_type == 'Commercial':
            base_cons = np.random.uniform(40, 100)
        else: # Residential
            base_cons = np.random.uniform(10, 35)
            
        # Determine fraud pattern style if fraudulent
        fraud_style = random.choice(['sudden_drop', 'high_variance', 'billing_mismatch', 'negative_resets', 'zero_consumption']) if is_fraud else 'none'
        
        # Let's say fraud starts after a random period (e.g., month 24 to 36)
        fraud_start_month = random.randint(20, 35)
        
        for m_idx, date_str in enumerate(date_strings):
            # Seasonal factor: high in summer (June-August, months 5, 6, 7 in 0-indexed year), low in winter
            month_num = (start_date + timedelta(days=30 * m_idx)).month
            # seasonality factor peaks at month 7 (July) and is lowest at month 1 (January)
            seasonal_factor = 1.0 + 0.35 * np.sin(2 * np.pi * (month_num - 4) / 12)
            
            # Normal baseline consumption
            cons = base_cons * seasonal_factor + np.random.normal(0, base_cons * 0.1)
            cons = max(0.1, cons) # Make sure it is positive for normal case
            
            # Apply fraud pattern if fraud is active
            if is_fraud and m_idx >= fraud_start_month:
                if fraud_style == 'sudden_drop':
                    # Consumption drops by 80% to 100%
                    cons = cons * np.random.uniform(0.0, 0.15)
                elif fraud_style == 'high_variance':
                    # Consumption fluctuates wildly
                    cons = cons * np.random.uniform(0.1, 2.5)
                elif fraud_style == 'billing_mismatch':
                    # Consumption is normal/high, but billing will be manipulated (handled below)
                    pass
                elif fraud_style == 'negative_resets':
                    # Meter resets result in negative/reset records
                    if random.random() < 0.2:
                        cons = -np.random.uniform(5, base_cons)
                elif fraud_style == 'zero_consumption':
                    # Active but zero consumption
                    cons = 0.0
            
            # Compute billing amount based on consumption
            # Base billing = $5 base + $1.5 per unit
            billing = 5.0 + 1.5 * cons
            
            # Mismatch fraud
            if is_fraud and m_idx >= fraud_start_month and fraud_style == 'billing_mismatch':
                # Billing is static and low, despite consumption
                billing = 5.0 + np.random.normal(0, 0.5)
                
            # Keep negative consumption values for fraud classification as requested (they flag resets)
            # but billing amount shouldn't be negative unless it's a credit, make it base fee at minimum
            if cons < 0:
                billing = 5.0
                
            records.append({
                'customer_id': cust_id,
                'date': date_str,
                'consumption': cons,
                'billing_amount': billing,
                'customer_type': cust_type,
                'meter_type': meter,
                'location': loc,
                'fraud_class': is_fraud
            })
            
    df = pd.DataFrame(records)
    
    # Introduce Noise: 1. Missing readings (NaN)
    missing_mask = np.random.rand(len(df)) < 0.02
    # Do not set fraud_class to NaN, only features
    df.loc[missing_mask, 'consumption'] = np.nan
    
    # Introduce Noise: 2. Outliers (e.g. keying errors, extremely high values)
    outlier_mask = np.random.rand(len(df)) < 0.005
    df.loc[outlier_mask, 'consumption'] = df.loc[outlier_mask, 'consumption'] * 10
    
    # Introduce Noise: 3. Duplicates (duplicate rows)
    dup_mask = np.random.rand(len(df)) < 0.01
    df_dups = df[dup_mask].copy()
    if len(df_dups) > 0:
        # Tweak date slightly or keep same to make exact duplicates
        df = pd.concat([df, df_dups], ignore_index=True)
        
    # Shuffle dataframe slightly while keeping customer records together mostly,
    # or just sort to make it messy then let cleanup sort it
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records for {num_customers} customers.")
    print(f"File saved to: {output_path}")
    print(f"Fraudulent customers: {num_customers - len(fraud_indices)} Normal, {len(fraud_indices)} Suspicious")

if __name__ == "__main__":
    generate_synthetic_data()
