import os
import json
import time
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

def prepare_training_data(features_df, feature_cols, test_size=0.20, random_state=42):
    """
    Splits features_df into train and test sets using stratified split based on fraud_class.
    Fits the StandardScaler on training data and transforms both train and test.
    Saves and returns standard datasets, scaler, and splits.
    """
    X = features_df[feature_cols].copy()
    
    # Handle NaN values inside features if any (though features.py should prevent them)
    X = X.fillna(0.0)
    
    y = features_df['fraud_class'].copy().astype(int)
    
    # Stratified train/test split
    # If there's only 1 class in the training data, we fallback to non-stratified
    unique_classes = y.nunique()
    if unique_classes > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        logger.warning("Only one class found in target label. Performing non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    # Standardize features (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column context
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    return {
        'X_train': X_train_scaled_df,
        'X_test': X_test_scaled_df,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }

def train_svm_model(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', random_state=42):
    """Trains an SVM model and logs training duration."""
    start_time = time.time()
    
    # SVC with class_weight='balanced' and probability=True
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight='balanced',
        probability=True,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    
    logger.info(f"SVM training complete in {duration:.4f}s")
    return model, duration

def train_knn_model(X_train, y_train, n_neighbors=5, weights='distance', metric='minkowski'):
    """Trains a KNN model and logs training duration."""
    start_time = time.time()
    
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    
    logger.info(f"KNN training complete in {duration:.4f}s")
    return model, duration

def save_model_artifacts(svm_model, knn_model, scaler, feature_cols, save_dir='models'):
    """Saves models, scaler, and feature list to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    svm_path = os.path.join(save_dir, 'svm_model.joblib')
    knn_path = os.path.join(save_dir, 'knn_model.joblib')
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    meta_path = os.path.join(save_dir, 'feature_metadata.json')
    
    joblib.dump(svm_model, svm_path)
    joblib.dump(knn_model, knn_path)
    joblib.dump(scaler, scaler_path)
    
    meta = {
        'feature_columns': list(feature_cols),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
        
    logger.info(f"Saved all ML artifacts to directory: {save_dir}")
    return {
        'svm_path': svm_path,
        'knn_path': knn_path,
        'scaler_path': scaler_path,
        'meta_path': meta_path
    }
