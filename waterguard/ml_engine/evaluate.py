import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Computes classification evaluation metrics for the predictions.
    Returns a dictionary of statistics.
    """
    # Force to integers
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # Ensure standard 2x2 shape even if one class is missing in test split
    tn, fp, fn, tp = 0, 0, 0, 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present
        present_class = y_true[0]
        val = cm[0, 0]
        if present_class == 0:
            tn = val
        else:
            tp = val
            
    # ROC AUC
    roc_auc = 0.5
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC: {str(e)}")
            
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    return metrics

def compare_models(svm_metrics, knn_metrics):
    """
    Compares SVM and KNN metrics and determines the best model based on F1-score (or Recall).
    F1-score is preferred for highly imbalanced fraud-detection datasets.
    """
    # Determine best model based on F1-score
    f1_svm = svm_metrics['f1']
    f1_knn = knn_metrics['f1']
    
    if f1_svm >= f1_knn:
        best_model = 'SVM'
        diff = f1_svm - f1_knn
    else:
        best_model = 'KNN'
        diff = f1_knn - f1_svm
        
    return {
        'best_model': best_model,
        'metric_preferred': 'F1-Score',
        'svm_f1': f1_svm,
        'knn_f1': f1_knn,
        'difference': float(diff)
    }
