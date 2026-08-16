import os
import numpy as np

# Set cohesive professional styling palette (Teal and Dark Navy/Blue)
PRIMARY_TEAL = '#008080'
SECONDARY_NAVY = '#1e293b'
ACCENT_BLUE = '#0284c7'
BG_LIGHT = '#f8fafc'
PALETTE = [PRIMARY_TEAL, SECONDARY_NAVY, ACCENT_BLUE, '#f59e0b', '#ef4444']

def _setup_styles():
    """Initializes matplotlib backend and settings dynamically inside drawing calls."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.facecolor': BG_LIGHT,
        'axes.facecolor': BG_LIGHT,
        'text.color': SECONDARY_NAVY,
        'axes.labelcolor': SECONDARY_NAVY,
        'xtick.color': SECONDARY_NAVY,
        'ytick.color': SECONDARY_NAVY,
        'font.family': 'sans-serif'
    })
    return plt, sns

def plot_confusion_matrix(tn, fp, fn, tp, save_path):
    """Generates a professional 2x2 confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt, sns = _setup_styles()
    
    cm = np.array([[tn, fp], [fn, tp]])
    group_names = ['True Neg\n(Normal)', 'False Pos\n(Flagged Normal)', 'False Neg\n(Missed Fraud)', 'True Pos\n(Flagged Fraud)']
    group_counts = [f"{value:d}" for value in cm.flatten()]
    
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap=sns.light_palette(PRIMARY_TEAL, as_cmap=True), 
        cbar=False, 
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    plt.title("Confusion Matrix (Model Predictions)", fontsize=13, pad=15, weight='bold')
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
    plt.ylabel("Actual Class", fontsize=11, labelpad=10)
    plt.xticks([0.5, 1.5], ['Normal', 'Suspicious/Fraud'])
    plt.yticks([0.5, 1.5], ['Normal', 'Suspicious/Fraud'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=BG_LIGHT)
    plt.close()
    return save_path

def plot_model_comparison(svm_metrics, knn_metrics, save_path):
    """Plots accuracy, precision, recall, and f1 score comparison for SVM vs KNN."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt, sns = _setup_styles()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    svm_vals = [svm_metrics[m] * 100 for m in metrics]
    knn_vals = [knn_metrics[m] * 100 for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, svm_vals, width, label='SVM', color=PRIMARY_TEAL)
    rects2 = ax.bar(x + width/2, knn_vals, width, label='KNN', color=SECONDARY_NAVY)
    
    ax.set_ylabel('Score (%)', fontsize=11, weight='bold')
    ax.set_title('Model Performance Comparison', fontsize=13, pad=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(frameon=True, facecolor=BG_LIGHT)
    ax.set_ylim(0, 105)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=BG_LIGHT)
    plt.close()
    return save_path

def plot_roc_curve(y_true, svm_prob, knn_prob, save_path):
    """Plots ROC curves for both models."""
    from sklearn.metrics import roc_curve, auc
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt, _ = _setup_styles()
    
    plt.figure(figsize=(6, 5))
    
    # SVM ROC
    if svm_prob is not None:
        fpr_svm, tpr_svm, _ = roc_curve(y_true, svm_prob)
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        plt.plot(fpr_svm, tpr_svm, color=PRIMARY_TEAL, lw=2, label=f'SVM ROC (AUC = {roc_auc_svm:.2f})')
        
    # KNN ROC
    if knn_prob is not None:
        fpr_knn, tpr_knn, _ = roc_curve(y_true, knn_prob)
        roc_auc_knn = auc(fpr_knn, tpr_knn)
        plt.plot(fpr_knn, tpr_knn, color=SECONDARY_NAVY, lw=2, label=f'KNN ROC (AUC = {roc_auc_knn:.2f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', labelpad=10)
    plt.ylabel('True Positive Rate', labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC)', pad=15, weight='bold')
    plt.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=BG_LIGHT)
    plt.close()
    return save_path
