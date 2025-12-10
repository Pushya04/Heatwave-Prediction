"""
Plotting utilities for ML models – generates comprehensive visualizations.
Corrected version (Fixed confusion matrix + fixed class report heatmaps)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# 1) CONFUSION MATRIX (FULLY FIXED VERSION)
# ---------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path, model_name):
    """Plot confusion matrix with raw counts only."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    # Plot without annotations first
    ax = sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=['No Heatwave', 'Heatwave'],
        yticklabels=['No Heatwave', 'Heatwave'],
        linewidths=3,
        linecolor='black',
        cbar_kws={'label': 'Count'}
    )

    # Manually add text annotations with proper colors
    # Determine threshold for text color (use median or middle value)
    threshold = cm.max() / 2.0
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Use white text for dark cells, black text for light cells
            text_color = 'white' if cm[i, j] > threshold else 'black'
            ax.text(j + 0.5, i + 0.5, f'{cm[i, j]:,}',
                   ha="center", va="center",
                   color=text_color,
                   fontsize=20, fontweight='bold')

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, fontweight='bold', color='black')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, fontweight='bold', color='black', rotation=0)

    plt.title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight='bold')
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')

    # Summary text under the plot
    total = cm.sum()
    acc = (cm[0,0] + cm[1,1]) / total

    plt.text(
        0.5,
        -0.15,
        f"Total Samples: {total:,} | Accuracy: {acc:.2%}",
        ha="center",
        fontsize=12,
        transform=plt.gca().transAxes,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "01_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# 2) ROC CURVE
# ---------------------------------------------------------
def plot_roc_curve(y_true, y_proba, save_path, model_name):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "02_roc_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Warning: ROC curve failed:", e)


# ---------------------------------------------------------
# 3) PRECISION–RECALL CURVE
# ---------------------------------------------------------
def plot_precision_recall_curve(y_true, y_proba, save_path, model_name):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve - {model_name}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "03_precision_recall_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Warning: PR curve failed:", e)


# ---------------------------------------------------------
# 4) CV FOLD BAR & BOX PLOTS
# ---------------------------------------------------------
def plot_cv_metrics_bar(fold_metrics, save_path, model_name):
    if len(fold_metrics) == 0:
        return

    metrics_array = np.array(fold_metrics)
    
    # Detect if we have 4 or 5 metrics (with or without ROC)
    n_metrics = metrics_array.shape[1]
    if n_metrics == 5:
        columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    else:
        columns = ['Accuracy', 'Precision', 'Recall', 'F1']

    df = pd.DataFrame(metrics_array, columns=columns)
    df['Fold'] = [f"Fold {i+1}" for i in range(len(fold_metrics))]
    df_melted = df.melt(id_vars=['Fold'], var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Fold', y='Score', hue='Metric')
    plt.title(f"Cross-Validation Metrics - {model_name}", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "04_cv_metrics_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_metrics_box(fold_metrics, save_path, model_name):
    if len(fold_metrics) == 0:
        return

    metrics_array = np.array(fold_metrics)
    
    # Detect if we have 4 or 5 metrics (with or without ROC)
    n_metrics = metrics_array.shape[1]
    if n_metrics == 5:
        columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    else:
        columns = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    df = pd.DataFrame(metrics_array, columns=columns)

    plt.figure(figsize=(10, 6))
    df.boxplot()
    plt.title(f"CV Metric Distribution - {model_name}", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "05_cv_metrics_box.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# 5) CLASSIFICATION REPORT HEATMAP (FIXED)
# ---------------------------------------------------------
def plot_classification_report_heatmap(y_true, y_pred, save_path, model_name):
    """Correctly plot class-wise precision/recall/F1 without transposing."""
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        target_names = ['No Heatwave', 'Heatwave']

        df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=target_names)

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            df,
            annot=False,
            cmap="YlGnBu",
            linewidths=2,
            linecolor="black"
        )

        # Manually add text annotations with proper colors
        # Determine threshold for text color based on data range
        data_min = df.min().min()
        data_max = df.max().max()
        threshold = (data_min + data_max) / 2.0
        
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                value = df.iloc[i, j]
                # Use white text for dark cells, black text for light cells
                text_color = 'white' if value > threshold else 'black'
                ax.text(j + 0.5, i + 0.5, f'{value:.4f}',
                       ha="center", va="center",
                       color=text_color,
                       fontsize=14, fontweight='bold')

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='bold', color='black')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold', color='black', rotation=0)

        plt.title(f"Classification Metrics by Class - {model_name}", fontsize=14, fontweight="bold")
        plt.ylabel("Class", fontsize=13, fontweight='bold')
        plt.xlabel("Metric", fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "06_classification_report_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Warning: Class heatmap failed:", e)


# ---------------------------------------------------------
# 6) PREDICTION PROBABILITY DISTRIBUTION
# ---------------------------------------------------------
def plot_prediction_distribution(y_proba, save_path, model_name):
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(y_proba, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(0.5, color='red', linestyle='--', label='Threshold 0.5')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(f"Probability Distribution - {model_name}", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "07_prediction_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Warning: Probability distribution failed:", e)


# ---------------------------------------------------------
# 7) FEATURE IMPORTANCE
# ---------------------------------------------------------
def plot_feature_importance(feature_importance, feature_names, save_path, model_name, top_n=15):
    try:
        idx = np.argsort(feature_importance)[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(idx)), feature_importance[idx], color='teal')
        plt.yticks(range(len(idx)), [feature_names[i] for i in idx], fontweight='bold')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "08_feature_importance.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Warning: Feature importance plot failed:", e)


# ---------------------------------------------------------
# 8) MASTER WRAPPER TO GENERATE ALL PLOTS
# ---------------------------------------------------------
def plot_all_classification(
    y_true,
    y_pred,
    y_proba,
    fold_metrics,
    save_path,
    model_name,
    feature_importance=None,
    feature_names=None
):
    ensure_dir(save_path)

    print(f"\n📊 Saving plots in: {save_path}")

    plot_confusion_matrix(y_true, y_pred, save_path, model_name)
    plot_roc_curve(y_true, y_proba, save_path, model_name)
    plot_precision_recall_curve(y_true, y_proba, save_path, model_name)
    plot_cv_metrics_bar(fold_metrics, save_path, model_name)
    plot_cv_metrics_box(fold_metrics, save_path, model_name)
    plot_classification_report_heatmap(y_true, y_pred, save_path, model_name)
    plot_prediction_distribution(y_proba, save_path, model_name)

    if feature_importance is not None and feature_names is not None:
        plot_feature_importance(feature_importance, feature_names, save_path, model_name)

    print(f"✅ Saved {len([f for f in os.listdir(save_path) if f.endswith('.png')])} images.")
