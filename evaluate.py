import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, r2_score, classification_report
)

RESULTS_DIR = "results"
CLS_SUMMARY = os.path.join(RESULTS_DIR, "model_summary_classification.csv")
REG_SUMMARY = os.path.join(RESULTS_DIR, "model_summary_regression.csv")


def detect_columns(df: pd.DataFrame):
    """
    Detects columns for classification and regression across different result file formats.
    Returns a dict with any of: y_true, y_pred, y_proba, y_true_hw, y_pred_hw, y_true_temp, y_pred_temp, thr_used.
    """
    cols = {c.lower(): c for c in df.columns}
    found = {}

    # Classification (direct)
    for k in ["y_true", "y_heatwave", "true", "actual"]:
        if k in cols:
            found["y_true"] = cols[k]; break
    for k in ["y_pred", "pred_heatwave", "pred", "prediction"]:
        if k in cols:
            found["y_pred"] = cols[k]; break
    for k in ["y_proba", "proba_heatwave", "proba", "probability"]:
        if k in cols:
            found["y_proba"] = cols[k]; break

    # Classification (ML7-style derived)
    if "y_true" not in found:
        if "y_true_hw" in cols: found["y_true_hw"] = cols["y_true_hw"]
    if "y_pred" not in found:
        if "y_pred_hw" in cols: found["y_pred_hw"] = cols["y_pred_hw"]

    # Regression (ML7)
    if "y_true_temp" in cols: found["y_true_temp"] = cols["y_true_temp"]
    if "y_pred_temp" in cols: found["y_pred_temp"] = cols["y_pred_temp"]
    if "thr_used" in cols:     found["thr_used"]     = cols["thr_used"]

    return found


def compute_classification_metrics(df: pd.DataFrame, colmap: dict):
    """
    Compute Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix.
    Works with either (y_true,y_pred[,y_proba]) or (y_true_hw,y_pred_hw[,y_pred_temp,thr_used]).
    Returns dict of metrics or None if insufficient columns.
    """
    # Prefer direct columns
    if "y_true" in colmap and "y_pred" in colmap:
        y_true = df[colmap["y_true"]].astype(int).values
        y_pred = df[colmap["y_pred"]].astype(int).values
        # optional probability/score
        y_proba = None
        if "y_proba" in colmap:
            y_proba = df[colmap["y_proba"]].astype(float).values

    # Else try ML7-style
    elif "y_true_hw" in colmap and "y_pred_hw" in colmap:
        y_true = df[colmap["y_true_hw"]].astype(int).values
        y_pred = df[colmap["y_pred_hw"]].astype(int).values
        # Optional ROC score from temperature distance above threshold
        y_proba = None
        if "y_pred_temp" in colmap and "thr_used" in colmap:
            score = df[colmap["y_pred_temp"]].astype(float).values - df[colmap["thr_used"]].astype(float).values
            # min-max scale to [0,1] for ROC-AUC stability
            smin, smax = float(np.min(score)), float(np.max(score))
            y_proba = (score - smin) / (smax - smin + 1e-9)
    else:
        return None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    except Exception:
        roc = np.nan
    cm = confusion_matrix(y_true, y_pred)

    return dict(
        Accuracy=acc, Precision=prec, Recall=rec, F1=f1, ROC_AUC=roc,
        Confusion_Matrix=cm
    )


def compute_regression_metrics(df: pd.DataFrame, colmap: dict):
    """
    Compute MAE and R² if y_true_temp and y_pred_temp are present (e.g., ML7).
    """
    if "y_true_temp" in colmap and "y_pred_temp" in colmap:
        y_true = df[colmap["y_true_temp"]].astype(float).values
        y_pred = df[colmap["y_pred_temp"]].astype(float).values
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return dict(MAE=mae, R2=r2)
    return None


def main():
    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    cls_rows, reg_rows = [], []

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    files.sort()

    if not files:
        print("No CSV files found in 'results/'.")
        return

    print("="*80)
    print("Evaluating all model outputs in 'results/'")
    print("="*80)

    for fname in files:
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"⚠️  Skipping {fname}: cannot read CSV ({e})")
            continue

        colmap = detect_columns(df)
        model_name = fname.replace(".csv", "")

        # Classification
        cls_metrics = compute_classification_metrics(df, colmap)
        if cls_metrics is not None:
            print("\n" + "-"*80)
            print(f"📊 Classification — {model_name}")
            print("-"*80)
            print(f"Accuracy:  {cls_metrics['Accuracy']:.4f}")
            print(f"Precision: {cls_metrics['Precision']:.4f}")
            print(f"Recall:    {cls_metrics['Recall']:.4f}")
            print(f"F1 Score:  {cls_metrics['F1']:.4f}")
            roc = cls_metrics['ROC_AUC']
            print(f"ROC AUC:   {('%.4f' % roc) if not np.isnan(roc) else 'NA'}")
            print("Confusion Matrix:\n", cls_metrics["Confusion_Matrix"])
            cls_rows.append(dict(
                Model=model_name,
                Accuracy=cls_metrics['Accuracy'],
                Precision=cls_metrics['Precision'],
                Recall=cls_metrics['Recall'],
                F1=cls_metrics['F1'],
                ROC_AUC=(np.nan if np.isnan(roc) else float(roc))
            ))

        # Regression
        reg_metrics = compute_regression_metrics(df, colmap)
        if reg_metrics is not None:
            print("\n" + "-"*80)
            print(f"📈 Regression — {model_name}")
            print("-"*80)
            print(f"MAE (°C):  {reg_metrics['MAE']:.3f}")
            print(f"R²:        {reg_metrics['R2']:.3f}")
            reg_rows.append(dict(
                Model=model_name,
                MAE=reg_metrics['MAE'],
                R2=reg_metrics['R2']
            ))

    # Save summaries
    if cls_rows:
        cls_df = pd.DataFrame(cls_rows).sort_values("Accuracy", ascending=False)
        cls_df.to_csv(CLS_SUMMARY, index=False)
        print("\n✅ Saved classification summary:", CLS_SUMMARY)
        print("\n🏆 Classification Summary:")
        print(cls_df.to_string(index=False))
    else:
        print("\nℹ️ No classification-compatible files found.")

    if reg_rows:
        reg_df = pd.DataFrame(reg_rows).sort_values("MAE", ascending=True)
        reg_df.to_csv(REG_SUMMARY, index=False)
        print("\n✅ Saved regression summary:", REG_SUMMARY)
        print("\n🏆 Regression Summary:")
        print(reg_df.to_string(index=False))
    else:
        print("\nℹ️ No regression-compatible files found.")


if __name__ == "__main__":
    main()
