# ml5_fast.py
import os, joblib, numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from ml_utils import load_and_prepare_data

BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR,  "ml5_fast_sampled.pkl")
PRED_PATH   = os.path.join(RESULTS_DIR, "ml5_fast_sampled_predictions.csv")


def main():
    print("=" * 70)
    print("ML5_FAST: MLPClassifier (FULL data, time-aware TEST)")
    print("=" * 70)

    # Use the same pipeline as ml1–ml4: time-aware train/test from ml_utils
    X_train, X_test, y_train, y_test, sw_train, feature_cols, test_df = load_and_prepare_data()
    print(f"Train rows: {X_train.shape[0]:,} | Test rows: {X_test.shape[0]:,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Positives (train): {y_train.sum():,}  ({100*y_train.mean():.2f}%)")
    print(f"Positives (test):  {y_test.sum():,}  ({100*y_test.mean():.2f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ===== 5-FOLD CROSS-VALIDATION =====
    from sklearn.model_selection import KFold
    print("\n----- 5-fold Cross-Validation -----")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_s), start=1):
        X_tr, X_val = X_train_s[tr_idx], X_train_s[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        model_fold = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        )
        model_fold.fit(X_tr, y_tr)
        y_pred_val = model_fold.predict(X_val)
        y_proba_val = model_fold.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred_val)
        prec = precision_score(y_val, y_pred_val, zero_division=0)
        rec = recall_score(y_val, y_pred_val, zero_division=0)
        f1 = f1_score(y_val, y_pred_val, zero_division=0)
        try:
            roc = roc_auc_score(y_val, y_proba_val)
        except Exception:
            roc = np.nan
        
        fold_metrics.append((acc, prec, rec, f1, roc))
        print(f"Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC={roc:.4f}")
    
    mets = np.array(fold_metrics)
    m_acc, m_prec, m_rec, m_f1, m_roc = mets.mean(axis=0)
    print("\nCV Averages over 5 folds:")
    print(f"Acc={m_acc:.4f}, Prec={m_prec:.4f}, Rec={m_rec:.4f}, F1={m_f1:.4f}, ROC={m_roc:.4f}")

    # ===== FINAL MODEL TRAINING =====
    print("\n----- Training Final Model on Full Train Set -----")
    # Simple, reasonably small MLP for fast run
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )

    model.fit(X_train_s, y_train)

    # --- TEST metrics (not validation) ---
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else np.nan
    cm   = confusion_matrix(y_test, y_pred)

    print("\n✅ TEST Results (heat-wave classification):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # --- Save model + predictions with SAME STRUCTURE as other fast models ---
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)

    out = test_df.copy()
    # ground-truth label column
    if "y_heatwave" not in out.columns:
        out["y_heatwave"] = y_test
    # prediction + probability (evaluate.py will pick these up via detect_columns)
    out["pred_heatwave"]   = y_pred
    out["proba_heatwave"]  = y_proba

    out.to_csv(PRED_PATH, index=False)
    print(f"\n✅ Saved model:       {MODEL_PATH}")
    print(f"✅ Saved predictions: {PRED_PATH}")
    
    # ===== GENERATE VISUALIZATIONS =====
    try:
        from plot_utils import plot_all_classification
        plot_dir = os.path.join(RESULTS_DIR, "ml5_metrics")
        plot_all_classification(
            y_true=y_test, y_pred=y_pred, y_proba=y_proba,
            fold_metrics=fold_metrics, save_path=plot_dir,
            model_name="MLP (ML5)"
        )
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots: {e}")
    
    print("\nDone (ML5_FAST).\n")


if __name__ == "__main__":
    main()
