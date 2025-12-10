import os, joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from ml_utils import load_and_prepare_data

# -------------------------------------------------------------------
# Directories (same style as ml1_fast / ml2_fast)
# -------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("ML3_FAST_FIXED: Logistic Regression (FULL data, time-aware TEST)")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1) Use canonical train/test split from ml_utils
    #    This gives you:
    #    - X_train, X_test: engineered features (incl. doy_sin, doy_cos, etc.)
    #    - y_train, y_test: heat-wave labels (0/1) for future_temp_c >= hw_threshold
    #    - sw_train       : sample weights (e.g., area weighting)
    #    - feature_cols   : list of feature column names
    #    - test_df        : original test rows with metadata (date, lat, lon, etc.)
    # ----------------------------------------------------------------
    X_train, X_test, y_train, y_test, sw_train, feature_cols, test_df = load_and_prepare_data()

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows : {len(X_test):,}")
    print(f"Features  : {len(feature_cols)}")

    # ===== 5-FOLD CROSS-VALIDATION =====
    from sklearn.model_selection import KFold
    print("\n----- 5-fold Cross-Validation -----")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        sw_tr = sw_train[tr_idx]
        
        model_fold = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=400,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ))
        ])
        model_fold.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
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

    # ----------------------------------------------------------------
    # 2) Define Logistic Regression model inside a scaling pipeline - FINAL TRAINING
    # ----------------------------------------------------------------
    print("\n----- Training Final Model on Full Train Set -----")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=400,
            class_weight="balanced",   # handle class imbalance
            n_jobs=-1,
            random_state=42
        ))
    ])

    # ----------------------------------------------------------------
    # 3) Train on FULL TRAIN SPLIT (time-aware), using sample weights
    # ----------------------------------------------------------------
    # sample weights are only meaningful for the classifier step
    model.fit(X_train, y_train, clf__sample_weight=sw_train)

    # ----------------------------------------------------------------
    # 4) Evaluate on HELD-OUT TEST SPLIT
    # ----------------------------------------------------------------
    print("\nEvaluating on HELD-OUT TEST split…")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of heat-wave (class 1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = np.nan
    cm = confusion_matrix(y_test, y_pred)

    print("\n✅ TEST Results (heat-wave classification):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {('%.4f' % roc) if not np.isnan(roc) else 'NA'}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # ----------------------------------------------------------------
    # 5) Save predictions in the SAME STRUCTURE as ml1_fast / ml2_fast
    # ----------------------------------------------------------------
    # test_df has metadata columns:
    #   date, year, month, day, day_of_year,
    #   latitude, longitude,
    #   temperature_anomaly, climatology, temperature_celsius,
    #   areal_weight, land_mask,
    #   temp_anom_prev_day, temp_anom_diff_1d,
    #   doy_sin, doy_cos,
    #   hw_threshold, future_temp_c, y_heatwave, ...
    #
    # We keep all of that and just add Logistic Regression outputs as:
    #   - pred_heatwave   (0/1)
    #   - proba_heatwave  (probability of class 1)
    # ----------------------------------------------------------------
    out = test_df.copy()

    # Ensure ground-truth label column is present
    if "y_heatwave" not in out.columns:
        out["y_heatwave"] = y_test

    out["pred_heatwave"]  = y_pred
    out["proba_heatwave"] = y_proba

    out_path = os.path.join(RESULTS_DIR, "ml3_fast_fixed_predictions.csv")
    out.to_csv(out_path, index=False)

    # ----------------------------------------------------------------
    # 6) Save model
    # ----------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "ml3_fast_fixed.pkl")
    joblib.dump(model, model_path)

    print(f"\n✅ Saved model      : {model_path}")
    print(f"✅ Saved predictions: {out_path}")
    
    # ===== GENERATE VISUALIZATIONS =====
    try:
        from plot_utils import plot_all_classification
        plot_dir = os.path.join(RESULTS_DIR, "ml3_metrics")
        plot_all_classification(
            y_true=y_test, y_pred=y_pred, y_proba=y_proba,
            fold_metrics=fold_metrics, save_path=plot_dir,
            model_name="Logistic Regression (ML3)"
        )
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots: {e}")
    
    print("\nDone.\n")


if __name__ == "__main__":
    main()
