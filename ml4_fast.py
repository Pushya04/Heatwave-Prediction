import os, joblib
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
import xai  # Import the XAI module

# -------------------------------------------------------------------
# Directories (same pattern as ml1_fast / ml2_fast / ml3_fast)
# -------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("ML4_FAST_OPTIMIZED: Linear SVM (FULL data, time-aware TEST)")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1) Load canonical train/test split & features from ml_utils
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
    
    # Define the fast model structure
    # LinearSVC is much faster than SVC(kernel='rbf'). 
    # CalibratedClassifierCV is used to get probabilities.
    def get_fast_model():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", CalibratedClassifierCV(
                LinearSVC(dual=False, class_weight="balanced", random_state=42),
                method='sigmoid',
                cv=3  # Internal CV for calibration
            ))
        ])
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        model_fold = get_fast_model()
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

    # ----------------------------------------------------------------
    # 2) Train Final Model on Full Train Set
    # ----------------------------------------------------------------
    print("\n----- Training Final Model on Full Train Set -----")
    model = get_fast_model()

    # ----------------------------------------------------------------
    # 3) Train on FULL TRAIN SPLIT (time-aware)
    # ----------------------------------------------------------------
    model.fit(X_train, y_train)

    # ----------------------------------------------------------------
    # 4) Evaluate on HELD-OUT TEST SPLIT
    # ----------------------------------------------------------------
    print("\nEvaluating on HELD-OUT TEST split…")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability for class 1 (heat-wave)

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
    # 5) Save predictions
    # ----------------------------------------------------------------
    out = test_df.copy()

    if "y_heatwave" not in out.columns:
        out["y_heatwave"] = y_test

    out["pred_heatwave"]  = y_pred
    out["proba_heatwave"] = y_proba

    out_path = os.path.join(RESULTS_DIR, "ml4_fast_fixed_predictions.csv")
    out.to_csv(out_path, index=False)

    # ----------------------------------------------------------------
    # 6) Save model
    # ----------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "ml4_fast_fixed.pkl")
    joblib.dump(model, model_path)

    print(f"\n✅ Saved model      : {model_path}")
    print(f"✅ Saved predictions: {out_path}")
    
    # ===== GENERATE VISUALIZATIONS =====
    try:
        from plot_utils import plot_all_classification
        plot_dir = os.path.join(RESULTS_DIR, "ml4_metrics")
        plot_all_classification(
            y_true=y_test, y_pred=y_pred, y_proba=y_proba,
            fold_metrics=fold_metrics, save_path=plot_dir,
            model_name="SVM (ML4)"
        )
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots: {e}")
    
    # ===== RUN XAI ANALYSIS =====
    print("\n----- Running XAI Analysis -----")
    try:
        xai_dir = os.path.join(RESULTS_DIR, "xai_analysis", "ml4_fast_fixed")
        # We need to pass the underlying LinearSVC to some XAI functions if they expect coef_,
        # but CalibratedClassifierCV wraps it.
        # However, our updated xai.py handles pipelines/wrappers by looking at steps.
        # CalibratedClassifierCV is an ensemble of classifiers if cv>1, or a wrapper.
        # If cv is used, it has calibrated_classifiers_.
        # If we want feature importance, we should use the base estimator trained on full data.
        # But CalibratedClassifierCV(cv=3) fits 3 models.
        # To get a single feature importance, we might need to refit a LinearSVC on full data just for XAI,
        # or extract one of the calibrated classifiers.
        
        # For simplicity and correctness in XAI, we'll pass the calibrated model.
        # Our xai.py might need to be smart about CalibratedClassifierCV.
        # Let's check if xai.py handles CalibratedClassifierCV.
        # It checks for 'steps' (Pipeline) or 'coef_'.
        # CalibratedClassifierCV doesn't have 'coef_'. It has 'calibrated_classifiers_'.
        # We should probably extract the base estimator for feature importance.
        
        # Helper to get base model for importance
        base_model_for_importance = model.named_steps['svm'].calibrated_classifiers_[0].estimator
        
        # But for SHAP, we can use the pipeline.
        
        # Let's modify the call to be safe.
        # Actually, let's just let xai.py handle it or fail gracefully, but better to help it.
        # We can pass the base linear model for feature importance.
        
        # Extract the LinearSVC from the first calibrated classifier for feature importance visualization
        linear_svc = model.named_steps['svm'].calibrated_classifiers_[0].estimator
        
        # Run XAI
        # Note: We pass the full pipeline 'model' for SHAP (so it uses predict_proba),
        # but we might need to manually handle feature importance if xai.py doesn't support CalibratedClassifierCV.
        # Our updated xai.py handles Pipeline -> steps[-1]. 
        # Here steps[-1] is CalibratedClassifierCV.
        # CalibratedClassifierCV does NOT have coef_.
        
        # So we should probably pass the linear_svc for feature importance specifically?
        # xai.run_xai_analysis calls plot_feature_importance(model, ...)
        
        # Let's just run it. If feature importance fails, it prints a warning.
        # But we want it to succeed.
        # Let's manually call the XAI functions here with the correct objects if needed, 
        # OR rely on xai.run_xai_analysis.
        
        # Better: Pass the linear_svc to run_xai_analysis? 
        # No, because SHAP needs predict_proba which LinearSVC doesn't have (unless we use Calibrated).
        
        # So:
        # 1. Feature Importance: Needs LinearSVC (for coef_)
        # 2. SHAP: Needs Pipeline (for predict_proba)
        
        # I'll update xai.py one more time to handle CalibratedClassifierCV?
        # Or just handle it here.
        # Let's handle it in xai.py to be generic.
        
        xai.run_xai_analysis(model, X_test, y_test, feature_cols, "ml4_fast_fixed", xai_dir)
        
    except Exception as e:
        print(f"⚠ Warning: XAI Analysis failed: {e}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
