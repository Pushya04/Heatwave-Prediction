import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy.stats import randint
from ml_utils import load_and_prepare_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("="*70)
    print("ML1_FAST: Random Forest (5-fold CV + Final TEST)")
    print("="*70)
    
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
        
        model_fold = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model_fold.fit(X_tr, y_tr, sample_weight=sw_tr)
        y_pred_val = model_fold.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred_val)
        prec = precision_score(y_val, y_pred_val, zero_division=0)
        rec = recall_score(y_val, y_pred_val, zero_division=0)
        f1 = f1_score(y_val, y_pred_val, zero_division=0)
        
        fold_metrics.append((acc, prec, rec, f1))
        print(f"Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    mets = np.array(fold_metrics)
    m_acc, m_prec, m_rec, m_f1 = mets.mean(axis=0)
    print("\nCV Averages over 5 folds:")
    print(f"Acc={m_acc:.4f}, Prec={m_prec:.4f}, Rec={m_rec:.4f}, F1={m_f1:.4f}")
    
    # ===== FINAL MODEL TRAINING ON FULL TRAIN SET =====
    print("\n----- Training Final Model on Full Train Set -----")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sw_train)
    y_pred = model.predict(X_test)
    
    # ===== TEST EVALUATION =====
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n✅ TEST Results (heat-wave classification):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    
    # ===== SAVE PREDICTIONS AND MODEL =====
    out = test_df.copy()
    if "y_heatwave" not in out.columns:
        out["y_heatwave"] = y_test
    out["pred_heatwave"] = y_pred
    
    out.to_csv(os.path.join(RESULTS_DIR, "ml1_fast_predictions.csv"), index=False)
    joblib.dump(model, os.path.join(MODELS_DIR, "ml1_fast.pkl"))
    
    print(f"\n✅ Saved model: {os.path.join(MODELS_DIR, 'ml1_fast.pkl')}")
    print(f"✅ Saved predictions: {os.path.join(RESULTS_DIR, 'ml1_fast_predictions.csv')}")
    
    # ===== GENERATE VISUALIZATIONS =====
    from plot_utils import plot_all_classification
    
    y_proba = model.predict_proba(X_test)[:, 1]
    feature_importance = model.feature_importances_
    plot_dir = os.path.join(RESULTS_DIR, "ml1_metrics")
    
    plot_all_classification(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        fold_metrics=fold_metrics,
        save_path=plot_dir,
        model_name="Random Forest (ML1)",
        feature_importance=feature_importance,
        feature_names=feature_cols
    )
    
    print("\nDone.\n")

if __name__ == "__main__":
    main()
