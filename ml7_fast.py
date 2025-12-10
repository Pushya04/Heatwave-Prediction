# ml7_fast.py
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def ensure_doy_features(df: pd.DataFrame) -> pd.DataFrame:
    if "doy_sin" in df.columns and "doy_cos" in df.columns:
        return df
    if "day_of_year" in df.columns:
        doy = df["day_of_year"].astype(float)
    else:
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            doy = d.dt.dayofyear.fillna(1).astype(float)
        else:
            doy = pd.Series(
                np.random.randint(1, 366, size=len(df)),
                index=df.index,
                dtype=float
            )
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    return df

def ensure_future_temp(df: pd.DataFrame) -> pd.DataFrame:
    if "future_temp_c" in df.columns:
        return df
    if "temperature_celsius" not in df.columns:
        raise ValueError("india.csv missing 'temperature_celsius' — cannot build future_temp_c.")
    if all(c in df.columns for c in ["latitude", "longitude"]):
        if "date" in df.columns:
            df = df.sort_values(["latitude", "longitude", "date"])
        else:
            df = df.sort_values(["latitude", "longitude"])
        df["future_temp_c"] = df.groupby(["latitude", "longitude"])["temperature_celsius"].shift(-10)
    else:
        if "date" in df.columns:
            df = df.sort_values("date")
        df["future_temp_c"] = df["temperature_celsius"].shift(-10)
    return df

def compute_hw_threshold(df: pd.DataFrame, perc: float = 90.0) -> pd.DataFrame:
    if "hw_threshold" in df.columns:
        return df
    if "month" in df.columns and "temperature_celsius" in df.columns:
        thr = df.groupby("month")["temperature_celsius"].quantile(perc / 100.0)
        glob_thr = df["temperature_celsius"].mean() + df["temperature_celsius"].std()
        thr = thr.to_dict()
        for m in range(1, 13):
            if m not in thr:
                thr[m] = glob_thr
        df["hw_threshold"] = df["month"].map(thr).astype(float)
    else:
        glob_thr = df["temperature_celsius"].quantile(perc / 100.0)
        df["hw_threshold"] = float(glob_thr)
    return df

def build_model():
    
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(32,),       
            activation="relu",
            learning_rate_init=3e-4,       
            alpha=3e-3,                   
            max_iter=200,                 
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=False
        ))
    ])
    model = AdaBoostRegressor(
        estimator=base,
        n_estimators=16,        
        learning_rate=0.8,     
        random_state=42
    )
    return model

def run_cv(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    print("\n----- 5-fold Cross-Validation (derived heat-wave classification) -----")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        df_val = df.iloc[val_idx].copy()

        model = build_model()
        model.fit(X_tr, y_tr)
        y_pred_val = model.predict(X_val)

        thr = df_val["hw_threshold"].astype(float).values
        y_true_hw = (y_val >= thr).astype(int)
        y_pred_hw = (y_pred_val >= thr).astype(int)

        acc = accuracy_score(y_true_hw, y_pred_hw)
        prec = precision_score(y_true_hw, y_pred_hw, zero_division=0)
        rec = recall_score(y_true_hw, y_pred_hw, zero_division=0)
        f1 = f1_score(y_true_hw, y_pred_hw, zero_division=0)
        try:
            roc = roc_auc_score(y_true_hw, y_pred_val)
        except Exception:
            roc = np.nan

        fold_metrics.append((acc, prec, rec, f1, roc))
        print(f"Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC={roc:.4f}")

    mets = np.array(fold_metrics)
    m_acc, m_prec, m_rec, m_f1, m_roc = mets.mean(axis=0)
    print("\nCV Averages over 5 folds:")
    print(f"Acc={m_acc:.4f}, Prec={m_prec:.4f}, Rec={m_rec:.4f}, F1={m_f1:.4f}, ROC={m_roc:.4f}")
    return fold_metrics

def main():
    print("=" * 70)
    print("ML7_FAST: AdaBoost + MLPRegressor (5-fold CV + 80/20 TEST, date-ordered output)")
    print("=" * 70)

    if not os.path.exists("india.csv"):
        raise FileNotFoundError("india.csv not found in current directory.")
    df = pd.read_csv("india.csv")

    df = ensure_doy_features(df)
    df = ensure_future_temp(df)
    df = compute_hw_threshold(df)
    df = df.dropna(subset=["future_temp_c"]).copy()
    df["y_heatwave"] = (df["future_temp_c"] >= df["hw_threshold"]).astype(int)

    feature_cols = [
        "temperature_anomaly", "climatology", "temperature_celsius",
        "areal_weight", "land_mask",
        "temp_anom_prev_day", "temp_anom_diff_1d",
        "doy_sin", "doy_cos",
        "latitude", "longitude",
        "day_of_year", "month"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_all = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).values
    y_all = df["future_temp_c"].astype(float).values

    # 5-fold CV on full data
    fold_metrics = run_cv(df, X_all, y_all, n_splits=5)

    # Final 80/20 TEST split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, np.arange(len(df)), test_size=0.2, random_state=42
    )
    test_df = df.iloc[idx_test].copy()

    print(f"\nTrain samples (test split): {len(X_train)}, Test samples: {len(X_test)}")

    model = build_model()
    print("\nTraining final model on TRAIN split...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nTEST Regression Results (T at t+10):")
    print(f"MAE: {mae:.3f} °C")
    print(f"R²:  {r2:.3f}")

    thr_series = test_df["hw_threshold"].astype(float).values
    y_true_cls = (y_test >= thr_series).astype(int)
    y_pred_cls = (y_pred >= thr_series).astype(int)

    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    rec = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    try:
        roc = roc_auc_score(y_true_cls, y_pred)
    except Exception:
        roc = np.nan
    cm = confusion_matrix(y_true_cls, y_pred_cls)

    print("\nTEST Derived Heat-wave Classification:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true_cls, y_pred_cls, digits=4))

    # ===== Build output DataFrame =====
    out = test_df.copy()
    out["y_heatwave"]    = y_true_cls
    out["pred_heatwave"] = y_pred_cls

    score = y_pred
    smin, smax = float(score.min()), float(score.max())
    if smax > smin:
        out["proba_heatwave"] = (score - smin) / (smax - smin + 1e-9)
    else:
        out["proba_heatwave"] = 0.5

    out["y_true_temp"] = y_test
    out["y_pred_temp"] = y_pred

    # ===== ORDER ROWS BY DATE (this fixes the weird shuffled date/year order) =====
    if "date" in out.columns:
        out["_sort_date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("_sort_date")
        out = out.drop(columns=["_sort_date"])
    elif {"year", "month", "day"}.issubset(out.columns):
        out = out.sort_values(["year", "month", "day"])

    # Save predictions; keep original column order + appended prediction columns
    out.to_csv("results/ml7_fast_predictions.csv", index=False)
    joblib.dump(model, "models/ml7_fast.pkl")

    print("\nSaved model: models/ml7_fast.pkl")
    print("Saved predictions (date-ordered): results/ml7_fast_predictions.csv")
    
    # ===== GENERATE VISUALIZATIONS =====
    # ===== GENERATE VISUALIZATIONS =====
    try:
        from plot_utils import plot_all_classification
        # Import XAI analysis
        from xai import run_xai_analysis
        
        plot_dir = os.path.join("results", "ml7_metrics")
        
        # 1. Standard Performance Plots
        plot_all_classification(
            y_true=y_true_cls,
            y_pred=y_pred_cls,
            y_proba=out["proba_heatwave"].values,
            fold_metrics=fold_metrics,
            save_path=plot_dir,
            model_name="AdaBoost + MLP Regressor (ML7)"
        )
        
        # 2. Advanced XAI Analysis (SHAP, Feature Importance)
        print("\n🔍 Running XAI Analysis (SHAP, Feature Importance)...")
        run_xai_analysis(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            model_name="ml7_fast",
            save_dir=os.path.join(plot_dir, "xai")
        )
        
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots/XAI: {e}")

if __name__ == "__main__":
    main()
