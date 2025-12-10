"""
XAI (Explainable AI) Analysis for ML Models - Optimized for Speed & Parallel Execution
Implements:
1. Feature Importance (Gini/gain + Permutation)
2. SHAP Summary (Global explanation)
3. SHAP Force/Decision Plots (Single-day explanation)
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
from joblib import Parallel, delayed

# Import data loader
from ml_utils import load_and_prepare_data

# Directories
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
XAI_DIR = os.path.join(RESULTS_DIR, "xai_analysis")
os.makedirs(XAI_DIR, exist_ok=True)

# Global data cache to avoid reloading for every process
_DATA_CACHE = None

def get_data():
    """Load data once and cache it."""
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = load_and_prepare_data()
    return _DATA_CACHE

def plot_feature_importance(model, feature_names, model_name, save_dir):
    """Plot built-in feature importance (Gini/gain or Coefficients)"""
    try:
        importances = None
        estimator = model

        # Unwrap Pipeline
        if hasattr(model, 'steps'):
            estimator = model.steps[-1][1]
        
        # Unwrap CalibratedClassifierCV
        if hasattr(estimator, 'calibrated_classifiers_'):
            # Use the first calibrated classifier's base estimator
            estimator = estimator.calibrated_classifiers_[0].estimator

        # Tree-based models
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        # Linear models (check for coef_)
        elif hasattr(estimator, 'coef_'):
            importances = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)

        if importances is not None:
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices], color='teal')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance (Magnitude)', fontweight='bold')
            plt.ylabel('Feature', fontweight='bold')
            plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model_name}_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print(f"ℹ️ [{model_name}] No feature importance/coefficients found.")
    except Exception as e:
        print(f"⚠️ [{model_name}] Feature importance plot failed: {e}")

def plot_permutation_importance(model, X_test, y_test, feature_names, model_name, save_dir):
    """Plot permutation importance"""
    try:
        # Reduced n_repeats to 5 for speed, n_jobs=1 to avoid nested parallelism contention
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=5, 
            random_state=42, 
            n_jobs=1 
        )
        
        indices = np.argsort(perm_importance.importances_mean)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), perm_importance.importances_mean[indices], 
                 xerr=perm_importance.importances_std[indices], color='coral')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Permutation Importance', fontweight='bold')
        plt.ylabel('Feature', fontweight='bold')
        plt.title(f'Permutation Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_permutation_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️ [{model_name}] Permutation importance failed: {e}")

def plot_shap_summary(model, X_test, feature_names, model_name, save_dir):
    """Generate SHAP summary plot (global explanation)"""
    try:
        # Convert memmap to regular array to avoid SHAP issues
        X_test = np.array(X_test)
        # Reduced sample size to 500 for speed
        n_samples = min(500, len(X_test))
        X_sample = X_test[:n_samples]
        
        explainer = None
        shap_values = None

        # Extract actual model if it's a pipeline
        estimator = model
        if hasattr(model, 'steps'):
            estimator = model.steps[-1][1]

        # 1. TreeExplainer (Fastest for Trees)
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        except Exception:
            # 2. LinearExplainer (Fast for Linear Models)
            try:
                # LinearExplainer requires independent features for background, or a masker
                # We'll use a small background sample
                X_background = shap.sample(X_test, 100)
                explainer = shap.LinearExplainer(estimator, X_background)
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                # 3. KernelExplainer (Slowest fallback)
                print(f"⚠️ [{model_name}] Tree/Linear Explainer failed, using KernelExplainer (Slow)...")
                X_background = shap.sample(X_test, 25) # Very small background for speed
                
                # For Pipeline, we need to pass the full predict_proba
                # If 'model' is a pipeline, use it directly
                predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                
                explainer = shap.KernelExplainer(predict_fn, X_background)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
        
        # Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_shap_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return explainer, shap_values
    except Exception as e:
        print(f"⚠️ [{model_name}] SHAP summary failed: {e}")
        return None, None

def plot_shap_force_plots(explainer, shap_values, X_test, feature_names, model_name, save_dir, n_examples=3):
    """Generate SHAP force plots for individual predictions"""
    if explainer is None or shap_values is None: return

    try:
        n_samples = min(n_examples, len(shap_values) if not isinstance(shap_values, list) else len(shap_values[1]))
        n_samples = min(n_samples, n_examples)
        
        for i in range(n_samples):
            try:
                # Extract SHAP values for this sample
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][i] if len(shap_values) > 1 else shap_values[0][i]
                else:
                    shap_vals = shap_values[i]
                
                # Get expected value
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = ev[1] if len(ev) > 1 else ev[0]
                if hasattr(ev, 'item'):
                    ev = ev.item()
                ev = float(ev)
                
                # Get feature values for this sample
                if isinstance(X_test, np.ndarray):
                    x_sample = X_test[i]
                else:
                    x_sample = X_test[i:i+1].flatten()

                # Create force plot using newer API without matplotlib=True
                plt.figure(figsize=(20, 3))
                fig = shap.force_plot(
                    ev,
                    shap_vals,
                    x_sample,
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.title(f'SHAP Force Plot - {model_name} (Sample {i+1})', fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{model_name}_force_plot_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
                plt.close('all')
            except Exception as e_inner:
                print(f"⚠️ [{model_name}] Force plot sample {i+1} failed: {e_inner}")
                continue
    except Exception as e:
        print(f"⚠️ [{model_name}] SHAP force plots failed: {e}")

def plot_shap_decision_plot(explainer, shap_values, X_test, feature_names, model_name, save_dir, n_examples=5):
    """Generate SHAP decision plot"""
    if explainer is None or shap_values is None: return

    try:
        # Get expected value
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) > 1 else ev[0]
        if hasattr(ev, 'item'):
            ev = ev.item()
        ev = float(ev)
        
        # Extract SHAP values
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
        
        # Limit to n_examples
        n_samples = min(n_examples, len(shap_vals))
        shap_vals = shap_vals[:n_samples]
        
        # Ensure it's 2D array
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)
        
        plt.figure(figsize=(10, 8))
        shap.decision_plot(ev, shap_vals, feature_names=feature_names, show=False)
        plt.title(f'SHAP Decision Plot - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_decision_plot.png'), dpi=300, bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        print(f"⚠️ [{model_name}] SHAP decision plot failed: {e}")

def run_xai_analysis(model, X_test, y_test, feature_names, model_name, save_dir):
    """
    Run all XAI analyses for a given model and data.
    Can be imported and used by other scripts (e.g., ml7_fast.py).
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"🚀 Starting XAI for: {model_name}")
    
    # Handle ensemble models (list of models)
    is_ensemble = isinstance(model, list)
    if is_ensemble:
        print(f"ℹ️  [{model_name}] Detected ensemble model with {len(model)} base models")
        # For feature importance, use the first model
        base_model_for_importance = model[0]
        
        # For SHAP, create a wrapper that averages predictions
        class EnsembleWrapper:
            def __init__(self, models):
                self.models = models
                self.classes_ = getattr(models[0], 'classes_', None)
            
            def fit(self, X, y):
                # Dummy fit method for compatibility
                return self
            
            def score(self, X, y):
                # Score method for permutation importance
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, self.predict(X))
            
            def predict(self, X):
                preds = [m.predict(X) for m in self.models]
                return np.mean(preds, axis=0)
            
            def predict_proba(self, X):
                probas = [m.predict_proba(X) for m in self.models]
                return np.mean(probas, axis=0)
        
        model_for_shap = EnsembleWrapper(model)
        model_for_feature_importance = base_model_for_importance
    else:
        model_for_shap = model
        model_for_feature_importance = model
    
    # 1. Feature Importance (Gini/Gain)
    plot_feature_importance(model_for_feature_importance, feature_names, model_name, save_dir)
    
    # 2. Permutation Importance
    plot_permutation_importance(model_for_shap, X_test, y_test, feature_names, model_name, save_dir)
    
    # 3. SHAP Analysis
    explainer, shap_values = plot_shap_summary(model_for_shap, X_test, feature_names, model_name, save_dir)
    
    # 4. SHAP Force & Decision Plots
    plot_shap_force_plots(explainer, shap_values, X_test, feature_names, model_name, save_dir)
    plot_shap_decision_plot(explainer, shap_values, X_test, feature_names, model_name, save_dir)
    
    print(f"✅ Finished XAI for: {model_name}")

def process_single_model(model_path):
    """Worker function to process a single model"""
    model_filename = os.path.basename(model_path)
    model_name = os.path.splitext(model_filename)[0]
    
    # Special handling for ml7_fast (uses different features)
    if model_name == "ml7_fast":
        try:
            # Load ml7's prediction file to get the correct test data
            pred_path = os.path.join(RESULTS_DIR, "ml7_fast_predictions.csv")
            if os.path.exists(pred_path):
                print(f"ℹ️  [{model_name}] Loading ml7-specific test data from predictions")
                df = pd.read_csv(pred_path)
                
                feature_cols = [
                    "temperature_anomaly", "climatology", "temperature_celsius",
                    "areal_weight", "land_mask", "temp_anom_prev_day", "temp_anom_diff_1d",
                    "doy_sin", "doy_cos", "latitude", "longitude", "day_of_year", "month"
                ]
                feature_cols = [c for c in feature_cols if c in df.columns]
                
                X_test = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).values
                y_test = df["y_true_temp"].values if "y_true_temp" in df.columns else df["future_temp_c"].values
            else:
                print(f"⚠️  [{model_name}] Predictions file not found, skipping")
                return
        except Exception as e:
            print(f"❌ Failed to load ml7 data: {e}")
            return
    else:
        # Load data (cached) for other models
        X_train, X_test, y_train, y_test, _, feature_cols, _ = get_data()
    
    # Convert memmap to regular arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Load model
    try:
        loaded_obj = joblib.load(model_path)
        # Handle models saved as dictionaries (like ml5 and ml6)
        if isinstance(loaded_obj, dict):
            if "model" in loaded_obj:
                model = loaded_obj["model"]
                print(f"ℹ️  [{model_name}] Extracted model from dictionary structure")
            elif "models" in loaded_obj:
                # ml6 saves ensemble as {"models": [...], "scaler": ...}
                model = loaded_obj["models"]
                print(f"ℹ️  [{model_name}] Extracted ensemble from dictionary structure")
            else:
                model = loaded_obj
        else:
            model = loaded_obj
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return
    
    # Create directory
    model_dir = os.path.join(XAI_DIR, model_name)
    
    # Run Analyses using the shared function
    run_xai_analysis(model, X_test, y_test, feature_cols, model_name, model_dir)

def main():
    print("\n" + "="*70)
    print("🚀 HIGH-PERFORMANCE XAI ANALYSIS STARTED")
    print("="*70)
    
    # Auto-discover all .pkl models
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    
    if not model_files:
        print("❌ No models found in models/ directory!")
        return

    print(f"Found {len(model_files)} models: {[os.path.basename(m) for m in model_files]}")
    
    # Load data once in main process to ensure it's available
    get_data()
    
    # Run in parallel using all available cores
    Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_model)(model_path) for model_path in model_files
    )
    
    print("\n" + "="*70)
    print("✅ ALL XAI ANALYSES COMPLETED")
    print(f"📁 Results saved in: {XAI_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
