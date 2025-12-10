"""

"""

import os
from typing import Tuple
import numpy as np
import pandas as pd


# === CONFIG ===
DATA_PATH = os.path.join(os.path.dirname(__file__), 'india.csv')
LEAD_DAYS = 10
MONTHS_TO_KEEP = (3, 4, 5, 6)  # March–June
HEATWAVE_PERCENTILE = 90
TEST_DATE_FRACTION = 0.2


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'day_of_year' not in df.columns or df['day_of_year'].isna().any():
        df['day_of_year'] = df['date'].dt.dayofyear
    two_pi = 2.0 * np.pi
    df['doy_sin'] = np.sin(two_pi * df['day_of_year'] / 366.0)
    df['doy_cos'] = np.cos(two_pi * df['day_of_year'] / 366.0)
    return df


def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-grid-cell monthly percentile thresholds."""
    valid = df[~df['temperature_celsius'].isna()].copy()
    if valid.empty:
        valid = df[~df['climatology'].isna()].copy()
        valid = valid.rename(columns={'climatology': 'temperature_celsius'})
    grp = valid.groupby(['latitude', 'longitude', 'month'], observed=True)['temperature_celsius']
    perc = grp.quantile(HEATWAVE_PERCENTILE / 100.0).reset_index(name='hw_threshold')
    df = df.merge(perc, on=['latitude', 'longitude', 'month'], how='left')
    return df


def build_future_target(df: pd.DataFrame) -> pd.DataFrame:
    """Build future temperature target at t+LEAD days."""
    df = df.sort_values(['latitude', 'longitude', 'date']).reset_index(drop=True)
    df['future_temp_c'] = (
        df.groupby(['latitude', 'longitude'], observed=True)['temperature_celsius']
          .shift(-LEAD_DAYS)
    )
    df['y_heatwave'] = (df['future_temp_c'] >= df['hw_threshold']).astype(int)
    return df


def time_aware_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-aware train/test split."""
    unique_dates = np.array(sorted(df['date'].dropna().unique()))
    if unique_dates.size == 0:
        raise RuntimeError('No valid dates parsed from india.csv')
    split_idx = int((1.0 - TEST_DATE_FRACTION) * unique_dates.size)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])
    train_df = df[df['date'].isin(train_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()
    
    if train_df.empty:
        split_idx = max(1, unique_dates.size - max(1, int(0.1 * unique_dates.size)))
        train_dates = set(unique_dates[:split_idx])
        train_df = df[df['date'].isin(train_dates)].copy()
        test_df = df[df['date'].isin(set(unique_dates[split_idx:]))].copy()
    if test_df.empty:
        test_dates = {unique_dates[-1]}
        test_df = df[df['date'].isin(test_dates)].copy()
        train_df = df[~df['date'].isin(test_dates)].copy()
    
    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Prepare features and handle missing values."""
    feature_cols = [
        'temperature_anomaly', 'temperature_celsius', 'climatology',
        'temp_anom_diff_1d', 'areal_weight', 'land_mask',
        'doy_sin', 'doy_cos', 'month'
    ]
    
    # Impute missing values
    for col in ['temperature_anomaly', 'temp_anom_diff_1d']:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0.0)
            test_df[col] = test_df[col].fillna(0.0)
    
    if 'climatology' in train_df.columns:
        clim_mean = train_df['climatology'].mean()
        if pd.isna(clim_mean):
            clim_mean = 25.0
        train_df['climatology'] = train_df['climatology'].fillna(clim_mean)
        test_df['climatology'] = test_df['climatology'].fillna(clim_mean)
    
    if 'temperature_celsius' in train_df.columns:
        missing = train_df['temperature_celsius'].isna()
        train_df.loc[missing, 'temperature_celsius'] = (
            train_df.loc[missing, 'temperature_anomaly'] + 
            train_df.loc[missing, 'climatology']
        )
        missing_test = test_df['temperature_celsius'].isna()
        test_df.loc[missing_test, 'temperature_celsius'] = (
            test_df.loc[missing_test, 'temperature_anomaly'] + 
            test_df.loc[missing_test, 'climatology']
        )
    
    for col in ['areal_weight', 'land_mask']:
        if col in train_df.columns:
            median_val = train_df[col].median()
            if pd.isna(median_val):
                median_val = 1.0 if col == 'areal_weight' else 1.0
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
    
    if 'doy_sin' not in train_df.columns or train_df['doy_sin'].isna().any():
        train_df = add_time_features(train_df)
        test_df = add_time_features(test_df)
    
    if train_df['month'].isna().any():
        train_df['month'] = pd.to_datetime(train_df['date'], errors='coerce').dt.month
    if test_df['month'].isna().any():
        test_df['month'] = pd.to_datetime(test_df['date'], errors='coerce').dt.month
    
    train_df = train_df.dropna(subset=feature_cols + ['y_heatwave'])
    test_df = test_df.dropna(subset=feature_cols + ['y_heatwave'])
    
    X_train = train_df[feature_cols].values
    y_train = train_df['y_heatwave'].astype(int).values
    sw_train = train_df['areal_weight'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['y_heatwave'].astype(int).values
    
    return X_train, X_test, y_train, y_test, sw_train, feature_cols, test_df


def load_and_prepare_data():
    """Load data and prepare for training."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"india.csv not found at {DATA_PATH}. Run i.py first.")
    
    df = pd.read_csv(DATA_PATH)
    
    required = {
        'date', 'latitude', 'longitude', 'month', 'day_of_year',
        'temperature_anomaly', 'climatology', 'temperature_celsius',
        'areal_weight', 'land_mask', 'temp_anom_diff_1d'
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    
    df = df[df['month'].isin(MONTHS_TO_KEEP)].copy()
    df = df.sort_values(['date', 'latitude', 'longitude']).reset_index(drop=True)
    
    df = add_time_features(df)
    df = compute_percentiles(df)
    df = build_future_target(df)
    
    df = df[~df['future_temp_c'].isna()].copy()
    df = df[~df['hw_threshold'].isna()].copy()
    df = df[~df['y_heatwave'].isna()].copy()
    
    train_df, test_df = time_aware_split(df)
    
    X_train, X_test, y_train, y_test, sw_train, feature_cols, test_df = prepare_features(
        train_df, test_df
    )
    
    return X_train, X_test, y_train, y_test, sw_train, feature_cols, test_df


