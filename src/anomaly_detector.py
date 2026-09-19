"""
Anomaly Detection and Outlier Handling Module
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any

def detect_iqr_anomalies(df: pd.DataFrame, col: str = 'Weekly_Sales', factor: float = 1.5) -> pd.Series:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return (df[col] < lower_bound) | (df[col] > upper_bound)

def detect_zscore_anomalies(df: pd.DataFrame, col: str = 'Weekly_Sales', threshold: float = 3.0) -> pd.Series:
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        return pd.Series(False, index=df.index)
    z_scores = (df[col] - mean) / std
    return z_scores.abs() > threshold

def detect_isolation_forest_anomalies(df: pd.DataFrame, feature_cols: list = None, contamination: float = 0.015, random_state: int = 42) -> pd.Series:
    if feature_cols is None:
        feature_cols = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100, n_jobs=-1)
    preds = iso.fit_predict(X)
    return preds == -1

def classify_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out['Is_Negative_Sale'] = df_out['Weekly_Sales'] < 0
    
    def dept_iqr(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
    
    df_out['Is_IQR_Anomaly'] = df_out.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(dept_iqr)
    
    conditions = [
        df_out['Weekly_Sales'] < 0,
        (df_out['Is_IQR_Anomaly']) & (df_out['IsHoliday']),
        (df_out['Is_IQR_Anomaly']) & (~df_out['IsHoliday']) & (df_out['Weekly_Sales'] > df_out['Weekly_Sales'].quantile(0.95)),
        df_out['Is_IQR_Anomaly']
    ]
    choices = [
        'Negative Return / Data Error',
        'Legitimate Holiday Surge',
        'Promotional / Unexplained Spike',
        'Structural Outlier'
    ]
    df_out['Anomaly_Category'] = np.select(conditions, choices, default='Normal')
    return df_out

def clean_anomalies_for_training(df: pd.DataFrame, clip_negatives: bool = True, winsorize_upper: float = 0.999) -> pd.DataFrame:
    df_clean = df.copy()
    if clip_negatives:
        df_clean['Weekly_Sales_Cleaned'] = df_clean['Weekly_Sales'].clip(lower=0)
    else:
        df_clean['Weekly_Sales_Cleaned'] = df_clean['Weekly_Sales']
        
    upper_limit = df_clean['Weekly_Sales_Cleaned'].quantile(winsorize_upper)
    df_clean['Weekly_Sales_Cleaned'] = df_clean['Weekly_Sales_Cleaned'].clip(upper=upper_limit)
    return df_clean
