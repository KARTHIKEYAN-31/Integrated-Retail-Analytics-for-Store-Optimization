"""
Demand Forecasting ML Models, Evaluation, and Tuning
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple

def calculate_wmae(y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray) -> float:
    weights = np.where(is_holiday, 5.0, 1.0)
    wmae = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    return float(wmae)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    is_hol_arr = np.array(is_holiday).astype(bool)
    
    wmae = calculate_wmae(y_true_arr, y_pred_arr, is_hol_arr)
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    
    non_zero = y_true_arr != 0
    mape = float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100)
    
    return {
        'WMAE': round(wmae, 2),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE_%': round(mape, 2)
    }

def train_ridge_model(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 10.0) -> Pipeline:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, max_depth: int = 16, random_state: int = 42) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf

def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, params: dict = None) -> lgb.LGBMRegressor:
    default_params = {
        'n_estimators': 350,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    if params:
        default_params.update(params)
        
    model = lgb.LGBMRegressor(**default_params)
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X_train, y_train)
    return model
