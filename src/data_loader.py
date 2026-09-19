"""
Data Loader Module for Integrated Retail Analytics
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def load_raw_data(data_dir: str = 'dataset') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_path = os.path.join(data_dir, 'sales data-set.csv')
    features_path = os.path.join(data_dir, 'Features data set.csv')
    stores_path = os.path.join(data_dir, 'stores data-set.csv')
    
    if not os.path.exists(sales_path) or not os.path.exists(features_path) or not os.path.exists(stores_path):
        raise FileNotFoundError(f'One or more dataset files missing in {data_dir}')
        
    sales_df = pd.read_csv(sales_path)
    features_df = pd.read_csv(features_path)
    stores_df = pd.read_csv(stores_path)
    
    sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%d/%m/%Y')
    features_df['Date'] = pd.to_datetime(features_df['Date'], format='%d/%m/%Y')
    
    return sales_df, features_df, stores_df

def merge_datasets(sales_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(sales_df, stores_df, on='Store', how='left')
    merged = pd.merge(merged, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
    merged = merged.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
    return merged

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'num_stores': int(df['Store'].nunique()) if 'Store' in df.columns else 0,
        'num_depts': int(df['Dept'].nunique()) if 'Dept' in df.columns else 0,
        'date_min': str(df['Date'].min()) if 'Date' in df.columns else '',
        'date_max': str(df['Date'].max()) if 'Date' in df.columns else '',
        'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        'duplicate_count': int(df.duplicated().sum())
    }
