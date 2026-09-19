"""
Feature Engineering and Data Preprocessing Pipeline
"""
import pandas as pd
import numpy as np
from typing import Tuple, List

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week.astype(int)
    data['Day'] = data['Date'].dt.day
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['Quarter'] = data['Date'].dt.quarter
    data['IsMonthStart'] = data['Date'].dt.is_month_start.astype(int)
    data['IsMonthEnd'] = data['Date'].dt.is_month_end.astype(int)
    data['Days_In_Month'] = data['Date'].dt.days_in_month
    
    data['Is_SuperBowl'] = ((data['Week'] == 6) & (data['IsHoliday'])).astype(int)
    data['Is_LaborDay'] = ((data['Week'] == 36) & (data['IsHoliday'])).astype(int)
    data['Is_Thanksgiving'] = ((data['Week'] == 47) & (data['IsHoliday'])).astype(int)
    data['Is_Christmas'] = ((data['Week'].isin([51, 52])) & (data['IsHoliday'])).astype(int)
    
    md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in md_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)
        else:
            data[col] = 0.0
            
    data['Total_MarkDown'] = data[md_cols].sum(axis=1)
    data['Has_MarkDown'] = (data['Total_MarkDown'] > 0).astype(int)
    data['MarkDown_Count'] = (data[md_cols] > 0).sum(axis=1)
    data['MarkDown_Max'] = data[md_cols].max(axis=1)
    
    data['CPI'] = data['CPI'].fillna(data.groupby('Store')['CPI'].transform('median'))
    data['Unemployment'] = data['Unemployment'].fillna(data.groupby('Store')['Unemployment'].transform('median'))
    
    data['Fuel_Price_to_CPI'] = data['Fuel_Price'] / (data['CPI'] + 1e-5)
    data['Size_Per_Dept'] = data['Size'] / 100.0
    data['Unemployment_x_Size'] = data['Unemployment'] * (data['Size'] / 10000.0)
    data['Temp_Deviation'] = data['Temperature'] - data.groupby('Store')['Temperature'].transform('mean')
    
    if 'Type' in data.columns:
        for t in ['A', 'B', 'C']:
            data[f'Type_{t}'] = (data['Type'] == t).astype(int)
            
    data['IsHoliday_Int'] = data['IsHoliday'].astype(int)
    
    data = data.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    data['Sales_Lag_1'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    data['Sales_Lag_2'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
    data['Sales_Lag_4'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
    data['Sales_Lag_52'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52)
    
    shifted_sales = data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    data['Sales_Rolling_Mean_4'] = shifted_sales.groupby([data['Store'], data['Dept']]).transform(lambda x: x.rolling(4, min_periods=1).mean())
    data['Sales_Rolling_Std_4'] = shifted_sales.groupby([data['Store'], data['Dept']]).transform(lambda x: x.rolling(4, min_periods=1).std()).fillna(0)
    data['Sales_Rolling_Mean_12'] = shifted_sales.groupby([data['Store'], data['Dept']]).transform(lambda x: x.rolling(12, min_periods=1).mean())
    data['Sales_Rolling_Max_4'] = shifted_sales.groupby([data['Store'], data['Dept']]).transform(lambda x: x.rolling(4, min_periods=1).max())
    data['Sales_Rolling_Min_4'] = shifted_sales.groupby([data['Store'], data['Dept']]).transform(lambda x: x.rolling(4, min_periods=1).min())
    
    dept_medians = data.groupby('Dept')['Weekly_Sales'].transform('median')
    for lag_col in ['Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_4', 'Sales_Lag_52', 'Sales_Rolling_Mean_4', 'Sales_Rolling_Std_4', 'Sales_Rolling_Mean_12', 'Sales_Rolling_Max_4', 'Sales_Rolling_Min_4']:
        data[lag_col] = data[lag_col].fillna(dept_medians).fillna(data['Weekly_Sales'].median())
        
    return data

def get_feature_columns() -> List[str]:
    return [
        'Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Year', 'Month', 'Week', 'Day', 'DayOfYear', 'Quarter', 'IsMonthStart', 'IsMonthEnd',
        'IsHoliday_Int', 'Is_SuperBowl', 'Is_LaborDay', 'Is_Thanksgiving', 'Is_Christmas',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Total_MarkDown', 'Has_MarkDown', 'MarkDown_Count',
        'Fuel_Price_to_CPI', 'Size_Per_Dept', 'Unemployment_x_Size', 'Temp_Deviation',
        'Type_A', 'Type_B', 'Type_C',
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_4', 'Sales_Lag_52',
        'Sales_Rolling_Mean_4', 'Sales_Rolling_Std_4', 'Sales_Rolling_Mean_12', 'Sales_Rolling_Max_4', 'Sales_Rolling_Min_4'
    ]
