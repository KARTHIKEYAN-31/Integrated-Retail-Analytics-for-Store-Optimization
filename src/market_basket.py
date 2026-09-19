"""
Market Basket / Departmental Affinity and Association Rule Mining
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from typing import Tuple, Dict, Any

def create_department_affinity_matrix(df: pd.DataFrame, threshold_percentile: float = 70.0, top_n_depts: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Select top N revenue departments to focus on high-impact retail associations
    top_depts = df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False).head(top_n_depts).index.tolist()
    df_filtered = df[df['Dept'].isin(top_depts)]
    
    pivot_sales = df_filtered.pivot_table(index=['Store', 'Date'], columns='Dept', values='Weekly_Sales', fill_value=0)
    corr_matrix = pivot_sales.corr()
    
    # Binary activation: True if dept sales in that store-week >= 70th percentile of that dept
    dept_thresholds = pivot_sales.apply(lambda col: np.percentile(col[col > 0], threshold_percentile) if len(col[col > 0]) > 0 else 0)
    binary_matrix = (pivot_sales >= dept_thresholds).astype(bool)
    binary_matrix.columns = [f'Dept_{c}' for c in binary_matrix.columns]
    
    return binary_matrix, corr_matrix

def mine_department_rules(binary_matrix: pd.DataFrame, min_support: float = 0.08, min_confidence: float = 0.25, min_lift: float = 1.1) -> pd.DataFrame:
    frequent_itemsets = fpgrowth(binary_matrix, min_support=min_support, use_colnames=True, max_len=2)
    if len(frequent_itemsets) == 0:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]
    if len(rules) == 0:
        # Fallback with slightly relaxed threshold to capture retail affinities
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
        rules = rules[rules['confidence'] >= 0.2]
        
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
    return rules
