"""
Customer, Store, and Department Segmentation Module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple, Dict, Any

def create_store_profiles(df: pd.DataFrame) -> pd.DataFrame:
    md_cols = [c for c in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'] if c in df.columns]
    df_temp = df.copy()
    df_temp['Total_MarkDown'] = df_temp[md_cols].fillna(0).sum(axis=1)
    
    profiles = df_temp.groupby('Store').agg(
        Total_Sales=('Weekly_Sales', 'sum'),
        Avg_Weekly_Sales=('Weekly_Sales', 'mean'),
        Std_Weekly_Sales=('Weekly_Sales', 'std'),
        Size=('Size', 'first'),
        Type=('Type', 'first'),
        Avg_Markdown=('Total_MarkDown', 'mean'),
        Holiday_Sales_Avg=('Weekly_Sales', lambda x: x[df_temp.loc[x.index, 'IsHoliday']].mean() if df_temp.loc[x.index, 'IsHoliday'].any() else x.mean()),
        NonHoliday_Sales_Avg=('Weekly_Sales', lambda x: x[~df_temp.loc[x.index, 'IsHoliday']].mean() if (~df_temp.loc[x.index, 'IsHoliday']).any() else x.mean()),
        Avg_CPI=('CPI', 'mean'),
        Avg_Unemployment=('Unemployment', 'mean')
    ).reset_index()
    
    profiles['Sales_Volatility_CV'] = profiles['Std_Weekly_Sales'] / profiles['Avg_Weekly_Sales']
    profiles['Holiday_Surge_Factor'] = profiles['Holiday_Sales_Avg'] / (profiles['NonHoliday_Sales_Avg'] + 1e-5)
    profiles['Sales_Per_SqFt'] = profiles['Total_Sales'] / profiles['Size']
    profiles['Markdown_Reliance'] = profiles['Avg_Markdown'] / (profiles['Avg_Weekly_Sales'] + 1e-5)
    
    return profiles

def perform_store_clustering(profiles_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans, StandardScaler, PCA, Dict[str, Any]]:
    feature_cols = ['Avg_Weekly_Sales', 'Size', 'Sales_Volatility_CV', 'Holiday_Surge_Factor', 'Sales_Per_SqFt', 'Markdown_Reliance']
    X = profiles_df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    
    sil_score = float(silhouette_score(X_scaled, labels))
    db_score = float(davies_bouldin_score(X_scaled, labels))
    
    result_df = profiles_df.copy()
    result_df['Cluster'] = labels
    result_df['PCA1'] = pca_coords[:, 0]
    result_df['PCA2'] = pca_coords[:, 1]
    
    # Sort cluster IDs by Average Sales descending to ensure reproducible names
    cluster_means = result_df.groupby('Cluster')['Avg_Weekly_Sales'].mean().sort_values(ascending=False)
    rank_map = {orig_id: rank for rank, orig_id in enumerate(cluster_means.index)}
    
    persona_names = {
        0: 'Flagship Megastores (High Volume / Scale)',
        1: 'Suburban High-Efficiency Outlets',
        2: 'Compact Community & Value Stores'
    }
    result_df['Cluster_Rank'] = result_df['Cluster'].map(rank_map)
    result_df['Cluster_Persona'] = result_df['Cluster_Rank'].map(persona_names)
    
    persona_dict = {int(k): v for k, v in result_df.groupby('Cluster')['Cluster_Persona'].first().to_dict().items()}
    
    metrics = {
        'silhouette_score': round(sil_score, 4),
        'davies_bouldin_score': round(db_score, 4),
        'explained_variance_ratio': [round(float(v), 4) for v in pca.explained_variance_ratio_],
        'cluster_personas': persona_dict
    }
    
    return result_df, kmeans, scaler, pca, metrics
