"""
End-to-End Execution Pipeline for Integrated Retail Analytics
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data, merge_datasets, get_data_summary
from src.anomaly_detector import detect_iqr_anomalies, detect_zscore_anomalies, detect_isolation_forest_anomalies, classify_anomalies, clean_anomalies_for_training
from src.segmentation import create_store_profiles, perform_store_clustering
from src.market_basket import create_department_affinity_matrix, mine_department_rules
from src.feature_engineering import engineer_all_features, get_feature_columns
from src.demand_forecaster import evaluate_predictions, train_ridge_model, train_rf_model, train_lightgbm_model
from src.strategy_optimizer import compute_inventory_parameters, generate_strategic_recommendations

# Configure plotting styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11, 'figure.autolayout': True})

def main():
    print("==========================================================")
    print("  INTEGRATED RETAIL ANALYTICS - END-TO-END PIPELINE")
    print("==========================================================")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    # ---------------------------------------------------------
    # STEP 1: Load and Merge Data
    # ---------------------------------------------------------
    print("\n[STEP 1/7] Loading and Merging Datasets...")
    sales_df, features_df, stores_df = load_raw_data('dataset')
    merged_df = merge_datasets(sales_df, features_df, stores_df)
    summary = get_data_summary(merged_df)
    print(f"  Merged Dataset Shape: {merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns")
    print(f"  Stores: {summary['num_stores']} | Departments: {summary['num_depts']}")
    print(f"  Date Range: {summary['date_min'][:10]} to {summary['date_max'][:10]}")
    
    # ---------------------------------------------------------
    # STEP 2: Anomaly Detection & Classification
    # ---------------------------------------------------------
    print("\n[STEP 2/7] Performing Anomaly Detection...")
    classified_df = classify_anomalies(merged_df)
    anomaly_counts = classified_df['Anomaly_Category'].value_counts().to_dict()
    print(f"  Anomaly Breakdown: {anomaly_counts}")
    
    # Clean dataset for robust model training
    cleaned_df = clean_anomalies_for_training(classified_df, clip_negatives=True, winsorize_upper=0.999)
    print("  Cleaned negative sales and extreme data glitches.")
    
    # ---------------------------------------------------------
    # STEP 3: Customer / Store Segmentation Analysis
    # ---------------------------------------------------------
    print("\n[STEP 3/7] Performing Store & Department Clustering...")
    store_profiles = create_store_profiles(merged_df)
    clustered_stores, kmeans_model, scaler_model, pca_model, cluster_metrics = perform_store_clustering(store_profiles, n_clusters=3, random_state=42)
    print(f"  Store Silhouette Score: {cluster_metrics['silhouette_score']} | Davies-Bouldin: {cluster_metrics['davies_bouldin_score']}")
    print(f"  Cluster Personas: {cluster_metrics['cluster_personas']}")
    
    # Save segmentation models
    joblib.dump(kmeans_model, 'models/kmeans_store_segmentation.pkl')
    joblib.dump(scaler_model, 'models/scaler_segmentation.pkl')
    clustered_stores.to_csv('reports/store_segmentation_profiles.csv', index=False)
    
    # ---------------------------------------------------------
    # STEP 4: Market Basket / Department Association Mining
    # ---------------------------------------------------------
    print("\n[STEP 4/7] Mining Market Basket & Department Association Rules...")
    binary_matrix, corr_matrix = create_department_affinity_matrix(merged_df, threshold_percentile=75.0)
    rules_df = mine_department_rules(binary_matrix, min_support=0.03, min_confidence=0.3, min_lift=1.1)
    print(f"  Discovered {len(rules_df)} strong departmental association rules.")
    if len(rules_df) > 0:
        print("  Top 3 Department Association Rules by Lift:")
        for idx, row in rules_df.head(3).iterrows():
            print(f"    [{row['antecedents_str']}] => [{row['consequents_str']}] (Lift: {row['lift']:.2f}, Conf: {row['confidence']:.2f}, Supp: {row['support']:.2f})")
    rules_df.to_csv('reports/market_basket_association_rules.csv', index=False)
    
    # ---------------------------------------------------------
    # STEP 5: Feature Engineering & Preprocessing
    # ---------------------------------------------------------
    print("\n[STEP 5/7] Engineering Advanced Features & Lags...")
    fe_df = engineer_all_features(cleaned_df)
    feature_cols = get_feature_columns()
    print(f"  Total Engineered Features: {len(feature_cols)}")
    
    # Out-of-time train/test split (split at 2012-02-03)
    train_mask = fe_df['Date'] < '2012-02-03'
    test_mask = fe_df['Date'] >= '2012-02-03'
    
    X_train = fe_df.loc[train_mask, feature_cols]
    y_train = fe_df.loc[train_mask, 'Weekly_Sales']
    is_holiday_train = fe_df.loc[train_mask, 'IsHoliday'].values
    
    X_test = fe_df.loc[test_mask, feature_cols]
    y_test = fe_df.loc[test_mask, 'Weekly_Sales']
    is_holiday_test = fe_df.loc[test_mask, 'IsHoliday'].values
    
    print(f"  Train Set: {len(X_train):,} samples (Feb 2010 - Jan 2012)")
    print(f"  Test Set:  {len(X_test):,} samples (Feb 2012 - Oct 2012)")
    
    # ---------------------------------------------------------
    # STEP 6: ML Model Implementation & Tuning
    # ---------------------------------------------------------
    print("\n[STEP 6/7] Training and Evaluating ML Demand Forecasters...")
    
    # Model 1: Regularized Ridge Regression
    print("  -> Training Model 1: Ridge Regression (Baseline)...")
    ridge_model = train_ridge_model(X_train, y_train, alpha=10.0)
    ridge_preds_train = ridge_model.predict(X_train)
    ridge_preds_test = ridge_model.predict(X_test)
    ridge_eval = evaluate_predictions(y_test, ridge_preds_test, is_holiday_test)
    print(f"     Ridge Test Results: WMAE={ridge_eval['WMAE']}, RMSE={ridge_eval['RMSE']}, R2={ridge_eval['R2']}")
    
    # Model 2: Random Forest Regressor
    print("  -> Training Model 2: Random Forest Regressor...")
    rf_model = train_rf_model(X_train, y_train, n_estimators=60, max_depth=16, random_state=42)
    rf_preds_test = rf_model.predict(X_test)
    rf_eval = evaluate_predictions(y_test, rf_preds_test, is_holiday_test)
    print(f"     Random Forest Test Results: WMAE={rf_eval['WMAE']}, RMSE={rf_eval['RMSE']}, R2={rf_eval['R2']}")
    
    # Model 3: LightGBM Regressor (State-of-the-Art)
    print("  -> Training Model 3: Tuned LightGBM Regressor...")
    lgb_params = {
        'n_estimators': 400,
        'learning_rate': 0.04,
        'num_leaves': 63,
        'max_depth': 8,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    lgb_model = train_lightgbm_model(X_train, y_train, X_test, y_test, params=lgb_params)
    lgb_preds_test = lgb_model.predict(X_test)
    lgb_eval = evaluate_predictions(y_test, lgb_preds_test, is_holiday_test)
    print(f"     LightGBM Test Results: WMAE={lgb_eval['WMAE']}, RMSE={lgb_eval['RMSE']}, R2={lgb_eval['R2']}")
    
    # Model Evaluation Summary
    model_comparison = {
        'Ridge Regression': ridge_eval,
        'Random Forest': rf_eval,
        'LightGBM Regressor (Best)': lgb_eval
    }
    with open('models/evaluation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(model_comparison, f, indent=4)
        
    with open('models/feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, indent=4)
        
    # Save the Best Performing Model (LightGBM)
    joblib.dump(lgb_model, 'models/best_forecasting_model.pkl')
    print("  Saved best model to 'models/best_forecasting_model.pkl'")
    
    # ---------------------------------------------------------
    # STEP 7: Strategy & Inventory Optimization
    # ---------------------------------------------------------
    print("\n[STEP 7/7] Generating Strategic Optimizations & Visualizations...")
    inv_df = compute_inventory_parameters(merged_df, lead_time_weeks=2.0, service_level_z=1.65)
    inv_df.to_csv('reports/inventory_optimization_recommendations.csv', index=False)
    strategic_playbook = generate_strategic_recommendations()
    with open('reports/strategic_recommendations.json', 'w', encoding='utf-8') as f:
        json.dump(strategic_playbook, f, indent=4)
        
    # ---------------------------------------------------------
    # GENERATE COMPREHENSIVE VISUAL CHARTS (Reports & EDA)
    # ---------------------------------------------------------
    print("  Rendering High-Resolution Visual Reports...")
    
    # 1. Weekly Sales Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(merged_df['Weekly_Sales'], bins=80, kde=True, color='#2b5c8f')
    plt.title("Distribution of Weekly Sales across All Stores and Departments", fontsize=13, weight='bold')
    plt.xlabel("Weekly Sales ($)")
    plt.ylabel("Frequency")
    plt.xlim(0, 100000)
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart1_weekly_sales_distribution.png", dpi=300)
    plt.close()
    
    # 2. Sales by Store Type
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    type_totals = merged_df.groupby('Type')['Weekly_Sales'].sum() / 1e6
    sns.barplot(x=type_totals.index, y=type_totals.values, palette=['#1f77b4', '#aec7e8', '#ff7f0e'])
    plt.title("Total Revenue by Store Type ($M)", weight='bold')
    plt.ylabel("Total Sales ($ Millions)")
    plt.subplot(1, 2, 2)
    sns.boxplot(data=merged_df, x='Type', y='Weekly_Sales', showfliers=False, palette=['#1f77b4', '#aec7e8', '#ff7f0e'])
    plt.title("Weekly Sales Spread by Store Type", weight='bold')
    plt.ylabel("Weekly Sales ($)")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart2_sales_by_store_type.png", dpi=300)
    plt.close()
    
    # 3. Total Sales by Store
    plt.figure(figsize=(14, 6))
    store_sales = merged_df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False) / 1e6
    sns.barplot(x=store_sales.index.astype(str), y=store_sales.values, palette="Blues_r")
    plt.title("Total Cumulative Sales by Store ($ Millions)", fontsize=14, weight='bold')
    plt.xlabel("Store ID (Sorted by Performance)")
    plt.ylabel("Total Sales ($M)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart3_sales_by_store.png", dpi=300)
    plt.close()
    
    # 4. Top 15 Departments by Sales
    plt.figure(figsize=(12, 6))
    top_depts = merged_df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False).head(15) / 1e6
    sns.barplot(x=top_depts.index.astype(str), y=top_depts.values, palette="crest_r")
    plt.title("Top 15 Highest Revenue Generating Departments ($ Millions)", fontsize=14, weight='bold')
    plt.xlabel("Department ID")
    plt.ylabel("Cumulative Sales ($M)")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart4_top_departments.png", dpi=300)
    plt.close()
    
    # 5. Holiday vs Non-Holiday Sales
    plt.figure(figsize=(9, 5))
    holiday_comp = merged_df.groupby('IsHoliday')['Weekly_Sales'].mean()
    sns.barplot(x=['Non-Holiday Week', 'Holiday Week'], y=holiday_comp.values, palette=['#7293cb', '#e15759'])
    plt.title("Average Weekly Sales: Holiday vs. Non-Holiday", fontsize=13, weight='bold')
    plt.ylabel("Average Weekly Sales ($)")
    for i, v in enumerate(holiday_comp.values):
        plt.text(i, v + 200, f"${v:,.2f}", ha='center', weight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart5_holiday_vs_nonholiday.png", dpi=300)
    plt.close()
    
    # 6. Overall Time Series Sales Trend
    plt.figure(figsize=(14, 5))
    weekly_total = merged_df.groupby('Date')['Weekly_Sales'].sum() / 1e6
    plt.plot(weekly_total.index, weekly_total.values, color='#2b5c8f', lw=2.2, label='Total Weekly Sales ($M)')
    plt.title("Store Network Total Sales Trend Over Time (2010 - 2012)", fontsize=14, weight='bold')
    plt.xlabel("Date")
    plt.ylabel("Total Network Sales ($M)")
    plt.axvline(pd.to_datetime('2010-11-26'), color='red', linestyle='--', alpha=0.7, label='Thanksgiving / Black Friday')
    plt.axvline(pd.to_datetime('2011-11-25'), color='red', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart6_time_series_trends.png", dpi=300)
    plt.close()
    
    # 7. MarkDown Impact Analysis
    plt.figure(figsize=(11, 5))
    md_sums = merged_df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].sum() / 1e6
    sns.barplot(x=md_sums.index, y=md_sums.values, palette="magma")
    plt.title("Total Promotional Markdown Spend by Category ($ Millions)", fontsize=13, weight='bold')
    plt.xlabel("MarkDown Type")
    plt.ylabel("Total Markdown Amount ($M)")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart7_markdown_impact.png", dpi=300)
    plt.close()
    
    # 8. Temperature vs Sales
    plt.figure(figsize=(10, 5))
    merged_df['Temp_Bin'] = pd.cut(merged_df['Temperature'], bins=8)
    temp_sales = merged_df.groupby('Temp_Bin', observed=False)['Weekly_Sales'].mean()
    sns.barplot(x=[str(b) for b in temp_sales.index], y=temp_sales.values, palette="coolwarm")
    plt.title("Average Weekly Sales across Temperature Bins (°F)", fontsize=13, weight='bold')
    plt.xlabel("Temperature Range (°F)")
    plt.ylabel("Mean Weekly Sales ($)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart8_temperature_vs_sales.png", dpi=300)
    plt.close()
    
    # 9. Fuel Price vs Sales
    plt.figure(figsize=(11, 5))
    fuel_sales = merged_df.groupby(pd.qcut(merged_df['Fuel_Price'], q=6, duplicates='drop'), observed=False)['Weekly_Sales'].mean()
    sns.barplot(x=[str(b) for b in fuel_sales.index], y=fuel_sales.values, palette="viridis")
    plt.title("Average Weekly Sales by Fuel Price Quantile ($/Gallon)", fontsize=13, weight='bold')
    plt.xlabel("Fuel Price Quantile ($)")
    plt.ylabel("Mean Weekly Sales ($)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart9_fuel_price_vs_sales.png", dpi=300)
    plt.close()
    
    # 10. CPI & Unemployment Impact
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.regplot(data=store_profiles, x='Avg_CPI', y='Avg_Weekly_Sales', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.title("Store Avg Sales vs Consumer Price Index (CPI)", weight='bold')
    plt.subplot(1, 2, 2)
    sns.regplot(data=store_profiles, x='Avg_Unemployment', y='Avg_Weekly_Sales', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.title("Store Avg Sales vs Unemployment Rate (%)", weight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart10_cpi_unemployment_scatter.png", dpi=300)
    plt.close()
    
    # 11. Store Size vs Sales
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=store_profiles, x='Size', y='Avg_Weekly_Sales', hue='Type', style='Type', s=130, palette=['#1f77b4', '#2ca02c', '#d62728'])
    plt.title("Store Physical Footprint (Size in Sq Ft) vs Average Weekly Sales", fontsize=13, weight='bold')
    plt.xlabel("Store Physical Size (Sq Ft)")
    plt.ylabel("Average Weekly Sales ($)")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart11_store_size_vs_sales.png", dpi=300)
    plt.close()
    
    # 12. Monthly Seasonality
    plt.figure(figsize=(11, 5))
    fe_df['Month_Name'] = fe_df['Date'].dt.strftime('%b')
    fe_df['Year_Str'] = fe_df['Date'].dt.year.astype(str)
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_sales = fe_df.groupby(['Month_Name', 'Year_Str'], observed=False)['Weekly_Sales'].mean().unstack().reindex(month_order)
    monthly_sales.plot(kind='bar', figsize=(11, 5), colormap='Blues')
    plt.title("Monthly Seasonality of Average Weekly Sales across Years", fontsize=13, weight='bold')
    plt.xlabel("Month")
    plt.ylabel("Average Weekly Sales ($)")
    plt.legend(title="Year")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart12_monthly_seasonality.png", dpi=300)
    plt.close()
    
    # 13. Anomaly Classification Breakdown
    plt.figure(figsize=(10, 5))
    sns.countplot(data=classified_df, y='Anomaly_Category', palette="Set2", order=classified_df['Anomaly_Category'].value_counts().index)
    plt.title("Classification Breakdown of Sales Data Patterns & Anomalies", fontsize=13, weight='bold')
    plt.xlabel("Record Count")
    plt.ylabel("Anomaly Category")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart13_anomaly_distribution.png", dpi=300)
    plt.close()
    
    # 14. Correlation Heatmap
    plt.figure(figsize=(11, 9))
    num_cols = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Total_MarkDown', 'IsHoliday_Int', 'Sales_Lag_1', 'Sales_Rolling_Mean_4']
    corr = fe_df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Comprehensive Numerical Correlation Heatmap", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/eda_chart14_correlation_heatmap.png", dpi=300)
    plt.close()
    
    # 15. Store Segmentation PCA Cluster Map
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clustered_stores, x='PCA1', y='PCA2', hue='Cluster_Persona', style='Cluster_Persona', s=160, palette="tab10")
    for idx, row in clustered_stores.iterrows():
        plt.text(row['PCA1'] + 0.08, row['PCA2'], str(int(row['Store'])), fontsize=9, alpha=0.85)
    plt.title("Store Clusters in 2D PCA Space (Strategic Segmentation)", fontsize=13, weight='bold')
    plt.xlabel(f"PCA Component 1 ({cluster_metrics['explained_variance_ratio'][0]*100:.1f}% Variance)")
    plt.ylabel(f"PCA Component 2 ({cluster_metrics['explained_variance_ratio'][1]*100:.1f}% Variance)")
    plt.legend(title="Store Segment Persona")
    plt.tight_layout()
    plt.savefig("reports/figures/store_segmentation_pca.png", dpi=300)
    plt.close()
    
    # 16. Model Comparison Bar Chart
    plt.figure(figsize=(12, 5))
    metrics_plot_df = pd.DataFrame(model_comparison).T[['WMAE', 'MAE', 'RMSE']]
    metrics_plot_df.plot(kind='bar', figsize=(11, 5), colormap='viridis')
    plt.title("Model Error Comparison on Test Set (Lower is Better)", fontsize=13, weight='bold')
    plt.ylabel("Error Score ($)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison_wmae_rmse.png", dpi=300)
    plt.close()
    
    # 17. Feature Importance of Best Model (LightGBM)
    plt.figure(figsize=(12, 7))
    feat_imp = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="crest_r")
    plt.title("Top 20 Predictive Features - LightGBM Demand Forecaster", fontsize=14, weight='bold')
    plt.xlabel("Feature Importance (Split Gain)")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png", dpi=300)
    plt.close()
    
    print("\n==========================================================")
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("==========================================================")

if __name__ == '__main__':
    main()
