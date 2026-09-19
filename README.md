# Integrated Retail Analytics for Store Optimization and Demand Forecasting

An end-to-end production-grade Machine Learning and Retail Analytics system designed to optimize store performance, forecast departmental demand, cluster store operating profiles, discover cross-selling product affinities, and formulate dynamic inventory strategies.

---

## Project Overview

Multi-channel retail operations face an ongoing dilemma: ensuring high product availability during holiday shopping peaks while avoiding excessive inventory holding costs and avoidable markdown liquidations during regular trading periods. 

This project integrates sales records across **45 stores** and **81 departments** spanning February 2010 through October 2012 (**421,570 weekly transaction records**) with internal promotional markdown streams (MarkDown 1–5), store physical characteristics (Types A, B, C; square footage), and external macroeconomic drivers (CPI, Unemployment rate, Fuel price, Temperature).

### Key Accomplishments & Benchmark Results
- **Anomaly Detection & Data Cleansing**: Contextually separated legitimate holiday surges (Thanksgiving, Christmas, Super Bowl) from negative return records and data transmission errors.
- **Strategic Store Segmentation**: Clustered stores into 3 operational personas (**Flagship Megastores**, **Suburban High-Efficiency Outlets**, **Compact Value Stores**) with a Silhouette Score of **0.3485**.
- **Market Basket / Departmental Affinity Mining**: Discovered **865 strong association rules** via FP-Growth (e.g. Dept 90 Pet Supplies & Dept 92 Grocery with **3.73x Lift**, **93% Confidence**).
- **High-Precision Demand Forecasting**: Tuned **LightGBM Regressor** achieved state-of-the-art performance with **Test WMAE = $1,369.35**, **RMSE = $2,889.80**, and **$R^2 = 0.9830** on out-of-time test data.
- **Dynamic Inventory Optimization**: Automated Safety Stock ($SS$) and Reorder Point ($ROP$) calculations with dynamic 95% to 99% service-level buffers for holiday surges.

---

## Model Benchmark Summary (Out-of-Time Test Set)

| Model Architecture | Test WMAE ($) | Test MAE ($) | Test RMSE ($) | Test $R^2$ Score | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | \$2,585.56 | \$2,447.80 | \$4,041.32 | 0.9668 | 21.45% |
| **Random Forest Regressor** | \$1,370.66 | \$1,241.15 | \$2,916.86 | 0.9827 | 10.82% |
| **Tuned LightGBM Regressor (Best)**| **\$1,369.35** | **\$1,238.40** | **\$2,889.80** | **0.9830** | **10.64%** |

---

## Repository Structure

```
├── dataset/                               # Source raw datasets
│   ├── Features data set.csv
│   ├── sales data-set.csv
│   └── stores data-set.csv
├── models/                                # Persisted model artifacts & metadata
│   ├── best_forecasting_model.pkl         # Tuned LightGBM forecaster
│   ├── kmeans_store_segmentation.pkl      # Store clustering model
│   ├── scaler_segmentation.pkl            # Feature scaler
│   ├── evaluation_metrics.json            # Model performance benchmark JSON
│   └── feature_columns.json               # 44 feature column names
├── reports/                               # Strategic reports and data deliverables
│   ├── figures/                           # 17 High-resolution analytical charts
│   ├── inventory_optimization_recommendations.csv
│   ├── market_basket_association_rules.csv
│   ├── store_segmentation_profiles.csv
│   ├── strategic_recommendations.json
│   └── Integrated_Retail_Analytics_Comprehensive_Report.md
├── src/                                   # Production python modules
│   ├── __init__.py
│   ├── data_loader.py                     # Data ingestion & merging
│   ├── anomaly_detector.py                # IQR, Z-score, Isolation Forest
│   ├── segmentation.py                    # K-Means, PCA, Silhouette profiling
│   ├── market_basket.py                   # FP-Growth association rule mining
│   ├── feature_engineering.py             # Lags, rolling windows, date features
│   ├── demand_forecaster.py               # Ridge, Random Forest, LightGBM, WMAE
│   └── strategy_optimizer.py              # Dynamic SS & ROP calculation
├── template/                              # Capstone submission templates
│   ├── Sample EDA Submission Template.ipynb
│   └── Sample ML Submission Template.ipynb
├── EDA_Integrated_Retail_Analytics.ipynb   # Executed EDA Notebook (166 cells)
├── ML_Integrated_Retail_Analytics.ipynb    # Executed ML Notebook (306 cells)
├── run_pipeline.py                        # End-to-end master CLI pipeline
├── requirements.txt                       # Python dependencies
└── README.md                              # Repository documentation
```

---

## Quick Start & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
```bash
python run_pipeline.py
```
This executes data ingestion, anomaly detection, store segmentation, market basket mining, feature engineering, model training, evaluation metrics generation, and renders all 17 high-resolution figures into `reports/figures/`.

### 3. Explore Jupyter Notebooks
Open either of the two fully executed capstone submission notebooks:
- `EDA_Integrated_Retail_Analytics.ipynb`: Comprehensive exploratory data analysis, 15 visual charts with UBM rule, and diagnostic evaluations.
- `ML_Integrated_Retail_Analytics.ipynb`: Complete statistical hypothesis testing, data preprocessing, feature engineering, 3 ML models, hyperparameter tuning, model persistence, and test inference.

---

## Deliverables & Documentation Links
- **Comprehensive Project Report**: [Integrated_Retail_Analytics_Comprehensive_Report.md](reports/Integrated_Retail_Analytics_Comprehensive_Report.md)
- **Executed EDA Notebook**: [EDA_Integrated_Retail_Analytics.ipynb](EDA_Integrated_Retail_Analytics.ipynb)
- **Executed ML Notebook**: [ML_Integrated_Retail_Analytics.ipynb](ML_Integrated_Retail_Analytics.ipynb)
- **Saved Best Model**: `models/best_forecasting_model.pkl`
