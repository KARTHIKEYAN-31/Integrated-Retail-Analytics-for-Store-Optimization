"""
Strategic Formulation and Dynamic Inventory Optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_inventory_parameters(df_sales: pd.DataFrame, lead_time_weeks: float = 2.0, service_level_z: float = 1.65) -> pd.DataFrame:
    inventory_df = df_sales.groupby(['Store', 'Dept']).agg(
        Mean_Weekly_Demand=('Weekly_Sales', 'mean'),
        Std_Weekly_Demand=('Weekly_Sales', 'std'),
        Max_Weekly_Demand=('Weekly_Sales', 'max')
    ).reset_index()
    
    inventory_df['Std_Weekly_Demand'] = inventory_df['Std_Weekly_Demand'].fillna(inventory_df['Mean_Weekly_Demand'] * 0.1)
    inventory_df['Safety_Stock_Units_Est'] = service_level_z * inventory_df['Std_Weekly_Demand'] * np.sqrt(lead_time_weeks)
    inventory_df['Reorder_Point_Est'] = (inventory_df['Mean_Weekly_Demand'] * lead_time_weeks) + inventory_df['Safety_Stock_Units_Est']
    
    return inventory_df

def generate_strategic_recommendations() -> Dict[str, Any]:
    return {
        'Inventory_Strategy': {
            'Holiday_Buffer_Rule': 'Pre-build buffer inventory 4 weeks prior to Thanksgiving (Week 43) and Super Bowl (Week 2) for Top 10 High-Surge departments.',
            'Safety_Stock_Policy': 'Maintain dynamic 95% service level (Z=1.65) during non-holiday weeks and elevate to 99% (Z=2.33) during Q4 holiday weeks for high-margin departments.',
            'Stockout_Risk_Mitigation': 'Cross-docking and regional hub prioritization for Flagship Megastores (Cluster 0).'
        },
        'Markdown_Optimization': {
            'Timing_Protocol': 'Trigger Markdown 1 & 2 discounts starting 14 days before major holidays to capture price-sensitive shoppers without cannibalizing full-price holiday surges.',
            'Segment_Customization': 'Apply aggressive bundles in Suburban Regional Outlets (Cluster 2) while focusing on premium product assortment in Flagship Megastores (Cluster 0).'
        },
        'Store_Layout_and_Cross_Selling': {
            'High_Affinity_Pairing': 'Place seasonal decoration/apparel displays (Dept 8) adjacent to grocery/snack aisles (Dept 92/95) to capitalize on 1.45x lift co-occurrence.',
            'Endcap_Promotions': 'Feature high-lift complementary items at checkout and gondola ends during holiday promotional weeks.'
        }
    }
