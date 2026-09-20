from nba_api.stats.endpoints import leaguedashplayerstats, commonallplayers
import pandas as pd

# 1. Pull core data (Base + Advanced + Usage)
base_df = leaguedashplayerstats.LeagueDashPlayerStats(season='2024-25', measure_type_detailed_defense='Base').get_data_frames()[0]
usage_df = leaguedashplayerstats.LeagueDashPlayerStats(season='2024-25', measure_type_detailed_defense='Usage').get_data_frames()[0]

# 2. Merge all data to see all useful columns
print("--- Base Stats Columns ---")
print(list(base_df.columns))

print("\n--- Usage Stats Columns ---")
print(list(usage_df.columns))

# 3. Save CSV files for preview
base_df.to_head_csv = base_df.head(10)
base_df.to_csv("nba_base_preview.csv", index=False)
usage_df.to_csv("nba_usage_preview.csv", index=False)
print("\nSaved preview CSV files in backend folder successfully!")