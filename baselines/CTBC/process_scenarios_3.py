import pandas as pd
import random

# Read CSV (skip Factor row names)
df = pd.read_csv("test_scenarios_scenario3.csv", skiprows=1, header=None)

df.columns = [
    "direction", "cloudiness", "precipitation", "precipitationdeposits", "windintensity", "timeofday",  
    "fogdensity", "fogdistance", "wetness", "roadfriction"  
]

# Function to extract numeric values
def extract_value(val):
    if isinstance(val, str) and val.startswith("roadfriction_"):
        return float(val.split("_")[-1])
    elif isinstance(val, str) and val.startswith("direction_"):
        return str(val.split("_")[-1])
    elif isinstance(val, str) and "_" in val:
        return int(val.split("_")[-1])
    return val

for col in df.columns:  
    df[col] = df[col].apply(extract_value)

# Reorder columns for clarity
# df = df[
#     [  
#      "timeofday", "direction", 
#      "roadfriction", "fogdensity", "precipitation", "precipitationdeposits",
#      "cloudiness", "windintensity", "wetness", "fogdistance"]
# ]

# Save to Excel
df.to_excel("processed_test_scenarios_scenario3.xlsx", index=False)

print("Excel file 'processed_test_scenarios_scenario3.xlsx' created successfully!")
