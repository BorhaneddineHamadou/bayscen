import pandas as pd
import random

# Read CSV (skip Factor row names)
df = pd.read_csv("test_scenarios_scenario1.csv", skiprows=1, header=None)

# Rename columns (first one is PathInteraction now)
df.columns = [
    "PathInteraction", "PrecipitationDeposits", "RoadFriction", "WindIntensity", 
    "Wetness", "Precipitation", "FogDistance", "FogDensity", "Cloudiness"
]

# Define the mappings
path_map = {
    "PathInteraction_c1": [
        {"StartEgo": "left", "GoalEgo": "right", "StartOther": "base", "GoalOther": "left"},
        {"StartEgo": "left", "GoalEgo": "right", "StartOther": "base", "GoalOther": "right"},
        {"StartEgo": "base", "GoalEgo": "left", "StartOther": "left", "GoalOther": "right"},
        {"StartEgo": "base", "GoalEgo": "right", "StartOther": "left", "GoalOther": "right"},
    ],
    "PathInteraction_c2": [
        {"StartEgo": "right", "GoalEgo": "left", "StartOther": "base", "GoalOther": "left"},
        {"StartEgo": "right", "GoalEgo": "base", "StartOther": "base", "GoalOther": "left"},
        {"StartEgo": "base", "GoalEgo": "left", "StartOther": "right", "GoalOther": "left"},
        {"StartEgo": "base", "GoalEgo": "left", "StartOther": "right", "GoalOther": "base"},
    ],
    "PathInteraction_c4": [
        {"StartEgo": "left", "GoalEgo": "right", "StartOther": "right", "GoalOther": "base"},
        {"StartEgo": "left", "GoalEgo": "base", "StartOther": "right", "GoalOther": "base"},
        {"StartEgo": "right", "GoalEgo": "base", "StartOther": "left", "GoalOther": "right"},
        {"StartEgo": "right", "GoalEgo": "base", "StartOther": "left", "GoalOther": "base"},
    ]
}

# Shuffle each set once → ensures diversity but random order
for k in path_map:
    random.shuffle(path_map[k])

# Keep counters for cycling
path_counters = {key: 0 for key in path_map}

def assign_path(row):
    path_type = row["PathInteraction"]
    if path_type in path_map:
        idx = path_counters[path_type]
        choice = path_map[path_type][idx]
        path_counters[path_type] = (idx + 1) % len(path_map[path_type])
        for key, val in choice.items():
            row[key] = val
    else:
        # If path type missing, fill with NaN
        row["StartEgo"] = row["GoalEgo"] = row["StartOther"] = row["GoalOther"] = None
    return row

# Apply mapping
df = df.apply(assign_path, axis=1)

# Function to extract numeric values
def extract_value(val):
    if isinstance(val, str) and val.startswith("RoadFriction_"):
        return float(val.split("_")[-1])
    elif isinstance(val, str) and val.startswith("PathInteraction_"):
        return str(val.split("_")[-1])
    elif isinstance(val, str) and "_" in val:
        return int(val.split("_")[-1])
    return val

for col in df.columns:  
    df[col] = df[col].apply(extract_value)

# Reorder columns for clarity
df = df[
    ["PathInteraction", "StartEgo", "GoalEgo", "StartOther", "GoalOther",
     "RoadFriction", "FogDensity", "Precipitation", "PrecipitationDeposits",
     "Cloudiness", "WindIntensity", "Wetness", "FogDistance"]
]

# Save to Excel
df.to_excel("processed_test_scenarios_scenario1.xlsx", index=False)

print("Excel file 'processed_test_scenarios_scenario1.xlsx' created successfully!")
