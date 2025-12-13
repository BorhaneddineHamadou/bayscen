import json
import pandas as pd
import os
import sys

def main():
    # 1. Get the directory where this script is located
    # This replaces os.getcwd() to make it robust when running from other locations
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Script running in: {current_dir}")

    # Helper function to load JSON files from the script's directory
    def load_json(filename):
        path = os.path.join(current_dir, filename)
        if not os.path.exists(path):
            print(f"Error: Could not find {filename} in {current_dir}")
            sys.exit(1)
        with open(path, 'r') as f:
            return json.load(f)

    print("Loading JSON files...")
    
    # Load all JSON files using the safe path
    conflict_points = load_json('conflic_points.json')
    ego_goal_locations = load_json('ego_goal_locations.json')
    ego_start_locations = load_json('ego_start_locations.json')
    other_goal_locations = load_json('other_goal_locations.json')
    other_start_locations = load_json('other_start_locations.json')
    weather_states = load_json('weather_states.json')

    # Combine all data into rows
    print("Processing data...")
    data = []
    
    # We assume all lists have the same length based on the original logic
    for i in range(len(conflict_points)):
        row = {
            "PathInteraction": conflict_points[i]["conflict_point"],
            "GoalEgo": ego_goal_locations[i]["ego_goal_location"],
            "WindIntensity": weather_states[i]["weather_states"]["wind_intensity"],
            "RoadFriction": weather_states[i]["weather_states"]["friction"],
            "FogDistance": weather_states[i]["weather_states"]["fog_distance"],
            "StartOther": other_start_locations[i]["other_start_location"],
            "PrecipitationDeposits": weather_states[i]["weather_states"]["precipitation_deposits"],
            "Cloudiness": weather_states[i]["weather_states"]["cloudiness"],
            "FogDensity": weather_states[i]["weather_states"]["fog_density"],
            "GoalOther": other_goal_locations[i]["other_goal_location"],
            "Precipitation": weather_states[i]["weather_states"]["precipitation"],
            "Wetness": weather_states[i]["weather_states"]["wetness"],
            "TimeOfDay": weather_states[i]["weather_states"]["sun_altitude_angle"],
            "StartEgo": ego_start_locations[i]["ego_start_location"]
        }
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # 2. Construct the relative path to 'generated_scenarios'
    # We go up 3 levels: collect_scenarios -> sitcov -> outputs -> scenario2_interfuser -> generated_scenarios
    target_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "generated_scenarios"))

    # 3. Create the directory if it doesn't exist
    if not os.path.exists(target_dir):
        print(f"Creating directory: {target_dir}")
        os.makedirs(target_dir)

    # 4. Define the full file path
    file_path = os.path.join(target_dir, "sitcov.xlsx")

    print(f"Saving Excel file to: {file_path}")

    # 5. Save the Excel file
    try:
        df.to_excel(file_path, index=False)
        print("Success! File saved.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()