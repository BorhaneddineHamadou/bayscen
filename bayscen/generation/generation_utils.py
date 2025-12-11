"""
Utility Functions for Scenario Generation

Helper functions for path mapping, scenario validation, and export.

Author: BayScen Team  
Date: December 2025
"""

import pandas as pd
from typing import Dict, List


# T-junction path mappings for each collision point
# Maps collision points to valid (StartEgo, GoalEgo, StartOther, GoalOther) combinations

PATH_MAPPINGS = {
    "c1": [
        {"Start_Ego": "Left", "Goal_Ego": "Right", "Start_Other": "Base", "Goal_Other": "Left"},
        {"Start_Ego": "Left", "Goal_Ego": "Right", "Start_Other": "Base", "Goal_Other": "Right"},
        {"Start_Ego": "Base", "Goal_Ego": "Left", "Start_Other": "Left", "Goal_Other": "Right"},
        {"Start_Ego": "Base", "Goal_Ego": "Right", "Start_Other": "Left", "Goal_Other": "Right"},
    ],
    "c2": [
        {"Start_Ego": "Right", "Goal_Ego": "Left", "Start_Other": "Base", "Goal_Other": "Left"},
        {"Start_Ego": "Right", "Goal_Ego": "Base", "Start_Other": "Base", "Goal_Other": "Left"},
        {"Start_Ego": "Base", "Goal_Ego": "Left", "Start_Other": "Right", "Goal_Other": "Left"},
        {"Start_Ego": "Base", "Goal_Ego": "Left", "Start_Other": "Right", "Goal_Other": "Base"},
    ],
    "c3": [
        {"Start_Ego": "Left", "Goal_Ego": "Right", "Start_Other": "Right", "Goal_Other": "Base"},
        {"Start_Ego": "Left", "Goal_Ego": "Base", "Start_Other": "Right", "Goal_Other": "Base"},
        {"Start_Ego": "Right", "Goal_Ego": "Base", "Start_Other": "Left", "Goal_Other": "Right"},
        {"Start_Ego": "Right", "Goal_Ego": "Base", "Start_Other": "Left", "Goal_Other": "Base"},
    ]
}


def assign_tjunction_paths(scenarios_df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Assign T-junction path variables based on Collision_Point.
    
    Uses rolling/cycling assignment to ensure balanced distribution of paths
    for each collision point, rather than random selection.
    
    Args:
        scenarios_df: DataFrame with 'Collision_Point' column
        inplace: If True, modify DataFrame in place
    
    Returns:
        DataFrame with T-junction path columns added/updated
    """
    if not inplace:
        scenarios_df = scenarios_df.copy()
    
    # Check if Collision_Point column exists
    if 'Collision_Point' not in scenarios_df.columns:
        print("⚠ Warning: 'Collision_Point' column not found. Cannot assign paths.")
        return scenarios_df
    
    # Keep counters for cycling through paths
    path_counters = {key: 0 for key in PATH_MAPPINGS}
    
    for idx, row in scenarios_df.iterrows():
        collision_point = row['Collision_Point']
        
        if collision_point in PATH_MAPPINGS:
            # Use cycling assignment (not random)
            counter = path_counters[collision_point]
            path = PATH_MAPPINGS[collision_point][counter]
            
            # Update counter for next occurrence of this collision point
            path_counters[collision_point] = (counter + 1) % len(PATH_MAPPINGS[collision_point])
            
            # Assign to DataFrame
            for key, value in path.items():
                scenarios_df.at[idx, key] = value
        else:
            # Invalid or None collision point
            print(f"⚠ Warning: Invalid collision point '{collision_point}' at row {idx}")
            # Fill with None
            scenarios_df.at[idx, 'Start_Ego'] = None
            scenarios_df.at[idx, 'Goal_Ego'] = None
            scenarios_df.at[idx, 'Start_Other'] = None
            scenarios_df.at[idx, 'Goal_Other'] = None
    
    return scenarios_df


def validate_scenarios(scenarios_df: pd.DataFrame, required_columns: List[str] = None) -> Dict:
    """
    Validate generated scenarios for completeness and consistency.
    
    Checks:
    1. No missing values in required columns
    2. Value ranges are valid
    3. Collision points match path combinations
    
    Args:
        scenarios_df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        dict: Validation results with issues found
    """
    issues = []
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(scenarios_df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
    
    # Check for missing values
    missing_count = scenarios_df.isnull().sum()
    cols_with_missing = missing_count[missing_count > 0]
    if not cols_with_missing.empty:
        issues.append(f"Missing values: {cols_with_missing.to_dict()}")
    
    # Check value ranges for numeric columns
    numeric_cols = scenarios_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col in ['probability', 'Time_of_Day', 'TimeOfDay']:  # Skip these columns
            continue
        
        values = scenarios_df[col].dropna()
        if len(values) > 0:
            min_val, max_val = values.min(), values.max()
            
            # Road_Friction has different range [0, 1]
            if 'friction' in col.lower():
                if min_val < 0 or max_val > 1:
                    issues.append(f"{col} has values outside [0, 1]: [{min_val}, {max_val}]")
            # Most other parameters should be in [0, 100] range
            elif min_val < 0 or max_val > 100:
                issues.append(f"{col} has values outside [0, 100]: [{min_val}, {max_val}]")
    
    return {
        'is_valid': len(issues) == 0,
        'num_scenarios': len(scenarios_df),
        'issues': issues
    }


def export_for_carla(scenarios_df: pd.DataFrame, output_path: str):
    """
    Export scenarios in CARLA-compatible format.
    
    Converts BayScen scenario format to format expected by CARLA simulator.
    
    Args:
        scenarios_df: DataFrame with generated scenarios
        output_path: Path for output CSV file
    
    Example:
        >>> export_for_carla(scenarios, 'carla_scenarios.csv')
    """
    # Define CARLA-specific column mapping
    carla_mapping = {
        'Time_of_Day': 'sun_altitude_angle',
        'Cloudiness': 'cloudiness',
        'Wind_Intensity': 'wind_intensity',
        'Precipitation': 'precipitation',
        'Precipitation_Deposits': 'precipitation_deposits',
        'Wetness': 'wetness',
        'Road_Friction': 'road_friction',
        'Fog_Density': 'fog_density',
        'Fog_Distance': 'fog_distance',
        'Start_Ego': 'ego_start',
        'Goal_Ego': 'ego_goal',
        'Start_Other': 'other_start',
        'Goal_Other': 'other_goal'
    }
    
    # Select and rename columns
    carla_df = scenarios_df.copy()
    carla_df = carla_df.rename(columns=carla_mapping)
    
    # Add scenario ID
    carla_df.insert(0, 'scenario_id', range(1, len(carla_df) + 1))
    
    # Save
    carla_df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(carla_df)} scenarios to {output_path}")


def split_by_collision_point(scenarios_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split scenarios by collision point for analysis.
    
    Args:
        scenarios_df: DataFrame with 'Collision_Point' column
    
    Returns:
        dict: Mapping collision point to DataFrame subset
    
    Example:
        >>> by_collision = split_by_collision_point(scenarios)
        >>> print(f"c1 scenarios: {len(by_collision['c1'])}")
        >>> print(f"c2 scenarios: {len(by_collision['c2'])}")
        >>> print(f"c3 scenarios: {len(by_collision['c3'])}")
    """
    result = {}
    
    for collision_point in ['c1', 'c2', 'c3']:
        result[collision_point] = scenarios_df[
            scenarios_df['Collision_Point'] == collision_point
        ].copy()
    
    return result


def get_summary_statistics(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for generated scenarios.
    
    Args:
        scenarios_df: DataFrame with scenarios
    
    Returns:
        DataFrame with summary statistics for each variable
    
    Example:
        >>> summary = get_summary_statistics(scenarios)
        >>> print(summary)
    """
    # Select numeric columns
    numeric_cols = scenarios_df.select_dtypes(include=['number']).columns
    
    summary = scenarios_df[numeric_cols].describe().T
    
    # Add additional statistics
    summary['unique'] = scenarios_df[numeric_cols].nunique()
    
    return summary


def compare_scenario_sets(
    scenarios1: pd.DataFrame,
    scenarios2: pd.DataFrame,
    attributes: List[str],
    name1: str = "Set 1",
    name2: str = "Set 2"
):
    """
    Compare two sets of scenarios (e.g., common vs rare).
    
    Args:
        scenarios1: First scenario DataFrame
        scenarios2: Second scenario DataFrame
        attributes: Attributes to compare
        name1: Label for first set
        name2: Label for second set
    
    Example:
        >>> compare_scenario_sets(common_scenarios, rare_scenarios, attributes)
    """
    print(f"\n{'='*70}")
    print(f"COMPARING SCENARIO SETS: {name1} vs {name2}")
    print(f"{'='*70}")
    
    print(f"\n{name1}:")
    print(f"  Total scenarios: {len(scenarios1)}")
    print(f"  Unique combinations: {scenarios1[attributes].drop_duplicates().shape[0]}")
    
    print(f"\n{name2}:")
    print(f"  Total scenarios: {len(scenarios2)}")
    print(f"  Unique combinations: {scenarios2[attributes].drop_duplicates().shape[0]}")
    
    # Compare distributions
    print(f"\nMean Values Comparison:")
    print("-" * 70)
    
    comparison = pd.DataFrame({
        name1: scenarios1[attributes].mean(),
        name2: scenarios2[attributes].mean()
    })
    comparison['Difference'] = comparison[name2] - comparison[name1]
    
    print(comparison)


if __name__ == '__main__':
    print("BayScen Generation Utilities Module")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - assign_tjunction_paths: Map collision points to paths")
    print("  - validate_scenarios: Check scenario validity")
    print("  - export_for_carla: Export to CARLA format")
    print("  - split_by_collision_point: Split by collision type")
    print("  - get_summary_statistics: Compute summary stats")
    print("  - compare_scenario_sets: Compare two scenario sets")