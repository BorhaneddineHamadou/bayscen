"""
Utility Functions for Scenario Generation

Helper functions for path assignment, scenario validation, and export.
"""

import pandas as pd
from typing import Dict, List


# T-junction path assignments for each conflict geometry state.
# Maps g1/g2/g3 to representative (Start_Ego, Goal_Ego, Start_Other, Goal_Other) tuples.
# Multiple path combinations are rotated to ensure balanced representation.

PATH_MAPPINGS: Dict[str, List[Dict[str, str]]] = {
    "g1": [
        {"Start_Ego": "Left",  "Goal_Ego": "Right", "Start_Other": "Base", "Goal_Other": "Left"},
        {"Start_Ego": "Left",  "Goal_Ego": "Right", "Start_Other": "Base", "Goal_Other": "Right"},
        {"Start_Ego": "Base",  "Goal_Ego": "Left",  "Start_Other": "Left", "Goal_Other": "Right"},
        {"Start_Ego": "Base",  "Goal_Ego": "Right", "Start_Other": "Left", "Goal_Other": "Right"},
    ],
    "g2": [
        {"Start_Ego": "Right", "Goal_Ego": "Left",  "Start_Other": "Base", "Goal_Other": "Left"},
        {"Start_Ego": "Right", "Goal_Ego": "Base",  "Start_Other": "Base", "Goal_Other": "Left"},
        {"Start_Ego": "Base",  "Goal_Ego": "Left",  "Start_Other": "Right", "Goal_Other": "Left"},
        {"Start_Ego": "Base",  "Goal_Ego": "Left",  "Start_Other": "Right", "Goal_Other": "Base"},
    ],
    "g3": [
        {"Start_Ego": "Left",  "Goal_Ego": "Right", "Start_Other": "Right", "Goal_Other": "Base"},
        {"Start_Ego": "Left",  "Goal_Ego": "Base",  "Start_Other": "Right", "Goal_Other": "Base"},
        {"Start_Ego": "Right", "Goal_Ego": "Base",  "Start_Other": "Left",  "Goal_Other": "Right"},
        {"Start_Ego": "Right", "Goal_Ego": "Base",  "Start_Other": "Left",  "Goal_Other": "Base"},
    ],
}


def assign_junction_paths(scenarios_df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Assign T-junction path variables based on Conflict_Geometry state.

    Cycles through the representative path combinations for each geometry state
    (g1/g2/g3) to ensure a balanced distribution of trajectory assignments.

    Args:
        scenarios_df : DataFrame with a 'Conflict_Geometry' column.
        inplace      : If True, modify the DataFrame in place.

    Returns:
        DataFrame with Start_Ego, Goal_Ego, Start_Other, Goal_Other columns added.
    """
    if not inplace:
        scenarios_df = scenarios_df.copy()

    if 'Conflict_Geometry' not in scenarios_df.columns:
        print("Warning: 'Conflict_Geometry' column not found — cannot assign paths.")
        return scenarios_df

    counters = {k: 0 for k in PATH_MAPPINGS}

    for idx, row in scenarios_df.iterrows():
        geom = row['Conflict_Geometry']

        if geom in PATH_MAPPINGS:
            c = counters[geom]
            path = PATH_MAPPINGS[geom][c % len(PATH_MAPPINGS[geom])]
            counters[geom] = c + 1
            for key, val in path.items():
                scenarios_df.at[idx, key] = val
        else:
            print(f"Warning: Unknown conflict geometry '{geom}' at row {idx}")
            for col in ['Start_Ego', 'Goal_Ego', 'Start_Other', 'Goal_Other']:
                scenarios_df.at[idx, col] = None

    return scenarios_df


def validate_scenarios(
    scenarios_df: pd.DataFrame,
    required_columns: List[str] = None,
) -> Dict:
    """
    Check generated scenarios for completeness and value ranges.

    Args:
        scenarios_df     : DataFrame of generated scenarios.
        required_columns : Columns that must be present (all non-null).

    Returns:
        dict with keys 'is_valid', 'num_scenarios', 'issues'.
    """
    issues = []

    if required_columns:
        missing = set(required_columns) - set(scenarios_df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

    null_counts = scenarios_df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Null values: {cols_with_nulls.to_dict()}")

    for col in scenarios_df.select_dtypes(include=['number']).columns:
        if col in ('probability',):
            continue
        values = scenarios_df[col].dropna()
        if values.empty:
            continue
        lo, hi = values.min(), values.max()
        if 'friction' in col.lower():
            if lo < 0 or hi > 1:
                issues.append(f"{col} out of [0,1]: [{lo}, {hi}]")
        elif col == 'Sun_Altitude_Angle':
            if lo < -90 or hi > 90:
                issues.append(f"{col} out of [-90,90]: [{lo}, {hi}]")
        else:
            if lo < 0 or hi > 100:
                issues.append(f"{col} out of [0,100]: [{lo}, {hi}]")

    return {
        'is_valid':      len(issues) == 0,
        'num_scenarios': len(scenarios_df),
        'issues':        issues,
    }


def export_for_carla(scenarios_df: pd.DataFrame, output_path: str):
    """
    Export generated scenarios in CARLA-compatible column format.

    Args:
        scenarios_df : DataFrame with BayScen scenario columns.
        output_path  : Path for the output CSV file.
    """
    carla_mapping = {
        'Sun_Altitude_Angle':    'sun_altitude_angle',
        'Cloudiness':            'cloudiness',
        'Wind_Intensity':        'wind_intensity',
        'Precipitation':         'precipitation',
        'Precipitation_Deposits':'precipitation_deposits',
        'Wetness':               'wetness',
        'Road_Friction':         'road_friction',
        'Fog_Density':           'fog_density',
        'Fog_Distance':          'fog_distance',
        'Start_Ego':             'ego_start',
        'Goal_Ego':              'ego_goal',
        'Start_Other':           'other_start',
        'Goal_Other':            'other_goal',
        'Cut_In_Direction':      'cut_in_direction',
    }

    carla_df = scenarios_df.copy().rename(columns=carla_mapping)
    carla_df.insert(0, 'scenario_id', range(1, len(carla_df) + 1))
    carla_df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(carla_df)} scenarios to {output_path}")


def split_by_conflict_geometry(scenarios_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split junction scenarios by conflict geometry state.

    Returns:
        dict mapping 'g1', 'g2', 'g3' → subset DataFrame.
    """
    return {
        g: scenarios_df[scenarios_df['Conflict_Geometry'] == g].copy()
        for g in ['g1', 'g2', 'g3']
    }


def get_summary_statistics(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns in the scenario set."""
    numeric = scenarios_df.select_dtypes(include=['number']).columns
    summary = scenarios_df[numeric].describe().T
    summary['unique'] = scenarios_df[numeric].nunique()
    return summary


def compare_scenario_sets(
    scenarios1: pd.DataFrame,
    scenarios2: pd.DataFrame,
    attributes: List[str],
    name1: str = "BayScen",
    name2: str = "BayScen-Common",
):
    """Print a mean-value comparison of two scenario sets."""
    print(f"\n{'=' * 70}")
    print(f"COMPARING SCENARIO SETS: {name1} vs {name2}")
    print(f"{'=' * 70}")

    print(f"\n{name1}: {len(scenarios1)} scenarios, "
          f"{scenarios1[attributes].drop_duplicates().shape[0]} unique combinations")
    print(f"{name2}: {len(scenarios2)} scenarios, "
          f"{scenarios2[attributes].drop_duplicates().shape[0]} unique combinations")

    import pandas as _pd
    comparison = _pd.DataFrame({
        name1: scenarios1[attributes].mean(),
        name2: scenarios2[attributes].mean(),
    })
    comparison['Difference'] = comparison[name2] - comparison[name1]
    print(f"\nMean values comparison:\n{comparison}")


if __name__ == '__main__':
    print("BayScen Generation Utilities")
    print("Available functions:")
    print("  assign_junction_paths    : map conflict geometry to T-junction paths")
    print("  validate_scenarios       : check scenario validity")
    print("  export_for_carla         : export to CARLA column format")
    print("  split_by_conflict_geometry: split by g1/g2/g3")
    print("  get_summary_statistics   : descriptive statistics")
    print("  compare_scenario_sets    : compare two scenario sets")
