"""
Scenario Quality Metrics for Generated Scenario Sets

Implements the physical plausibility metric used in the new BayScen paper:

    Physical Plausibility (RQ2)
        Seven physical constraints (C1–C6) derived from Hao et al.
        (arXiv:2311.10937) check whether generated scenarios respect
        real-world physical relationships among environmental parameters.
        The violation rate and per-constraint breakdown match Table IV;
        the Clean Critical Rate (plausible critical scenarios) matches Table V.

Full evaluation (TISA coverage, collision rates, failure profiling) is
implemented in the standalone scripts under evaluation/scripts/.

References:
    Paper Section III   : Evaluation setup
    Paper Table II      : Physical plausibility constraints (Hao et al.)
    Paper Tables IV & V : Physical violation rate and Clean Critical Rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ============================================================================
# PHYSICAL PLAUSIBILITY CONSTRAINTS
# Derived from Hao et al. (arXiv:2311.10937), Table II of the paper.
# Parameters are expected to be in [0, 100] except Road_Friction ([0, 1]).
# ============================================================================

def check_physical_plausibility(
    df: pd.DataFrame,
    epsilon: float = 10.0,
    precip_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    Check all physical plausibility constraints (C1–C6) for each scenario.

    Constraints (Hao et al. 2023):
        C1  P > 20  ⟹  D > 0               precipitation causes deposits
        C2  P > 20  ⟹  W > 0               precipitation causes wetness
        C3a W < 40  ⟹  F ≤ 1 − W/200       wetness reduces friction (low regime)
        C3b W ≥ 40  ⟹  F ≤ 0.6             wetness reduces friction (high regime)
        C4  |L − (100−G)| ≤ ε               fog density determines fog distance
        C5  N ≥ 60  ⟹  G ≤ 40             high wind disperses fog
        C6  P > 20  ⟹  C > 0               rain requires cloud cover

    Column name mapping used here:
        P = Precipitation          D = Precipitation_Deposits
        W = Wetness                F = Road_Friction  (in [0, 1])
        G = Fog_Density            L = Fog_Distance
        N = Wind_Intensity         C = Cloudiness

    Args:
        df              : DataFrame of generated scenarios.
        epsilon         : Tolerance for C4 (default 10).
        precip_threshold: Precipitation threshold for C1, C2, C6 (default 20).

    Returns:
        DataFrame of same length with boolean columns 'C1'–'C6', 'any_violation',
        and 'physically_plausible'.
    """
    result = pd.DataFrame(index=df.index)

    # Retrieve columns (handle both present and missing gracefully)
    def col(name, default=0.0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    P = col('Precipitation')
    D = col('Precipitation_Deposits')
    W = col('Wetness')
    F = col('Road_Friction')
    G = col('Fog_Density')
    L = col('Fog_Distance')
    N = col('Wind_Intensity')
    C = col('Cloudiness')

    # C1: precipitation → deposits
    result['C1'] = (P > precip_threshold) & (D <= 0)

    # C2: precipitation → wetness
    result['C2'] = (P > precip_threshold) & (W <= 0)

    # C3a: low-wetness friction bound  W < 40 ⟹ F ≤ 1 − W/200
    mask_3a = W < 40
    result['C3a'] = mask_3a & (F > 1.0 - W / 200.0)

    # C3b: high-wetness friction bound  W ≥ 40 ⟹ F ≤ 0.6
    mask_3b = W >= 40
    result['C3b'] = mask_3b & (F > 0.6)

    # C4: fog density determines fog distance  |L − (100−G)| ≤ ε
    result['C4'] = (L - (100.0 - G)).abs() > epsilon

    # C5: high wind disperses fog  N ≥ 60 ⟹ G ≤ 40
    result['C5'] = (N >= 60) & (G > 40)

    # C6: rain requires cloud cover  P > 20 ⟹ C > 0
    result['C6'] = (P > precip_threshold) & (C <= 0)

    result['any_violation'] = (
        result['C1'] | result['C2'] | result['C3a'] | result['C3b'] |
        result['C4'] | result['C5'] | result['C6']
    )
    result['physically_plausible'] = ~result['any_violation']

    return result


def physical_plausibility_summary(
    df: pd.DataFrame,
    epsilon: float = 10.0,
    precip_threshold: float = 20.0,
) -> Dict:
    """
    Compute summary statistics for physical plausibility (Table IV equivalent).

    Args:
        df              : DataFrame of generated scenarios.
        epsilon         : Tolerance for constraint C4.
        precip_threshold: Precipitation threshold for C1, C2, C6.

    Returns:
        dict with per-constraint violation rates and overall violation/plausibility rates.
    """
    checks = check_physical_plausibility(df, epsilon, precip_threshold)

    summary = {}
    for col in ['C1', 'C2', 'C3a', 'C3b', 'C4', 'C5', 'C6']:
        summary[f'{col}_violation_rate'] = checks[col].mean() * 100

    summary['overall_violation_rate']    = checks['any_violation'].mean() * 100
    summary['physically_plausible_rate'] = checks['physically_plausible'].mean() * 100
    summary['n_scenarios']               = len(df)
    summary['n_violations']              = int(checks['any_violation'].sum())
    summary['n_plausible']               = int(checks['physically_plausible'].sum())

    return summary


def clean_critical_rate(
    df: pd.DataFrame,
    collision_mask: np.ndarray,
    plausibility_mask: np.ndarray,
) -> Dict:
    """
    Compute the Clean Critical Rate (Table V): proportion of critical (collision)
    scenarios that are also physically plausible.

    Args:
        df               : Scenario DataFrame.
        collision_mask   : Boolean array — True if a collision occurred.
        plausibility_mask: Boolean array — True if physically plausible.

    Returns:
        dict with counts and clean critical rate percentage.
    """
    n_critical  = int(collision_mask.sum())
    n_clean     = int((collision_mask & plausibility_mask).sum())
    rate        = (n_clean / n_critical * 100) if n_critical > 0 else 0.0

    return {
        'n_critical':           n_critical,
        'n_clean_critical':     n_clean,
        'clean_critical_rate':  rate,
    }


# ============================================================================
# DISTRIBUTION ANALYSIS (supplementary utility)
# ============================================================================

def compute_attribute_distributions(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    attributes: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Compare marginal distributions between real-world data and generated scenarios.

    Args:
        real_df      : DataFrame of real-world observations.
        generated_df : DataFrame of generated scenarios.
        attributes   : List of attribute names to compare.

    Returns:
        Dict mapping attribute name → DataFrame with columns ['Real', 'Generated'].
    """
    distribution_tables = {}
    for attr in attributes:
        if attr not in real_df.columns or attr not in generated_df.columns:
            continue
        unique_vals = sorted(real_df[attr].dropna().unique().tolist())
        table = pd.DataFrame(index=unique_vals)
        for label, df in [('Real', real_df), ('Generated', generated_df)]:
            counts = df[attr].value_counts(normalize=True) * 100
            table[label] = [counts.get(v, 0.0) for v in unique_vals]
        distribution_tables[attr] = table.fillna(0.0)
    return distribution_tables


# ============================================================================
# COMPREHENSIVE EVALUATION (per generated scenario set)
# ============================================================================

def evaluate_scenarios(
    generated_df: pd.DataFrame,
    print_summary: bool = True,
    epsilon: float = 10.0,
    precip_threshold: float = 20.0,
    real_data_path: Optional[str] = None,
    attributes: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate generated scenarios for physical plausibility and (optionally)
    attribute distribution similarity to real-world data.

    Args:
        generated_df    : DataFrame of generated scenarios.
        print_summary   : Print summary table to stdout.
        epsilon         : Tolerance for physical constraint C4.
        precip_threshold: Precipitation threshold for C1, C2, C6.
        real_data_path  : (Optional) path to real-world data CSV for distribution analysis.
        attributes      : (Optional) attributes for distribution analysis.

    Returns:
        dict with 'plausibility', 'plausibility_checks', and optionally 'distributions'.
    """
    results = {}

    # Physical plausibility
    plaus_summary = physical_plausibility_summary(generated_df, epsilon, precip_threshold)
    plaus_checks  = check_physical_plausibility(generated_df, epsilon, precip_threshold)
    results['plausibility']        = plaus_summary
    results['plausibility_checks'] = plaus_checks

    # Optional: distribution analysis
    if real_data_path and attributes:
        try:
            real_df = pd.read_csv(real_data_path)
            if 'Time_of_Day' in real_df.columns and 'Sun_Altitude_Angle' not in real_df.columns:
                real_df = real_df.rename(columns={'Time_of_Day': 'Sun_Altitude_Angle'})
            results['distributions'] = compute_attribute_distributions(
                real_df, generated_df, attributes
            )
        except Exception as e:
            print(f"  ⚠ Distribution analysis failed: {e}")

    if print_summary:
        print("\n" + "=" * 70)
        print("PHYSICAL PLAUSIBILITY EVALUATION")
        print("=" * 70)
        print(f"  Scenarios evaluated : {plaus_summary['n_scenarios']}")
        print(f"  Physically plausible: {plaus_summary['n_plausible']} "
              f"({plaus_summary['physically_plausible_rate']:.1f}%)")
        print(f"  Any violation       : {plaus_summary['n_violations']} "
              f"({plaus_summary['overall_violation_rate']:.1f}%)")
        print("\n  Per-constraint violation rates:")
        for c in ['C1', 'C2', 'C3a', 'C3b', 'C4', 'C5', 'C6']:
            print(f"    {c}: {plaus_summary[f'{c}_violation_rate']:.1f}%")
        print("=" * 70)

    return results


if __name__ == '__main__':
    # Quick demo with synthetic data
    np.random.seed(42)
    n = 200
    demo = pd.DataFrame({
        'Fog_Density':            np.random.choice([0, 20, 40, 60, 80, 100], n),
        'Fog_Distance':           np.random.choice([0, 20, 40, 60, 80, 100], n),
        'Cloudiness':             np.random.choice([0, 20, 40, 60, 80, 100], n),
        'Precipitation':          np.random.choice([0, 20, 40, 60, 80, 100], n),
        'Precipitation_Deposits': np.random.choice([0, 20, 40, 60, 80, 100], n),
        'Wetness':                np.random.choice([0, 20, 40, 60, 80, 100], n),
        'Road_Friction':          np.random.choice([0.1, 0.2, 0.4, 0.8, 1.0], n),
        'Wind_Intensity':         np.random.choice([0, 20, 40, 60, 80, 100], n),
    })
    print("Demo: physical plausibility on random data")
    evaluate_scenarios(demo, print_summary=True)
