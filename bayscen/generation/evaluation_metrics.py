"""
Evaluation Metrics for Generated Scenarios

This module implements metrics to evaluate the quality of generated test scenarios:
1. Realism: Distance-based similarity using cKDTree (memory-efficient)
2. Coverage: 3-way combinatorial coverage of parameter interactions
3. Criticality: TTC and collision analysis with realism filtering

References:
    Paper Section IV-B: Evaluation Metrics

Author: BayScen Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from itertools import combinations
from typing import Dict, List, Tuple, Set
import json
import os


# =============================================================================
# REALISM METRICS (Memory-Efficient with cKDTree)
# =============================================================================

def compute_realism(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    attributes: List[str]
) -> Tuple[float, np.ndarray]:
    """
    Compute realism percentage using cKDTree (memory-efficient).
    
    A generated scenario is considered realistic if its distance to the
    nearest real-world scenario is less than the maximum nearest-neighbor
    distance observed within the real-world data.
    
    Args:
        real_df: DataFrame with real-world observations
        generated_df: DataFrame with generated scenarios
        attributes: List of attributes to use for distance calculation
    
    Returns:
        tuple: (realism_percentage, is_realistic_mask)
            - realism_percentage: Percentage of realistic scenarios (0-100)
            - is_realistic_mask: Boolean array indicating which scenarios are realistic
    
    Method:
        1. Build KD-tree for real-world data
        2. Compute nearest-neighbor distances within real data
        3. Use max NN distance as threshold
        4. Check if generated scenarios fall within threshold
    
    Example:
        >>> realism_pct, mask = compute_realism(real_data, generated, attributes)
        >>> print(f"Realism: {realism_pct:.1f}%")
        >>> realistic_scenarios = generated[mask]
    
    References:
        Paper Section IV-B-2: Realism Metric
    """
    # Convert to numpy arrays
    real_arr = real_df[attributes].to_numpy()
    gen_arr = generated_df[attributes].to_numpy()
    
    # Build KD-tree for real data (memory-efficient)
    tree = cKDTree(real_arr)
    
    # Compute nearest-neighbor distances within real data
    # k=2 because closest neighbor to a point is itself (distance=0)
    real_dists, _ = tree.query(real_arr, k=2)
    # The actual nearest neighbor distance is at index 1
    baseline_threshold = np.nanmax(real_dists[:, 1])
    
    # Compute distances from generated scenarios to nearest real scenario
    gen_dists, _ = tree.query(gen_arr, k=1)
    
    # Boolean mask: which generated scenarios are within threshold
    is_realistic_mask = gen_dists < baseline_threshold
    
    # Percentage of realistic scenarios
    realism_percentage = is_realistic_mask.mean() * 100
    
    return realism_percentage, is_realistic_mask


def compute_attribute_distributions(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    attributes: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Compute single-attribute distributions for comparison.
    
    Compares the marginal distributions of each attribute between
    real-world data and generated scenarios.
    
    Args:
        real_df: DataFrame with real-world observations
        generated_df: DataFrame with generated scenarios
        attributes: List of attribute names to compare
    
    Returns:
        dict: Mapping attribute name to DataFrame with distributions
            Index: Unique attribute values
            Columns: ['Real', 'Generated'] with percentages
    
    Example:
        >>> dist = compute_attribute_distributions(real, generated, ['Cloudiness'])
        >>> print(dist['Cloudiness'])
                Real  Generated
        0      15.2       14.8
        20     18.3       19.1
        ...
    """
    datasets = {
        "Real": real_df,
        "Generated": generated_df
    }
    
    distribution_tables = {}
    
    for attr in attributes:
        # Extract unique values from real dataset
        unique_values = sorted(real_df[attr].dropna().unique().tolist())
        table = pd.DataFrame(index=unique_values)
        
        # Compute distributions for each dataset
        for name, df in datasets.items():
            counts = df[attr].value_counts(normalize=True) * 100
            table[name] = table.index.map(lambda x: counts.get(x, 0))
        
        distribution_tables[attr] = table.fillna(0)
    
    return distribution_tables


# =============================================================================
# 3-WAY COVERAGE METRICS
# =============================================================================

def get_triples_from_scenarios(
    df: pd.DataFrame,
    parameter_columns: List[str],
    parameter_domains: Dict[str, list]
) -> Set[Tuple]:
    """
    Extract all unique 3-way parameter combinations from scenarios.
    
    Args:
        df: DataFrame with scenarios
        parameter_columns: List of parameter names
        parameter_domains: Dict mapping parameter names to valid values
    
    Returns:
        set: Set of tuples (param1, val1, param2, val2, param3, val3)
    
    Example:
        >>> triples = get_triples_from_scenarios(df, params, domains)
        >>> print(f"Found {len(triples)} unique 3-way combinations")
    """
    all_triples = set()
    
    # Iterate over all 3-parameter combinations
    for param1, param2, param3 in combinations(parameter_columns, 3):
        for _, row in df.iterrows():
            val1 = row[param1]
            val2 = row[param2]
            val3 = row[param3]
            
            # Only include if values are in valid domains
            if (val1 in parameter_domains[param1] and 
                val2 in parameter_domains[param2] and
                val3 in parameter_domains[param3]):
                all_triples.add((param1, val1, param2, val2, param3, val3))
    
    return all_triples


def get_all_possible_triples(
    parameter_columns: List[str],
    parameter_domains: Dict[str, list]
) -> Set[Tuple]:
    """
    Generate all theoretically possible 3-way combinations.
    
    Args:
        parameter_columns: List of parameter names
        parameter_domains: Dict mapping parameter names to valid values
    
    Returns:
        set: Set of all possible 3-way combinations
    
    Example:
        >>> all_triples = get_all_possible_triples(params, domains)
        >>> print(f"Total possible: {len(all_triples)}")
    """
    all_possible = set()
    
    for param1, param2, param3 in combinations(parameter_columns, 3):
        domain1 = parameter_domains[param1]
        domain2 = parameter_domains[param2]
        domain3 = parameter_domains[param3]
        
        for val1 in domain1:
            for val2 in domain2:
                for val3 in domain3:
                    all_possible.add((param1, val1, param2, val2, param3, val3))
    
    return all_possible


def compute_threeway_coverage(
    generated_df: pd.DataFrame,
    real_df: pd.DataFrame,
    parameter_columns: List[str],
    parameter_domains: Dict[str, list]
) -> Dict:
    """
    Compute 3-way combinatorial coverage metrics.
    
    Args:
        generated_df: DataFrame with generated scenarios
        real_df: DataFrame with real-world data
        parameter_columns: List of parameter names to analyze
        parameter_domains: Dict mapping parameter names to valid values
    
    Returns:
        dict: Coverage statistics
            - 'triples_covered': Number of 3-way combinations covered
            - 'all_triples_pct': % of all possible combinations covered
            - 'real_triples_pct': % of real-world combinations covered
            - 'precision': % of covered combinations that exist in real data
            - 'recall': % of real combinations that are covered
            - 'f1_score': Harmonic mean of precision and recall
    
    Example:
        >>> coverage = compute_threeway_coverage(gen, real, params, domains)
        >>> print(f"3-way coverage: {coverage['real_triples_pct']:.1f}%")
    
    References:
        Paper Section IV-B-3: 3-Way Coverage Metric
    """
    # Get 3-way combinations
    real_triples = get_triples_from_scenarios(real_df, parameter_columns, parameter_domains)
    generated_triples = get_triples_from_scenarios(generated_df, parameter_columns, parameter_domains)
    all_possible_triples = get_all_possible_triples(parameter_columns, parameter_domains)
    
    # Compute metrics
    coverage_all = len(generated_triples) / len(all_possible_triples) * 100
    coverage_real = len(generated_triples & real_triples) / len(real_triples) * 100
    
    precision = (len(generated_triples & real_triples) / len(generated_triples) * 100 
                 if len(generated_triples) > 0 else 0)
    recall = (len(generated_triples & real_triples) / len(real_triples) * 100 
              if len(real_triples) > 0 else 0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'triples_covered': len(generated_triples),
        'all_triples_pct': coverage_all,
        'real_triples_pct': coverage_real,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_real_triples': len(real_triples),
        'total_possible_triples': len(all_possible_triples)
    }


# =============================================================================
# CRITICALITY METRICS (TTC & Collision Analysis)
# =============================================================================

def load_simulation_results(
    scenario_name: str,
    json_folder: str,
    num_runs: int = 3
) -> Tuple[Dict, Dict]:
    """
    Load TTC and collision data from simulation JSON files.
    
    Args:
        scenario_name: Name of the scenario (folder name)
        json_folder: Root folder containing simulation results
        num_runs: Number of simulation runs (default: 3)
    
    Returns:
        tuple: (run_TTC, run_collision)
            - run_TTC: Dict mapping run number to list of TTC values
            - run_collision: Dict mapping run number to list of collision bools
    
    Example:
        >>> ttc, coll = load_simulation_results('bayscen', 'results/', 3)
        >>> print(f"Run 1 collisions: {sum(coll[1])}")
    """
    scenario_json_root = os.path.join(json_folder, scenario_name)
    
    run_TTC = {r: [] for r in range(1, num_runs + 1)}
    run_collision = {r: [] for r in range(1, num_runs + 1)}
    
    for r in range(1, num_runs + 1):
        run_folder = os.path.join(scenario_json_root, f"run{r}")
        min_ttc_path = os.path.join(run_folder, "min_ttc_log.json")
        results_path = os.path.join(run_folder, "run_results.json")
        
        try:
            with open(min_ttc_path, 'r') as f:
                ttc_list = json.load(f)
            with open(results_path, 'r') as f:
                results_list = json.load(f)
        except Exception as e:
            print(f"⚠ Warning: Could not load JSONs for {scenario_name} run{r}: {e}")
            # Fill with defaults
            ttc_list = [{'min_ttc': 100}] * 648  # Assume 648 scenarios
            results_list = [{'collision_occurred': False}] * 648
        
        for ttc_entry, result_entry in zip(ttc_list, results_list):
            collision = result_entry["collision_occurred"]
            ttc_value = 0 if collision else ttc_entry["min_ttc"]
            run_TTC[r].append(ttc_value)
            run_collision[r].append(collision)
    
    return run_TTC, run_collision


def compute_criticality_metrics(
    scenarios_df: pd.DataFrame,
    run_TTC: Dict,
    run_collision: Dict,
    is_realistic_mask: np.ndarray = None
) -> Dict:
    """
    Compute criticality metrics with optional realism filtering.
    
    Args:
        scenarios_df: DataFrame with scenarios
        run_TTC: Dict mapping run number to TTC values
        run_collision: Dict mapping run number to collision booleans
        is_realistic_mask: Optional boolean mask for realistic scenarios
    
    Returns:
        dict: Criticality statistics
            - 'critical_ttc_count': Number with TTC < 0.5s
            - 'critical_ttc_realistic': Number realistic with TTC < 0.5s
            - 'critical_ttc_realism_pct': % realistic among critical TTC
            - 'collision_1_3_count': Scenarios with ≥1 collision
            - 'collision_2_3_count': Scenarios with ≥2 collisions
            - 'collision_3_3_count': Scenarios with 3 collisions
            - 'collision_1_3_realistic': Realistic scenarios with ≥1 collision
            - 'collision_2_3_realistic': Realistic scenarios with ≥2 collisions
            - 'collision_3_3_realistic': Realistic scenarios with 3 collisions
            - 'collision_1_3_realism_pct': % realistic among ≥1 collision
            - 'collision_2_3_realism_pct': % realistic among ≥2 collisions
            - 'collision_3_3_realism_pct': % realistic among 3 collisions
    
    Example:
        >>> metrics = compute_criticality_metrics(df, ttc, coll, mask)
        >>> print(f"Critical TTC: {metrics['critical_ttc_count']}")
        >>> print(f"Realistic: {metrics['critical_ttc_realism_pct']:.1f}%")
    """
    df = scenarios_df.copy()
    
    # Add TTC and collision data to DataFrame
    num_runs = len(run_TTC)
    for r in range(1, num_runs + 1):
        df[f"run{r}_TTC"] = run_TTC[r]
        df[f"run{r}_collision"] = run_collision[r]
    
    # Compute collision counts
    collision_sum = df[[f"run{r}_collision" for r in range(1, num_runs + 1)]].sum(axis=1)
    df["collision_1_3"] = collision_sum >= 1
    df["collision_2_3"] = collision_sum >= 2
    df["collision_3_3"] = collision_sum == 3
    
    # Compute mean TTC and critical threshold
    df["mean_TTC"] = df[[f"run{r}_TTC" for r in range(1, num_runs + 1)]].mean(axis=1)
    df["critical_TTC"] = df["mean_TTC"] < 0.5
    
    # Add realism mask if provided
    if is_realistic_mask is not None:
        df["is_realistic"] = is_realistic_mask
    else:
        df["is_realistic"] = True  # Assume all realistic if mask not provided
    
    # Calculate statistics
    total_scenarios = len(df)
    
    # Critical TTC
    crit_ttc_count = df['critical_TTC'].sum()
    crit_ttc_real_count = df[df['critical_TTC']]['is_realistic'].sum()
    crit_ttc_real_pct = (crit_ttc_real_count / crit_ttc_count * 100) if crit_ttc_count > 0 else 0.0
    
    # Collision 1/3
    coll_1_3_count = df['collision_1_3'].sum()
    coll_1_3_real_count = df[df['collision_1_3']]['is_realistic'].sum()
    coll_1_3_real_pct = (coll_1_3_real_count / coll_1_3_count * 100) if coll_1_3_count > 0 else 0.0
    
    # Collision 2/3
    coll_2_3_count = df['collision_2_3'].sum()
    coll_2_3_real_count = df[df['collision_2_3']]['is_realistic'].sum()
    coll_2_3_real_pct = (coll_2_3_real_count / coll_2_3_count * 100) if coll_2_3_count > 0 else 0.0
    
    # Collision 3/3
    coll_3_3_count = df['collision_3_3'].sum()
    coll_3_3_real_count = df[df['collision_3_3']]['is_realistic'].sum()
    coll_3_3_real_pct = (coll_3_3_real_count / coll_3_3_count * 100) if coll_3_3_count > 0 else 0.0
    
    return {
        'total_scenarios': total_scenarios,
        'critical_ttc_count': int(crit_ttc_count),
        'critical_ttc_realistic': int(crit_ttc_real_count),
        'critical_ttc_realism_pct': crit_ttc_real_pct,
        'collision_1_3_count': int(coll_1_3_count),
        'collision_1_3_realistic': int(coll_1_3_real_count),
        'collision_1_3_realism_pct': coll_1_3_real_pct,
        'collision_2_3_count': int(coll_2_3_count),
        'collision_2_3_realistic': int(coll_2_3_real_count),
        'collision_2_3_realism_pct': coll_2_3_real_pct,
        'collision_3_3_count': int(coll_3_3_count),
        'collision_3_3_realistic': int(coll_3_3_real_count),
        'collision_3_3_realism_pct': coll_3_3_real_pct
    }


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_scenarios(
    generated_df: pd.DataFrame,
    real_data_path: str,
    attributes: List[str],
    parameter_domains: Dict[str, list] = None,
    json_folder: str = None,
    scenario_name: str = None,
    print_summary: bool = True
) -> Dict:
    """
    Comprehensive evaluation of generated scenarios.
    """
    # Load real data
    real_df = pd.read_csv(real_data_path)
    
    # Clean column names
    real_df.columns = [col.replace('_', '') for col in real_df.columns]
    generated_df = generated_df.copy()
    generated_df.columns = [col.replace('_', '') for col in generated_df.columns]
    
    # Handle TimeOfDay naming variations
    if 'TimeofDay' in real_df.columns:
        real_df.rename(columns={'TimeofDay': 'TimeOfDay'}, inplace=True)
    if 'TimeofDay' in generated_df.columns:
        generated_df.rename(columns={'TimeofDay': 'TimeOfDay'}, inplace=True)
    
    results = {}
    
    # 1. REALISM
    available_attributes = [attr for attr in attributes 
                           if attr in generated_df.columns and attr in real_df.columns]
    
    if len(available_attributes) > 0:
        realism_pct, realism_mask = compute_realism(real_df, generated_df, available_attributes)
        results['realism'] = realism_pct
        results['realism_mask'] = realism_mask
        results['distributions'] = compute_attribute_distributions(real_df, generated_df, available_attributes)
    else:
        print("⚠ Warning: No matching attributes for realism computation")
        results['realism'] = 100.0
        results['realism_mask'] = np.ones(len(generated_df), dtype=bool)
        results['distributions'] = {}
    
    # 2. 3-WAY COVERAGE
    if parameter_domains is not None:
        # Clean parameter domain keys to match cleaned column names
        cleaned_domains = {}
        for key, values in parameter_domains.items():
            cleaned_key = key.replace('_', '')
            cleaned_domains[cleaned_key] = values
        
        param_cols = list(cleaned_domains.keys())
        
        # Only use parameters that exist in both dataframes
        param_cols = [p for p in param_cols if p in generated_df.columns and p in real_df.columns]
        
        if len(param_cols) > 0:
            coverage = compute_threeway_coverage(generated_df, real_df, param_cols, cleaned_domains)
            results['coverage_3way'] = coverage
        else:
            print("⚠ Warning: No matching parameters for 3-way coverage")
            results['coverage_3way'] = None
    
    # 3. CRITICALITY (if simulation results provided)
    if json_folder is not None and scenario_name is not None:
        try:
            run_TTC, run_collision = load_simulation_results(scenario_name, json_folder)
            criticality = compute_criticality_metrics(
                generated_df, 
                run_TTC, 
                run_collision,
                results.get('realism_mask', None)
            )
            results['criticality'] = criticality
        except Exception as e:
            print(f"⚠ Warning: Could not load simulation results: {e}")
            results['criticality'] = None
    
    # PRINT SUMMARY
    if print_summary:
        print("\n" + "=" * 70)
        print("SCENARIO EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n1. REALISM")
        print(f"   Overall: {results['realism']:.2f}%")
        print(f"   ({results['realism']:.2f}% of scenarios within real-world distribution)")
        
        if 'coverage_3way' in results and results['coverage_3way'] is not None:
            cov = results['coverage_3way']
            print(f"\n2. 3-WAY COVERAGE")
            print(f"   Triples covered: {cov['triples_covered']}")
            print(f"   All possible: {cov['all_triples_pct']:.2f}%")
            print(f"   Real-world: {cov['real_triples_pct']:.2f}%")
            print(f"   Precision: {cov['precision']:.2f}%")
            print(f"   Recall: {cov['recall']:.2f}%")
            print(f"   F1 Score: {cov['f1_score']:.2f}")
        
        if 'criticality' in results and results['criticality'] is not None:
            crit = results['criticality']
            print(f"\n3. CRITICALITY")
            print(f"   Total scenarios: {crit['total_scenarios']}")
            print(f"\n   Critical TTC (<0.5s): {crit['critical_ttc_count']}")
            print(f"      Realistic: {crit['critical_ttc_realistic']} ({crit['critical_ttc_realism_pct']:.1f}%)")
            print(f"\n   Collisions (≥1/3): {crit['collision_1_3_count']}")
            print(f"      Realistic: {crit['collision_1_3_realistic']} ({crit['collision_1_3_realism_pct']:.1f}%)")
            print(f"\n   Collisions (≥2/3): {crit['collision_2_3_count']}")
            print(f"      Realistic: {crit['collision_2_3_realistic']} ({crit['collision_2_3_realism_pct']:.1f}%)")
            print(f"\n   Collisions (3/3): {crit['collision_3_3_count']}")
            print(f"      Realistic: {crit['collision_3_3_realistic']} ({crit['collision_3_3_realism_pct']:.1f}%)")
        
        print("\n" + "=" * 70)
    
    return results


def compare_methods(
    methods_dict: Dict[str, pd.DataFrame],
    real_data_path: str,
    attributes: List[str],
    parameter_domains: Dict[str, list]
) -> pd.DataFrame:
    """
    Compare multiple methods using all metrics.
    
    Args:
        methods_dict: Dict mapping method name to DataFrame
        real_data_path: Path to real-world data CSV
        attributes: Attributes for realism
        parameter_domains: Domains for 3-way coverage
    
    Returns:
        DataFrame: Comparison table with all metrics
    
    Example:
        >>> methods = {
        ...     'BayScen': bayscen_df,
        ...     'Random': random_df,
        ...     'PICT': pict_df
        ... }
        >>> comparison = compare_methods(methods, 'data.csv', attrs, domains)
        >>> print(comparison)
    """
    # Load real data
    real_df = pd.read_csv(real_data_path)
    real_df.columns = [col.replace('_', '') for col in real_df.columns]
    if 'TimeofDay' in real_df.columns:
        real_df.rename(columns={'TimeofDay': 'TimeOfDay'}, inplace=True)
    
    results = []
    
    for method_name, method_df in methods_dict.items():
        print(f"Evaluating {method_name}...")
        
        # Clean columns
        method_df = method_df.copy()
        method_df.columns = [col.replace('_', '') for col in method_df.columns]
        if 'TimeofDay' in method_df.columns:
            method_df.rename(columns={'TimeofDay': 'TimeOfDay'}, inplace=True)
        
        # Compute realism
        available_attrs = [a for a in attributes if a in method_df.columns and a in real_df.columns]
        realism_pct, _ = compute_realism(real_df, method_df, available_attrs)
        
        # Compute 3-way coverage
        param_cols = list(parameter_domains.keys())
        coverage = compute_threeway_coverage(method_df, real_df, param_cols, parameter_domains)
        
        results.append({
            'Method': method_name,
            'N Scenarios': len(method_df),
            'Realism (%)': realism_pct,
            'Triples Covered': coverage['triples_covered'],
            'All Triples (%)': coverage['all_triples_pct'],
            'Real Triples (%)': coverage['real_triples_pct'],
            'Precision (%)': coverage['precision'],
            'Recall (%)': coverage['recall'],
            'F1 Score': coverage['f1_score']
        })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("BayScen Evaluation Metrics Module (Updated)")
    print("=" * 70)
    print("\nNew features:")
    print("  ✓ cKDTree-based realism (memory-efficient)")
    print("  ✓ 3-way combinatorial coverage")
    print("  ✓ Criticality metrics with realism filtering")
    print("  ✓ Collision 3/3 tracking")
    print("\nAvailable functions:")
    print("  - compute_realism: Distance-based similarity with cKDTree")
    print("  - compute_threeway_coverage: 3-way parameter interaction coverage")
    print("  - compute_criticality_metrics: TTC and collision analysis")
    print("  - evaluate_scenarios: Comprehensive evaluation")
    print("  - compare_methods: Multi-method comparison")