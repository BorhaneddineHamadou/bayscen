"""
BayScen Evaluation Module

Complete evaluation system for comparing scenario generation methods.
Replicates all results reported in the paper.

Metrics computed:
1. Realism (cKDTree-based)
2. 3-way combinatorial coverage (Precision, Recall, F1)
3. Criticality (TTC, Collisions with realism filtering)

Author: BayScen Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from scipy.spatial import cKDTree
from itertools import combinations
from typing import Dict, List, Tuple, Set
from tqdm import tqdm


# =============================================================================
# REALISM METRICS
# =============================================================================

def compute_realism(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    attributes: List[str]
) -> Tuple[float, np.ndarray]:
    """
    Compute realism using cKDTree (memory-efficient).
    
    Returns:
        (realism_percentage, is_realistic_mask)
    """
    real_arr = real_df[attributes].to_numpy()
    gen_arr = generated_df[attributes].to_numpy()
    
    tree = cKDTree(real_arr)
    
    # Compute threshold from real data NN distances
    real_dists, _ = tree.query(real_arr, k=2)
    baseline_threshold = np.nanmax(real_dists[:, 1])
    
    # Check generated scenarios
    gen_dists, _ = tree.query(gen_arr, k=1)
    is_realistic_mask = gen_dists < baseline_threshold
    
    realism_percentage = is_realistic_mask.mean() * 100
    
    return realism_percentage, is_realistic_mask


# =============================================================================
# 3-WAY COVERAGE METRICS
# =============================================================================

def get_triples_from_scenarios(
    df: pd.DataFrame,
    parameter_columns: List[str],
    parameter_domains: Dict[str, list]
) -> Set[Tuple]:
    """Extract all unique 3-way parameter combinations."""
    all_triples = set()
    
    for param1, param2, param3 in combinations(parameter_columns, 3):
        for _, row in df.iterrows():
            val1 = row[param1]
            val2 = row[param2]
            val3 = row[param3]
            
            if (val1 in parameter_domains[param1] and 
                val2 in parameter_domains[param2] and
                val3 in parameter_domains[param3]):
                all_triples.add((param1, val1, param2, val2, param3, val3))
    
    return all_triples


def get_all_possible_triples(
    parameter_columns: List[str],
    parameter_domains: Dict[str, list]
) -> Set[Tuple]:
    """Generate all theoretically possible 3-way combinations."""
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
    
    Returns metrics for paper reporting.
    """
    real_triples = get_triples_from_scenarios(real_df, parameter_columns, parameter_domains)
    generated_triples = get_triples_from_scenarios(generated_df, parameter_columns, parameter_domains)
    all_possible_triples = get_all_possible_triples(parameter_columns, parameter_domains)
    
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
        'f1_score': f1
    }


# =============================================================================
# SIMULATION RESULTS LOADING
# =============================================================================

def load_simulation_results(
    method_name: str,
    json_folder: Path,
    num_scenarios: int,
    num_runs: int = 3
) -> Tuple[Dict, Dict]:
    """
    Load TTC and collision data from JSON files.
    
    Args:
        method_name: Method folder name (e.g., 'bayscen', 'random')
        json_folder: Root folder with execution results
        num_scenarios: Expected number of scenarios
        num_runs: Number of runs (default: 3)
    
    Returns:
        (run_TTC, run_collision) dictionaries
    """
    method_folder = json_folder / method_name
    
    run_TTC = {r: [] for r in range(1, num_runs + 1)}
    run_collision = {r: [] for r in range(1, num_runs + 1)}
    
    for r in range(1, num_runs + 1):
        run_folder = method_folder / f"run{r}"
        min_ttc_path = run_folder / "min_ttc_log.json"
        results_path = run_folder / "run_results.json"
        
        try:
            with open(min_ttc_path, 'r') as f:
                ttc_list = json.load(f)
            with open(results_path, 'r') as f:
                results_list = json.load(f)
            
            # Process each scenario
            for ttc_entry, result_entry in zip(ttc_list, results_list):
                collision = result_entry["collision_occurred"]
                # If collision occurred, TTC = 0
                ttc_value = 0 if collision else ttc_entry["min_ttc"]
                run_TTC[r].append(ttc_value)
                run_collision[r].append(collision)
                
        except Exception as e:
            print(f"  ⚠ Warning: Could not load JSONs for {method_name} run{r}: {e}")
            # Fill with defaults
            run_TTC[r] = [100.0] * num_scenarios
            run_collision[r] = [False] * num_scenarios
    
    return run_TTC, run_collision


# =============================================================================
# CRITICALITY METRICS
# =============================================================================

def compute_criticality_metrics(
    run_TTC: Dict,
    run_collision: Dict,
    is_realistic_mask: np.ndarray = None
) -> Dict:
    """
    Compute criticality metrics from simulation results.
    
    Returns all metrics reported in paper:
    - Mean TTC
    - Critical TTC (<0.5s) count and realism
    - Collision 2/3 count and realism
    - Collision 3/3 count and realism
    """
    num_scenarios = len(run_TTC[1])
    num_runs = len(run_TTC)
    
    # Create temporary DataFrame for calculations
    df = pd.DataFrame()
    
    for r in range(1, num_runs + 1):
        df[f"run{r}_TTC"] = run_TTC[r]
        df[f"run{r}_collision"] = run_collision[r]
    
    # Compute collision counts
    collision_sum = df[[f"run{r}_collision" for r in range(1, num_runs + 1)]].sum(axis=1)
    df["collision_1_3"] = collision_sum >= 1
    df["collision_2_3"] = collision_sum >= 2
    df["collision_3_3"] = collision_sum == 3
    
    # Compute mean TTC
    df["mean_TTC"] = df[[f"run{r}_TTC" for r in range(1, num_runs + 1)]].mean(axis=1)
    df["critical_TTC"] = df["mean_TTC"] < 0.5
    
    # Add realism mask
    if is_realistic_mask is not None:
        df["is_realistic"] = is_realistic_mask
    else:
        df["is_realistic"] = True
    
    # Calculate statistics
    # Critical TTC
    crit_ttc_count = df['critical_TTC'].sum()
    crit_ttc_real_count = df[df['critical_TTC']]['is_realistic'].sum()
    crit_ttc_real_pct = (crit_ttc_real_count / crit_ttc_count * 100) if crit_ttc_count > 0 else 0.0
    
    # Collision 2/3
    coll_2_3_count = df['collision_2_3'].sum()
    coll_2_3_real_count = df[df['collision_2_3']]['is_realistic'].sum()
    coll_2_3_real_pct = (coll_2_3_real_count / coll_2_3_count * 100) if coll_2_3_count > 0 else 0.0
    
    # Collision 3/3
    coll_3_3_count = df['collision_3_3'].sum()
    coll_3_3_real_count = df[df['collision_3_3']]['is_realistic'].sum()
    coll_3_3_real_pct = (coll_3_3_real_count / coll_3_3_count * 100) if coll_3_3_count > 0 else 0.0
    
    # Mean TTC (overall)
    mean_ttc = df["mean_TTC"].mean()
    
    return {
        'total_scenarios': num_scenarios,
        'mean_ttc': mean_ttc,
        'critical_ttc_count': int(crit_ttc_count),
        'critical_ttc_realistic': int(crit_ttc_real_count),
        'critical_ttc_realism_pct': crit_ttc_real_pct,
        'collision_2_3_count': int(coll_2_3_count),
        'collision_2_3_realistic': int(coll_2_3_real_count),
        'collision_2_3_realism_pct': coll_2_3_real_pct,
        'collision_3_3_count': int(coll_3_3_count),
        'collision_3_3_realistic': int(coll_3_3_real_count),
        'collision_3_3_realism_pct': coll_3_3_real_pct
    }


# =============================================================================
# COMPREHENSIVE METHOD EVALUATION
# =============================================================================

def evaluate_method(
    method_name: str,
    scenario_excel: Path,
    json_folder: Path,
    real_data: pd.DataFrame,
    attributes: List[str],
    parameter_domains: Dict[str, list]
) -> Dict:
    """
    Complete evaluation for one method.
    
    Args:
        method_name: Name of the method
        scenario_excel: Path to Excel file with generated scenarios
        json_folder: Folder with simulation results (JSON files)
        real_data: DataFrame with real-world data
        attributes: Environmental attributes for realism
        parameter_domains: Parameter domains for 3-way coverage
    
    Returns:
        Dictionary with all metrics
    """
    print(f"\nEvaluating {method_name}...")
    
    # Load scenarios
    try:
        scenarios_df = pd.read_excel(scenario_excel, engine='openpyxl')
    except Exception as e:
        print(f"  ✗ Error loading {scenario_excel}: {e}")
        return None
    
    # Clean column names
    scenarios_df.columns = [col.replace('_', '') for col in scenarios_df.columns]
    
    num_scenarios = len(scenarios_df)
    print(f"  Scenarios: {num_scenarios}")
    
    # 1. REALISM
    available_attrs = [a for a in attributes if a in scenarios_df.columns and a in real_data.columns]
    if len(available_attrs) > 0:
        realism_pct, realism_mask = compute_realism(real_data, scenarios_df, available_attrs)
    else:
        print(f"  ⚠ No matching attributes for realism")
        realism_pct = 0.0
        realism_mask = np.zeros(num_scenarios, dtype=bool)
    
    # 2. 3-WAY COVERAGE
    param_cols = [p for p in parameter_domains.keys() if p in scenarios_df.columns and p in real_data.columns]
    if len(param_cols) > 0:
        coverage = compute_threeway_coverage(scenarios_df, real_data, param_cols, parameter_domains)
    else:
        print(f"  ⚠ No matching parameters for coverage")
        coverage = {'triples_covered': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    
    # 3. CRITICALITY (from simulation results)
    try:
        run_TTC, run_collision = load_simulation_results(method_name, json_folder, num_scenarios)
        criticality = compute_criticality_metrics(run_TTC, run_collision, realism_mask)
    except Exception as e:
        print(f"  ⚠ Could not load simulation results: {e}")
        criticality = None
    
    # Compile results
    results = {
        'method': method_name,
        'n_scenarios': num_scenarios,
        'realism_pct': realism_pct,
        'coverage': coverage,
        'criticality': criticality
    }
    
    print(f"  ✓ Complete")
    
    return results


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def evaluate_all_methods(
    scenario_folder: Path,
    json_folder: Path,
    real_data_path: Path,
    methods: List[str],
    attributes: List[str],
    parameter_domains: Dict[str, list],
    output_file: Path = None
) -> pd.DataFrame:
    """
    Evaluate all methods and generate comparison table.
    
    Args:
        scenario_folder: Folder with Excel files
        json_folder: Folder with JSON execution results
        real_data_path: Path to real-world CSV data
        methods: List of method names
        attributes: Environmental attributes
        parameter_domains: Parameter domains
        output_file: Optional path to save results
    
    Returns:
        DataFrame with all results
    """
    print("=" * 70)
    print("BAYSCEN EVALUATION - ALL METHODS")
    print("=" * 70)
    
    # Load real data
    real_data = pd.read_csv(real_data_path)
    real_data.columns = [col.replace('_', '') for col in real_data.columns]
    if 'TimeofDay' in real_data.columns:
        real_data.rename(columns={'TimeofDay': 'TimeOfDay'}, inplace=True)
    
    print(f"\nReal data: {len(real_data)} observations")
    print(f"Methods to evaluate: {len(methods)}")
    print(f"Attributes: {attributes}")
    
    # Evaluate each method
    all_results = []
    
    for method in tqdm(methods, desc="Evaluating methods"):
        # Find Excel file for this method
        excel_file = scenario_folder / f"{method}.xlsx"
        if not excel_file.exists():
            # Try with _scenarios suffix
            excel_file = scenario_folder / f"{method}_scenarios.xlsx"
        
        if not excel_file.exists():
            print(f"\n  ✗ File not found for {method}")
            continue
        
        # Evaluate
        result = evaluate_method(
            method_name=method,
            scenario_excel=excel_file,
            json_folder=json_folder,
            real_data=real_data,
            attributes=attributes,
            parameter_domains=parameter_domains
        )
        
        if result is not None:
            all_results.append(result)
    
    # Create summary DataFrame
    summary = create_summary_table(all_results)
    
    # Save if requested
    if output_file is not None:
        summary.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")
    
    return summary


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """Create formatted summary table from results."""
    rows = []
    
    for r in results:
        cov = r['coverage']
        crit = r['criticality'] if r['criticality'] is not None else {}
        
        row = {
            'Method': r['method'],
            'N Scenarios': r['n_scenarios'],
            'Realism (%)': round(r['realism_pct'], 2),
            'Triples Covered': cov.get('triples_covered', 0),
            'All Triples (%)': round(cov.get('all_triples_pct', 0), 2),
            'Real Triples (%)': round(cov.get('real_triples_pct', 0), 2),
            'Precision (%)': round(cov.get('precision', 0), 2),
            'Recall (%)': round(cov.get('recall', 0), 2),
            'F1 Score': round(cov.get('f1_score', 0), 2),
            'Mean TTC': round(crit.get('mean_ttc', 0), 3),
            'Critical TTC': crit.get('critical_ttc_count', 0),
            'Critical TTC Realism (%)': round(crit.get('critical_ttc_realism_pct', 0), 1),
            'Collision 2/3': crit.get('collision_2_3_count', 0),
            'Collision 2/3 Realism (%)': round(crit.get('collision_2_3_realism_pct', 0), 1),
            'Collision 3/3': crit.get('collision_3_3_count', 0),
            'Collision 3/3 Realism (%)': round(crit.get('collision_3_3_realism_pct', 0), 1)
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)

# =============================================================================
# PAPER TABLE GENERATION
# =============================================================================

def generate_table_ii(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table II: Safety-Critical Scenario Discovery (RQ1: Effectiveness)
    
    Reports both absolute counts and normalized rates for fair comparison.
    
    Args:
        results_df: Results from evaluate_all_methods
    
    Returns:
        DataFrame formatted for Table II
    """
    table = []
    
    for _, row in results_df.iterrows():
        crit = row
        n = crit['N Scenarios']
        
        # TTC < 0.5
        ttc_count = crit['Critical TTC']
        ttc_rate = (ttc_count / n * 100) if n > 0 else 0
        
        # Collision ≥2/3
        coll_2_3_count = crit['Collision 2/3']
        coll_2_3_rate = (coll_2_3_count / n * 100) if n > 0 else 0
        
        # Collision 3/3
        coll_3_3_count = crit['Collision 3/3']
        coll_3_3_rate = (coll_3_3_count / n * 100) if n > 0 else 0
        
        table.append({
            'Method': row['Method'],
            'N': n,
            'TTC<0.5 Count (#)': ttc_count,
            'TTC<0.5 Rate (%)': round(ttc_rate, 1),
            'Collision (≥2/3) Count (#)': coll_2_3_count,
            'Collision (≥2/3) Rate (%)': round(coll_2_3_rate, 1),
            'Collision (3/3) Count (#)': coll_3_3_count,
            'Collision (3/3) Rate (%)': round(coll_3_3_rate, 1)
        })
    
    return pd.DataFrame(table)


def generate_table_iii(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table III: Realism Analysis of Discovered Safety-Critical Scenarios
    
    Args:
        results_df: Results from evaluate_all_methods
    
    Returns:
        DataFrame formatted for Table III
    """
    table = []
    
    for _, row in results_df.iterrows():
        # Overall Realism
        overall_realism = row['Realism (%)']
        
        # Mean TTC < 0.5
        ttc_count = row['Critical TTC']
        ttc_realistic = row['Critical TTC Realism (%)']
        ttc_realistic_count = int(ttc_count * ttc_realistic / 100) if ttc_realistic > 0 else 0
        
        # Collisions ≥2/3
        coll_count = row['Collision 2/3']
        coll_realistic = row['Collision 2/3 Realism (%)']
        coll_realistic_count = int(coll_count * coll_realistic / 100) if coll_realistic > 0 else 0
        
        table.append({
            'Method': row['Method'],
            'Overall Realism (%)': round(overall_realism, 1),
            'Mean TTC < 0.5 Count': int(ttc_count),
            'Mean TTC < 0.5 Realistic (#)': ttc_realistic_count,
            'Mean TTC < 0.5 Realism (%)': round(ttc_realistic, 1),
            'Collisions (≥2/3) Count': int(coll_count),
            'Collisions (≥2/3) Realistic (#)': coll_realistic_count,
            'Collisions (≥2/3) Realism(%)': round(coll_realistic, 1)
        })
    
    return pd.DataFrame(table)


def generate_table_iv(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table IV: 3-Way Coverage Quality Analysis
    
    Args:
        results_df: Results from evaluate_all_methods
    
    Returns:
        DataFrame formatted for Table IV
    """
    table = []
    
    for _, row in results_df.iterrows():
        # Get values - handle both old and new column names
        if 'All Triples (%)' in row:
            all_triples = row['All Triples (%)']
            real_triples = row['Real Triples (%)']
        else:
            # Column names might be different, extract from nested dict
            all_triples = row.get('all_triples_pct', 0)
            real_triples = row.get('real_triples_pct', 0)
        
        precision = row['Precision (%)']
        f1 = row['F1 Score']
        n = row['N Scenarios']
        
        table.append({
            'Method': row['Method'],
            'All triples': f"{all_triples:.1f}%",
            'Real triples': f"{real_triples:.1f}%",
            'Precision': f"{precision:.1f}%",
            'F1': round(f1, 1),
            'N': int(n)
        })
    
    return pd.DataFrame(table)


def save_paper_tables(
    results_df: pd.DataFrame,
    output_prefix: str = "paper_table",
    output_dir: Path = None
) -> None:
    """
    Generate and save all three paper tables.
    
    Args:
        results_df: Results from evaluate_all_methods
        output_prefix: Prefix for output files
        output_dir: Directory to save files (default: results/)
    """
    # Default to results/ folder
    if output_dir is None:
        output_dir = Path("results")
    
    # Create results directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Generate tables
    table_ii = generate_table_ii(results_df)
    table_iii = generate_table_iii(results_df)
    table_iv = generate_table_iv(results_df)
    
    # Save as CSV in results folder
    table_ii.to_csv(output_dir / f"{output_prefix}_II_effectiveness.csv", index=False)
    table_iii.to_csv(output_dir / f"{output_prefix}_III_realism.csv", index=False)
    table_iv.to_csv(output_dir / f"{output_prefix}_IV_coverage.csv", index=False)
    
    print(f"\n✓ Saved Table II to {output_dir / f'{output_prefix}_II_effectiveness.csv'}")
    print(f"✓ Saved Table III to {output_dir / f'{output_prefix}_III_realism.csv'}")
    print(f"✓ Saved Table IV to {output_dir / f'{output_prefix}_IV_coverage.csv'}")
    
    # Print tables
    print("\n" + "="*80)
    print("TABLE II: Safety-Critical Scenario Discovery (RQ1: Effectiveness)")
    print("="*80)
    print(table_ii.to_string(index=False))
    
    print("\n" + "="*80)
    print("TABLE III: Realism Analysis of Discovered Safety-Critical Scenarios")
    print("="*80)
    print(table_iii.to_string(index=False))
    
    print("\n" + "="*80)
    print("TABLE IV: 3-Way Coverage Quality Analysis")
    print("="*80)
    print(table_iv.to_string(index=False))

if __name__ == '__main__':
    print("BayScen Evaluation Module")
    print("=" * 70)
    print("\nThis module provides complete evaluation for paper replication.")
    print("\nKey functions:")
    print("  - evaluate_method: Evaluate single method")
    print("  - evaluate_all_methods: Batch evaluation")
    print("  - compute_realism: cKDTree-based realism")
    print("  - compute_threeway_coverage: 3-way coverage metrics")
    print("  - compute_criticality_metrics: TTC & collision analysis")