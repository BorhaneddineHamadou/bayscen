"""
Command-Line Interface for BayScen Evaluation

Evaluate all methods and generate paper-ready results.

Usage:
    python evaluate.py --scenario 1
    python evaluate.py --scenario 2
    python evaluate.py --scenario 1 --output results_scenario1.csv

Author: BayScen Team
Date: December 2025
"""

import argparse
from pathlib import Path
import sys

from metrics import (
    evaluate_all_methods,
    save_paper_tables
)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BayScen and baseline methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Evaluate Scenario 1
  python evaluate.py --scenario 1
  
  # Evaluate Scenario 2 with custom output
  python evaluate.py --scenario 2 --output my_results.csv
  
  # Specify custom data path
  python evaluate.py --scenario 1 --real-data ../data/custom_data.csv
        '''
    )
    
    parser.add_argument(
        '--scenario',
        type=int,
        required=True,
        choices=[1, 2],
        help='Scenario number (1 or 2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file (default: results_scenario{N}.csv)'
    )
    
    parser.add_argument(
        '--real-data',
        type=str,
        default=None,
        help='Path to real-world data CSV (default: ../data/processed/bayscen_final_data.csv)'
    )
    
    parser.add_argument(
        '--scenarios-folder',
        type=str,
        default=None,
        help='Folder with scenario Excel files (default: Scenario{N} Generated Scenarios)'
    )
    
    parser.add_argument(
        '--json-folder',
        type=str,
        default=None,
        help='Folder with execution JSON files (default: Scenario{N} Execution Results (JSON))'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    eval_dir = Path(__file__).parent
    scenario = args.scenario
    
    # Default paths
    if args.scenarios_folder is None:
        scenarios_folder = eval_dir / f"Scenario{scenario} Generated Scenarios"
    else:
        scenarios_folder = Path(args.scenarios_folder)
    
    if args.json_folder is None:
        json_folder = eval_dir / f"Scenario{scenario} Execution Results (JSON)"
    else:
        json_folder = Path(args.json_folder)
    
    if args.real_data is None:
        real_data_path = eval_dir.parent / "data" / "processed" / "bayscen_final_data.csv"
    else:
        real_data_path = Path(args.real_data)
    
    if args.output is None:
        output_file = eval_dir / "results" / f"results_scenario{scenario}.csv"
    else:
        output_file = Path(args.output)
    
    # Verify paths exist
    if not scenarios_folder.exists():
        print(f"✗ Error: Scenarios folder not found: {scenarios_folder}")
        sys.exit(1)
    
    if not json_folder.exists():
        print(f"✗ Error: JSON folder not found: {json_folder}")
        sys.exit(1)
    
    if not real_data_path.exists():
        print(f"✗ Error: Real data not found: {real_data_path}")
        sys.exit(1)
    
    # Define methods
    methods = ['bayscen', 'random', 'sitcov', 'PICT_3w', 'PICT_2w', 'CTBC']
    
    # Define attributes based on scenario
    if scenario == 1:
        attributes = [
            'Cloudiness', 'WindIntensity', 'Precipitation', 'PrecipitationDeposits',
            'Wetness', 'FogDensity', 'RoadFriction', 'FogDistance'
        ]
        parameter_domains = {
            'Cloudiness': [0, 20, 40, 60, 80, 100],
            'WindIntensity': [0, 20, 40, 60, 80, 100],
            'Precipitation': [0, 20, 40, 60, 80, 100],
            'PrecipitationDeposits': [0, 20, 40, 60, 80, 100],
            'Wetness': [0, 20, 40, 60, 80, 100],
            'RoadFriction': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            'FogDensity': [0, 20, 40, 60, 80, 100],
            'FogDistance': [0, 20, 40, 60, 80, 100]
        }
    else:  # Scenario 2
        attributes = [
            'TimeOfDay', 'Cloudiness', 'WindIntensity', 'Precipitation', 
            'PrecipitationDeposits', 'Wetness', 'FogDensity', 
            'RoadFriction', 'FogDistance'
        ]
        parameter_domains = {
            'TimeOfDay': [-90, -60, -30, 0, 30, 60, 90],
            'Cloudiness': [0, 20, 40, 60, 80, 100],
            'WindIntensity': [0, 20, 40, 60, 80, 100],
            'Precipitation': [0, 20, 40, 60, 80, 100],
            'PrecipitationDeposits': [0, 20, 40, 60, 80, 100],
            'Wetness': [0, 20, 40, 60, 80, 100],
            'RoadFriction': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            'FogDensity': [0, 20, 40, 60, 80, 100],
            'FogDistance': [0, 20, 40, 60, 80, 100]
        }
    
    print(f"\n{'='*70}")
    print(f"BAYSCEN EVALUATION - SCENARIO {scenario}")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Scenarios folder: {scenarios_folder}")
    print(f"  JSON folder: {json_folder}")
    print(f"  Real data: {real_data_path}")
    print(f"  Output: {output_file}")
    print(f"  Methods: {', '.join(methods)}")
    
    # Run evaluation
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results = evaluate_all_methods(
        scenario_folder=scenarios_folder,
        json_folder=json_folder,
        real_data_path=real_data_path,
        methods=methods,
        attributes=attributes,
        parameter_domains=parameter_domains,
        output_file=output_dir / f"results_scenario{scenario}.csv"
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    print(results.to_string(index=False))
    
    # Generate paper tables
    print(f"\n{'='*70}")
    print("GENERATING PAPER TABLES")
    print(f"{'='*70}")
    
    save_paper_tables(
        results, 
        output_prefix=f"paper_scenario{scenario}",
        output_dir=Path("results")  # ADD THIS
    )
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Complete results: results/results_scenario{scenario}.csv")
    print(f"✓ Table II: results/paper_scenario{scenario}_II_effectiveness.csv")
    print(f"✓ Table III: results/paper_scenario{scenario}_III_realism.csv")
    print(f"✓ Table IV: results/paper_scenario{scenario}_IV_coverage.csv")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())