"""
BayScen Scenario Generation Script

Main script for generating test scenarios for both Scenario 1 and Scenario 2.
Supports command-line execution with scenario selection.

Usage:
    python generate_scenarios.py --scenario 1 --mode rare
    python generate_scenarios.py --scenario 2 --mode common

References:
    Paper Section IV: Evaluation

Author: BayScen Team
Date: December 2025
"""

import argparse
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from generation.scenario_generator import BayesianScenarioGenerator
from generation.evaluation_metrics import evaluate_scenarios
from abstraction.abstract_variables import LEAF_NODES


class ScenarioGenerationPipeline:
    """
    Complete pipeline for generating and evaluating test scenarios.
    
    Handles both Scenario 1 (Vehicle-Vehicle) and Scenario 2 (Vehicle-Cyclist).
    """
    
    def __init__(self, scenario: int, mode: str = 'rare'):
        """
        Initialize the generation pipeline.
        
        Args:
            scenario: Scenario number (1 or 2)
            mode: Generation mode ('common' or 'rare')
                - 'common': Prioritize typical scenarios
                - 'rare': Prioritize edge cases (recommended for testing)
        """
        if scenario not in [1, 2]:
            raise ValueError(f"Invalid scenario: {scenario}. Must be 1 or 2")
        
        if mode not in ['common', 'rare']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'common' or 'rare'")
        
        self.scenario = scenario
        self.mode = mode
        self.prefer_rare = (mode == 'rare')
        
        # Define paths
        self.script_dir = Path(__file__).parent
        self.model_dir = self.script_dir.parent / "modeling" / "models"
        self.data_dir = self.script_dir.parent.parent / "data"
        self.output_dir = self.script_dir / "generated_scenarios"
        self.output_dir.mkdir(exist_ok=True)
        
        # Define model path
        self.model_path = self.model_dir / f"scenario{scenario}_full_bayesian_network.pkl"
        
        # Define concrete variables based on scenario
        self.concrete_variables = self._get_concrete_variables()
        
        # Define abstracted variables (from abstract_variables.py)
        self.abstracted_variables = LEAF_NODES
        
    def _get_concrete_variables(self):
        """Get list of concrete variables for the scenario."""
        base_variables = [
            "Cloudiness",
            "Wind_Intensity",
            "Precipitation",
            "Precipitation_Deposits",
            "Wetness",
            "Fog_Density",
            "Road_Friction",
            "Fog_Distance"
        ]
        
        # Scenario 2 includes Time_of_Day
        if self.scenario == 2:
            base_variables = ["Time_of_Day"] + base_variables
        
        # Add T-junction variables
        tjunction_variables = ["Start_Ego", "Goal_Ego", "Start_Other", "Goal_Other"]
        
        return base_variables + tjunction_variables
    
    def load_model(self):
        """Load the trained Bayesian Network."""
        print(f"\n{'='*70}")
        print(f"LOADING MODEL - SCENARIO {self.scenario}")
        print(f"{'='*70}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Please train the model first using bn_parametrization.py"
            )
        
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Loaded model from {self.model_path}")
        print(f"  Total nodes: {len(model.nodes())}")
        print(f"  Total edges: {len(model.edges())}")
        
        return model
    
    def create_generator(self, model):
        """Create the scenario generator."""
        print(f"\n{'='*70}")
        print(f"INITIALIZING GENERATOR")
        print(f"{'='*70}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Prefer rare scenarios: {self.prefer_rare}")
        print(f"Abstracted variables: {list(self.abstracted_variables.keys())}")
        print(f"Concrete variables: {len(self.concrete_variables)} variables")
        
        generator = BayesianScenarioGenerator(
            model=model,
            leaf_nodes=self.abstracted_variables,
            initial_nodes=self.concrete_variables,
            similarity_threshold=0.1,
            n_samples=100000,
            use_sampling=True,
            prefer_rare=self.prefer_rare
        )
        
        return generator
    
    def generate(self):
        """Execute the complete generation pipeline."""
        print(f"\n{'='*70}")
        print(f"BAYSCEN SCENARIO GENERATION PIPELINE")
        print(f"{'='*70}")
        print(f"Scenario: {self.scenario}")
        print(f"Mode: {self.mode}")
        print(f"{'='*70}\n")
        
        # Load model
        model = self.load_model()
        
        # Create generator
        generator = self.create_generator(model)
        
        # Generate scenarios
        scenarios = generator.generate_scenarios()
        
        # Save scenarios
        output_filename = f"scenario{self.scenario}_{self.mode}_scenarios.csv"
        output_path = self.output_dir / output_filename
        generator.save_scenarios(scenarios, str(output_path))
        
        return scenarios, output_path
    
    def evaluate(self, scenarios):
        """Evaluate generated scenarios."""
        print(f"\n{'='*70}")
        print(f"EVALUATING SCENARIOS")
        print(f"{'='*70}")
        
        # Define evaluation attributes (concrete environmental variables only)
        eval_attributes = [
            "Cloudiness",
            "Wind_Intensity",
            "Precipitation",
            "Precipitation_Deposits",
            "Wetness",
            "Fog_Density",
            "Road_Friction",
            "Fog_Distance"
        ]
        
        # Scenario 2 includes Time_of_Day
        if self.scenario == 2:
            eval_attributes = ["Time_of_Day"] + eval_attributes
        
        # Define parameter domains for 3-way coverage
        parameter_domains = {
            'Cloudiness': [0, 20, 40, 60, 80, 100],
            'Wind_Intensity': [0, 20, 40, 60, 80, 100],
            'Precipitation': [0, 20, 40, 60, 80, 100],
            'Precipitation_Deposits': [0, 20, 40, 60, 80, 100],
            'Wetness': [0, 20, 40, 60, 80, 100],
            'Road_Friction': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            'Fog_Density': [0, 20, 40, 60, 80, 100],
            'Fog_Distance': [0, 20, 40, 60, 80, 100]
        }
        
        if self.scenario == 2:
            parameter_domains['Time_of_Day'] = [-90, -60, -30, 0, 30, 60, 90]
        
        # Path to real data
        real_data_path = self.data_dir / "bayscen.csv"
        
        if not real_data_path.exists():
            print(f"⚠ Real data not found at {real_data_path}")
            print("Skipping evaluation...")
            return None
        
        # Evaluate (realism + 3-way coverage)
        results = evaluate_scenarios(
            generated_df=scenarios,
            real_data_path=str(real_data_path),
            attributes=eval_attributes,
            parameter_domains=parameter_domains,
            print_summary=True
        )
        
        # Save evaluation results
        eval_filename = f"scenario{self.scenario}_{self.mode}_evaluation.pkl"
        eval_path = self.output_dir / eval_filename
        
        with open(eval_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n✓ Evaluation results saved to {eval_path}")
        
        return results
    
    def run(self):
        """Run the complete pipeline."""
        try:
            # Generate scenarios
            scenarios, output_path = self.generate()
            
            # Evaluate scenarios
            results = self.evaluate(scenarios)
            
            # Summary
            print(f"\n{'='*70}")
            print(f"PIPELINE COMPLETE")
            print(f"{'='*70}")
            print(f"✓ Generated {len(scenarios)} scenarios")
            print(f"✓ Saved to: {output_path}")
            if results:
                print(f"✓ Realism: {results['realism']:.1f}%")
                if 'coverage_3way' in results:
                    print(f"✓ 3-way Coverage (Real): {results['coverage_3way']['real_triples_pct']:.1f}%")
                    print(f"✓ F1 Score: {results['coverage_3way']['f1_score']:.2f}")
            print(f"{'='*70}\n")
            
            return scenarios, results
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Generate test scenarios for BayScen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate rare (edge case) scenarios for Scenario 1
  python generate_scenarios.py --scenario 1 --mode rare
  
  # Generate common (typical) scenarios for Scenario 2
  python generate_scenarios.py --scenario 2 --mode common
  
  # Quick test with Scenario 1
  python generate_scenarios.py
        '''
    )
    
    parser.add_argument(
        '--scenario',
        type=int,
        default=1,
        choices=[1, 2],
        help='Scenario number: 1 (Vehicle-Vehicle) or 2 (Vehicle-Cyclist)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='rare',
        choices=['common', 'rare'],
        help='Generation mode: common (typical) or rare (edge cases)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ScenarioGenerationPipeline(
        scenario=args.scenario,
        mode=args.mode
    )
    
    scenarios, results = pipeline.run()
    
    if scenarios is not None:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    sys.exit(main())