"""
BayScen Scenario Generation Script

Generates test scenarios for all three NHTSA scenarios:
    S1 — Vehicle–Vehicle Junction (left & right turn)
    S2 — Vehicle–Cyclist Junction (crossing paths)
    S3 — Vehicle–Vehicle Cut-In  (highway lateral encroachment)

Usage:
    # BayScen (rarity-prioritized, default)
    python generate_scenarios.py --scenario 1 --mode rare
    python generate_scenarios.py --scenario 2 --mode rare
    python generate_scenarios.py --scenario 3 --mode rare

    # BayScen-Common ablation (common scenario selection)
    python generate_scenarios.py --scenario 1 --mode common
    python generate_scenarios.py --scenario 2 --mode common
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from generation.scenario_generator import BayesianScenarioGenerator
from generation.evaluation_metrics import evaluate_scenarios
from abstraction.abstract_variables import LEAF_NODES, LEAF_NODES_S3


class ScenarioGenerationPipeline:
    """
    End-to-end pipeline: load BN → configure generator → generate → evaluate → save.
    """

    # Environmental variables present in each scenario
    _ENV_VARS_S1 = [
        "Cloudiness", "Wind_Intensity", "Precipitation",
        "Precipitation_Deposits", "Wetness", "Fog_Density",
        "Road_Friction", "Fog_Distance",
    ]
    _ENV_VARS_S2 = ["Sun_Altitude_Angle"] + _ENV_VARS_S1
    _ENV_VARS_S3 = _ENV_VARS_S2  # S3 also uses Sun_Altitude_Angle

    def __init__(self, scenario: int, mode: str = 'rare'):
        """
        Args:
            scenario : 1 (Vehicle–Vehicle), 2 (Vehicle–Cyclist), 3 (Cut-In).
            mode     : 'rare' (rarity-prioritized, BayScen) or
                       'common' (most-common selection, BayScen-Common ablation).
        """
        if scenario not in [1, 2, 3]:
            raise ValueError(f"Invalid scenario {scenario}. Must be 1, 2, or 3.")
        if mode not in ['rare', 'common']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'rare' or 'common'.")

        self.scenario    = scenario
        self.mode        = mode
        self.prefer_rare = (mode == 'rare')

        self.script_dir = Path(__file__).parent
        self.model_dir  = self.script_dir.parent / "modeling" / "models"
        self.output_dir = self.script_dir / "generated_scenarios"
        self.output_dir.mkdir(exist_ok=True)

        self.model_path  = self.model_dir / f"scenario{scenario}_full_bayesian_network.pkl"
        self.concrete_variables  = self._concrete_variables()
        self.capability_leaf_nodes = LEAF_NODES_S3 if scenario == 3 else LEAF_NODES

    def _concrete_variables(self) -> list:
        """Build the list of concrete variables to sample for this scenario."""
        env = {
            1: self._ENV_VARS_S1,
            2: self._ENV_VARS_S2,
            3: self._ENV_VARS_S3,
        }[self.scenario]

        if self.scenario in [1, 2]:
            # Junction scenarios: include T-junction position variables
            geo_vars = ["Start_Ego", "Goal_Ego", "Start_Other", "Goal_Other"]
        else:
            # Cut-in: binary concrete variable retained as-is
            geo_vars = ["Cut_In_Direction"]

        return env + geo_vars

    # ------------------------------------------------------------------

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run bn_parametrization.py --scenario {self.scenario} to train it."
            )
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded model: {self.model_path}")
        print(f"  Nodes: {len(model.nodes())}  Edges: {len(model.edges())}")
        return model

    def create_generator(self, model) -> BayesianScenarioGenerator:
        print(f"\nMode            : {self.mode.upper()}")
        print(f"Capability nodes : {list(self.capability_leaf_nodes.keys())}")
        print(f"Concrete vars    : {len(self.concrete_variables)}")
        return BayesianScenarioGenerator(
            model=model,
            leaf_nodes=self.capability_leaf_nodes,
            initial_nodes=self.concrete_variables,
            n_samples=100_000,
            use_sampling=True,
            prefer_rare=self.prefer_rare,
        )

    def generate(self):
        model     = self.load_model()
        generator = self.create_generator(model)
        scenarios = generator.generate_scenarios()
        out_name  = f"scenario{self.scenario}_{self.mode}_scenarios.csv"
        out_path  = self.output_dir / out_name
        generator.save_scenarios(scenarios, str(out_path))
        return scenarios, out_path

    def evaluate(self, scenarios: 'pd.DataFrame'):
        """Quick physical plausibility check on the generated scenario set."""
        print(f"\n{'=' * 70}")
        print("PHYSICAL PLAUSIBILITY EVALUATION")
        print(f"{'=' * 70}")
        results = evaluate_scenarios(scenarios, print_summary=True)

        eval_path = self.output_dir / f"scenario{self.scenario}_{self.mode}_plausibility.pkl"
        with open(eval_path, 'wb') as f:
            pickle.dump(results['plausibility'], f)
        print(f"✓ Plausibility summary saved to {eval_path}")
        return results

    def run(self):
        """Execute the full pipeline (generate → evaluate → report)."""
        print(f"\n{'=' * 70}")
        print(f"BAYSCEN PIPELINE  |  Scenario {self.scenario}  |  Mode: {self.mode.upper()}")
        print(f"{'=' * 70}\n")

        try:
            scenarios, out_path = self.generate()
            results = self.evaluate(scenarios)

            print(f"\n{'=' * 70}")
            print("PIPELINE COMPLETE")
            print(f"{'=' * 70}")
            print(f"  Scenarios     : {len(scenarios)}")
            print(f"  Saved to      : {out_path}")
            plaus = results.get('plausibility', {})
            if plaus:
                print(f"  Plausible     : {plaus.get('physically_plausible_rate', 0):.1f}%")
                print(f"  Any violation : {plaus.get('overall_violation_rate', 0):.1f}%")
            print(f"{'=' * 70}\n")
            return scenarios, results

        except Exception as exc:
            print(f"\n❌ ERROR: {exc}")
            import traceback; traceback.print_exc()
            return None, None


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate BayScen test scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BayScen rarity-prioritized (recommended)
  python generate_scenarios.py --scenario 1 --mode rare
  python generate_scenarios.py --scenario 2 --mode rare
  python generate_scenarios.py --scenario 3 --mode rare

  # BayScen-Common ablation
  python generate_scenarios.py --scenario 1 --mode common
        """,
    )
    parser.add_argument(
        '--scenario', type=int, default=1, choices=[1, 2, 3],
        help='1=Vehicle–Vehicle, 2=Vehicle–Cyclist, 3=Cut-In',
    )
    parser.add_argument(
        '--mode', type=str, default='rare', choices=['rare', 'common'],
        help="'rare' = rarity-prioritized (BayScen), 'common' = BayScen-Common ablation",
    )
    args = parser.parse_args()

    pipeline = ScenarioGenerationPipeline(args.scenario, args.mode)
    scenarios, results = pipeline.run()
    return 0 if scenarios is not None else 1


if __name__ == "__main__":
    sys.exit(main())
