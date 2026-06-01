"""
Bayesian Network Parameterization for BayScen

Trains the complete Bayesian Network for scenario generation:
  1. Loads real-world meteorological data and the learned BN structure.
  2. Fits conditional probability distributions (BDeu prior, ESS=5).
  3. Extends the base BN with ISO 34503-grounded capability variables as leaf nodes.
  4. Adds scenario-specific variables:
       S1/S2 (junction): Start/Goal positions + Conflict_Geometry
       S3     (cut-in) : Cut_In_Direction (binary, concrete — no further abstraction)

Supported scenarios:
    1  Vehicle–Vehicle Junction (S1)   8 env vars
    2  Vehicle–Cyclist Junction (S2)   9 env vars (adds Sun_Altitude_Angle)
    3  Vehicle–Vehicle Cut-In   (S3)   9 env vars (adds Sun_Altitude_Angle)

References:
    Paper Section II-B: Modeling the Scenario Space as a Bayesian Network
    Paper Section II-C: ODD-Grounded Capability Abstraction
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import product

from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

# Abstraction modules
sys.path.append(str(Path(__file__).parent.parent / "abstraction"))
from mapping_functions import STANDARD_VALUES, MAP_TO_STANDARD
from abstraction_cpd import (
    compute_capability_cpd,
    create_conflict_geometry_cpd,
    extend_bayesian_network,
)
from abstract_variables import (
    SENSOR_PERCEPTION,
    SURFACE_TRACTION,
    LATERAL_STABILITY,
    CONFLICT_GEOMETRY,
)

from bn_utils import save_model, load_model, print_model_summary


class BayesianNetworkParametrizer:
    """
    Complete parameterization pipeline for BayScen Bayesian Networks.

    Pipeline stages:
      1. Load and preprocess weather data.
      2. Parse the learned BN structure.
      3. Fit base environmental CPDs (BDeu, ESS=5).
      4. Extend with ISO 34503-grounded capability leaf nodes.
      5a. (S1/S2) Add T-junction position variables and Conflict_Geometry.
      5b. (S3)    Add Cut_In_Direction (binary concrete variable).
      6. Save all intermediate and final models.
    """

    def __init__(self, data_path: str, structure_path: str, scenario: int = 1):
        """
        Args:
            data_path      : Path to CSV file with processed weather data.
            structure_path : Path to text file with the learned BN structure.
            scenario       : 1 (Vehicle–Vehicle), 2 (Vehicle–Cyclist), 3 (Cut-In).
        """
        if scenario not in [1, 2, 3]:
            raise ValueError(f"Invalid scenario: {scenario}. Must be 1, 2, or 3.")

        self.data_path      = data_path
        self.structure_path = structure_path
        self.scenario       = scenario
        self.data           = None
        self.edges          = None
        self.base_model     = None
        self.extended_model = None
        self.full_model     = None

        # Base environmental variables
        self.base_variables = [
            "Cloudiness",
            "Wind_Intensity",
            "Precipitation",
            "Precipitation_Deposits",
            "Wetness",
            "Fog_Density",
            "Road_Friction",
            "Fog_Distance",
        ]
        # S2 and S3 add Sun_Altitude_Angle (ISO §10.4(d))
        if scenario in [2, 3]:
            self.base_variables.append("Sun_Altitude_Angle")

        # Build capability structure from abstract_variables definitions
        self.capability_structure = self._build_capability_structure()

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _build_capability_structure(self) -> dict:
        """
        Assemble the capability variable structure from abstract_variables.py.
        Returns dict: capability_name → [(parent, relationship, weight), ...]
        """
        structure = defaultdict(list)

        for var in [SENSOR_PERCEPTION, SURFACE_TRACTION, LATERAL_STABILITY]:
            # Sensor_Perception adapts its parent list based on available vars
            if hasattr(var, 'get_parents_for_scenario'):
                parents_info = var.get_parents_for_scenario(self.base_variables)
            else:
                parents_info = var.parents

            for parent_info in parents_info:
                if len(parent_info) == 3:
                    parent, rel, w = parent_info
                    if parent in self.base_variables:
                        structure[var.name].append((parent, rel, w))

        return dict(structure)

    # ------------------------------------------------------------------
    # PIPELINE STEPS
    # ------------------------------------------------------------------

    def load_data(self):
        """Load and select relevant variables from the weather CSV."""
        print("\n" + "=" * 70)
        print(f"STEP 1: LOADING DATA (Scenario {self.scenario})")
        print("=" * 70)

        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} observations from {self.data_path}")

        # Handle backward-compatible column naming (Time_of_Day → Sun_Altitude_Angle)
        if 'Time_of_Day' in df.columns and 'Sun_Altitude_Angle' not in df.columns:
            df = df.rename(columns={'Time_of_Day': 'Sun_Altitude_Angle'})
            print("  (Renamed column Time_of_Day → Sun_Altitude_Angle for consistency)")

        self.data = df[self.base_variables]
        print(f"✓ Selected {len(self.base_variables)} variables: {', '.join(self.base_variables)}")

    def load_structure(self):
        """Parse the learned BN structure from a text file."""
        print("\n" + "=" * 70)
        print("STEP 2: LOADING NETWORK STRUCTURE")
        print("=" * 70)

        with open(self.structure_path, 'r') as f:
            lines = f.readlines()

        self.edges = []
        reading_edges = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                if '# EDGES' in line:
                    reading_edges = True
                continue
            if reading_edges and '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    parent = parts[0].strip()
                    child  = parts[1].strip()
                    if parent in self.base_variables and child in self.base_variables:
                        self.edges.append((parent, child))

        print(f"✓ Loaded structure from {self.structure_path}")
        print(f"  Edges ({len(self.edges)}):")
        for p, c in self.edges:
            print(f"    {p} → {c}")

    def fit_base_model(self):
        """Fit base BN using Bayesian estimation (BDeu prior, ESS=5)."""
        print("\n" + "=" * 70)
        print("STEP 3: FITTING BASE MODEL")
        print("=" * 70)

        self.base_model = BayesianNetwork(self.edges)
        print("Estimating CPDs (BDeu prior, ESS=5)...")
        self.base_model.fit(
            self.data,
            estimator=BayesianEstimator,
            prior_type='BDeu',
            equivalent_sample_size=5,
        )

        if self.base_model.check_model():
            print(f"✓ Base model valid — {len(self.base_model.nodes())} nodes, "
                  f"{len(self.base_model.edges())} edges")
        else:
            raise ValueError("Base model validation failed")

    def extend_with_capabilities(self):
        """Add ISO 34503-grounded capability variables as leaf nodes."""
        print("\n" + "=" * 70)
        print("STEP 4: ADDING CAPABILITY VARIABLES (ISO 34503:2023)")
        print("=" * 70)

        for cap, parents in self.capability_structure.items():
            print(f"\n  {cap}:")
            for parent, rel, w in parents:
                print(f"    ← {parent} ({rel}, weight={w:.4f})")

        self.extended_model = extend_bayesian_network(
            self.base_model,
            self.capability_structure,
            STANDARD_VALUES,
        )
        print(f"\n✓ Extended model — {len(self.extended_model.nodes())} nodes")

    def add_junction_variables(self):
        """
        (S1, S2) Add T-junction position variables and Conflict_Geometry.

        Variables added:
            Start_Ego, Goal_Ego, Start_Other, Goal_Other — uniform over {Left, Right, Base}
            Conflict_Geometry — deterministic from trajectory intersection geometry
        """
        print("\n" + "=" * 70)
        print("STEP 5a: ADDING JUNCTION VARIABLES")
        print("=" * 70)

        existing_edges = list(self.extended_model.edges())
        existing_cpds  = self.extended_model.get_cpds()

        self.full_model = BayesianNetwork()
        if existing_edges:
            self.full_model.add_edges_from(existing_edges)
        else:
            self.full_model.add_nodes_from(self.extended_model.nodes())

        position_vars  = ['Start_Ego', 'Goal_Ego', 'Start_Other', 'Goal_Other']
        conflict_var   = CONFLICT_GEOMETRY.name  # 'Conflict_Geometry'

        self.full_model.add_nodes_from(position_vars + [conflict_var])
        for pv in position_vars:
            self.full_model.add_edge(pv, conflict_var)

        for cpd in existing_cpds:
            self.full_model.add_cpds(cpd)

        locations = ['Left', 'Right', 'Base']
        for var in position_vars:
            cpd = TabularCPD(
                variable=var, variable_card=3,
                values=[[1/3], [1/3], [1/3]],
                state_names={var: locations},
            )
            self.full_model.add_cpds(cpd)
            print(f"  ✓ {var} (uniform over {locations})")

        conflict_rules = CONFLICT_GEOMETRY.define_conflict_logic()
        cpd_conflict   = create_conflict_geometry_cpd(conflict_rules)
        self.full_model.add_cpds(cpd_conflict)
        n_conflicts = sum(1 for v in conflict_rules.values() if v is not None)
        print(f"  ✓ Conflict_Geometry  ({n_conflicts} conflicting trajectories)")

        if self.full_model.check_model():
            print("\n✓ Full model (S1/S2) validated")
        else:
            raise ValueError("Full junction model validation failed")

    def add_cutin_variables(self):
        """
        (S3) Add Cut_In_Direction as a concrete binary variable.

        Cut-in direction is retained as-is (not abstracted): the binary
        Left/Right state is already fully expressive for the cut-in geometry.
        """
        print("\n" + "=" * 70)
        print("STEP 5b: ADDING CUT-IN VARIABLE")
        print("=" * 70)

        existing_edges = list(self.extended_model.edges())
        existing_cpds  = self.extended_model.get_cpds()

        self.full_model = BayesianNetwork()
        if existing_edges:
            self.full_model.add_edges_from(existing_edges)
        else:
            self.full_model.add_nodes_from(self.extended_model.nodes())

        self.full_model.add_node('Cut_In_Direction')

        for cpd in existing_cpds:
            self.full_model.add_cpds(cpd)

        cpd_cutin = TabularCPD(
            variable='Cut_In_Direction', variable_card=2,
            values=[[0.5], [0.5]],
            state_names={'Cut_In_Direction': ['Left', 'Right']},
        )
        self.full_model.add_cpds(cpd_cutin)
        print("  ✓ Cut_In_Direction (uniform over {Left, Right})")

        if self.full_model.check_model():
            print("\n✓ Full model (S3) validated")
        else:
            raise ValueError("Full cut-in model validation failed")

    def save_models(self, base_path=None, extended_path=None, full_path=None):
        """Save base, extended, and full models to disk."""
        print("\n" + "=" * 70)
        print("STEP 6: SAVING MODELS")
        print("=" * 70)

        output_dir = Path(__file__).parent / "models"
        output_dir.mkdir(exist_ok=True)

        n = self.scenario
        if base_path     is None: base_path     = output_dir / f"scenario{n}_fitted_bayesian_network.pkl"
        if extended_path is None: extended_path = output_dir / f"scenario{n}_extended_bayesian_network.pkl"
        if full_path     is None: full_path     = output_dir / f"scenario{n}_full_bayesian_network.pkl"

        if self.base_model:     save_model(self.base_model,     str(base_path))
        if self.extended_model: save_model(self.extended_model, str(extended_path))
        if self.full_model:
            save_model(self.full_model, str(full_path))
            print(f"\n✓ FINAL MODEL: {full_path}")

    def print_summary(self):
        """Print a summary of the full model."""
        print("\n" + "=" * 70)
        print(f"FINAL MODEL SUMMARY — SCENARIO {self.scenario}")
        print("=" * 70)

        if self.full_model:
            print_model_summary(self.full_model)
            print(f"\nEnvironmental variables ({len(self.base_variables)}): "
                  + ", ".join(self.base_variables))
            caps = list(self.capability_structure.keys())
            print(f"Capability variables ({len(caps)}): " + ", ".join(caps))
            if self.scenario in [1, 2]:
                print("Junction variables (5): "
                      "Start_Ego, Goal_Ego, Start_Other, Goal_Other, Conflict_Geometry")
            else:
                print("Cut-in variable (1): Cut_In_Direction")

    def run_full_pipeline(self):
        """Execute the complete parameterization pipeline."""
        print("\n" + "=" * 70)
        print(f"BAYESIAN NETWORK PARAMETERIZATION — SCENARIO {self.scenario}")
        print("=" * 70)

        self.load_data()
        self.load_structure()
        self.fit_base_model()
        self.extend_with_capabilities()

        if self.scenario in [1, 2]:
            self.add_junction_variables()
        else:
            self.add_cutin_variables()

        self.save_models()
        self.print_summary()

        print("\n" + "=" * 70)
        print("✓ PARAMETERIZATION COMPLETE")
        print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Bayesian Network for BayScen scenario generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bn_parametrization.py --scenario 1   # Vehicle–Vehicle junction
  python bn_parametrization.py --scenario 2   # Vehicle–Cyclist junction
  python bn_parametrization.py --scenario 3   # Vehicle–Vehicle cut-in
        """,
    )
    parser.add_argument(
        '--scenario', type=int, default=1, choices=[1, 2, 3],
        help='Scenario: 1=Vehicle–Vehicle, 2=Vehicle–Cyclist, 3=Cut-In',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_path  = script_dir.parent.parent / "data" / "processed" / "bayscen_final_data.csv"

    # Scenario 3 uses the same structure as Scenario 2 (both have Sun_Altitude_Angle)
    struct_name = f"scenario{args.scenario}_structure.txt"
    structure_path = (script_dir / "structure_learning" / "learned_structures" / struct_name)

    if not structure_path.exists():
        fallback = script_dir / "structure_learning" / "learned_structures" / "scenario2_structure.txt"
        if args.scenario == 3 and fallback.exists():
            print(f"Note: Using scenario2_structure.txt as base for Scenario 3")
            structure_path = fallback
        else:
            print(f"Error: Structure file not found — {structure_path}")
            return

    if not data_path.exists():
        print(f"Error: Data file not found — {data_path}")
        return

    p = BayesianNetworkParametrizer(str(data_path), str(structure_path), args.scenario)
    p.run_full_pipeline()


if __name__ == "__main__":
    main()
