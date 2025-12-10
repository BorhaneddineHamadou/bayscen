"""
Bayesian Network Parametrization for BayScen Scenarios

This script trains a complete Bayesian Network for scenario generation, including:
1. Learning structure from real-world weather data
2. Estimating parameters using Bayesian estimation (BDeu prior)
3. Adding abstracted variables (Road_Surface, Vehicle_Stability, Visibility)
4. Adding junction intersection variables (Start/Goal positions, Collision_Point)

Supports both Scenario 1 (Vehicle-Vehicle) and Scenario 2 (Vehicle-Cyclist with Time_of_Day).

References:
    Paper Section III-B: Modeling the Scenario Space as a Bayesian Network
    Paper Section III-C: Effect-Based Abstraction
    Paper Section IV-A-3: Abstracted Variable Definition

Author: BayScen Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import pickle
from itertools import product
from collections import defaultdict
from pathlib import Path

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD

# Import abstraction utilities
import sys
sys.path.append(str(Path(__file__).parent.parent / "abstraction"))
from mapping_functions import (
    STANDARD_VALUES,
    MAP_TO_STANDARD,
    MAP_TO_ORIGINAL
)
from abstraction_cpd import (
    compute_abstracted_cpd,
    create_collision_cpd,
    extend_bayesian_network
)
from abstract_variables import (
    ABSTRACT_VARIABLES,
    VISIBILITY,
    ROAD_SURFACE,
    VEHICLE_STABILITY,
    COLLISION_POINT
)

# Import BN utilities
from bn_utils import save_model, load_model, print_model_summary


class BayesianNetworkParametrizer:
    """
    Handles the complete parametrization pipeline for BayScen Bayesian Networks.
    
    Supports two scenarios:
    - Scenario 1 (Vehicle-Vehicle): 8 environmental variables
    - Scenario 2 (Vehicle-Cyclist): 9 environmental variables (includes Time_of_Day)
    
    Pipeline stages:
    1. Load and preprocess weather data
    2. Define network structure (from learned structure)
    3. Fit base environmental variables with Bayesian estimation
    4. Extend with abstracted variables (from abstract_variables.py)
    5. Add T-junction variables (positions and Collision_Point)
    """
    
    def __init__(self, data_path: str, structure_path: str, scenario: int = 1):
        """
        Initialize the parametrizer.
        
        Args:
            data_path: Path to CSV file with weather data
            structure_path: Path to text file with learned BN structure
            scenario: Scenario number (1 for Vehicle-Vehicle, 2 for Vehicle-Cyclist)
        
        Raises:
            ValueError: If scenario is not 1 or 2
        """
        if scenario not in [1, 2]:
            raise ValueError(f"Invalid scenario: {scenario}. Must be 1 or 2.")
        
        self.data_path = data_path
        self.structure_path = structure_path
        self.scenario = scenario
        self.data = None
        self.edges = None
        self.base_model = None
        self.extended_model = None
        self.full_model = None
        
        # Define base variables based on scenario
        self.base_variables = [
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
        if scenario == 2:
            self.base_variables.append("Time_of_Day")
        
        # Extract abstracted edges from abstract_variables.py
        # This is the single source of truth for abstraction structure
        self.abstracted_edges = self._extract_abstracted_edges()
        
    def _extract_abstracted_edges(self):
        """
        Extract abstracted variable structure from abstract_variables.py.
        Automatically adapts to available variables (handles Time_of_Day for Scenario 2).
        """
        edges = []
        
        # Get abstracted variables
        abstracted_vars = [VISIBILITY, ROAD_SURFACE, VEHICLE_STABILITY]
        
        for var in abstracted_vars:
            # Check if variable has dynamic parent selection
            if hasattr(var, 'get_parents_for_scenario'):
                # Use scenario-aware parent selection (for Visibility with Time_of_Day)
                parents_info = var.get_parents_for_scenario(self.base_variables)
            else:
                # Use static parents
                parents_info = var.parents
            
            for parent_info in parents_info:
                if len(parent_info) == 3:
                    parent_name, relationship, weight = parent_info
                    # Only add edge if parent exists in base variables
                    if parent_name in self.base_variables:
                        edges.append((parent_name, var.name, relationship, weight))
                else:
                    # Skip deterministic parents (like for Collision_Point)
                    continue
        
        return edges
        
    def load_data(self):
        """
        Load weather data from CSV and select relevant variables.
        
        Expected CSV columns: all base_variables plus potentially extra columns
        """
        print("\n" + "=" * 70)
        print(f"STEP 1: LOADING DATA (Scenario {self.scenario})")
        print("=" * 70)
        
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded data from {self.data_path}")
        print(f"  Total observations: {len(df)}")
        
        # Select only relevant variables for this scenario
        self.data = df[self.base_variables]
        print(f"✓ Selected {len(self.base_variables)} base variables")
        print(f"  Variables: {', '.join(self.base_variables)}")
        
    def load_structure(self):
        """
        Load the learned Bayesian Network structure from text file.
        
        Structure file format:
            # NODES (8 variables)
            Cloudiness, Wind_Intensity, ...
            
            # EDGES (9 directed arcs)
            Parent -> Child
        """
        print("\n" + "=" * 70)
        print("STEP 2: LOADING NETWORK STRUCTURE")
        print("=" * 70)
        
        with open(self.structure_path, 'r') as f:
            lines = f.readlines()
        
        # Parse edges from structure file
        self.edges = []
        reading_edges = False
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                if '# EDGES' in line:
                    reading_edges = True
                continue
            
            # Parse edge: "Parent -> Child"
            if reading_edges and '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    parent = parts[0].strip()
                    child = parts[1].strip()
                    # Only include edges for variables in this scenario
                    if parent in self.base_variables and child in self.base_variables:
                        self.edges.append((parent, child))
        
        print(f"✓ Loaded structure from {self.structure_path}")
        print(f"  Total edges: {len(self.edges)}")
        print("\n  Edges:")
        for parent, child in self.edges:
            print(f"    {parent} -> {child}")
    
    def fit_base_model(self):
        """
        Fit the base Bayesian Network using Bayesian parameter estimation.
        
        Uses BDeu prior with equivalent sample size = 5 for smoothing,
        as described in Paper Section III-B-3.
        """
        print("\n" + "=" * 70)
        print("STEP 3: FITTING BASE MODEL")
        print("=" * 70)
        
        # Create model with structure
        self.base_model = BayesianNetwork(self.edges)
        
        # Fit using Bayesian estimation with BDeu prior
        print("Estimating parameters with Bayesian estimation (BDeu prior, ESS=5)...")
        self.base_model.fit(
            self.data,
            estimator=BayesianEstimator,
            prior_type='BDeu',
            equivalent_sample_size=5
        )
        
        print("✓ Base model fitted successfully")
        print(f"  Model has {len(self.base_model.nodes())} nodes")
        print(f"  Model has {len(self.base_model.edges())} edges")
        
        # Validate
        if self.base_model.check_model():
            print("✓ Model validation passed")
        else:
            raise ValueError("✗ Model validation failed!")
    
    def extend_with_abstractions(self):
        """
        Extend the base model with abstracted variables.
        
        Adds three abstracted leaf nodes:
        - Road_Surface: Consolidates traction effects
        - Vehicle_Stability: Consolidates destabilizing forces  
        - Visibility: Consolidates perception degradation
        
        Structure is extracted from abstract_variables.py to avoid duplication.
        
        See Paper Section III-C for abstraction methodology.
        """
        print("\n" + "=" * 70)
        print("STEP 4: ADDING ABSTRACTED VARIABLES")
        print("=" * 70)
        
        # Group edges by child (abstracted variable)
        abstracted_structure = defaultdict(list)
        for parent, child, relationship, weight in self.abstracted_edges:
            abstracted_structure[child].append((parent, relationship, weight))
        
        print("Abstracted variables structure (from abstract_variables.py):")
        for child, parents_info in abstracted_structure.items():
            print(f"\n  {child}:")
            for parent, rel, weight in parents_info:
                print(f"    <- {parent} ({rel}, weight={weight})")
        
        # Extend the model
        print("\nComputing CPDs for abstracted variables...")
        self.extended_model = extend_bayesian_network(
            self.base_model,
            abstracted_structure,
            STANDARD_VALUES
        )
        
        print(f"\n✓ Extended model has {len(self.extended_model.nodes())} nodes")
        print(f"  Added: {', '.join(abstracted_structure.keys())}")
    
    def add_tjunction_variables(self):
        """
        Add T-junction intersection variables to the model.
        
        Adds:
        - Start_Ego, Goal_Ego: Ego vehicle start/goal positions
        - Start_Other, Goal_Other: Adversary vehicle start/goal positions
        - Collision_Point: Deterministic collision point (c1, c2, c3, None)
        
        See Paper Section IV-A-3 and Figure 3 for T-junction layout.
        """
        print("\n" + "=" * 70)
        print("STEP 5: ADDING T-JUNCTION VARIABLES")
        print("=" * 70)
        
        # Get existing edges and CPDs
        existing_edges = list(self.extended_model.edges())
        existing_cpds = self.extended_model.get_cpds()
        
        # Create new model
        self.full_model = BayesianNetwork()
        
        # Add original structure
        if existing_edges:
            self.full_model.add_edges_from(existing_edges)
        else:
            self.full_model.add_nodes_from(self.extended_model.nodes())
        
        # Get T-junction node names from abstract_variables.py
        position_vars = ['Start_Ego', 'Goal_Ego', 'Start_Other', 'Goal_Other']
        collision_var = COLLISION_POINT.name
        tjunction_nodes = position_vars + [collision_var]
        
        self.full_model.add_nodes_from(tjunction_nodes)
        
        # Add edges: Collision_Point depends on all four position variables
        tjunction_edges = [
            ('Start_Ego', collision_var),
            ('Goal_Ego', collision_var),
            ('Start_Other', collision_var),
            ('Goal_Other', collision_var)
        ]
        self.full_model.add_edges_from(tjunction_edges)
        
        print(f"Added {len(tjunction_nodes)} T-junction nodes")
        print(f"  Position variables: {', '.join(position_vars)}")
        print(f"  Collision variable: {collision_var}")
        
        # Add original CPDs
        for cpd in existing_cpds:
            self.full_model.add_cpds(cpd)
        
        # Create CPDs for position variables (uniform distributions)
        print("\nCreating CPDs for position variables...")
        for var in position_vars:
            cpd = self._create_position_cpd(var)
            self.full_model.add_cpds(cpd)
            print(f"  ✓ {var}")
        
        # Create CPD for Collision_Point
        print(f"\nCreating CPD for {collision_var}...")
        collision_rules = self._define_collision_logic()
        cpd_collision = create_collision_cpd(collision_rules)
        self.full_model.add_cpds(cpd_collision)
        print(f"  ✓ {collision_var}")
        
        # Validate
        if self.full_model.check_model():
            print("\n✓ Full model validation passed")
        else:
            raise ValueError("✗ Full model validation failed!")
        
        # Summary
        collision_count = sum(1 for v in collision_rules.values() if v is not None)
        print(f"\nCollision scenarios: {collision_count}/{len(collision_rules)} result in collisions")
    
    def _create_position_cpd(self, variable_name: str) -> TabularCPD:
        """
        Create uniform CPD for position variables.
        
        Each position (Left, Right, Base) has equal probability 1/3.
        
        Args:
            variable_name: Name of position variable
            
        Returns:
            TabularCPD with uniform distribution
        """
        locations = ['Left', 'Right', 'Base']
        cpd_values = [[1.0/3], [1.0/3], [1.0/3]]
        
        cpd = TabularCPD(
            variable=variable_name,
            variable_card=3,
            values=cpd_values,
            state_names={variable_name: locations}
        )
        
        return cpd
    
    def _define_collision_logic(self) -> dict:
        """
        Define collision point mapping based on trajectory intersections.
        
        T-junction layout (viewed from above):
               Base
                |
        Left ---+--- Right
        
        Collision points:
        - c1: Right lane of main road
        - c2: Center of junction
        - c3: Left lane of main road
        
        Returns:
            Dictionary mapping (start_ego, goal_ego, start_other, goal_other) 
            to collision point ('c1', 'c2', 'c3', or None)
        """
        collision_rules = {}
        locations = ['Left', 'Right', 'Base']
        
        # Define which collision points each path traverses
        path_collision_points = {
            ('Left', 'Right'): ['c1', 'c3'],
            ('Right', 'Left'): ['c2'],
            ('Base', 'Left'): ['c1', 'c2'],
            ('Base', 'Right'): ['c1'],
            ('Left', 'Base'): ['c3'],
            ('Right', 'Base'): ['c2', 'c3'],
        }
        
        for start_ego, goal_ego, start_other, goal_other in product(locations, repeat=4):
            # Invalid scenarios: vehicle doesn't move or both start at same location
            if start_ego == goal_ego or start_other == goal_other or start_ego == start_other:
                collision_rules[(start_ego, goal_ego, start_other, goal_other)] = None
                continue
            
            # Get collision points for each path
            ego_points = path_collision_points.get((start_ego, goal_ego), [])
            other_points = path_collision_points.get((start_other, goal_other), [])
            
            # Find intersection
            common_points = set(ego_points) & set(other_points)
            
            if common_points:
                # Choose first common point (consistent ordering)
                collision_point = sorted(common_points)[0]
                collision_rules[(start_ego, goal_ego, start_other, goal_other)] = collision_point
            else:
                collision_rules[(start_ego, goal_ego, start_other, goal_other)] = None
        
        return collision_rules
    
    def save_models(self, base_path: str = None, extended_path: str = None, 
                    full_path: str = None):
        """
        Save all trained models to disk.
        
        Args:
            base_path: Path for base model (default: 'scenario{N}_fitted_bayesian_network.pkl')
            extended_path: Path for extended model (default: 'scenario{N}_extended_bayesian_network.pkl')
            full_path: Path for full model (default: 'scenario{N}_full_bayesian_network.pkl')
        """
        print("\n" + "=" * 70)
        print("STEP 6: SAVING MODELS")
        print("=" * 70)
        
        # Default paths - ALL scenario-specific
        output_dir = Path(__file__).parent / "models"
        output_dir.mkdir(exist_ok=True)
        
        if base_path is None:
            base_path = output_dir / f"scenario{self.scenario}_fitted_bayesian_network.pkl"
        if extended_path is None:
            extended_path = output_dir / f"scenario{self.scenario}_extended_bayesian_network.pkl"
        if full_path is None:
            full_path = output_dir / f"scenario{self.scenario}_full_bayesian_network.pkl"
        
        # Save models
        if self.base_model:
            save_model(self.base_model, str(base_path))
        
        if self.extended_model:
            save_model(self.extended_model, str(extended_path))
        
        if self.full_model:
            save_model(self.full_model, str(full_path))
            print(f"\n✓ FINAL MODEL: {full_path}")
    
    def print_summary(self):
        """Print comprehensive summary of the full model."""
        print("\n" + "=" * 70)
        print(f"FINAL MODEL SUMMARY - SCENARIO {self.scenario}")
        print("=" * 70)
        
        if self.full_model:
            print_model_summary(self.full_model)
            
            print("\nVariable Categories:")
            print(f"  Environmental ({len(self.base_variables)}): " + ", ".join(self.base_variables))
            
            # Extract abstracted variable names
            abstracted_names = list(set(child for _, child, _, _ in self.abstracted_edges))
            print(f"  Abstracted ({len(abstracted_names)}): " + ", ".join(sorted(abstracted_names)))
            
            print("  T-Junction (5): Start_Ego, Goal_Ego, Start_Other, Goal_Other, Collision_Point")
    
    def run_full_pipeline(self):
        """Execute the complete parametrization pipeline."""
        print("\n" + "=" * 70)
        print(f"BAYESIAN NETWORK PARAMETRIZATION - SCENARIO {self.scenario}")
        print("=" * 70)
        
        # Execute all steps
        self.load_data()
        self.load_structure()
        self.fit_base_model()
        self.extend_with_abstractions()
        self.add_tjunction_variables()
        self.save_models()
        self.print_summary()
        
        print("\n" + "=" * 70)
        print("✓ PARAMETRIZATION COMPLETE")
        print("=" * 70)


def main():
    """
    Main entry point for BayScen BN parametrization.
    
    Usage:
        python bn_parametrization.py              # Scenario 1 (default)
        python bn_parametrization.py --scenario 2 # Scenario 2
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Bayesian Network for BayScen scenario generation'
    )
    parser.add_argument(
        '--scenario',
        type=int,
        default=1,
        choices=[1, 2],
        help='Scenario number: 1 (Vehicle-Vehicle) or 2 (Vehicle-Cyclist with Time_of_Day)'
    )
    args = parser.parse_args()
    
    # Define paths
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / "data" / "processed" / "bayscen_final_data.csv"
    
    # Use scenario-specific structure file if it exists, otherwise use generic
    structure_filename = f"scenario{args.scenario}_structure.txt"
    structure_path = script_dir / "structure_learning" / "learned_structures" / structure_filename
    
    # Fallback to scenario1_structure.txt if scenario-specific doesn't exist
    if not structure_path.exists() and args.scenario == 2:
        print(f"Warning: {structure_filename} not found, using scenario1_structure.txt")
        structure_path = script_dir / "structure_learning" / "learned_structures" / "scenario1_structure.txt"
    
    # Check if paths exist
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure bayscen_final_data.csv is in the correct location.")
        return
    
    if not structure_path.exists():
        print(f"Error: Structure file not found at {structure_path}")
        print("Please ensure the structure file is in the correct location.")
        return
    
    # Run parametrization
    parametrizer = BayesianNetworkParametrizer(
        data_path=str(data_path),
        structure_path=str(structure_path),
        scenario=args.scenario
    )
    
    parametrizer.run_full_pipeline()


if __name__ == "__main__":
    main()