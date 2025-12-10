"""
Abstracted Variable CPD Computation

This module implements the CPD computation for abstracted variables using
weighted aggregation and soft similarity functions.

Key Functions:
- compute_abstracted_cpd: Creates CPDs for effect-based abstractions
- create_collision_cpd: Creates deterministic CPD for Collision_Point
- extend_bayesian_network: Extends a BN with abstracted variables

References:
    Paper Section III-C: Effect-Based Abstraction
    Paper Equation (1): Weighted aggregation
    Paper Equation (2): Soft similarity function

Author: BayScen Team
Date: December 2025
"""

import numpy as np
from itertools import product
from collections import defaultdict
from typing import Dict, List, Tuple

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from mapping_functions import STANDARD_VALUES, MAP_TO_STANDARD


def compute_abstracted_cpd(
    child_name: str,
    parents_info: List[Tuple[str, str, float]],
    fitted_model: BayesianNetwork,
    standard_values: List[int] = STANDARD_VALUES,
    sigma: float = 25.0
) -> TabularCPD:
    """
    Compute CPD for an abstracted variable using weighted combination + soft similarity.
    
    This implements the aggregation method described in Paper Section III-C-3:
    1. For each parent configuration, compute weighted combination (Eq. 1)
    2. Apply soft similarity to determine probability distribution (Eq. 2)
    3. Normalize to ensure valid probability distribution
    
    Args:
        child_name: Name of the abstracted variable
        parents_info: List of (parent_name, relationship, weight) tuples
            - relationship: 'normal' (co-varies) or 'inverse' (inverse relationship)
            - weight: Contribution weight (should sum to 1.0 across parents)
        fitted_model: Fitted BayesianNetwork containing parent variables
        standard_values: Discrete states for abstracted variable [0, 20, 40, 60, 80, 100]
        sigma: Smoothness parameter for soft similarity (default=25)
            - Larger σ → smoother distribution
            - Smaller σ → sharper distribution
    
    Returns:
        TabularCPD for the abstracted variable
    
    Example:
        >>> parents_info = [
        ...     ('Fog_Density', 'inverse', 0.45),
        ...     ('Fog_Distance', 'normal', 0.45),
        ...     ('Precipitation', 'inverse', 0.1)
        ... ]
        >>> cpd = compute_abstracted_cpd('Visibility', parents_info, model)
    """
    parent_names = [p[0] for p in parents_info]
    
    # Get actual state values for each parent from the fitted model
    parent_state_values = {}
    for parent in parent_names:
        parent_cpd = fitted_model.get_cpds(parent)
        parent_state_values[parent] = parent_cpd.state_names[parent]
    
    # Generate all combinations of parent values
    parent_value_lists = [parent_state_values[parent] for parent in parent_names]
    parent_combinations = list(product(*parent_value_lists))
    num_combinations = len(parent_combinations)
    
    # Initialize CPD table
    num_states = len(standard_values)
    cpd_values = np.zeros((num_states, num_combinations))
    
    # Compute CPD for each parent configuration
    for col_idx, parent_values in enumerate(parent_combinations):
        # Step 1: Compute weighted combination (Paper Equation 1)
        weighted_sum = 0.0
        total_weight = 0.0
        
        for parent_idx, (parent_name, relationship, weight) in enumerate(parents_info):
            # Get parent value and map to standard scale if needed
            parent_val_original = parent_values[parent_idx]
            map_func = MAP_TO_STANDARD.get(parent_name)
            parent_val_standard = map_func(parent_val_original) if map_func else parent_val_original
            
            # Handle ambiguous mappings (take mean if list returned)
            if isinstance(parent_val_standard, list):
                parent_val_standard = np.mean(parent_val_standard)
            
            # Apply relationship transformation
            if relationship == "inverse":
                parent_val_standard = 100 - parent_val_standard
            elif relationship != "normal":
                raise ValueError(f"Invalid relationship: {relationship}. Must be 'normal' or 'inverse'")
            
            # Accumulate weighted sum
            weighted_sum += weight * parent_val_standard
            total_weight += weight
        
        # Normalize by total weight
        combined_value = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Step 2: Apply soft similarity function (Paper Equation 2)
        # P(child = c | parents) ∝ exp(-|c - combined_value| / σ)
        for row_idx, child_value in enumerate(standard_values):
            distance = abs(child_value - combined_value)
            similarity = np.exp(-distance / sigma)
            cpd_values[row_idx, col_idx] = similarity
        
        # Step 3: Normalize column to sum to 1
        column_sum = np.sum(cpd_values[:, col_idx])
        if column_sum > 0:
            cpd_values[:, col_idx] /= column_sum
        else:
            # Fallback to uniform if all zeros
            cpd_values[:, col_idx] = 1.0 / num_states
    
    # Create TabularCPD
    evidence_card = [len(parent_state_values[parent]) for parent in parent_names]
    
    cpd = TabularCPD(
        variable=child_name,
        variable_card=num_states,
        values=cpd_values,
        evidence=parent_names,
        evidence_card=evidence_card,
        state_names={child_name: standard_values, **parent_state_values}
    )
    
    return cpd


def create_collision_cpd(collision_rules: Dict[Tuple[str, str, str, str], str]) -> TabularCPD:
    """
    Create deterministic CPD for Collision_Point based on trajectory geometry.
    
    The collision point is determined by the intersection of ego and adversary
    trajectories through the T-junction. This is a deterministic mapping based
    on geometric analysis (see Paper Figure 3).
    
    Args:
        collision_rules: Dictionary mapping (start_ego, goal_ego, start_other, goal_other)
            to collision point ('c1', 'c2', 'c3', or None)
    
    Returns:
        TabularCPD with deterministic probabilities (0 or 1 for each configuration)
    
    State Meanings:
        - c1: Right lane collision (e.g., Left→Right crosses Base→Right)
        - c2: Center junction collision (e.g., Right→Left crosses Base→Left)
        - c3: Left lane collision (e.g., Left→Base crosses Right→Base)
        - None: No collision (paths don't intersect or invalid scenario)
    
    Example:
        >>> rules = define_collision_logic()
        >>> cpd = create_collision_cpd(rules)
    """
    locations = ['Left', 'Right', 'Base']
    collision_points = ['c1', 'c2', 'c3', 'None']
    
    # Total combinations: 3^4 = 81
    n_cols = len(locations) ** 4
    
    # Initialize CPD table (all zeros)
    cpd_values = np.zeros((len(collision_points), n_cols))
    
    # Generate all parent combinations in consistent order
    parent_combinations = list(product(locations, repeat=4))
    
    # Fill CPD based on collision rules
    for col_idx, (start_ego, goal_ego, start_other, goal_other) in enumerate(parent_combinations):
        # Invalid scenarios: vehicle doesn't move
        if start_ego == goal_ego or start_other == goal_other:
            cpd_values[collision_points.index('None'), col_idx] = 1.0
            continue
        
        # Look up collision point from rules
        collision = collision_rules.get((start_ego, goal_ego, start_other, goal_other))
        
        if collision is not None:
            # Collision occurs at specific point
            collision_idx = collision_points.index(collision)
            cpd_values[collision_idx, col_idx] = 1.0
        else:
            # No collision (paths don't intersect)
            cpd_values[collision_points.index('None'), col_idx] = 1.0
    
    # Verify all columns sum to 1
    assert np.allclose(cpd_values.sum(axis=0), 1.0), "CPD columns don't sum to 1"
    
    # Create TabularCPD
    cpd = TabularCPD(
        variable='Collision_Point',
        variable_card=len(collision_points),
        values=cpd_values,
        evidence=['Start_Ego', 'Goal_Ego', 'Start_Other', 'Goal_Other'],
        evidence_card=[len(locations)] * 4,
        state_names={
            'Collision_Point': collision_points,
            'Start_Ego': locations,
            'Goal_Ego': locations,
            'Start_Other': locations,
            'Goal_Other': locations
        }
    )
    
    return cpd


def extend_bayesian_network(
    fitted_model: BayesianNetwork,
    abstracted_structure: Dict[str, List[Tuple[str, str, float]]],
    standard_values: List[int] = STANDARD_VALUES
) -> BayesianNetwork:
    """
    Extend a fitted Bayesian Network with abstracted variables.
    
    Creates a new BN that includes:
    1. All nodes and edges from the original model
    2. New abstracted variables as leaf nodes
    3. Edges from concrete parents to abstracted children
    4. CPDs for abstracted variables computed via weighted aggregation
    
    Args:
        fitted_model: Already fitted BayesianNetwork from pgmpy
        abstracted_structure: Dict mapping child → list of (parent, relationship, weight)
            Example: {
                'Visibility': [
                    ('Fog_Density', 'inverse', 0.45),
                    ('Fog_Distance', 'normal', 0.45),
                    ('Precipitation', 'inverse', 0.1)
                ]
            }
        standard_values: Discrete states for abstracted variables
    
    Returns:
        Extended BayesianNetwork with abstracted nodes
    
    Raises:
        ValueError: If model validation fails
    
    Example:
        >>> abstracted_structure = {
        ...     'Road_Surface': [
        ...         ('Road_Friction', 'normal', 0.6),
        ...         ('Wetness', 'inverse', 0.2),
        ...         ('Precipitation_Deposits', 'inverse', 0.2)
        ...     ]
        ... }
        >>> extended_model = extend_bayesian_network(base_model, abstracted_structure)
    """
    # Get existing structure
    existing_edges = list(fitted_model.edges())
    
    # Create new edges for abstracted variables
    new_edges = []
    for child, parents_info in abstracted_structure.items():
        for parent, _, _ in parents_info:
            # Verify parent exists in original model
            if parent not in fitted_model.nodes():
                raise ValueError(f"Parent '{parent}' not found in fitted model for child '{child}'")
            new_edges.append((parent, child))
    
    # Combine all edges
    all_edges = existing_edges + new_edges
    
    # Create extended model
    extended_model = BayesianNetwork(all_edges)
    
    # Copy CPDs from fitted model for existing nodes
    print("Copying CPDs from original model...")
    for node in fitted_model.nodes():
        cpd = fitted_model.get_cpds(node)
        extended_model.add_cpds(cpd)
        print(f"  ✓ {node}")
    
    # Add CPDs for abstracted nodes
    print("\nComputing CPDs for abstracted variables...")
    for child, parents_info in abstracted_structure.items():
        print(f"  Computing {child}...")
        cpd = compute_abstracted_cpd(child, parents_info, fitted_model, standard_values)
        extended_model.add_cpds(cpd)
        print(f"  ✓ {child}")
    
    # Validate the extended model
    if extended_model.check_model():
        print("\n✓ Extended model is valid!")
    else:
        raise ValueError("✗ Extended model validation failed!")
    
    return extended_model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_abstracted_structure(abstracted_structure: Dict[str, List[Tuple[str, str, float]]]):
    """
    Print a human-readable summary of the abstracted variable structure.
    
    Args:
        abstracted_structure: Dict mapping child → list of (parent, relationship, weight)
    """
    print("=" * 70)
    print("ABSTRACTED VARIABLES STRUCTURE")
    print("=" * 70)
    
    for child, parents_info in abstracted_structure.items():
        print(f"\n{child}:")
        total_weight = sum(w for _, _, w in parents_info)
        
        for parent, relationship, weight in parents_info:
            percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
            symbol = "↓" if relationship == "normal" else "↑"
            print(f"  {symbol} {parent} ({relationship}, {weight:.2f} = {percentage:.1f}%)")
        
        print(f"  Total weight: {total_weight:.2f}")


def validate_abstracted_structure(
    abstracted_structure: Dict[str, List[Tuple[str, str, float]]],
    fitted_model: BayesianNetwork
) -> bool:
    """
    Validate that an abstracted structure is well-formed.
    
    Checks:
    1. All parents exist in the fitted model
    2. All relationships are 'normal' or 'inverse'
    3. All weights are positive
    4. Weights sum to approximately 1.0 for each child
    
    Args:
        abstracted_structure: Structure to validate
        fitted_model: Base model containing parent variables
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    for child, parents_info in abstracted_structure.items():
        # Check parents exist
        for parent, relationship, weight in parents_info:
            if parent not in fitted_model.nodes():
                raise ValueError(f"Parent '{parent}' not found in model for child '{child}'")
            
            if relationship not in ['normal', 'inverse']:
                raise ValueError(f"Invalid relationship '{relationship}' for {child}")
            
            if weight <= 0:
                raise ValueError(f"Invalid weight {weight} for {parent}→{child}")
        
        # Check weights sum to ~1.0
        total_weight = sum(w for _, _, w in parents_info)
        if not np.isclose(total_weight, 1.0, atol=0.01):
            print(f"Warning: Weights for '{child}' sum to {total_weight:.3f}, not 1.0")
    
    return True


if __name__ == "__main__":
    print("This module provides CPD computation functions for abstracted variables.")
    print("Import and use these functions in your parametrization pipeline.")
    print("\nKey functions:")
    print("  - compute_abstracted_cpd: Create CPDs with weighted aggregation")
    print("  - create_collision_cpd: Create deterministic collision CPD")
    print("  - extend_bayesian_network: Add abstractions to a fitted BN")