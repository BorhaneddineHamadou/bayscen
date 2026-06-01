"""
Capability Variable CPD Computation

Implements the uniform-weight aggregation CPD for BayScen capability variables.

Each capability variable is a leaf node whose CPD is computed by:
  1. For every parent configuration, computing the uniform-weight average of the
     normalised (0–100) parent values after applying directional transformations
     (inverse: 100 − value).
  2. Assigning probability mass 1.0 to the K=6 discrete level [0,20,40,60,80,100]
     nearest to the computed average, and 0.0 to all other levels.

This "concentrating" CPD encodes the abstraction faithfully: conditioning on a
target capability level a* induces a well-defined posterior P(X|A=a*) from which
concrete parameter assignments are drawn via ancestral sampling.

References:
    Paper Section II-C-4: Aggregation and Discretization
    Paper Section II-E-1: Conditional Sampling

Key functions:
    compute_capability_cpd      – CPD for a continuous-parent capability variable
    create_conflict_geometry_cpd – deterministic CPD for Conflict_Geometry
    extend_bayesian_network      – add capability leaf nodes to a fitted BN
"""

import numpy as np
from itertools import product
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from mapping_functions import STANDARD_VALUES, MAP_TO_STANDARD


# ============================================================================
# CORE CPD COMPUTATION
# ============================================================================

def compute_capability_cpd(
    child_name: str,
    parents_info: List[Tuple[str, str, float]],
    fitted_model: BayesianNetwork,
    standard_values: List[int] = STANDARD_VALUES,
) -> TabularCPD:
    """
    Compute the CPD for a capability variable using uniform-weight aggregation.

    For each combination of parent values the CPD concentrates all probability
    mass on the K=6 discrete level [0, 20, 40, 60, 80, 100] that is nearest to
    the uniform-weight average of the (normalised, direction-adjusted) parent
    values. This implements the aggregation rule described in Section II-C-4.

    Args:
        child_name    : Name of the capability variable (leaf node).
        parents_info  : List of (parent_name, relationship, weight) tuples.
                        relationship: 'normal' (co-varies positively) or
                                      'inverse' (co-varies negatively).
                        weight: contribution weight (all equal → uniform).
        fitted_model  : Fitted BayesianNetwork whose nodes are the parents.
        standard_values: Discrete states for the child, default [0,20,40,60,80,100].

    Returns:
        TabularCPD for the capability variable.

    Example:
        >>> parents = [
        ...     ('Fog_Density',   'inverse', 0.25),
        ...     ('Fog_Distance',  'normal',  0.25),
        ...     ('Cloudiness',    'inverse', 0.25),
        ...     ('Precipitation', 'inverse', 0.25),
        ... ]
        >>> cpd = compute_capability_cpd('Sensor_Perception', parents, model)
    """
    parent_names = [p[0] for p in parents_info]

    # Retrieve actual state values for each parent
    parent_state_values: Dict[str, list] = {}
    for parent in parent_names:
        parent_cpd = fitted_model.get_cpds(parent)
        parent_state_values[parent] = parent_cpd.state_names[parent]

    parent_value_lists = [parent_state_values[p] for p in parent_names]
    parent_combinations = list(product(*parent_value_lists))
    num_combinations = len(parent_combinations)

    num_states = len(standard_values)
    cpd_values = np.zeros((num_states, num_combinations))

    for col_idx, parent_values in enumerate(parent_combinations):
        # --- Step 1: uniform-weight average of normalised parent values ---
        transformed = []
        for parent_idx, (parent_name, relationship, _weight) in enumerate(parents_info):
            raw_val = parent_values[parent_idx]

            # Map to standard [0, 100] scale if needed (e.g. Road_Friction)
            map_func = MAP_TO_STANDARD.get(parent_name)
            std_val = map_func(raw_val) if map_func else raw_val

            if isinstance(std_val, (list, tuple)):
                std_val = float(np.mean(std_val))
            else:
                std_val = float(std_val)

            # Directional adjustment
            if relationship == 'inverse':
                std_val = 100.0 - std_val
            elif relationship != 'normal':
                raise ValueError(
                    f"Relationship must be 'normal' or 'inverse', got '{relationship}'"
                )

            transformed.append(std_val)

        average = float(np.mean(transformed))  # uniform weights

        # --- Step 2: nearest K=6 level (hard assignment) ---
        nearest_idx = int(np.argmin([abs(v - average) for v in standard_values]))
        cpd_values[nearest_idx, col_idx] = 1.0

    # Create TabularCPD
    evidence_card = [len(parent_state_values[p]) for p in parent_names]
    cpd = TabularCPD(
        variable=child_name,
        variable_card=num_states,
        values=cpd_values,
        evidence=parent_names,
        evidence_card=evidence_card,
        state_names={child_name: standard_values, **parent_state_values},
    )

    return cpd


def create_conflict_geometry_cpd(
    conflict_rules: Dict[Tuple[str, str, str, str], Optional[str]]
) -> TabularCPD:
    """
    Create the deterministic CPD for Conflict_Geometry based on trajectory geometry.

    Maps all (start_ego, goal_ego, start_other, goal_other) combinations to the
    conflict geometry state g1, g2, g3, or None (no conflict).

    Args:
        conflict_rules: Dict from ConflictGeometry.define_conflict_logic().

    Returns:
        TabularCPD with deterministic (0/1) probabilities.

    State meanings:
        g1  : right-lane conflict
        g2  : centre-junction conflict
        g3  : left-lane conflict
        None: no trajectory intersection (invalid for generation; filtered out)
    """
    locations = ['Left', 'Right', 'Base']
    geometry_states = ['g1', 'g2', 'g3', 'None']

    n_cols = len(locations) ** 4
    cpd_values = np.zeros((len(geometry_states), n_cols))

    parent_combinations = list(product(locations, repeat=4))

    for col_idx, (start_ego, goal_ego, start_other, goal_other) in enumerate(parent_combinations):
        if start_ego == goal_ego or start_other == goal_other:
            cpd_values[geometry_states.index('None'), col_idx] = 1.0
            continue

        geom = conflict_rules.get((start_ego, goal_ego, start_other, goal_other))
        if geom is not None:
            cpd_values[geometry_states.index(geom), col_idx] = 1.0
        else:
            cpd_values[geometry_states.index('None'), col_idx] = 1.0

    assert np.allclose(cpd_values.sum(axis=0), 1.0), "CPD columns do not sum to 1"

    cpd = TabularCPD(
        variable='Conflict_Geometry',
        variable_card=len(geometry_states),
        values=cpd_values,
        evidence=['Start_Ego', 'Goal_Ego', 'Start_Other', 'Goal_Other'],
        evidence_card=[len(locations)] * 4,
        state_names={
            'Conflict_Geometry': geometry_states,
            'Start_Ego':   locations,
            'Goal_Ego':    locations,
            'Start_Other': locations,
            'Goal_Other':  locations,
        },
    )

    return cpd


# ============================================================================
# EXTEND BAYESIAN NETWORK
# ============================================================================

def extend_bayesian_network(
    fitted_model: BayesianNetwork,
    capability_structure: Dict[str, List[Tuple[str, str, float]]],
    standard_values: List[int] = STANDARD_VALUES,
) -> BayesianNetwork:
    """
    Extend a fitted BN with ISO 34503-grounded capability variables as leaf nodes.

    Creates a new BN that includes:
      1. All nodes and edges from the original model (with their CPDs).
      2. New capability leaf nodes with edges from their concrete parent parameters.
      3. CPDs for capability nodes computed via uniform-weight aggregation.

    Args:
        fitted_model       : Already-fitted BayesianNetwork (from pgmpy).
        capability_structure: Dict mapping capability variable name →
                              list of (parent_name, relationship, weight) tuples.
        standard_values    : Discrete states for capability variables.

    Returns:
        Extended BayesianNetwork with capability leaf nodes.

    Raises:
        ValueError: If a parent is missing from fitted_model or validation fails.

    Example:
        >>> structure = {
        ...     'Surface_Traction': [
        ...         ('Road_Friction',          'normal',  1/3),
        ...         ('Wetness',                'inverse', 1/3),
        ...         ('Precipitation_Deposits', 'inverse', 1/3),
        ...     ]
        ... }
        >>> extended = extend_bayesian_network(base_model, structure)
    """
    existing_edges = list(fitted_model.edges())

    new_edges = []
    for child, parents_info in capability_structure.items():
        for parent, _rel, _w in parents_info:
            if parent not in fitted_model.nodes():
                raise ValueError(
                    f"Parent '{parent}' not found in fitted model for capability '{child}'"
                )
            new_edges.append((parent, child))

    extended_model = BayesianNetwork(existing_edges + new_edges)

    print("Copying CPDs from base model...")
    for node in fitted_model.nodes():
        extended_model.add_cpds(fitted_model.get_cpds(node))
        print(f"  ✓ {node}")

    print("\nComputing CPDs for capability variables (uniform-weight aggregation)...")
    for child, parents_info in capability_structure.items():
        print(f"  Computing {child}...")
        cpd = compute_capability_cpd(child, parents_info, fitted_model, standard_values)
        extended_model.add_cpds(cpd)
        print(f"  ✓ {child}")

    if extended_model.check_model():
        print("\n✓ Extended model validated successfully")
    else:
        raise ValueError("Extended model validation failed — check CPD consistency")

    return extended_model


# ============================================================================
# UTILITIES
# ============================================================================

def print_capability_structure(
    capability_structure: Dict[str, List[Tuple[str, str, float]]]
):
    """Print a human-readable summary of the capability variable structure."""
    print("=" * 70)
    print("CAPABILITY VARIABLES STRUCTURE (ISO 34503:2023)")
    print("=" * 70)

    for child, parents_info in capability_structure.items():
        print(f"\n{child}:")
        for parent, relationship, weight in parents_info:
            direction = "↑ co-varies" if relationship == 'normal' else "↓ inverse"
            print(f"  {parent:30s}  {direction}  (weight={weight:.4f})")


if __name__ == "__main__":
    print("abstraction_cpd.py — BayScen capability CPD module")
    print("\nKey functions:")
    print("  compute_capability_cpd      : uniform-weight aggregation CPD")
    print("  create_conflict_geometry_cpd: deterministic conflict geometry CPD")
    print("  extend_bayesian_network     : add capability leaves to a fitted BN")
