"""
BayScen Generation Module

Rarity-prioritized diverse scenario generation for autonomous vehicle testing
using Bayesian Networks and ISO 34503-grounded capability abstraction.

Quick Start:
    >>> from generation.scenario_generator import BayesianScenarioGenerator
    >>> from generation.evaluation_metrics import evaluate_scenarios
    >>>
    >>> generator = BayesianScenarioGenerator(model, leaf_nodes, initial_nodes)
    >>> scenarios = generator.generate_scenarios()

Command Line:
    python generate_scenarios.py --scenario 1 --mode rare
"""

from .scenario_generator import BayesianScenarioGenerator
from .evaluation_metrics import (
    check_physical_plausibility,
    physical_plausibility_summary,
    clean_critical_rate,
    compute_attribute_distributions,
    evaluate_scenarios,
)
from .generation_utils import (
    assign_junction_paths,
    validate_scenarios,
    export_for_carla,
    split_by_conflict_geometry,
    get_summary_statistics,
)

__all__ = [
    'BayesianScenarioGenerator',
    'check_physical_plausibility',
    'physical_plausibility_summary',
    'clean_critical_rate',
    'compute_attribute_distributions',
    'evaluate_scenarios',
    'assign_junction_paths',
    'validate_scenarios',
    'export_for_carla',
    'split_by_conflict_geometry',
    'get_summary_statistics',
]

__version__ = '2.0.0'
__author__ = 'BayScen Team'
