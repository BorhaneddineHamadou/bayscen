"""
BayScen Generation Module

Scenario generation for autonomous vehicle testing using Bayesian Networks.

Main components:
- scenario_generator: Core generation algorithm
- evaluation_metrics: Quality metrics (realism, coverage, diversity)
- generate_scenarios: Command-line interface
- generation_utils: Helper functions

Quick Start:
    >>> from generation.scenario_generator import BayesianScenarioGenerator
    >>> from generation.evaluation_metrics import evaluate_scenarios
    >>> 
    >>> # Load model and generate
    >>> generator = BayesianScenarioGenerator(model, leaf_nodes, initial_nodes)
    >>> scenarios = generator.generate_scenarios()
    >>> 
    >>> # Evaluate
    >>> results = evaluate_scenarios('data.csv', scenarios, attributes)

Command Line:
    $ python generate_scenarios.py --scenario 1 --mode rare

See README.md for detailed documentation.
"""

from .scenario_generator import BayesianScenarioGenerator
from .evaluation_metrics import (
    compute_attribute_distributions,
    compute_realism,
    compute_threeway_coverage,
    compute_criticality_metrics,
    load_simulation_results,
    evaluate_scenarios,
    compare_methods
)
from .generation_utils import (
    assign_tjunction_paths,
    validate_scenarios,
    export_for_carla,
    split_by_collision_point,
    get_summary_statistics
)

__all__ = [
    'BayesianScenarioGenerator',
    'compute_attribute_distributions',
    'compute_realism',
    'compute_threeway_coverage',
    'compute_criticality_metrics',
    'load_simulation_results',
    'evaluate_scenarios',
    'compare_methods',
    'assign_tjunction_paths',
    'validate_scenarios',
    'export_for_carla',
    'split_by_collision_point',
    'get_summary_statistics'
]

__version__ = '1.0.0'
__author__ = 'BayScen Team'