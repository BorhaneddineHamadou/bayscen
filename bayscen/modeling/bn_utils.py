"""
Bayesian Network Utility Functions

This module contains utility functions for working with Bayesian Networks,
including model saving/loading and CPD visualization.

Author: BayScen Team
Date: December 2025
"""

import pickle
import pandas as pd
import numpy as np
from typing import Optional
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def save_model(model: BayesianNetwork, filename: str) -> None:
    """
    Save a fitted Bayesian Network model to disk.
    
    Args:
        model: The BayesianNetwork object to save
        filename: Path where the model will be saved (e.g., 'model.pkl')
    
    Example:
        >>> save_model(bn_model, 'fitted_model.pkl')
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filename}")


def load_model(filename: str) -> BayesianNetwork:
    """
    Load a fitted Bayesian Network model from disk.
    
    Args:
        filename: Path to the saved model file
    
    Returns:
        The loaded BayesianNetwork object
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    
    Example:
        >>> model = load_model('fitted_model.pkl')
    """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"✗ Error: {filename} not found")
        raise


def print_cpd_as_dataframe(cpd: TabularCPD, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Convert a TabularCPD to a pandas DataFrame for readable display.
    
    This function reshapes the CPD values into a DataFrame where:
    - Rows represent parent variable combinations (if any)
    - Columns represent the child variable states
    - Values are the conditional probabilities
    
    Args:
        cpd: TabularCPD object from pgmpy
        max_rows: Maximum number of rows to display (None for all)
    
    Returns:
        DataFrame representation of the CPD
    
    Example:
        >>> cpd = model.get_cpds('Visibility')
        >>> df = print_cpd_as_dataframe(cpd)
        >>> print(df.head())
    """
    # Get CPD components
    values = cpd.values
    variables = cpd.variables
    state_names = cpd.state_names
    
    # Get parent variables and their states
    evidence_vars = variables[1:]  # All variables except the first (child)
    evidence_states = [state_names[var] for var in evidence_vars]
    
    # Create MultiIndex for parent combinations
    if evidence_states:
        index = pd.MultiIndex.from_product(evidence_states, names=evidence_vars)
    else:
        index = None
    
    # Get child variable states for columns
    columns = state_names[cpd.variable]
    
    # Reshape values array to match DataFrame structure
    if len(variables) > 1:
        # Transpose axes: (child_states, parent1_states, parent2_states, ...) 
        # -> (parent1_states, parent2_states, ..., child_states)
        order = list(range(1, len(variables))) + [0]
        df_values = values.transpose(order).reshape(-1, values.shape[0])
    else:
        # No parents: simple 2D array
        df_values = values.T
    
    # Create DataFrame
    cpd_df = pd.DataFrame(
        data=df_values,
        index=index,
        columns=columns
    )
    
    if max_rows:
        return cpd_df.head(max_rows)
    
    return cpd_df


def print_model_summary(model: BayesianNetwork) -> None:
    """
    Print a comprehensive summary of a Bayesian Network model.
    
    Includes:
    - Number of nodes and edges
    - List of all variables
    - Model validity check
    
    Args:
        model: BayesianNetwork object to summarize
    
    Example:
        >>> print_model_summary(my_model)
    """
    print("=" * 70)
    print("BAYESIAN NETWORK SUMMARY")
    print("=" * 70)
    print(f"Total nodes: {len(model.nodes())}")
    print(f"Total edges: {len(model.edges())}")
    print(f"\nNodes: {sorted(model.nodes())}")
    print(f"\nModel valid: {model.check_model()}")
    print("=" * 70)


def get_cpd_by_variable(model: BayesianNetwork, variable_name: str) -> Optional[TabularCPD]:
    """
    Retrieve the CPD for a specific variable in the model.
    
    Args:
        model: BayesianNetwork object
        variable_name: Name of the variable
    
    Returns:
        TabularCPD object for the variable, or None if not found
    
    Example:
        >>> cpd = get_cpd_by_variable(model, 'Road_Surface')
        >>> if cpd:
        >>>     print(cpd)
    """
    try:
        return model.get_cpds(variable_name)
    except ValueError:
        print(f"Warning: Variable '{variable_name}' not found in model")
        return None


def validate_model_structure(model: BayesianNetwork, expected_nodes: list = None) -> bool:
    """
    Validate that a Bayesian Network has the expected structure.
    
    Args:
        model: BayesianNetwork to validate
        expected_nodes: Optional list of node names that should be present
    
    Returns:
        True if valid, False otherwise
    
    Example:
        >>> nodes = ['Cloudiness', 'Wind_Intensity', 'Precipitation']
        >>> is_valid = validate_model_structure(model, nodes)
    """
    # Check if model is a valid DAG
    if not model.check_model():
        print("✗ Model validation failed: Invalid CPDs or structure")
        return False
    
    # Check for expected nodes if provided
    if expected_nodes:
        model_nodes = set(model.nodes())
        expected_set = set(expected_nodes)
        
        if not expected_set.issubset(model_nodes):
            missing = expected_set - model_nodes
            print(f"✗ Missing expected nodes: {missing}")
            return False
    
    print("✓ Model structure is valid")
    return True