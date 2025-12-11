"""
Bayesian Scenario Generator

This module implements the BayScen scenario generation algorithm using:
1. Combinatorial coverage over abstracted variables
2. Conditional sampling from Bayesian Networks
3. Rarity prioritization for edge case discovery
4. Max-min diversity selection

References:
    Paper Section III-D: Generation Stage
    Paper Algorithm 1: Rarity-Prioritized Diverse Scenario Generation

Author: BayScen Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from itertools import product
from scipy.spatial.distance import euclidean
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import logging

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

# Suppress pgmpy warnings
logging.getLogger("pgmpy").setLevel(logging.ERROR)


class BayesianScenarioGenerator:
    """
    Generate diverse and realistic test scenarios from a Bayesian Network.
    
    This class implements the BayScen generation algorithm that:
    1. Generates all combinations of abstracted variable values (combinatorial coverage)
    2. For each combination, samples concrete scenarios from the BN (conditional sampling)
    3. Prioritizes rare but valid parameter combinations (rarity prioritization)
    4. Selects diverse scenarios using max-min distance (diversity selection)
    
    The result is a comprehensive test suite that achieves high coverage of the
    operational design domain while maintaining realism and discovering edge cases.
    """
    
    def __init__(
        self,
        model: BayesianNetwork,
        leaf_nodes: dict,
        initial_nodes: list,
        similarity_threshold: float = 0.1,
        n_samples: int = 100000,
        use_sampling: bool = True,
        prefer_rare: bool = False
    ):
        """
        Initialize the scenario generator.
        
        Args:
            model: Trained Bayesian Network with CPDs
            leaf_nodes: Dict mapping abstracted variable names to their values
                Example: {
                    'Visibility': [0, 20, 40, 60, 80, 100],
                    'Road_Surface': [0, 20, 40, 60, 80, 100],
                    'Vehicle_Stability': [0, 20, 40, 60, 80, 100],
                    'Collision_Point': ['c1', 'c2', 'c3']
                }
            initial_nodes: List of concrete variable names to sample
                Example: ['Cloudiness', 'Wind_Intensity', 'Precipitation', ...]
            similarity_threshold: Threshold for considering scenarios as similar (0-1)
                Lower values → stricter diversity requirements
            n_samples: Number of samples for likelihood-weighted sampling
                More samples → better approximation of posterior distribution
            use_sampling: If True, use Monte Carlo sampling (recommended for speed)
                If False, use exhaustive search (only feasible for small spaces)
            prefer_rare: If True, prioritize rare scenarios instead of common ones
                This helps discover edge cases and corner cases
        
        Example:
            >>> generator = BayesianScenarioGenerator(
            ...     model=trained_bn,
            ...     leaf_nodes={'Visibility': [0,20,40,60,80,100], ...},
            ...     initial_nodes=['Cloudiness', 'Wind_Intensity', ...],
            ...     prefer_rare=True  # For edge case discovery
            ... )
            >>> scenarios = generator.generate_scenarios()
        """
        self.model = model
        self.leaf_nodes = leaf_nodes
        self.initial_nodes = initial_nodes
        self.similarity_threshold = similarity_threshold
        self.n_samples = n_samples
        self.use_sampling = use_sampling
        self.prefer_rare = prefer_rare
        self.inference = VariableElimination(model)
        self.generated_scenarios = []
        
    def generate_leaf_combinations(self):
        """
        Generate all combinations of abstracted variable values.
        
        This implements the combinatorial coverage phase (Paper Section III-D).
        For example, with 3 abstracted variables having 6 values each and 
        Collision_Point with 3 values: 6 × 6 × 6 × 3 = 648 combinations.
        
        Returns:
            tuple: (leaf_names, combinations)
                - leaf_names: List of abstracted variable names
                - combinations: List of all value combinations
        """
        node_names = list(self.leaf_nodes.keys())
        node_values = [self.leaf_nodes[node] for node in node_names]
        
        combinations = list(product(*node_values))
        
        print(f"Generated {len(combinations)} abstracted variable combinations")
        return node_names, combinations
    
    def infer_initial_parameters(self, evidence: dict):
        """
        Infer concrete parameter values given abstracted variable evidence.
        
        This implements conditional sampling from the BN (Paper Section III-E-1).
        Uses either likelihood-weighted sampling (fast, approximate) or 
        exhaustive search (slow, exact).
        
        Args:
            evidence: Dictionary of abstracted variable values
                Example: {'Visibility': 60, 'Road_Surface': 80, ...}
            
        Returns:
            list: List of (params_dict, probability) tuples
                Sorted by probability (descending if prefer_rare=False)
        
        Example:
            >>> evidence = {'Visibility': 60, 'Collision_Point': 'c1'}
            >>> candidates = generator.infer_initial_parameters(evidence)
            >>> # Returns top 100 most/least likely parameter combinations
        """
        if self.use_sampling:
            return self._likelihood_weighted_sampling(evidence, n_samples=self.n_samples, top_k=100)
        else:
            return self._exhaustive_search(evidence, top_k=100)

    def _likelihood_weighted_sampling(self, evidence: dict, n_samples: int = 100000, top_k: int = 100):
        """
        Perform likelihood-weighted sampling from the Bayesian Network.
        
        This is the recommended method for sampling as it:
        1. Respects BN structure and CPDs
        2. Scales to large parameter spaces
        3. Naturally prioritizes high-probability regions
        
        Args:
            evidence: Abstracted variable values (conditioning set)
            n_samples: Number of samples to generate
            top_k: Number of top configurations to return
        
        Returns:
            list: Top-k parameter configurations with probabilities
        """
        sampler = BayesianModelSampling(self.model)
        
        # Convert evidence dict to pgmpy State objects
        evidence_list = [State(var, val) for var, val in evidence.items()]
        
        try:
            # Sample from P(concrete_vars | abstracted_vars)
            samples = sampler.likelihood_weighted_sample(
                evidence=evidence_list,
                size=n_samples
            )
        except Exception as e:
            print(f"⚠ Sampling failed for {evidence}: {e}")
            return []
        
        # Count occurrences of each configuration
        config_counts = Counter()
        for _, row in samples.iterrows():
            config = tuple((node, row[node]) for node in self.initial_nodes)
            config_counts[config] += 1
        
        # Normalize to probabilities
        total = sum(config_counts.values())
        results = []
        
        # Sort by rarity if prefer_rare=True, otherwise by commonness
        sorted_configs = (
            config_counts.most_common()[::-1] if self.prefer_rare 
            else config_counts.most_common()
        )
        
        for config, count in sorted_configs[:top_k]:
            params_dict = dict(config)
            prob = count / total
            results.append((params_dict, prob))
        
        return results
    
    def _exhaustive_search(self, evidence: dict, top_k: int = 100):
        """
        Exhaustive search using Variable Elimination (fallback method).
        
        Warning: Only feasible for small parameter spaces (<10k combinations).
        For large spaces, use likelihood-weighted sampling instead.
        
        Args:
            evidence: Abstracted variable values
            top_k: Number of configurations to return
        
        Returns:
            list: Top-k parameter configurations with exact probabilities
        """
        # Get all possible values for concrete variables
        initial_values = {}
        for node in self.initial_nodes:
            cpd = self.model.get_cpds(node)
            initial_values[node] = cpd.state_names[node]
        
        # Generate all combinations
        initial_combos = list(product(*[initial_values[node] for node in self.initial_nodes]))
        
        # Calculate probability for each combination
        results = []
        for combo in initial_combos:
            initial_dict = {self.initial_nodes[i]: combo[i] for i in range(len(self.initial_nodes))}
            
            # Create full evidence
            full_evidence = evidence.copy()
            full_evidence.update(initial_dict)
            
            # Calculate joint probability
            try:
                prob = self._calculate_joint_probability(full_evidence)
                if prob > 0:
                    results.append((initial_dict, prob))
            except:
                continue
        
        # Sort by rarity/commonness
        results.sort(key=lambda x: x[1], reverse=not self.prefer_rare)
        return results[:top_k]
    
    def _calculate_joint_probability(self, evidence: dict) -> float:
        """
        Calculate joint probability of evidence using chain rule.
        
        P(evidence) = ∏ P(var | parents(var)) for all vars in evidence
        
        Args:
            evidence: Complete variable assignment
        
        Returns:
            float: Joint probability
        """
        try:
            prob = 1.0
            for node in evidence:
                cpd = self.model.get_cpds(node)
                parents = self.model.get_parents(node)
                
                if not parents:
                    # No parents: use marginal probability
                    node_idx = list(cpd.state_names[node]).index(evidence[node])
                    prob *= cpd.values[node_idx]
                else:
                    # Has parents: use conditional probability
                    parent_evidence = {p: evidence[p] for p in parents if p in evidence}
                    
                    if len(parent_evidence) == len(parents):
                        query_result = self.inference.query(
                            variables=[node],
                            evidence=parent_evidence,
                            show_progress=False
                        )
                        node_idx = list(cpd.state_names[node]).index(evidence[node])
                        prob *= query_result.values[node_idx]
            
            return prob
        except Exception as e:
            return 0.0
    
    def normalize_values(self, params_dict: dict) -> np.ndarray:
        """
        Normalize parameter values to [0, 1] for distance calculation.
        
        This ensures all parameters contribute equally to diversity measurement
        regardless of their original scales.
        
        Args:
            params_dict: Dictionary of parameter values
        
        Returns:
            np.ndarray: Normalized parameter vector
        """
        vector = []
        for node in sorted(params_dict.keys()):
            value = params_dict[node]
            
            if isinstance(value, (int, float)):
                # Numeric: assume 0-100 range
                vector.append(value / 100.0)
            else:
                # Categorical: use index-based encoding
                cpd = self.model.get_cpds(node)
                states = list(cpd.state_names[node])
                if value in states:
                    vector.append(states.index(value) / len(states))
                else:
                    vector.append(0.0)
        
        return np.array(vector)
    
    def is_similar_to_existing(self, candidate_params: dict) -> bool:
        """
        Check if candidate is too similar to existing scenarios.
        
        Uses Euclidean distance in normalized parameter space.
        
        Args:
            candidate_params: Candidate parameter values
        
        Returns:
            bool: True if similar to any existing scenario
        """
        if not self.generated_scenarios:
            return False
        
        candidate_vector = self.normalize_values(candidate_params)
        
        for scenario in self.generated_scenarios:
            existing_params = {node: scenario[node] for node in self.initial_nodes}
            existing_vector = self.normalize_values(existing_params)
            
            # Calculate normalized Euclidean distance
            distance = euclidean(candidate_vector, existing_vector)
            max_distance = np.sqrt(len(candidate_vector))
            normalized_distance = distance / max_distance
            
            if normalized_distance < self.similarity_threshold:
                return True
        
        return False

    def select_diverse_scenario(self, candidates: list):
        """
        Select scenario that maximizes diversity (max-min distance criterion).
        
        This implements diversity-aware selection (Paper Section III-E-3).
        Among all candidates, selects the one that is most different from
        all existing scenarios (measured by minimum distance).
        
        Args:
            candidates: List of (params_dict, probability) tuples
        
        Returns:
            tuple: (selected_params, probability)
        
        Example:
            If existing scenarios occupy regions A and B, this selects
            the candidate farthest from both A and B.
        """
        if not candidates:
            return None, 0.0
        
        # First scenario: take highest/lowest probability
        if not self.generated_scenarios:
            return candidates[0]
        
        best_candidate = None
        max_min_distance = -1.0
        
        # For each candidate, find its minimum distance to existing scenarios
        for params_dict, prob in candidates:
            candidate_vector = self.normalize_values(params_dict)
            
            min_distance_to_existing = float('inf')
            for scenario in self.generated_scenarios:
                existing_params = {node: scenario[node] for node in self.initial_nodes}
                existing_vector = self.normalize_values(existing_params)
                
                distance = euclidean(candidate_vector, existing_vector)
                min_distance_to_existing = min(min_distance_to_existing, distance)
            
            # Select candidate with largest minimum distance
            if min_distance_to_existing > max_min_distance:
                max_min_distance = min_distance_to_existing
                best_candidate = (params_dict, prob)
        
        # Fallback to most/least probable if all candidates are identical
        if best_candidate is None:
            return candidates[0]
        
        return best_candidate

    def generate_scenarios(self) -> pd.DataFrame:
        """
        Execute the complete BayScen generation algorithm.
        ...
        """
        leaf_names, leaf_combinations = self.generate_leaf_combinations()
        
        mode_str = "RARE (Edge Cases)" if self.prefer_rare else "COMMON (Typical Cases)"
        print(f"\n{'='*70}")
        print(f"BAYSCEN SCENARIO GENERATION")
        print(f"{'='*70}")
        print(f"Mode: {mode_str}")
        print(f"Total combinations: {len(leaf_combinations)}")
        print(f"Diversity threshold: {self.similarity_threshold}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        start_time = datetime.now()
        
        # Process each abstracted variable combination
        for idx, combo in enumerate(tqdm(leaf_combinations, desc="Generating scenarios", unit="combo")):
            # Create evidence for abstracted variables
            evidence = {leaf_names[i]: combo[i] for i in range(len(leaf_names))}
            
            # Sample concrete parameters from BN
            candidates = self.infer_initial_parameters(evidence)
            
            if not candidates:
                tqdm.write(f"⚠ Warning: No valid parameters found for combination {idx+1}")
                continue
            
            # Select diverse scenario
            selected_params, probability = self.select_diverse_scenario(candidates)
            
            if selected_params is None:
                tqdm.write(f"⚠ Warning: Using fallback for combination {idx+1}")
                selected_params, probability = candidates[0]
            
            # Create complete scenario
            scenario = selected_params.copy()
            scenario.update(evidence)
            scenario['probability'] = probability
            
            self.generated_scenarios.append(scenario)
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time: {elapsed_time}")
        print(f"Scenarios generated: {len(self.generated_scenarios)}")
        avg = elapsed_time.total_seconds() / len(leaf_combinations)
        print(f"Average time per scenario: {avg:.2f}s")
        print(f"{'='*70}\n")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.generated_scenarios)
        
        # Reorder columns: concrete vars, abstracted vars, probability
        column_order = self.initial_nodes + leaf_names + ['probability']
        df = df[column_order]
        
        # ASSIGN T-JUNCTION PATHS BASED ON COLLISION_POINT
        print("Assigning T-junction paths based on Collision_Point...")
        from .generation_utils import assign_tjunction_paths
        df = assign_tjunction_paths(df, inplace=True)
        print("✓ T-junction paths assigned\n")
        
        return df
    
    def save_scenarios(self, df: pd.DataFrame, filename: str = 'generated_scenarios.csv'):
        """
        Save generated scenarios to CSV file.
        
        Args:
            df: DataFrame containing scenarios
            filename: Output filename (.csv or .xlsx)
        
        Example:
            >>> scenarios = generator.generate_scenarios()
            >>> generator.save_scenarios(scenarios, 'scenario1_test_suite.csv')
        """
        if filename.endswith('.xlsx'):
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Scenarios', index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets['Scenarios']
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(col)
                    ) + 2
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
        else:
            df.to_csv(filename, index=False)
        
        print(f"✓ Scenarios saved to {filename}")