"""
BayScen Scenario Generator

Implements Algorithm 1 (Rarity-Prioritized Diverse Scenario Generation):
  1. For each capability configuration t ∈ T (exhaustive combinatorial coverage):
     a. Draw N=100,000 candidates from P(X | t) via likelihood-weighted sampling.
     b. Rank candidates by empirical frequency (ascending → rarest first).
     c. Select the candidate maximising min-distance to all already-generated
        scenarios (max-min diversity criterion).
  2. Return the accumulated scenario set S.

References:
    Paper Section II-D: Coverage
    Paper Section II-E: Generation
    Paper Algorithm 1:  Rarity-Prioritized Diverse Scenario Generation
"""

import logging
from collections import Counter
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from pgmpy.factors.discrete import State
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

logging.getLogger("pgmpy").setLevel(logging.ERROR)


class BayesianScenarioGenerator:
    """
    Generate diverse, rarity-prioritized test scenarios from a Bayesian Network.

    Given a trained BN (fitted + extended with capability leaf nodes), this class:
      - Enumerates all capability configurations T = ×_{i} V_i.
      - For each configuration t, samples concrete scenarios from P(X | t).
      - Selects the rarest candidate that maximises diversity over the existing set.
    """

    def __init__(
        self,
        model: BayesianNetwork,
        leaf_nodes: dict,
        initial_nodes: list,
        n_samples: int = 100_000,
        use_sampling: bool = True,
        prefer_rare: bool = True,
    ):
        """
        Args:
            model         : Trained BN with capability leaf nodes included.
            leaf_nodes    : Dict mapping capability variable names → list of states.
                            E.g. {'Sensor_Perception': [0,20,40,60,80,100], ...}
            initial_nodes : List of concrete variable names to include in output.
            n_samples     : Candidates per capability configuration (default 100,000).
            use_sampling  : Use likelihood-weighted sampling (True, recommended)
                            or exhaustive enumeration (False, small spaces only).
            prefer_rare   : If True, rank by rarity (edge-case discovery — default).
                            If False, rank by commonality (BayScen-Common ablation).
        """
        self.model        = model
        self.leaf_nodes   = leaf_nodes
        self.initial_nodes = initial_nodes
        self.n_samples    = n_samples
        self.use_sampling = use_sampling
        self.prefer_rare  = prefer_rare
        self.inference    = VariableElimination(model)
        self.generated_scenarios: list = []

    # ------------------------------------------------------------------
    # PHASE 1 — COVERAGE: enumerate capability configurations
    # ------------------------------------------------------------------

    def generate_capability_combinations(self):
        """
        Enumerate all capability configurations T = ×_{a_i ∈ A} V_i.

        Returns:
            (cap_names, combinations): list of variable names and all value tuples.
        """
        cap_names  = list(self.leaf_nodes.keys())
        cap_values = [self.leaf_nodes[n] for n in cap_names]
        combinations = list(product(*cap_values))
        print(f"Capability configurations: {len(combinations)} "
              f"({'×'.join(str(len(v)) for v in cap_values)})")
        return cap_names, combinations

    # ------------------------------------------------------------------
    # PHASE 2 — GENERATION: conditional sampling
    # ------------------------------------------------------------------

    def infer_concrete_parameters(self, evidence: dict) -> list:
        """
        Sample concrete parameters from P(X | t) and rank by rarity.

        Args:
            evidence : Dict of capability variable values (the target configuration t).

        Returns:
            List of (params_dict, empirical_probability) tuples sorted by rarity
            (ascending probability if prefer_rare=True).
        """
        if self.use_sampling:
            return self._likelihood_weighted_sampling(evidence)
        return self._exhaustive_search(evidence)

    def _likelihood_weighted_sampling(self, evidence: dict) -> list:
        """Draw N samples from P(X | evidence) and rank by empirical frequency."""
        sampler = BayesianModelSampling(self.model)
        evidence_states = [State(var, val) for var, val in evidence.items()]

        try:
            samples = sampler.likelihood_weighted_sample(
                evidence=evidence_states, size=self.n_samples
            )
        except Exception as e:
            print(f"  ⚠ Sampling failed for {evidence}: {e}")
            return []

        config_counts: Counter = Counter()
        for _, row in samples.iterrows():
            config = tuple((node, row[node]) for node in self.initial_nodes)
            config_counts[config] += 1

        total = sum(config_counts.values())

        # Sort ascending (rarest first) or descending (most common first)
        sorted_items = config_counts.most_common()
        if self.prefer_rare:
            sorted_items = sorted_items[::-1]

        results = []
        for config, count in sorted_items[:100]:
            results.append((dict(config), count / total))

        return results

    def _exhaustive_search(self, evidence: dict) -> list:
        """Exhaustive enumeration — only feasible for small concrete spaces."""
        initial_values = {
            node: self.model.get_cpds(node).state_names[node]
            for node in self.initial_nodes
        }
        combos = list(product(*[initial_values[n] for n in self.initial_nodes]))

        results = []
        for combo in combos:
            params = dict(zip(self.initial_nodes, combo))
            full_ev = {**evidence, **params}
            try:
                prob = self._joint_probability(full_ev)
                if prob > 0:
                    results.append((params, prob))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=not self.prefer_rare)
        return results[:100]

    def _joint_probability(self, evidence: dict) -> float:
        """Compute joint probability using the chain rule over the BN."""
        prob = 1.0
        for node in evidence:
            cpd     = self.model.get_cpds(node)
            parents = self.model.get_parents(node)
            if not parents:
                idx   = list(cpd.state_names[node]).index(evidence[node])
                prob *= float(cpd.values[idx])
            else:
                parent_ev = {p: evidence[p] for p in parents if p in evidence}
                if len(parent_ev) == len(parents):
                    qr   = self.inference.query([node], evidence=parent_ev,
                                               show_progress=False)
                    idx  = list(cpd.state_names[node]).index(evidence[node])
                    prob *= float(qr.values[idx])
        return prob

    # ------------------------------------------------------------------
    # PHASE 2 — SELECTION: max-min diversity
    # ------------------------------------------------------------------

    def _normalise(self, params_dict: dict) -> np.ndarray:
        """Normalise concrete parameter values to [0, 1] for distance computation."""
        vec = []
        for node in sorted(params_dict):
            val = params_dict[node]
            if isinstance(val, (int, float)):
                vec.append(val / 100.0)
            else:
                states = list(self.model.get_cpds(node).state_names[node])
                vec.append(states.index(val) / max(len(states) - 1, 1))
        return np.array(vec)

    def select_diverse_scenario(self, candidates: list):
        """
        Select the candidate that maximises its minimum distance to all
        previously generated scenarios (max-min diversity criterion, Eq. 4).

        For the very first scenario, the rarest candidate is chosen directly.
        """
        if not candidates:
            return None, 0.0
        if not self.generated_scenarios:
            return candidates[0]  # first: rarest

        best, best_dist = None, -1.0
        for params, prob in candidates:
            cand_vec  = self._normalise(params)
            min_dist  = min(
                euclidean(cand_vec, self._normalise(
                    {n: s[n] for n in self.initial_nodes}
                ))
                for s in self.generated_scenarios
            )
            if min_dist > best_dist:
                best_dist, best = min_dist, (params, prob)

        return best if best is not None else candidates[0]

    # ------------------------------------------------------------------
    # MAIN GENERATION LOOP
    # ------------------------------------------------------------------

    def generate_scenarios(self) -> pd.DataFrame:
        """
        Execute the full BayScen generation algorithm (Algorithm 1).

        Returns:
            DataFrame with one row per generated scenario containing concrete
            parameters, capability variable values, and empirical probability.
        """
        cap_names, cap_combos = self.generate_capability_combinations()

        mode_str = "RARE (rarity-prioritized)" if self.prefer_rare else "COMMON (BayScen-Common ablation)"
        print(f"\n{'=' * 70}")
        print("BAYSCEN SCENARIO GENERATION")
        print(f"{'=' * 70}")
        print(f"Mode            : {mode_str}")
        print(f"Configurations  : {len(cap_combos)}")
        print(f"Samples/config  : {self.n_samples:,}")
        print(f"Start           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}\n")

        start_time = datetime.now()

        for idx, combo in enumerate(tqdm(cap_combos, desc="Generating", unit="cfg")):
            evidence = {cap_names[i]: combo[i] for i in range(len(cap_names))}
            candidates = self.infer_concrete_parameters(evidence)

            if not candidates:
                tqdm.write(f"  ⚠ No candidates for config {idx + 1}/{len(cap_combos)} — skipping")
                continue

            selected, prob = self.select_diverse_scenario(candidates)
            if selected is None:
                selected, prob = candidates[0]

            scenario = {**selected, **evidence, 'probability': prob}
            self.generated_scenarios.append(scenario)

        elapsed = datetime.now() - start_time
        print(f"\n{'=' * 70}")
        print("GENERATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"End             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed         : {elapsed}")
        print(f"Scenarios       : {len(self.generated_scenarios)}")
        if self.generated_scenarios:
            avg = elapsed.total_seconds() / len(cap_combos)
            print(f"Avg / config    : {avg:.2f}s")
        print(f"{'=' * 70}\n")

        df = pd.DataFrame(self.generated_scenarios)
        col_order = self.initial_nodes + cap_names + ['probability']
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]

        # Assign T-junction paths for junction scenarios
        if 'Conflict_Geometry' in df.columns:
            print("Assigning T-junction paths from Conflict_Geometry...")
            from .generation_utils import assign_junction_paths
            df = assign_junction_paths(df, inplace=True)
            print("✓ Paths assigned\n")

        return df

    def save_scenarios(self, df: pd.DataFrame, filename: str = 'generated_scenarios.csv'):
        """Save generated scenarios to CSV (or XLSX)."""
        if filename.endswith('.xlsx'):
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Scenarios', index=False)
                ws = writer.sheets['Scenarios']
                for i, col in enumerate(df.columns):
                    w = max(df[col].astype(str).str.len().max(), len(col)) + 2
                    ws.column_dimensions[chr(65 + i)].width = min(w, 50)
        else:
            df.to_csv(filename, index=False)
        print(f"✓ Scenarios saved to {filename}")
