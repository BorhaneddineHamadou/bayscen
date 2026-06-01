# BayScen — Generation Module

Rarity-prioritized diverse scenario generation from trained Bayesian Networks.

## Overview

The generation module implements **Algorithm 1** from the paper. For each
capability configuration t ∈ T (exhaustive combinatorial coverage):

1. Draw N = 100,000 candidates from P(X | t) via likelihood-weighted sampling.
2. Rank candidates by empirical rarity (ascending probability).
3. Select the candidate that maximises its minimum distance to all existing
   scenarios (max-min diversity, Eq. 4).

## Directory Structure

```
generation/
├── scenario_generator.py           # Algorithm 1 core implementation
├── generate_scenarios.py           # CLI interface (S1, S2, S3)
├── evaluation_metrics.py           # Physical plausibility checks (C1–C6)
├── generation_utils.py             # Path assignment, export helpers
├── scenario_generation_tutorial.ipynb # Interactive generation guide
├── README.md
└── generated_scenarios/            # Output CSVs (created on first run)
    ├── scenario1_rare_scenarios.csv
    ├── scenario1_common_scenarios.csv
    ├── scenario2_rare_scenarios.csv
    ├── scenario2_common_scenarios.csv
    └── scenario3_rare_scenarios.csv
```

## Quick Start

### Command Line

```bash
# BayScen (rarity-prioritized — recommended)
python generate_scenarios.py --scenario 1 --mode rare   # S1: 648 scenarios
python generate_scenarios.py --scenario 2 --mode rare   # S2: 648 scenarios
python generate_scenarios.py --scenario 3 --mode rare   # S3: 216 scenarios

# BayScen-Common ablation (common scenario selection)
python generate_scenarios.py --scenario 1 --mode common
```

### Python API

```python
import pickle
from generation.scenario_generator import BayesianScenarioGenerator
from abstraction.abstract_variables import LEAF_NODES, LEAF_NODES_S3

with open('modeling/models/scenario1_full_bayesian_network.pkl', 'rb') as f:
    model = pickle.load(f)

generator = BayesianScenarioGenerator(
    model=model,
    leaf_nodes=LEAF_NODES,          # or LEAF_NODES_S3 for Scenario 3
    initial_nodes=['Cloudiness', 'Wind_Intensity', ...],
    prefer_rare=True,               # False → BayScen-Common ablation
)

scenarios = generator.generate_scenarios()
generator.save_scenarios(scenarios, 'scenario1_rare_scenarios.csv')
```

## Capability Configurations (Combinatorial Space)

### Scenarios 1 & 2 (junction) — 648 configurations

| Capability Variable | States | Count |
|--------------------|--------|-------|
| Conflict_Geometry (g) | g1, g2, g3 | 3 |
| Sensor_Perception (a_perc) | 0, 20, 40, 60, 80, 100 | 6 |
| Surface_Traction (a_trac) | 0, 20, 40, 60, 80, 100 | 6 |
| Lateral_Stability (a_stab) | 0, 20, 40, 60, 80, 100 | 6 |
| **Total** | | **3 × 6 × 6 × 6 = 648** |

### Scenario 3 (cut-in) — 216 configurations

| Capability Variable | States | Count |
|--------------------|--------|-------|
| Sensor_Perception (a_perc) | 0, 20, 40, 60, 80, 100 | 6 |
| Surface_Traction (a_trac) | 0, 20, 40, 60, 80, 100 | 6 |
| Lateral_Stability (a_stab) | 0, 20, 40, 60, 80, 100 | 6 |
| **Total** | | **6 × 6 × 6 = 216** |

Note: For S3, `Cut_In_Direction` (Left/Right) is retained as a concrete
variable and sampled from the BN; it is not part of the combinatorial suite.

## Algorithm Steps

### 1. Combinatorial Coverage

```
T = {g1,g2,g3} × {0,20,40,60,80,100}³    [S1/S2, 648 configurations]
T =              {0,20,40,60,80,100}³    [S3,    216 configurations]
```

### 2. Conditional Sampling

For each t = (a_perc=v*, a_trac=v*, a_stab=v*, g=v*):

```python
samples = sampler.likelihood_weighted_sample(
    evidence=t, size=100_000
)
```

### 3. Rarity-Prioritized Selection

```python
# Sort by ascending empirical frequency → rarest configurations first
candidates = sorted(config_counts, key=lambda x: x[1])[:100]
```

### 4. Max-Min Diversity (Eq. 4)

```python
s* = argmax_{s ∈ C} min_{s' ∈ S} d(s, s')
```

where d is normalised Euclidean distance over the concrete parameter space.

## Output CSV Format

Each generated scenario CSV contains:

| Column group | Columns |
|---|---|
| Concrete environmental | `Cloudiness`, `Wind_Intensity`, `Precipitation`, `Precipitation_Deposits`, `Wetness`, `Fog_Density`, `Road_Friction`, `Fog_Distance`, `Sun_Altitude_Angle` (S2/S3) |
| Trajectory (S1/S2) | `Start_Ego`, `Goal_Ego`, `Start_Other`, `Goal_Other` |
| Trajectory (S3) | `Cut_In_Direction` |
| Capability variables | `Sensor_Perception`, `Surface_Traction`, `Lateral_Stability`, `Conflict_Geometry` (S1/S2) |
| Generation metadata | `probability` (empirical probability of the concrete configuration) |

## Physical Plausibility Evaluation

After generation, `evaluation_metrics.py` checks constraints C1–C6 from
Hao et al. (arXiv:2311.10937):

```python
from generation.evaluation_metrics import physical_plausibility_summary

summary = physical_plausibility_summary(scenarios_df)
print(f"Physically plausible: {summary['physically_plausible_rate']:.1f}%")
```

Full evaluation (TISA coverage, collision rates, failure profiling) is in
`evaluation/scripts/`.

## Performance

Typical generation times (single CPU core, no parallelisation):

| Scenario | Configurations | Typical time |
|----------|---------------|-------------|
| S1 Vehicle–Vehicle | 648 | ~80 min |
| S2 Vehicle–Cyclist | 648 | ~80 min |
| S3 Cut-In | 216 | ~45 min |

Since each capability configuration is sampled independently, the process is
embarrassingly parallel — wall time scales inversely with available cores.

## References

- Paper Algorithm 1: Rarity-Prioritized Diverse Scenario Generation
- Paper Section II-D: Coverage — combinatorial testing on capability variables
- Paper Section II-E: Generation — conditional sampling and diversity selection
