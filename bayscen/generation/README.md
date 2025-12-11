# BayScen Generation Module

This module implements the complete BayScen scenario generation algorithm for autonomous vehicle testing.

## Overview

The generation module transforms trained Bayesian Networks into comprehensive test suites by:

1. **Combinatorial Coverage**: Systematically exploring all combinations of abstracted variables
2. **Conditional Sampling**: Sampling concrete parameters from the Bayesian Network
3. **Rarity Prioritization**: Focusing on rare but valid edge cases
4. **Diversity Selection**: Ensuring diverse coverage using max-min distance

## Directory Structure

```
generation/
├── scenario_generator.py        # Main generation algorithm
├── evaluation_metrics.py         # Realism, coverage, and criticality metrics
├── generate_scenarios.py         # Command-line interface
├── generation_utils.py           # Helper functions
├── README.md                     # This file
└── generated_scenarios/          # Output directory (created on first run)
    ├── scenario1_rare_scenarios.csv
    ├── scenario1_common_scenarios.csv
    ├── scenario2_rare_scenarios.csv
    └── scenario2_common_scenarios.csv
```

## Quick Start

### Command Line (Recommended)

```bash
# Generate edge case scenarios for Scenario 1 (Vehicle-Vehicle)
cd bayscen/generation
python generate_scenarios.py --scenario 1 --mode rare

# Generate typical scenarios for Scenario 2 (Vehicle-Cyclist)
python generate_scenarios.py --scenario 2 --mode common
```

### Python API

```python
from bayscen.generation.scenario_generator import BayesianScenarioGenerator
from bayscen.generation.evaluation_metrics import evaluate_scenarios
from bayscen.abstraction.abstract_variables import LEAF_NODES
import pickle

# Load trained model
with open('modeling/models/scenario1_full_bayesian_network.pkl', 'rb') as f:
    model = pickle.load(f)

# Define parameters
concrete_vars = ['Cloudiness', 'Wind_Intensity', 'Precipitation', ...]
abstracted_vars = LEAF_NODES  # From abstract_variables.py

# Create generator
generator = BayesianScenarioGenerator(
    model=model,
    leaf_nodes=abstracted_vars,
    initial_nodes=concrete_vars,
    prefer_rare=True  # For edge case discovery
)

# Generate scenarios
scenarios = generator.generate_scenarios()

# Save
generator.save_scenarios(scenarios, 'test_scenarios.csv')

# Evaluate
results = evaluate_scenarios(
    real_data_path='data/processed/bayscen_final_data.csv',
    generated_df=scenarios,
    attributes=['Cloudiness', 'Wind_Intensity', ...]
)
```

## Algorithm Overview

### 1. Combinatorial Coverage Phase

Generates all combinations of abstracted variable values:

```python
# Example: 648 combinations for Scenario 1
Visibility: [0, 20, 40, 60, 80, 100]        # 6 values
Road_Surface: [0, 20, 40, 60, 80, 100]      # 6 values  
Vehicle_Stability: [0, 20, 40, 60, 80, 100] # 6 values
Collision_Point: ['c1', 'c2', 'c3']         # 3 values

Total combinations: 6 × 6 × 6 × 3 = 648
```

### 2. Conditional Sampling Phase

For each abstracted combination, sample concrete parameters:

```python
# Example: Given Visibility=60, Road_Surface=80, ...
# Sample from: P(Cloudiness, Wind_Intensity, ... | Visibility=60, Road_Surface=80, ...)

# Uses likelihood-weighted sampling (100,000 samples)
samples = sampler.likelihood_weighted_sample(
    evidence={'Visibility': 60, 'Road_Surface': 80, ...},
    size=100000
)
```

### 3. Rarity Prioritization Phase

```python
# Mode: 'rare' (edge cases) or 'common' (typical cases)

if prefer_rare:
    # Sort by ascending probability (rarest first)
    candidates = sorted_by_probability[:100]  # Bottom 100
else:
    # Sort by descending probability (most common first)
    candidates = sorted_by_probability[:100]  # Top 100
```

### 4. Diversity Selection Phase

Select scenario that maximizes minimum distance to existing scenarios:

```python
# Max-min diversity criterion
best_scenario = arg max_{s in candidates} min_{s' in generated} distance(s, s')

# Ensures scenarios are spread across parameter space
```

## Configuration Parameters

### Scenario Generator

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.1 | Distance threshold for diversity (0-1)<br>Lower → stricter diversity |
| `n_samples` | 100000 | Number of samples for likelihood weighting<br>More → better approximation |
| `use_sampling` | True | Use Monte Carlo sampling (recommended)<br>False → exhaustive search |
| `prefer_rare` | False | Prioritize rare scenarios<br>True → edge cases, False → typical cases |

### Example Configurations

**Edge Case Discovery (Recommended for Testing)**
```python
generator = BayesianScenarioGenerator(
    model=model,
    leaf_nodes=abstracted_vars,
    initial_nodes=concrete_vars,
    similarity_threshold=0.1,  # Moderate diversity
    n_samples=100000,          # Good approximation
    use_sampling=True,         # Fast
    prefer_rare=True           # Focus on edge cases
)
```

## Evaluation Metrics

### 1. Realism

Measures how close generated scenarios are to real-world distribution:

```python
realism = compute_realism(real_data, generated, attributes)
# Returns: Percentage of scenarios within real-world threshold (0-100%)
```

### 2. Coverage

Measures how much of real-world space is covered:

```python
coverage = compute_coverage(real_data, generated, attributes)
# Returns: {
#   'coverage_percentage': 94.2,
#   'num_covered': 942,
#   'total_real_unique': 1000
# }
```

## Output Files

### Generated Scenarios CSV

Contains all generated test scenarios:

```csv
Cloudiness,Wind_Intensity,...,Visibility,Road_Surface,...,probability
60,40,...,60,80,...,0.0234
80,20,...,80,100,...,0.0891
...
```

**Columns:**
- **Concrete variables**: Sampled environmental parameters
- **Abstracted variables**: Target coverage values
- **probability**: Likelihood of scenario (from BN)

### Evaluation Results PKL

Contains comprehensive evaluation metrics:

```python
{
    'distributions': {...},      # Per-attribute distribution tables
    'realism': 87.5,            # Realism percentage
    'coverage': {...},          # Coverage statistics
    'num_unique': 648           # Number of unique scenarios
}
```

## Usage Examples

### Example 1: Generate Both Modes

```bash
# Generate rare (edge case) scenarios
python generate_scenarios.py --scenario 1 --mode rare

# Generate common (typical) scenarios
python generate_scenarios.py --scenario 1 --mode common

# Compare results
python -c "
import pandas as pd
rare = pd.read_csv('generated_scenarios/scenario1_rare_scenarios.csv')
common = pd.read_csv('generated_scenarios/scenario1_common_scenarios.csv')
print(f'Rare mean probability: {rare.probability.mean():.4f}')
print(f'Common mean probability: {common.probability.mean():.4f}')
"
```

### Example 2: Custom Generation

```python
from generation.scenario_generator import BayesianScenarioGenerator
import pickle

# Load model
with open('modeling/models/scenario2_full_bayesian_network.pkl', 'rb') as f:
    model = pickle.load(f)

# Custom configuration
generator = BayesianScenarioGenerator(
    model=model,
    leaf_nodes={
        'Visibility': [0, 20, 40, 60, 80, 100],
        'Road_Surface': [0, 20, 40, 60, 80, 100],
        'Vehicle_Stability': [0, 20, 40, 60, 80, 100],
        'Collision_Point': ['c1', 'c2', 'c3']
    },
    initial_nodes=['Time_of_Day', 'Cloudiness', ...],
    similarity_threshold=0.15,  # More relaxed diversity
    n_samples=50000,            # Faster generation
    prefer_rare=True
)

# Generate
scenarios = generator.generate_scenarios()
print(f"Generated {len(scenarios)} scenarios")
```

## Differences Between Scenarios

| Feature | Scenario 1 (Vehicle-Vehicle) | Scenario 2 (Vehicle-Cyclist) |
|---------|------------------------------|------------------------------|
| **Concrete Variables** | 8 environmental + 4 junction | 9 environmental + 4 junction |
| **Includes Time_of_Day** | No | Yes (lighting conditions) |
| **Total Combinations** | 648 (6×6×6×3) | 648 (6×6×6×3) |
| **Visibility Parents** | Fog_Density, Fog_Distance, Precipitation | + Time_of_Day |
| **Use Case** | Vehicle-vehicle conflicts | Vehicle-cyclist conflicts |

## Performance

**Typical generation time (on standard hardware):**
- Scenario 1 (648 scenarios): ~90-120 minutes
- Scenario 2 (648 scenarios): ~100-130 minutes

**Optimization tips:**
- Reduce `n_samples` for faster generation (trade-off: lower quality)

## Paper References

This implementation corresponds to:

- **Section III-D**: Generation Stage
  - Combinatorial coverage of abstracted variables
  - Conditional sampling from Bayesian Networks

- **Section III-E**: Sampling and Selection
  - Rarity prioritization for edge cases
  - Max-min diversity criterion

- **Algorithm 1**: Rarity-Prioritized Diverse Scenario Generation
  - Complete generation pipeline

- **Section IV-B**: Evaluation Metrics
  - Realism, coverage, and diversity measures

## Contact

For questions about the generation module:
- Open an issue on GitHub