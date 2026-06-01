# BayScen — Modeling Module

Bayesian Network structure learning and parameterization for BayScen.

## Overview

The BN serves two roles in BayScen (Paper Section II-B):

1. **Realism**: Generated scenarios inherit the statistical regularities of
   43,000+ real-world meteorological observations — they are realistic by
   construction.

2. **Conditional Sampling**: Conditioning on a target capability configuration t
   induces a well-defined posterior P(X | t) from which concrete CARLA parameters
   are drawn via ancestral sampling.

## Directory Structure

```
modeling/
├── README.md
├── bn_parametrization.py              # Full parameterization pipeline
├── bn_parametrization_example.ipynb  # Training walkthrough (interactive)
├── bn_utils.py                        # Model I/O utilities
├── models/                            # Trained BN models (output)
│   ├── scenario1_full_bayesian_network.pkl
│   ├── scenario2_full_bayesian_network.pkl
│   └── scenario3_full_bayesian_network.pkl
└── structure_learning/
    ├── bicamml_priors.txt             # Bi-CaMML prior specification
    ├── domain_knowledge_priors.md     # Documentation: causal relationships & priors
    └── learned_structures/
        ├── scenario1_structure.txt    # Learned DAG for S1 (8 variables)
        ├── scenario2_structure.txt    # Learned DAG for S2 (9 variables)
        └── scenario3_structure.txt    # Learned DAG for S3 (9 variables)
```

## Methodology

### Step 1: Structure Learning with Bi-CaMML

We use **Bi-CaMML** (Bayesian Causal MML) with the error-tolerant
knowledge-guided framework of Ban et al. (2025):

- **Data-driven learning**: learns from 43,000+ observations (Frost API)
- **Domain knowledge priors**: encodes physical causal relationships as soft
  ancestral constraints (encouraged but overridable by data)
- **Error-tolerant**: data overrides a prior if observational evidence is strong

See `structure_learning/domain_knowledge_priors.md` for the complete prior
specification.

### Step 2: Bayesian Parameter Estimation

With the structure fixed, CPDs are estimated using:
- **BDeu prior** (Heckerman et al. 1995)
- **Equivalent sample size = 5** (Bayesian smoothing, Silander et al. 2012)

### Step 3: Capability Variable Extension

Capability leaf nodes are added per ISO 34503:2023 (Paper Section II-C-4):
- CPD computed by **uniform-weight aggregation** over contributing parents
- Probability mass concentrated on the K=6 discrete level nearest the average

## Environmental Variables

### Scenario 1 (Vehicle–Vehicle) — 8 variables

| Variable | CARLA values | ISO clause |
|----------|-------------|------------|
| Cloudiness | [0, 20, 40, 60, 80, 100] | §10.4(c) |
| Wind_Intensity | [0, 20, 40, 60, 80, 100] | §10.2.3 |
| Precipitation | [0, 20, 40, 60, 80, 100] | §10.2.4 |
| Precipitation_Deposits | [0, 20, 40, 60, 80, 100] | §9.3.7 |
| Wetness | [0, 20, 40, 60, 80, 100] | §9.3.7 |
| Fog_Density | [0, 20, 40, 60, 80, 100] | §10.3 |
| Fog_Distance | [0, 20, 40, 60, 80, 100] | §10.3 |
| Road_Friction | [0.1, 0.2, 0.4, 0.8, 1.0] | §9.3.7 |

### Scenarios 2 & 3 — 9 variables (adds Sun_Altitude_Angle)

| Variable | CARLA values | ISO clause |
|----------|-------------|------------|
| Sun_Altitude_Angle | [-90, -60, -30, 0, 30, 60, 90]° | §10.4(d) |
| + all 8 variables from S1 | | |

**Note**: `Sun_Altitude_Angle` was named `Time_of_Day` in earlier versions of
the code. The data collection script (`data/collect.py`) may output a column
named `Time_of_Day`; `bn_parametrization.py` automatically renames it.

## Learned BN Structures

### Scenario 1 (8 variables, 9 edges)

```
Cloudiness            → Precipitation
Wind_Intensity        → Fog_Density
Precipitation         → Precipitation_Deposits
Precipitation         → Fog_Density
Precipitation_Deposits→ Wetness
Precipitation_Deposits→ Road_Friction
Wetness               → Road_Friction
Fog_Density           → Fog_Distance
Fog_Density           → Wetness
```

### Scenarios 2 & 3 (9 variables, 11 edges)

Extends S1 with:
```
Sun_Altitude_Angle    → Cloudiness
Sun_Altitude_Angle    → Wind_Intensity
```

## Usage

### Command Line

```bash
# Scenario 1 (Vehicle–Vehicle junction)
python bn_parametrization.py --scenario 1

# Scenario 2 (Vehicle–Cyclist junction; adds Sun_Altitude_Angle)
python bn_parametrization.py --scenario 2

# Scenario 3 (Vehicle–Vehicle cut-in; same structure as S2)
python bn_parametrization.py --scenario 3
```

### Python API

```python
import pickle

with open('models/scenario1_full_bayesian_network.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Nodes : {sorted(model.nodes())}")
print(f"Edges : {sorted(model.edges())}")
print(f"Valid : {model.check_model()}")
```

## Output Files

| File | Description |
|------|-------------|
| `scenario{N}_fitted_bayesian_network.pkl` | Base environmental BN |
| `scenario{N}_extended_bayesian_network.pkl` | + capability leaf nodes |
| `scenario{N}_full_bayesian_network.pkl` | + position/trajectory variables |

The `full` model is the input to the generation pipeline.

## References

1. Ban et al. (2025). "Integrating large language model for improved causal discovery."
   *IEEE Transactions on Artificial Intelligence*.
2. Wallace, Korb & Dai (1996). "Causal discovery via MML." *ICML*.
3. Heckerman, Geiger & Chickering (1995). "Learning Bayesian networks."
   *Machine Learning*, 20(3):197–243.
4. Silander, Kontkanen & Myllymäki (2012). "On sensitivity of the MAP BN
   structure to the equivalent sample size." *arXiv:1206.5293*.
