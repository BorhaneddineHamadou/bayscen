# BayScen Bayesian Network Modeling

This directory contains the Bayesian Network structure learning and parameterization components of BayScen.

## Overview

BayScen uses Bayesian Networks (BNs) to model probabilistic dependencies between environmental parameters. The BN serves two critical purposes:

1. **Realism**: Ensures generated scenarios respect naturalistic dependencies observed in real-world weather data
2. **Conditional Sampling**: Enables sampling scenarios conditioned on target abstracted variable values

## Directory Structure

```
bayscen/modeling/
├── README.md                          # This file
├── structure_learning/
│   ├── llm_prompts.md                 # LLM prompts for structure elicitation
│   ├── bicamml_priors.txt             # Expert priors for Bi-CaMML
│   └── learned_structures/
│       ├── scenario1_structure.txt    # Learned BN for Scenario 1 (Vehicle-Vehicle)
│       └── scenario2_structure.txt    # Learned BN for Scenario 2 (Vehicle-Cyclist)
├── bn_parametrization_scenario1.py    # Parameterization for Scenario 1
├── bn_parametrization_scenario2.py    # Parameterization for Scenario 2
└── models/                            # Saved trained BN models
    ├── scenario1_full_bayesian_network.pkl
    └── scenario2_full_bayesian_network.pkl
```

## Methodology

### Step 1: Structure Learning with Bi-CaMML

We use **Bi-CaMML** (Bayesian Causal MML) for structure learning, which combines:

1. **Data-driven learning**: Learns from 43,000+ observations from Frost API
2. **LLM-guided priors**: Incorporates domain knowledge from GPT-4 as soft constraints
3. **Error-tolerant approach**: Robust to LLM inaccuracies

**Reference**: Ban et al. (2025). "Integrating large language model for improved causal discovery." *IEEE Transactions on Artificial Intelligence*.

**Installation**: 
```bash
# Bi-CaMML is available from the official repository
git clone https://github.com/CausalAILab/Bi-CaMML
cd Bi-CaMML
# Follow installation instructions in repository
```

#### Using Bi-CaMML for BayScen

1. **Prepare data**: Use `data/processed/bayscen_final_data.csv` (preprocessed Frost API data)

2. **Load priors**: Copy contents from `structure_learning/bicamml_priors.txt` into Bi-CaMML's "Expert Priors" interface

3. **Run structure learning**: Bi-CaMML will output the learned DAG structure

4. **Save structure**: Save the edge list to `structure_learning/learned_structures/`

### Step 2: Bayesian Parameter Estimation

Once the structure is learned, we estimate Conditional Probability Distributions (CPDs) using:

- **Bayesian Estimator** with BDeu prior
- **Equivalent sample size** = 5 (for smoothing)
- **pgmpy library** for implementation

## Variables

### Scenario 1 (Vehicle-Vehicle) - 8 Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `Cloudiness` | [0, 20, 40, 60, 80, 100] | Cloud coverage percentage |
| `Wind_Intensity` | [0, 20, 40, 60, 80, 100] | Wind speed intensity |
| `Precipitation` | [0, 20, 40, 60, 80, 100] | Precipitation intensity |
| `Precipitation_Deposits` | [0, 20, 40, 60, 80, 100] | Accumulated precipitation |
| `Wetness` | [0, 20, 40, 60, 80, 100] | Road surface wetness |
| `Fog_Density` | [0, 20, 40, 60, 80, 100] | Fog thickness |
| `Fog_Distance` | [0, 20, 40, 60, 80, 100] | Visibility distance |
| `Road_Friction` | [0.1, 0.2, 0.4, 0.8, 1.0] | Road-tire friction coefficient |

### Scenario 2 (Vehicle-Cyclist) - 9 Variables

All variables from Scenario 1, plus:

| Variable | Values | Description |
|----------|--------|-------------|
| `Time_of_Day` | [-90, -60, -30, 0, 30, 60, 90] | Sun altitude angle (degrees) |

**Note**: Negative values indicate night (sun below horizon)

## Learned BN Structure

### Scenario 1 Structure

```
Cloudiness → Precipitation
Wind_Intensity → Fog_Density
Precipitation → Precipitation_Deposits
Precipitation → Fog_Density
Precipitation_Deposits → Wetness
Precipitation_Deposits → Road_Friction
Wetness → Road_Friction
Fog_Density → Fog_Distance
Fog_Density → Wetness
```

**Key Properties**:
- No cycles (valid DAG)
- Captures physical causality (e.g., precipitation causes wetness)
- 9 edges connecting 8 variables

### Scenario 2 Structure

Same as Scenario 1, plus:

```
Time_of_Day → Cloudiness
Time_of_Day → Wind_Intensity
```

**Key Properties**:
- Extends Scenario 1 with temporal effects
- Time of day influences cloud formation and wind patterns
- 11 edges connecting 9 variables

## Usage

### Training a Bayesian Network

```python
from bayscen.modeling.bn_parametrization_scenario1 import train_bn_scenario1
from bayscen.modeling.bn_parametrization_scenario2 import train_bn_scenario2

# Train for Scenario 1 (Vehicle-Vehicle)
model_s1 = train_bn_scenario1(
    data_path='data/processed/bayscen_final_data.csv',
    save_path='bayscen/modeling/models/scenario1_full_bayesian_network.pkl'
)

# Train for Scenario 2 (Vehicle-Cyclist)
model_s2 = train_bn_scenario2(
    data_path='data/processed/bayscen_final_data.csv',
    save_path='bayscen/modeling/models/scenario2_full_bayesian_network.pkl'
)
```

### Loading a Trained Model

```python
from bayscen.modeling.bn_utils import load_bn_model

model = load_bn_model('bayscen/modeling/models/scenario1_full_bayesian_network.pkl')

# Verify model
print(f"Nodes: {model.nodes()}")
print(f"Edges: {model.edges()}")
print(f"Is valid: {model.check_model()}")
```

## LLM-Guided Structure Learning

The structure learning process uses GPT-4 to derive causal relationships. Key prompts include:

1. **Variable Definition**: Understanding what each variable represents
2. **Causal Extraction**: Identifying direct causal relationships
3. **Causal Validation**: Verifying proposed edges
4. **Soft Constraints**: Converting relationships to ancestral constraints

See `structure_learning/llm_prompts.md` for complete prompts.

## Expert Priors for Bi-CaMML

The priors encode domain knowledge as soft constraints:

**Tier Constraints** (temporal ordering):
```
Time_of_Day < Cloudiness Wind_Intensity Precipitation < 
Fog_Density Fog_Distance < Wetness Precipitation_Deposits < 
Road_Friction
```

**Arc Constraints**:
```
Precipitation => Wetness (0.99999)
Precipitation => Precipitation_Deposits (0.99999)
Wetness => Road_Friction (0.99999)
Precipitation_Deposits => Road_Friction (0.99999)
Fog_Density => Fog_Distance (0.99999)
...
```

These constraints guide but don't force the structure learning—data can override them if evidence is strong.

## Model Validation

Both models are validated through:

1. **DAG Check**: Ensures no cycles
2. **CPD Consistency**: All probabilities sum to 1

## Integration with BayScen Pipeline

The learned BN integrates with BayScen's generation pipeline:

```
1. Load trained BN model
2. For each abstract configuration from combinatorial testing:
   a. Map abstract values to concrete variable conditions
   b. Sample N=100,000 scenarios from conditional distribution
   c. Select rarest configurations (prioritize edge cases)
   d. Apply diversity-aware selection (max-min distance)
3. Output final scenario set
```

## Files Generated

After training, the following files are created:

- `models/scenario1_full_bayesian_network.pkl`: Trained BN for Scenario 1 (~2 MB)
- `models/scenario2_full_bayesian_network.pkl`: Trained BN for Scenario 2 (~2.5 MB)

These can be loaded directly without retraining.

## References

1. Ban et al. (2025). "Integrating large language model for improved causal discovery." *IEEE Transactions on Artificial Intelligence*.
2. Wallace et al. (1996). "Causal discovery via MML." *ICML*.
3. Heckerman et al. (1995). "Learning Bayesian networks: The combination of knowledge and statistical data." *Machine Learning*, 20(3):197-243.

## Contact

For questions about the Bayesian Network modeling, please open a GitHub issue.