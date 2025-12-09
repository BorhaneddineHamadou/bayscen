# CTBC Baseline Method

This folder contains the implementation and execution instructions for generating test scenarios using **CTBC (Combinatorial Testing Based on Complexity)**.

## About CTBC

CTBC is a pairwise combinatorial testing method that uses importance-weighted values to prioritize high-complexity scenarios. The method assigns importance indices to parameter values based on their contribution to system failures, then generates test cases that emphasize combinations of high-importance values.

**Original Paper:**
```bibtex
@article{gao2019test,
  title={A test scenario automatic generation strategy for intelligent driving systems},
  author={Gao, Feng and Duan, Jianli and He, Yingdong and Wang, Zilong},
  journal={Mathematical Problems in Engineering},
  volume={2019},
  number={1},
  pages={3737486},
  year={2019},
  publisher={Wiley Online Library}
}
```

## Overview

Our CTBC implementation consists of two main phases:

1. **Importance Index Calculation**: Estimate importance indices for each parameter value based on data-driven collision rates
2. **Test Scenario Generation**: Generate pairwise combinatorial test cases weighted by importance indices using Bayesian optimization

## Folder Structure

```
CTBC/
├── Bayesian_optimization_of_CTBC.py          # Main scenario generation script (from the source)
├── process_scenarios.py                       # Post-processing for Scenario 1
├── process_scenarios_2.py                     # Post-processing for Scenario 2
├── parameters_scenario1.txt                   # Parameter definitions for Scenario 1
├── parameters_scenario2.txt                   # Parameter definitions for Scenario 2
├── Importance_Indices_Approximation/
│   ├── Scenario_1/                           # Importance index estimation for Scenario 1
│   └── Scenario_2/                           # Importance index estimation for Scenario 2
└── README.md                                 # This file
```

## Adaptation for BayScen Evaluation

### Importance Index Estimation

The original CTBC paper derived importance indices for Lane Departure Warning systems using the Analytic Hierarchy Process (AHP). For our autonomous vehicle junction scenarios, we adapted the method by calculating **data-driven importance indices** from preliminary simulations.

**Methodology:**

For each value $v$ of variable $P$, we computed the collision rate $CR_{P_v}$ across all simulation runs where $P = v$, then normalized these rates to obtain importance indices:

$$I_{P_v} = \frac{CR_{P_v}}{\sum_{i=1}^{K} CR_i}$$

where $K$ is the total number of variable values across all variables.

This data-driven approach ensures that importance indices reflect the actual failure-inducing characteristics of parameter values in our specific testing scenarios, rather than relying on domain expert judgments.

**Implementation:**
- **Scenario 1 (Vehicle-Vehicle)**: 3,888 preliminary simulations → importance indices in `Importance_Indices_Approximation/Scenario_1/`
- **Scenario 2 (Vehicle-Cyclist)**: 3,888 preliminary simulations → importance indices in `Importance_Indices_Approximation/Scenario_2/`

## Usage

### Step 1: Estimate Importance Indices (Optional)

If you want to recalculate importance indices from scratch:

```bash
cd Importance_Indices_Approximation/Scenario_1/
# Run the estimation scripts for Scenario 1

cd ../Scenario_2/
# Run the estimation scripts for Scenario 2
```

**Note:** Pre-computed importance indices are already included in the repository.

### Step 2: Generate Test Scenarios

Run the Bayesian optimization script to generate test scenarios:

```bash
python Bayesian_optimization_of_CTBC.py
```

**Important:** Before running, modify line 21 to select the correct parameter file:

```python
# For Scenario 1 (Vehicle-Vehicle):
lines = open('parameters_scenario1.txt').readlines()

# For Scenario 2 (Vehicle-Cyclist):
lines = open('parameters_scenario2.txt').readlines()
```

**Output:** This generates a CSV file containing test scenarios (e.g., `test_scenarios_scenario1.csv` or `test_scenarios_scenario2.csv`).

### Step 3: Post-Process Scenarios

Convert the generated CSV to Excel format and expand collision point constraints:

```bash
# For Scenario 1:
python process_scenarios.py

# For Scenario 2:
python process_scenarios_2.py
```

**What this does:**
1. Reads the CSV file generated in Step 2
2. Extracts numeric values from factor names (e.g., `RoadFriction_0.4` → `0.4`)
3. Expands `PathInteraction` values into concrete `StartEgo`, `GoalEgo`, `StartOther`, `GoalOther` combinations
4. Randomly assigns valid geometric configurations for each collision point
5. Reorders columns for clarity
6. Saves to Excel format (e.g., `processed_test_scenarios_scenario1.xlsx`)

**Manual Adjustment Required:** After post-processing, manually verify and adjust the collision point assignments if needed:
- Ensure `PathInteraction` values (c1, c2, c4) map correctly to geometric configurations
- Verify `Base`, `Left`, `Right` values are consistent with your junction setup

## Post-Processing Details

The post-processing script performs the following transformations:

### 1. Collision Point Expansion

Each `PathInteraction` value is mapped to multiple valid geometric configurations:

```python
path_map = {
    "PathInteraction_c1": [
        {"StartEgo": "left", "GoalEgo": "right", "StartOther": "base", "GoalOther": "left"},
        {"StartEgo": "left", "GoalEgo": "right", "StartOther": "base", "GoalOther": "right"},
        # ... (4 total configurations for c1)
    ],
    "PathInteraction_c2": [
        # ... (4 configurations for c2)
    ],
    "PathInteraction_c4": [
        # ... (4 configurations for c4)
    ]
}
```

The script **shuffles** each set once for diversity, then **cycles** through configurations to ensure varied assignment.

### 2. Value Extraction

Extracts numeric values from CTBC's factor naming convention:
- `RoadFriction_0.4` → `0.4`
- `FogDensity_60` → `60`
- `PathInteraction_c1` → `c1`

### 3. Column Reordering

Reorders columns for consistency with other baseline methods:

```python
[
    "TimeOfDay", "PathInteraction", 
    "StartEgo", "GoalEgo", "StartOther", "GoalOther",
    "RoadFriction", "FogDensity", "Precipitation", "PrecipitationDeposits",
    "Cloudiness", "WindIntensity", "Wetness", "FogDistance"
]
```

## Expected Outputs

For the BayScen paper experiments:

| Scenario | Generated Tests | Importance Source |
|----------|-----------------|-------------------|
| Scenario 1 (V-V) | 95 | 3,888 simulations |
| Scenario 2 (V-C) | 165 | 3,888 simulations |

## Key Differences from Original CTBC

1. **Importance Indices**: Data-driven (collision rates) instead of AHP expert judgments
2. **Domain**: Autonomous vehicle junction scenarios instead of Lane Departure Warning
3. **Parameters**: 12-13 environmental/geometric parameters instead of highway lane-keeping parameters
4. **Implementation**: Python with Bayesian optimization instead of original implementation

## References

- **Original Paper**: Gao, F., Duan, J., He, Y., & Wang, Z. (2019). "A test scenario automatic generation strategy for intelligent driving systems." *Mathematical Problems in Engineering*, 2019(1), 3737486.

## Citation

If you use this CTBC implementation in your research, please cite both the original paper and the BayScen paper:

```bibtex
@article{gao2019test,
  title={A test scenario automatic generation strategy for intelligent driving systems},
  author={Gao, Feng and Duan, Jianli and He, Yingdong and Wang, Zilong},
  journal={Mathematical Problems in Engineering},
  volume={2019},
  number={1},
  pages={3737486},
  year={2019},
  publisher={Wiley Online Library}
}
```

## Troubleshooting

**Issue:** Generated scenarios have missing or incorrect collision point configurations
- **Solution:** Verify the `path_map` dictionary in `process_scenarios.py` matches your junction geometry

**Issue:** Importance indices seem unrealistic
- **Solution:** Check the preliminary simulation results in `Importance_Indices_Approximation/` folders

**Issue:** Bayesian optimization fails to converge
- **Solution:** Adjust optimization parameters in `Bayesian_optimization_of_CTBC.py` (e.g., number of iterations, acquisition function)