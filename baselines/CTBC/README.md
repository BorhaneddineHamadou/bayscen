# CTBC — Combinatorial Testing Based on Complexity

Importance-weighted pairwise combinatorial testing adapted for autonomous vehicle junction and cut-in scenarios.

**Original paper:**
```bibtex
@article{gao2019test,
  title={A test scenario automatic generation strategy for intelligent driving systems},
  author={Gao, Feng and Duan, Jianli and He, Yingdong and Wang, Zilong},
  journal={Mathematical Problems in Engineering},
  volume={2019}, number={1}, pages={3737486}, year={2019},
  publisher={Wiley Online Library}
}
```

---

## Overview

CTBC assigns importance indices to parameter values based on their contribution to system failures, then generates pairwise test cases that emphasise high-importance combinations. We adapted the original method (designed for Lane Departure Warning) to autonomous vehicle scenarios by replacing AHP expert weights with **data-driven importance indices** derived from preliminary collision-rate simulations.

### Importance index formula

For each value $v$ of parameter $P$:

$$I_{P_v} = \frac{CR_{P_v}}{\sum_{i=1}^{K} CR_i}$$

where $CR_{P_v}$ is the collision rate when $P = v$, and $K$ is the total number of values across all parameters.

---

## Folder Structure

```
CTBC/
├── Bayesian_optimization_of_CTBC.py              # Main scenario generation script
├── process_scenarios.py                           # Post-processing for Scenario 1
├── process_scenarios_2.py                         # Post-processing for Scenario 2
├── process_scenarios_3.py                         # Post-processing for Scenario 3
├── parameters_scenario1.txt                       # Importance indices for S1
├── parameters_scenario2.txt                       # Importance indices for S2
├── parameters_scenario3.txt                       # Importance indices for S3
├── test_scenarios_scenario{1,2,3}.csv             # Raw CTBC output (pre-generated)
├── processed_test_scenarios_scenario{1,2,3}.xlsx  # Ready-to-execute scenario suites
├── Importance Indices Approximation/
│   ├── Scenario 1/   # 3,888 preliminary simulations → importance indices
│   ├── Scenario 2/   # 3,888 preliminary simulations → importance indices
│   └── Scenario 3/   # Preliminary simulations → importance indices
└── README.md
```

---

## Scenarios

| Scenario | Conflict type | Parameters | Generated tests |
|----------|---------------|-----------|----------------|
| S1 Vehicle-Vehicle Junction | PathInteraction c1/c2/c4 + 8 env | 95 |
| S2 Vehicle-Cyclist Junction | PathInteraction c1/c2/c4 + 9 env (+TimeOfDay) | 165 |
| S3 Vehicle-Vehicle Cut-In | Direction left/right + 9 env (+TimeOfDay) | see xlsx |

S3 replaces the junction PathInteraction axis with a binary Direction axis (left/right cut-in), keeping the same 9 environmental parameters as S2.

---

## Usage

### Step 1: Generate test scenarios (optional — pre-generated files included)

Edit `Bayesian_optimization_of_CTBC.py` line 21 to select the parameter file:

```python
# Scenario 1:  lines = open('parameters_scenario1.txt').readlines()
# Scenario 2:  lines = open('parameters_scenario2.txt').readlines()
# Scenario 3:  lines = open('parameters_scenario3.txt').readlines()
```

```bash
python Bayesian_optimization_of_CTBC.py
```

### Step 2: Post-process to executable format

```bash
python process_scenarios.py    # Scenario 1  → processed_test_scenarios_scenario1.xlsx
python process_scenarios_2.py  # Scenario 2  → processed_test_scenarios_scenario2.xlsx
python process_scenarios_3.py  # Scenario 3  → processed_test_scenarios_scenario3.xlsx
```

Post-processing:
1. Extracts numeric values from CTBC factor names (`RoadFriction_0.4` → `0.4`)
2. Expands `PathInteraction` values into concrete start/goal route combinations (S1/S2)
3. Maps `Direction` values for cut-in direction (S3)
4. Reorders columns for consistency with other methods

---

## Adaptation Notes

| Aspect | Original CTBC | This adaptation |
|--------|--------------|-----------------|
| Importance indices | AHP expert judgement | Data-driven collision rates |
| Domain | Lane Departure Warning | AV junction & cut-in scenarios |
| Parameters | Highway lane-keeping | 8–9 env + geometry |
| Scenario types | Single | S1 (junction), S2 (junction + ToD), S3 (cut-in) |

---

## References

- Gao, F., Duan, J., He, Y., & Wang, Z. (2019). "A test scenario automatic generation strategy for intelligent driving systems." *Mathematical Problems in Engineering*, 2019(1), 3737486.
