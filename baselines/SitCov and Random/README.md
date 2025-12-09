# SitCov and Random Sampling Baseline Methods

This folder contains instructions for executing the **SitCov (Situation Coverage)** and **Random Sampling** baseline methods using the original implementation from the SitCov repository.

## About the Methods

### SitCov (Situation Coverage)

SitCov is a coverage-based weighted sampling method that prioritizes under-tested variable combinations to achieve near-uniform distribution across the parameter space. The method iteratively selects scenarios that maximize coverage of unexplored or under-represented parameter value combinations.

**Original Paper:**
```bibtex
@inproceedings{tahir2021intersection,
  title={Intersection focused situation coverage-based verification and validation framework for autonomous vehicles implemented in CARLA},
  author={Tahir, Zaid and Alexander, Rob},
  booktitle={International Conference on Modelling and Simulation for Autonomous Systems},
  pages={191--212},
  year={2021},
  organization={Springer}
}
```

**Official Repository:** [https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA](https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA)

### Random Sampling

Random Sampling serves as a naive baseline that uniformly samples across all valid variable combinations without any coverage optimization or importance weighting. This baseline helps assess whether more sophisticated methods provide meaningful improvements over simple random exploration.

## Why Use the Original Repository?

We use the original SitCov repository for both SitCov and Random Sampling because:

1. **Scenario Runner Compatibility**: The repository uses the same CARLA Scenario Runner version required for consistent execution across all baseline methods
2. **Consistent Implementation**: Ensures fair comparison by using the same infrastructure for scenario execution and data collection
3. **Validated Framework**: The original implementation has been validated and tested by the authors

## Setup and Execution

### Prerequisites

- CARLA 0.9.10
- Python 3.7+
- CARLA Scenario Runner (version specified in SitCov repository)

### Step 1: Clone the SitCov Repository

```bash
git clone https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA.git
cd Situation-Coverage-based-AV-Testing-Framework-in-CARLA
```

### Step 2: Install Dependencies

Follow the installation instructions in the original repository's README:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Scenarios

Adapt the scenario configurations to match our evaluation setup:

1. **Scenario 1 (Vehicle-Vehicle)**: Configure parameters for non-signalized junction vehicle-vehicle interactions
2. **Scenario 2 (Vehicle-Cyclist)**: Configure parameters for vehicle-cyclist interactions with Time-of-Day parameter

Ensure parameter definitions match those in:
- `../PICT/parameters_scenario1.txt`
- `../PICT/parameters_scenario2.txt`

### Step 4: Execute Methods

Follow the execution instructions from the original repository to run:

**For SitCov:**
```bash
# Follow SitCov-specific execution steps from the repository
# Generate N=648 scenarios for fair comparison with BayScen
```

**For Random Sampling:**
```bash
# Follow Random sampling execution steps from the repository
# Generate N=648 scenarios for fair comparison with BayScen
```

### Step 5: Collect Results

The execution will generate results including:
- Scenario parameter configurations
- Collision events
- Time-to-Collision (TTC) measurements
- Coverage metrics

## Scenario Budget

For fair comparison across all methods in the BayScen evaluation:

| Method | Scenario Count | Rationale |
|--------|---------------|-----------|
| SitCov | 648 | Match BayScen's combinatorial suite size |
| Random Sampling | 648 | Match BayScen's combinatorial suite size |

Both methods can generate arbitrary numbers of scenarios; we set N=648 to ensure fair comparison with BayScen's full combinatorial coverage of the abstracted parameter space (3 × 6 × 6 × 6 = 648).

## Parameter Space

Both methods operate on the same 12-13 dimensional concrete parameter space:

**Scenario 1 (Vehicle-Vehicle) - 12 parameters:**
- 4 intersection variables: StartEgo, GoalEgo, StartOther, GoalOther (3 values each)
- 8 environmental variables: RoadFriction (6 bins), FogDensity (6 bins), FogDistance (6 bins), Precipitation (6 bins), PrecipitationDeposits (6 bins), Cloudiness (6 bins), WindIntensity (6 bins), Wetness (6 bins)

**Scenario 2 (Vehicle-Cyclist) - 13 parameters:**
- Same as Scenario 1 + TimeOfDay (7 bins for sun altitude angle)

## Key Differences from Other Baselines

| Method | Strategy | Coverage Guarantee | Realism Modeling |
|--------|----------|-------------------|------------------|
| **Random** | Uniform sampling | Probabilistic only | None |
| **SitCov** | Weighted sampling for uniform coverage | Near-uniform distribution | None |
| **PICT** | Combinatorial t-way coverage | Complete t-way coverage | None |
| **CTBC** | Importance-weighted pairwise | Pairwise coverage | Implicit (via weights) |
| **BayScen** | BN-guided + abstraction + combinatorial | Complete abstracted coverage | Explicit (BN dependencies) |

## Expected Outputs

For the BayScen paper experiments:

| Scenario | Method | Generated Tests | Mean TTC < 0.5s | Collisions (≥2/3) | Collision Realism |
|----------|--------|----------------|-----------------|-------------------|-------------------|
| **Scenario 1 (V-V)** | Random | 648 | 12 (1.9%) | 44 (6.8%) | 34.1% |
| **Scenario 1 (V-V)** | SitCov | 648 | 13 (2.0%) | 34 (5.2%) | 44.1% |
| **Scenario 2 (V-C)** | Random | 648 | 3 (0.5%) | 10 (1.5%) | 30.0% |
| **Scenario 2 (V-C)** | SitCov | 648 | 18 (2.8%) | 19 (2.9%) | 42.1% |

## Notes on Implementation

### SitCov Execution

SitCov dynamically adjusts sampling weights based on coverage statistics:
- Initially samples uniformly across all parameter combinations
- Tracks frequency of each parameter value occurrence
- Increases weights for under-represented combinations
- Converges toward near-uniform coverage distribution

### Random Sampling Execution

Random Sampling provides a pure baseline:
- No coverage optimization
- No importance weighting
- Uniform probability for all valid combinations
- Serves as lower bound for effectiveness comparison

## Reproducibility

To ensure reproducibility:

1. **Set random seeds**: Use fixed random seeds for both methods
2. **Parameter consistency**: Verify parameter definitions match across all baselines
3. **Execution environment**: Use the same CARLA version (0.9.10) and Scenario Runner
4. **Repetitions**: Execute each scenario 3 times to account for stochasticity
5. **System under test**: Use the same autonomous driving system (e.g., InterFuser) across all methods

## References

- **SitCov Paper**: Tahir, Z., & Alexander, R. (2021). "Intersection focused situation coverage-based verification and validation framework for autonomous vehicles implemented in CARLA." In *International Conference on Modelling and Simulation for Autonomous Systems* (pp. 191-212). Springer.
- **SitCov Repository**: [https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA](https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA)
- **CARLA Simulator**: Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017). "CARLA: An open urban driving simulator." In *Conference on robot learning* (pp. 1-16). PMLR.

## Citation

If you use SitCov or Random Sampling from this setup in your research, please cite:

```bibtex
@inproceedings{tahir2021intersection,
  title={Intersection focused situation coverage-based verification and validation framework for autonomous vehicles implemented in CARLA},
  author={Tahir, Zaid and Alexander, Rob},
  booktitle={International Conference on Modelling and Simulation for Autonomous Systems},
  pages={191--212},
  year={2021},
  organization={Springer}
}

@inproceedings{dosovitskiy2017carla,
  title={CARLA: An open urban driving simulator},
  author={Dosovitskiy, Alexey and Ros, German and Codevilla, Felipe and Lopez, Antonio and Koltun, Vladlen},
  booktitle={Conference on robot learning},
  pages={1--16},
  year={2017},
  organization={PMLR}
}
```

## Troubleshooting

**Issue:** Scenario Runner version mismatch
- **Solution:** Use the exact Scenario Runner version specified in the SitCov repository

**Issue:** Parameter definitions don't match BayScen evaluation
- **Solution:** Cross-reference with `parameters_scenario1.txt` and `parameters_scenario2.txt` in the PICT folder

**Issue:** Different collision counts than reported in paper
- **Solution:** Verify you're using the same autonomous driving system (InterFuser) and random seed

## Support

For issues specific to the SitCov implementation, please refer to the original repository or contact the authors.

For issues related to BayScen evaluation setup or parameter configurations, please refer to the main BayScen repository.