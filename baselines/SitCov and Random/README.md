# SitCov and Random Sampling Baseline Methods

This folder contains information about the **SitCov (Situation Coverage)** and **Random Sampling** baseline methods used in the BayScen evaluation.

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

## Implementation in BayScen Framework

**Our framework includes complete implementations of both SitCov and Random Sampling methods.** All necessary code, scripts, and pre-generated scenarios are provided in the `experiments/` directory.

### What's Included

1. **Full Implementation**: SitCov framework is integrated into our experimental framework
2. **Pre-generated Scenarios**: Excel files containing 648 scenarios for each method (already generated and ready to use)
3. **Execution Scripts**: `run_sitcov.py` and `run_random.py` for regenerating scenarios if needed
4. **Collection Scripts**: Tools to save generated scenarios as Excel files for repeated execution

### Getting Started

**For complete setup and execution instructions, please refer to the main experiments README:**

**See `experiments/README.md` for detailed instructions**

### Quick Overview

**To run experiments with pre-generated scenarios (recommended):**
```bash
# Navigate to scenario folder
cd experiments/scenario_runner-0.9.10/scenario1_interfuser

# Run SitCov scenarios
python run_simulation.py --test_method sitcov --run_number 4

# Run Random scenarios  
python run_simulation.py --test_method random --run_number 4
```

**To regenerate scenarios (optional) and execute them:**
```bash
# Navigate to scenario folder
cd experiments/scenario_runner-0.9.10/scenario1_interfuser

# Regenerate SitCov scenarios
python run_sitcov.py --run_number 4

# Regenerate Random scenarios
python run_random.py --run_number 4

# Save as Excel files
python outputs/sitcov/collect_scenarios/save_sitcov_scenarios.py
python outputs/random/collect_scenarios/save_random_scenarios.py
```

## Why Use the Original SitCov Approach?

We adapted the original SitCov repository implementation because:

1. **Scenario Runner Compatibility**: Uses the same CARLA Scenario Runner version required for consistent execution across all baseline methods
2. **Consistent Implementation**: Ensures fair comparison by using the same infrastructure for scenario execution and data collection
3. **Validated Framework**: The original implementation has been validated and tested by the authors

## References

- **SitCov Paper**: Tahir, Z., & Alexander, R. (2021). "Intersection focused situation coverage-based verification and validation framework for autonomous vehicles implemented in CARLA." In *International Conference on Modelling and Simulation for Autonomous Systems* (pp. 191-212). Springer.
- **SitCov Repository**: [https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA](https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA)

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
```

## Support

For setup, execution, and configuration questions, please refer to:
- **Main experiments README**: `experiments/README.md` (complete setup and execution guide)
- **Original SitCov repository**: For implementation details and methodology questions
- **BayScen repository**: For issues related to our adapted implementation or evaluation setup