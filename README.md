# BayScen: Realistic and Comprehensive Scenario Generation for Autonomous Vehicle Testing via Bayesian Networks

[![Paper](https://img.shields.io/badge/Paper-ICST%202026-blue)](https://anonymous.4open.science/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.10-green)](https://carla.readthedocs.io/en/0.9.10/)
[![Python](https://img.shields.io/badge/Python-3.7+-orange)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Anonymous submission to ICST 2026**

---

## Overview

BayScen is a novel scenario generation framework for autonomous vehicle (AV) testing that addresses the fundamental trade-off between **coverage**, **criticality**, and **realism**. By combining Bayesian Networks with effect-based parameter abstraction and rarity-prioritized sampling, BayScen generates comprehensive test suites that are both safety-critical and realistic.

### Key Innovations

**Effect-Based Abstraction**: Reduces parameter space while preserving physical relationships  
**Bayesian Network Integration**: Models real-world weather dependencies using data from Norwegian Meteorological Institute  
**Rarity-Prioritized Generation**: Focuses on edge cases while maintaining realism through probabilistic filtering  
**Comprehensive Evaluation**: Validated on NHTSA pre-crash scenarios using CARLA Simulator and InterFuser agent

### Performance Highlights

| Metric | BayScen | Best Baseline | Improvement |
|--------|---------|---------------|-------------|
| Critical Scenarios (TTC < 0.5s) | **21-28%** | 2-3% | **7-14×** |
| Collision Detection (≥2/3 runs) | **19-29%** | 5-7% | **3-4×** |
| Realism (Critical Events) | **71-76%** | 34-44% | **1.7-2.2×** |
| 3-Way Coverage F1 Score | **0.99-1.00** | 0.47-0.84 | **Complete** |

---

## 🚀 Quick Start

### Prerequisites

- **OS**: Windows 10/11 (for CARLA simulation)
- **GPU**: CUDA-capable GPU
- **Python**: 3.7+
- **Anaconda**: Latest version

### Installation (5 minutes)
```bash
# Clone repository
git clone [REPOSITORY_URL]
cd BayScen

# Create environment
conda env create -f environment.yml
conda activate bayscen
```

### Generate Scenarios (2 minutes)
```bash
# Generate scenarios for Scenario 1 (Vehicle-Vehicle)
cd bayscen/generation
python generate_scenarios.py --scenario 1 --mode rare

# Output: generated_scenarios/scenario1_rare_scenarios.csv
```

### Run Complete Evaluation (Optional, 6-8 hours)
```bash
# See experiments/README.md for full CARLA setup
cd experiments
python evaluate.py --scenario 1
```

---

## 📁 Repository Structure
```
BayScen/
├── 📂 bayscen/                    # Core framework modules
│   ├── abstraction/               # Effect-based parameter abstractions
│   ├── modeling/                  # Bayesian Network training
│   └── generation/                # Scenario generation & selection
│
├── 📂 data/                       # Real-world data collection
│   ├── collect.py                 # Frost API data fetching
│   ├── process.py                 # Weather data processing
│   └── processed/                 # Final datasets for BN training
│
├── 📂 baselines/                  # Baseline method implementations
│   ├── PICT/                      # Pairwise & 3-way combinatorial testing
│   ├── CTBC/                      # Combinatorial testing with base choice
│   └── SitCov_and_Random/         # Coverage-based & random sampling
│
├── 📂 experiments/                # CARLA simulation experiments
│   ├── scenario_runner-0.9.10/    # Modified ScenarioRunner
│   │   ├── scenario1_interfuser/  # Vehicle-Vehicle scenarios
│   │   └── scenario2_interfuser/  # Vehicle-Cyclist scenarios
│   ├── InterFuser/                # End-to-end driving agent
│   └── environment.yml            # Simulation environment spec
│
├── 📂 evaluation/                 # Metrics & paper results
│   ├── metrics.py                 # Realism, coverage, criticality
│   ├── evaluate.py                # Command-line evaluation
│   └── evaluation_tutorial.ipynb # Interactive analysis
│
├── 📄 README.md                   # This file
```

---

### 4. Evaluation Framework

**Metrics Computed:**
- ✅ **Realism**: Distance-based similarity to real-world distributions
- ✅ **3-Way Coverage**: Precision, Recall, F1 for combinatorial interactions
- ✅ **Criticality**: TTC < 0.5s, Collision (≥2/3 runs), Collision (3/3 runs)
- ✅ **Realism-Filtered Criticality**: Critical events that are also realistic

---

## Reproducibility

All experiments and results from the paper can be fully reproduced.

### Step 1: Data Collection (Optional - Pre-collected data provided)
```bash
cd data
python cli.py full --config config.yaml
# Output: processed/bayscen_final_data.csv
```

### Step 2: Train Bayesian Networks
```bash
cd bayscen/modeling
python bn_parametrization.py --scenario 1
python bn_parametrization.py --scenario 2
# Output: models/scenario{1,2}_full_bayesian_network.pkl
```

### Step 3: Generate Scenarios
```bash
cd bayscen/generation
python generate_scenarios.py --scenario 1 --mode rare
python generate_scenarios.py --scenario 2 --mode rare
# Output: generated_scenarios/scenario{1,2}_rare_scenarios.csv
```

### Step 4: Run Simulations (Requires CARLA Setup)
```bash
cd experiments
# Follow experiments/README.md for complete CARLA installation
python scenario_runner-0.9.10/scenario1_interfuser/run_simulation.py \
    --test_method bayscen --run_number 4
```

### Step 5: Evaluate Results
```bash
cd evaluation
python evaluate.py --scenario 1
python evaluate.py --scenario 2
# Output: results/paper_scenario{1,2}_{II,III,IV}.csv
```

---

## Module Documentation

Each module contains detailed documentation:

| Module | Purpose | Documentation |
|--------|---------|---------------|
| `bayscen/` | Core framework | [README](bayscen/README.md) |
| `data/` | Real-world data collection | [README](data/README.md) |
| `baselines/` | Baseline implementations | [PICT](baselines/PICT/README.md), [CTBC](baselines/CTBC/README.md), [SitCov](baselines/SitCov_and_Random/README.md) |
| `experiments/` | CARLA simulation setup | [README](experiments/README.md) |
| `evaluation/` | Metrics & analysis | [README](evaluation/README.md) |

**Tutorials Available:**
- 📓 `data/data_collection_tutorial.ipynb` - Data collection walkthrough
- 📓 `bayscen/modeling/bn_parametrization_example.ipynb` - BN training guide
- 📓 `bayscen/generation/scenario_generation_tutorial.ipynb` - Generation guide
- 📓 `evaluation/evaluation_tutorial.ipynb` - Evaluation & results analysis

---

## Baseline Methods

All baseline implementations are included for complete reproducibility:

### 1. PICT (Pairwise/3-Way Combinatorial Testing)
- **Tool**: Microsoft PICT
- **Variants**: 2-way, 3-way
- **Folder**: `baselines/PICT/`

### 2. CTBC (Combinatorial Testing with Base Choice)
- **Implementation**: CAgen tool
- **Folder**: `baselines/CTBC/`

### 3. SitCov (Situation Coverage)
- **Paper**: Tahir & Alexander (2021)
- **Repository**: [Original Implementation](https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA)
- **Tests**: 648 scenarios
- **Folder**: `baselines/SitCov_and_Random/`

### 4. Random Sampling
- **Strategy**: Uniform random sampling
- **Tests**: 648 scenarios
- **Folder**: `baselines/SitCov_and_Random/`

---

## 📝 Citation

If you use BayScen in your research, please cite:
```bibtex
@inproceedings{bayscen2026,
  title={BayScen: Bayesian Network-Based Scenario Generation for Autonomous Vehicle Testing},
  author={Anonymous},
  booktitle={International Conference on Software Testing, Verification and Validation (ICST)},
  year={2026},
  note={Under Review}
}
```

**Related Work:**
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

---

## Support

For questions or issues create an issue in this repository.

---

## 🙏 Acknowledgments

- **CARLA Simulator**: Open-source autonomous driving simulator
- **Norwegian Meteorological Institute**: Real-world weather data via Frost API
- **BI-CaMML**: Bayesian Network structure learning algorithm
- **InterFuser**: End-to-end autonomous driving model
- **Baseline Authors**: Original implementations of SitCov, PICT, and CTBC

---

**Built with ❤️ for safer autonomous vehicles**