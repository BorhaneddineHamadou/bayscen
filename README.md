# BayScen: Realistic and Comprehensive Scenario Generation for Autonomous Vehicle Testing via Bayesian Networks

**Anonymous submission to ICST 2026**

## Overview
This repository contains the implementation of BayScen, a scenario generation framework that integrates Bayesian Networks with parameter abstraction for autonomous vehicle testing.

## Features
- Bayesian Network structure learning with LLM-guided constraints
- Parameter abstraction
- Rarity-prioritized diverse scenario generation
- Evaluation on NHTSA pre-crash scenarios in CARLA

## Installation

### Prerequisites
- Python 3.8+
- CARLA 0.9.10
- Additional dependencies in `requirements.txt`

### Setup
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Generate Scenarios
```bash
python experiments/run_scenario1.py --config experiments/configs/scenario1_config.yaml
```

### 2. Evaluate Results
```bash
python experiments/evaluate_metrics.py --results results/scenario1/
```

## Reproducibility

To reproduce the results from the paper:

1. **Download real world data** (or use provided sample):
```bash
   python data/collect_paper_data.py
```

2. **Generate tables and figures**:
```bash
   python evaluation/generate_paper_results.py
```

Expected runtime: ~X hours on [specify hardware]

## Repository Structure

- `bayscen/`: Core framework implementation
  - `modeling/`: Bayesian Network construction
  - `abstraction/`: Effect-based parameter grouping
  - `generation/`: Combinatorial testing and Scenario sampling and selection
- `baselines/`: Implementations of comparison methods
- `experiments/`: Scripts for running evaluations
- `evaluation/`: Metrics and analysis code

## Citation

If you find this work useful, please cite:
```bibtex
[Will be added after acceptance]
```

## License

[Specify license - MIT recommended for academic code]

## Contact

For questions about this anonymous submission, please use the GitHub issues.