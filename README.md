# BayScen: Standards-Grounded Scenario Generation for Autonomous Vehicle Testing via Bayesian Networks and Capability Abstraction

[![CARLA](https://img.shields.io/badge/CARLA-0.9.10-green)](https://carla.readthedocs.io/en/0.9.10/)
[![Python](https://img.shields.io/badge/Python-3.7+-orange)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

BayScen is a scenario generation framework for autonomous vehicle (AV) testing that resolves the fundamental **coverage–realism–criticality trilemma**. Existing methods satisfy at most one of these objectives simultaneously: combinatorial testing achieves coverage but generates physically implausible scenarios; search-based methods find critical failures but violate naturalistic driving dependencies; data-driven methods maintain realism but cannot escape their training distribution.

BayScen navigates this trilemma through two principled mechanisms:

1. **ISO 34503:2023-grounded capability abstraction** — collapses millions of raw environmental parameter combinations into three genuinely distinct degradation regimes (sensor perception, surface traction, lateral stability), making exhaustive combinatorial coverage tractable while preserving all safety-relevant distinctions.

2. **Bayesian Network learned from real-world observations** — encodes probabilistic dependencies among environmental parameters from 43,000+ hourly meteorological records, ensuring generated scenarios respect the statistical regularities of naturalistic driving.

### Key Results

| Metric | BayScen | Best Baseline | Notes |
|--------|---------|---------------|-------|
| Clean Critical Rate (physically plausible critical scenarios) | **38.8%** | 14.8% | ×8 vs. AvFuzzer (5.4%) |
| Physical Violation Rate | **52.8%** | 93.2–94.3% | 37× reduction vs. non-BN baselines |
| ODD Coverage (covIS) | **62.1–64.3%** | 47.5–51.8% | +12–17 pp lead over AvFuzzer |
| Failure-Region Diversity (areaBugs, S1) | **28.5–28.9** | 17.1–21.1 | vs. AvFuzzer |

Evaluated across **3 NHTSA pre-crash scenarios**, **2 ADS architectures** (InterFuser end-to-end, Modular rule-based), and **6 baselines** including AvFuzzer as the strongest adversarial comparator.

---

## Core Contributions

**1. BayScen framework** — integrates Bayesian Networks with ISO 34503-grounded capability abstraction and rarity-prioritized combinatorial testing, balancing coverage, realism, and criticality within a unified framework rather than trading one against the others.

**2. Scenario-agnostic capability abstraction** — three capability variables derived from ISO 34503:2023 (sensor perception, surface traction, lateral stability) apply unchanged across all evaluated scenario types, positioning the abstraction as a reusable foundation for ODD-driven test design.

**3. Comprehensive empirical evaluation** — across three NHTSA scenarios, two ADS architectures, and six baselines. BayScen's critical scenarios are ×8 more likely to be physically plausible than the strongest adversarial baseline, and capability-organized failure profiling reveals architecture-specific vulnerability patterns invisible to methods reporting only aggregate collision counts.

---

## Repository Structure

```
BayScen/
├── 📂 bayscen/                    # Core framework modules
│   ├── abstraction/               # ISO 34503-grounded capability abstraction
│   │   ├── abstract_variables.py  # Capability variable definitions
│   │   ├── abstraction_cpd.py     # Uniform-weight aggregation CPDs
│   │   └── mapping_functions.py   # Scale converters
│   ├── modeling/                  # BN structure learning & parameterization
│   │   ├── bn_parametrization.py  # Training pipeline (S1, S2, S3)
│   │   ├── bn_utils.py            # Model I/O utilities
│   │   └── structure_learning/    # Bi-CaMML priors & learned structures
│   └── generation/                # Rarity-prioritized scenario generation
│       ├── scenario_generator.py  # Algorithm 1 core
│       ├── generate_scenarios.py  # CLI interface
│       ├── evaluation_metrics.py  # Physical plausibility (C1–C6)
│       └── generation_utils.py    # Path assignment & export helpers
│
├── 📂 data/                       # Real-world meteorological data
│   ├── collect.py                 # Frost API data collection
│   ├── process.py                 # Weather data processing & discretization
│   └── processed/                 # Final datasets for BN training (43,000+ obs.)
│
├── 📂 baselines/                  # All baseline implementations
│   ├── CTBC/                      # Importance-weighted combinatorial testing (S1/S2/S3)
│   ├── PICT/                      # Pairwise & 3-way combinatorial testing (S1/S2/S3)
│   ├── Random/                    # Uniform random sampling (S1/S2/S3)
│   │   └── run_random.py          # --scenario 1|2|3  --seed 42|123|7
│   ├── SitCov/                    # Situation Coverage weighted sampling (S1/S2/S3)
│   │   └── run_sitcov.py          # --scenario 1|2|3  --seed 42|123|7
│   └── AvFuzzer/                  # Genetic adversarial search (S1/S2/S3)
│       └── run_avfuzzer.py        # --scenario 1|2|3  --seed 42|123|7  [--resume]
│
├── 📂 evaluation/                 # Metrics, scripts & pre-computed results
│   ├── simulation results/       # Raw CARLA simulation outputs (all methods)
│   │   ├── Interfuser/            # Results for InterFuser ADS
│   │   │   ├── Scenario 1/        # S1: Vehicle–Vehicle junction
│   │   │   ├── Scenario 2/        # S2: Vehicle–Cyclist junction
│   │   │   └── Scenario 3/        # S3: Vehicle–Vehicle cut-in
│   │   └── Modular/               # Results for Modular ADS
│   │       ├── Scenario 1/
│   │       ├── Scenario 2/
│   │       └── Scenario 3/
│   ├── scripts/                   # One script per research question
│   ├── tisa/                      # TISA coverage metric implementation
│   └── results/                   # Pre-computed paper outputs (Tables III–VII)
│
└── 📄 README.md                   # This file
```

---

## Experimental Setup

### Scenarios

| ID | Description | Conflict Type | Actor Types |
|----|-------------|---------------|-------------|
| S1 | Vehicle–Vehicle Junction | Left-turn & right-turn sub-cases | Two vehicles |
| S2 | Vehicle–Cyclist Junction | Crossing paths | Vehicle + pedalcyclist |
| S3 | Vehicle–Vehicle Cut-In | Highway lateral encroachment | Two vehicles (left/right) |

### Systems Under Test

- **InterFuser**: End-to-end transformer fusing multi-view RGB and LiDAR; top performer on CARLA 0.9.10 leaderboard.
- **Modular ADS**: Rule-based pipeline with explicit separation of perception (SSD MobileNet), planning (A*), and control (PID).

### Baselines

| Baseline | Folder | Paradigm | Notes |
|----------|--------|----------|-------|
| Random Sampling | `baselines/Random/` | Uninformed | Uniform over all variable combinations |
| SitCov | `baselines/SitCov/` | Coverage-based | Prioritizes under-tested values (Tahir & Alexander 2021) |
| PICT-2w | `baselines/PICT/` | Combinatorial | All pairwise combinations |
| PICT-3w | `baselines/PICT/` | Combinatorial | All three-way combinations |
| CTBC | `baselines/CTBC/` | Combinatorial | Importance-weighted pairwise (Gao et al. 2019) |
| AvFuzzer | `baselines/AvFuzzer/` | Adversarial search | Genetic algorithm optimizing for collision induction (Li et al. 2020) |
| BayScen-Common | `bayscen/generation/` | Ablation | BayScen with common (not rare) scenario selection; isolates rarity contribution |

All methods use matched scenario budgets (N=648 for junction, N=216 for cut-in).

### Capability Abstraction (ISO 34503:2023)

| Capability Variable | Symbol | Contributing Parameters | ISO Clause |
|--------------------|--------|------------------------|------------|
| Sensor Perception | a_perc | Fog density, fog distance, cloudiness, precipitation, sun altitude angle (S2/S3) | §10.2.4, §10.3, §10.4 |
| Surface Traction | a_trac | Road friction, wetness, precipitation deposits | §9.3.7 |
| Lateral Stability | a_stab | Wind intensity | §10.2.3 |
| Conflict Geometry | g | Ego/adversary start & goal positions | — (scenario-level) |

---

## Reproducing the Paper Results

### Prerequisites

- Python 3.7+
- CARLA 0.9.10 (required for re-running simulations; pre-collected results are provided)
- `pip install pgmpy pandas numpy scipy tqdm openpyxl`

### Step 1: Data Collection *(optional — pre-collected data included)*

```bash
cd data
python collect.py --config config.yaml
python process.py
# Output: processed/bayscen_final_data.csv  (43,000+ hourly observations)
```

### Step 2: Train Bayesian Networks

```bash
cd bayscen/modeling
python bn_parametrization.py --scenario 1   # Vehicle–Vehicle
python bn_parametrization.py --scenario 2   # Vehicle–Cyclist
python bn_parametrization.py --scenario 3   # Cut-In
# Output: models/scenario{1,2,3}_full_bayesian_network.pkl
```

### Step 3: Generate Scenarios

```bash
cd bayscen/generation

# BayScen (rarity-prioritized)
python generate_scenarios.py --scenario 1 --mode rare
python generate_scenarios.py --scenario 2 --mode rare
python generate_scenarios.py --scenario 3 --mode rare

# BayScen-Common ablation
python generate_scenarios.py --scenario 1 --mode common
python generate_scenarios.py --scenario 2 --mode common
```

### Step 4: Run Simulations *(requires CARLA 0.9.10)*

Simulation execution scripts for each baseline are in their respective `baselines/` subfolder.
Pre-collected results for all methods × scenarios × ADS combinations are
already provided in `evaluation/simulation results/`.

```bash
# Example: run all baselines for Scenario 1, seed 42
python baselines/Random/run_random.py   --scenario 1 --seed 42
python baselines/SitCov/run_sitcov.py  --scenario 1 --seed 42
python baselines/AvFuzzer/run_avfuzzer.py --scenario 1 --seed 42 \
    --min_ttc_path /path/to/min_ttc_log.json \
    --results_path /path/to/run_results.json
# PICT and CTBC use pre-generated xlsx files (see baselines/PICT/ and baselines/CTBC/)
```

### Step 5: Evaluate

```bash
cd evaluation

# RQ1 — Collision rate & TTC-critical rate (Table III, Figure 3)
python scripts/criticality.py --results "simulation results" --output results/rq1/rq1_raw_data.xlsx

# RQ2 — Physical violation rate (Table IV)
python scripts/physical_plausibility.py --results "simulation results" --output results/rq2/physical_plausibility.xlsx

# RQ2 — Clean Critical Rate (Table V)
python scripts/critical_plausibility.py --results "simulation results" --output results/rq2/critical_plausibility.xlsx

# RQ3 — TISA coverage metrics: areaIS, areaBugs, covIS (Table VI)
python scripts/coverage_analysis.py --results "simulation results" --tisa tisa/tisa_matlab --output results/rq3/tisa_coverage.xlsx --matlab /path/to/matlab

# RQ4 — Capability-level failure profiling (Figure 4, Table VII)
python scripts/failure_characterization.py --results "simulation results" --output results/rq4/failure_characterization.xlsx
```

---

## Evaluation Metrics

| Metric | Description | Tables/Figures |
|--------|-------------|----------------|
| **Collision Rate** | % scenarios causing collision in ≥2/3 runs | Table III, Fig. 3 |
| **Physical Violation Rate** | % scenarios violating ≥1 physical constraint (Hao et al.) | Table IV |
| **Clean Critical Rate** | % critical scenarios that are also physically plausible | Table V |
| **areaIS** | ODD feature-space diversity of the full test suite (TISA) | Table VI |
| **areaBugs** | Diversity of the failure-inducing ODD region (TISA) | Table VI |
| **covIS** | Proportion of feasible ODD explored (TISA) | Table VI |
| **Capability Profile** | Collision rate vs. degradation level per capability variable | Fig. 4, Table VII |

Physical plausibility uses the independently developed constraint taxonomy of Hao et al. (arXiv:2311.10937), covering fog–distance consistency (C4), precipitation–surface dependencies (C1–C2), wetness–friction relationships (C3a/C3b), wind–fog incompatibility (C5), and rain–cloud cover (C6).

ODD coverage uses the TISA metrics of Neelofar & Aleti (ICSE 2024), purpose-built for black-box AI system testing and empirically validated to correlate with fault detection.

---

## Module Documentation

| Module | Purpose | Documentation |
|--------|---------|---------------|
| `bayscen/` | Core framework | [README](bayscen/README.md) |
| `data/` | Meteorological data collection | [README](data/README.md) |
| `baselines/Random/` | Uniform random sampling | [README](baselines/Random/README.md) |
| `baselines/SitCov/` | Situation Coverage weighted sampling | [README](baselines/SitCov/README.md) |
| `baselines/PICT/` | Pairwise & 3-way combinatorial testing | [README](baselines/PICT/README.md) |
| `baselines/CTBC/` | Importance-weighted combinatorial testing | [README](baselines/CTBC/README.md) |
| `baselines/AvFuzzer/` | Genetic adversarial search | [README](baselines/AvFuzzer/README.md) |
| `evaluation/` | Metrics & analysis | [README](evaluation/README.md) |

**Tutorials:**

- `data/data_collection_tutorial.ipynb` — Frost API data collection walkthrough
- `bayscen/modeling/bn_parametrization_example.ipynb` — BN structure learning & parameterization
- `bayscen/generation/scenario_generation_tutorial.ipynb` — Rarity-prioritized generation guide

---

## Acknowledgments

- **CARLA Simulator** — Open-source autonomous driving simulator
- **Norwegian Meteorological Institute** — Real-world weather data via Frost API
- **Bi-CaMML** — Bayesian Network structure learning with soft ancestral constraints (Ban et al. 2025)
- **InterFuser** — End-to-end autonomous driving model (Shao et al., CoRL 2023)
- **Hao et al.** — Physical plausibility constraint taxonomy (arXiv:2311.10937)
- **Neelofar & Aleti** — TISA coverage metrics (ICSE 2024)
- **Baseline authors** — SitCov (Tahir & Alexander 2021), PICT, CTBC, AvFuzzer (Li et al. 2020)
