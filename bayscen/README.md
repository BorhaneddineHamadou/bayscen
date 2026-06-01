# BayScen — Core Framework

This folder contains the three core modules of the BayScen framework.

## Structure

```
bayscen/
├── abstraction/     # ISO 34503:2023-grounded capability abstraction
├── modeling/        # Bayesian Network structure learning & parameterization
├── generation/      # Rarity-prioritized diverse scenario generation
└── README.md
```

---

### 1. Abstraction Module

**Purpose:** Define the ISO 34503:2023-grounded capability variables that group
concrete CARLA parameters by the ADS capability they degrade.

**Key files:**

| File | Description |
|------|-------------|
| `abstract_variables.py` | Capability variable definitions (Sensor_Perception, Surface_Traction, Lateral_Stability, Conflict_Geometry) and their ISO grounding |
| `abstraction_cpd.py` | CPD computation for capability leaf nodes (uniform-weight aggregation) |
| `mapping_functions.py` | Scale converters for Road_Friction and Sun_Altitude_Angle |

**Capability variables (ISO 34503:2023):**

| Variable | Symbol | ISO Clause | Contributing Parameters |
|----------|--------|------------|------------------------|
| Sensor Perception | a_perc | §10.2.4, §10.3, §10.4 | Fog density, fog distance, cloudiness, precipitation, sun altitude angle (S2/S3) |
| Surface Traction  | a_trac | §9.3.7 | Road friction, wetness, precipitation deposits |
| Lateral Stability | a_stab | §10.2.3 | Wind intensity |
| Conflict Geometry | g      | — | Ego/adversary start & goal positions (junction scenarios) |

---

### 2. Modeling Module

**Purpose:** Train Bayesian Networks from real-world meteorological data.

**Key files:**

| File | Description |
|------|-------------|
| `bn_parametrization.py` | Full parameterization pipeline (supports S1, S2, S3) |
| `bn_parametrization_example.ipynb` | Training walkthrough |
| `bn_utils.py` | Model save/load utilities |
| `models/` | Saved trained BN models |
| `structure_learning/` | Bi-CaMML priors and learned structures |

**Usage:**
```bash
python modeling/bn_parametrization.py --scenario 1   # Vehicle–Vehicle
python modeling/bn_parametrization.py --scenario 2   # Vehicle–Cyclist
python modeling/bn_parametrization.py --scenario 3   # Cut-In
```

---

### 3. Generation Module

**Purpose:** Generate rarity-prioritized, diverse test scenarios from trained BNs.

**Key files:**

| File | Description |
|------|-------------|
| `scenario_generator.py` | Core Algorithm 1 implementation |
| `generate_scenarios.py` | CLI interface (supports S1, S2, S3) |
| `evaluation_metrics.py` | Physical plausibility checks (C1–C6) |
| `generation_utils.py` | Path assignment, export utilities |
| `scenario_generation_tutorial.ipynb` | Interactive generation guide |

**Usage:**
```bash
# BayScen (rarity-prioritized)
python generation/generate_scenarios.py --scenario 1 --mode rare
python generation/generate_scenarios.py --scenario 2 --mode rare
python generation/generate_scenarios.py --scenario 3 --mode rare

# BayScen-Common ablation
python generation/generate_scenarios.py --scenario 1 --mode common
```

---

## Complete Workflow

```bash
# Step 1: Train Bayesian Networks
cd bayscen/modeling
python bn_parametrization.py --scenario 1
python bn_parametrization.py --scenario 2
python bn_parametrization.py --scenario 3

# Step 2: Generate scenarios
cd ../generation
python generate_scenarios.py --scenario 1 --mode rare
python generate_scenarios.py --scenario 2 --mode rare
python generate_scenarios.py --scenario 3 --mode rare
```

Output scenario CSVs are written to `generation/generated_scenarios/`.
