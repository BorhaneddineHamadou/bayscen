# BayScen Package Structure

This folder contains the three core modules of the BayScen framework.

## Modules

```
bayscen/
├── abstraction/     # Parameter abstraction definitions
├── modeling/        # Bayesian Network training
├── generation/      # Scenario generation and evaluation
└── README.md        # This file
```

### 1. Abstraction Module

**Purpose:** Define effect-based parameter abstractions.

**Key files:**
- `abstract_variables.py` - Definitions of abstracted variables and their associated utility functions
- `abstraction_cpd.py` - Functions to extend the BN with abstracted variables and define their corresponding CPDs 
- `mapping_functions.py` - Mapping utilities to convert non-standard parameter scales, used when computing CPDs for abstracted variables

---

### 2. Modeling Module

**Purpose:** Train Bayesian Networks from real-world data.

**Key files:**
- `bn_parametrization.py` - BN training script
- `bn_parametrization_example.ipynb` - Training walkthrough
- `bn_utils.py` - Utility functions
- `models/` - Trained BN models (output)
- `README.md` - Module documentation

**Usage:**
```bash
python modeling/bn_parametrization.py --scenario 1
# Output: modeling/models/scenario1_full_bayesian_network.pkl
```

---

### 3. Generation Module

**Purpose:** Generate test scenarios using trained Bayesian Networks.

**Key files:**
- `scenario_generator.py` - Core generation algorithm
- `evaluation_metrics.py` - Quality metrics
- `generate_scenarios.py` - CLI interface
- `generation_utils.py` - Helper functions
- `scenario_generation_tutorial.ipynb` - Interactive guide
- `generated_scenarios/` - Output folder
- `README.md` - Module documentation

**Usage:**
```bash
python generation/generate_scenarios.py --scenario 1 --mode rare
# Output: generation/generated_scenarios/scenario1_rare_scenarios.csv
```

---

## Complete Workflow

```bash
# Step 1: Abstraction (definitions already provided)
# See: abstraction/abstract_variables.py

# Step 2: Train Bayesian Network
cd modeling
python bn_parametrization.py --scenario 1

# Step 3: Generate scenarios
cd ../generation
python generate_scenarios.py --scenario 1 --mode rare
```

## Module Documentation

Each module has detailed README with:
- Purpose and functionality
- Usage examples
- Output formats

Refer to individual module READMEs for complete documentation.