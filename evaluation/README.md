# Evaluation Module

Complete evaluation system for comparing scenario generation methods. Computes all metrics reported in the paper.

## Structure

```
evaluation/
├── metrics.py                                    # Core evaluation functions
├── evaluate.py                                   # Command-line interface
├── evaluation_tutorial.ipynb                     # Interactive notebook
├── README.md                                     # This file
├── results/                                      # Output folder (auto-created)
│   ├── results_scenario{N}.csv                  # Complete results
│   ├── paper_scenario{N}_II_effectiveness.csv   # Table II
│   ├── paper_scenario{N}_III_realism.csv        # Table III
│   └── paper_scenario{N}_IV_coverage.csv        # Table IV
├── Scenario1 Generated Scenarios/               # Excel files (user provides)
├── Scenario1 Execution Results (JSON)/          # Simulation results (user provides)
├── Scenario2 Generated Scenarios/               # Excel files (user provides)
└── Scenario2 Execution Results (JSON)/          # Simulation results (user provides)
```

## Quick Start

```bash
# Evaluate Scenario 1 - generates 4 CSV files in results/
python evaluate.py --scenario 1

# Evaluate Scenario 2
python evaluate.py --scenario 2
```

**Output:** All CSV files saved in `results/` folder.

## Data Requirements

### Excel Files (Generated Scenarios)
Location: `Scenario{N} Generated Scenarios/`

Required files:
- `bayscen.xlsx`
- `random.xlsx`
- `sitcov.xlsx`
- `PICT_3w.xlsx`
- `PICT_2w.xlsx`
- `CTBC.xlsx`

Columns: `TimeOfDay` (Scenario 2 only), `Cloudiness`, `WindIntensity`, `Precipitation`, `PrecipitationDeposits`, `Wetness`, `RoadFriction`, `FogDensity`, `FogDistance`, `Visibility`, `RoadSurface`, `VehicleStability`, `PathInteraction`, `probability`, `StartEgo`, `GoalEgo`, `StartOther`, `GoalOther`

### JSON Files (Execution Results)
Location: `Scenario{N} Execution Results (JSON)/{method}/run{1,2,3}/`

Required files per run:
- `min_ttc_log.json` - Format: `[{"min_ttc": 2.45}, ...]`
- `run_results.json` - Format: `[{"collision_occurred": false, "run_duration": 271.8}, ...]`

### Real-World Data
Location: `../data/processed/bayscen_final_data.csv`

Reference data for realism computation.

## Metrics Computed

### 1. Realism
Distance-based similarity to real-world data using cKDTree.

### 2. 3-Way Coverage
Combinatorial coverage with Precision, Recall, and F1 Score.

### 3. Criticality
- Mean TTC across all runs
- Critical TTC (<0.5s) with realism filtering
- Collision 2/3 (≥2 runs collided) with realism filtering
- Collision 3/3 (all runs collided) with realism filtering

## Output Files

All files saved in `results/` folder:

### 1. Complete Results
`results_scenario{N}.csv` - All raw metrics

### 2. Table II - Effectiveness
`paper_scenario{N}_II_effectiveness.csv`

Columns: Method, N, TTC<0.5 Count (#), TTC<0.5 Rate (%), Collision (≥2/3) Count (#), Collision (≥2/3) Rate (%), Collision (3/3) Count (#), Collision (3/3) Rate (%)

### 3. Table III - Realism Analysis
`paper_scenario{N}_III_realism.csv`

Columns: Method, Overall Realism (%), Mean TTC < 0.5 Count, Mean TTC < 0.5 Realistic (#), Mean TTC < 0.5 Realism (%), Collisions (≥2/3) Count, Collisions (≥2/3) Realistic (#), Collisions (≥2/3) Realism(%)

### 4. Table IV - Coverage Quality
`paper_scenario{N}_IV_coverage.csv`

Columns: Method, All triples, Real triples, Precision, F1, N

## Tutorial Notebook

`evaluation_tutorial.ipynb` provides a complete walkthrough:

1. Configuration and setup
2. Run evaluation (one command)
3. View Table II (displayed in notebook)
4. View Table III (displayed in notebook)
5. View Table IV (displayed in notebook)
6. Save all tables to CSV

## Expected Runtime

- Scenario 1: ~6-8 minutes (6 methods)
- Scenario 2: ~7-9 minutes (6 methods)
- Single method: ~2-3 seconds

### Wrong metric values
Verify:
1. Number of scenarios matches JSON entries
2. All 3 runs present (run1, run2, run3)
3. TimeOfDay present for Scenario 2
4. Parameter domains match your data