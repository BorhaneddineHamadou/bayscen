# BayScen — Evaluation

This folder contains all scripts, simulation results, and pre-computed outputs needed to reproduce the four research questions from the paper.

---

## Folder Structure

```
evaluation/
├── 📂 simulation results/            # Raw simulation outputs (input to all scripts)
│   ├── Interfuser/
│   │   ├── Scenario 1/            # S1: Vehicle–Vehicle junction
│   │   ├── Scenario 2/            # S2: Vehicle–Cyclist junction
│   │   └── Scenario 3/            # S3: Vehicle–Vehicle cut-in
│   └── Modular/
│       ├── Scenario 1/
│       ├── Scenario 2/
│       └── Scenario 3/
│
├── 📂 scripts/                    # One script per RQ
│   ├── criticality.py             # RQ1 – Collision rate & TTC-critical rate
│   ├── physical_plausibility.py   # RQ2 – Physical violation rate (Table IV)
│   ├── critical_plausibility.py   # RQ2 – Clean Critical Rate (Table V)
│   ├── coverage_analysis.py       # RQ3 – TISA coverage metrics (Table VI)
│   └── failure_characterization.py# RQ4 – Capability-level failure profiling (Fig. 4, Table VII)
│
├── 📂 tisa/                       # TISA metric implementation
│   ├── tisa.py                    # Python wrapper
│   └── tisa_matlab/
│       └── ISA-code/
│           └── InstanceSpace/     # MATLAB InstanceSpace toolbox (Neelofar & Aleti 2024)
│
└── 📂 results/                    # Pre-computed paper outputs
    ├── rq1/
    │   ├── rq1_raw_data.xlsx
    │   ├── rq1_summary.xlsx
    │   ├── rq1_collision_rate.png
    │   ├── rq1_ttc_critical_rate.png
    │   └── rq1_combined.png
    ├── rq2/
    │   ├── physical_plausibility.xlsx
    │   └── critical_plausibility.xlsx
    ├── rq3/
    │   └── tisa_coverage.xlsx
    └── rq4/
        ├── failure_characterization.xlsx
        ├── generate_rq4_figure.py
        ├── rq4_failure_characterization.pdf
        └── rq4_failure_characterization.png
```

---

## CSV Format (`simulation results/`)

Each CSV file follows the naming convention:

```
{method}_{scenario}_{sut}_run{N}.csv
```

For example: `bayscen_scenario1_interfuser_run2.csv`

Every CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `feature_FogDensity` | CARLA fog density [0–100] |
| `feature_FogDistance` | CARLA fog distance [0–100] |
| `feature_Cloudiness` | Cloud cover [0–100] |
| `feature_Precipitation` | Rainfall intensity [0–100] |
| `feature_PrecipitationDeposits` | Surface water accumulation [0–100] |
| `feature_Wetness` | Road wetness [0–100] |
| `feature_RoadFriction` | Surface friction coefficient [0.1–1.0] |
| `feature_WindIntensity` | Wind speed [0–100] |
| `feature_TimeOfDay` | Sun altitude angle [-90°–90°] *(Scenario 2 only)* |
| `PathInteraction` | Conflict geometry state (c1/c2/c4) *(Scenarios 1 & 2 only)* |
| `Collision` | Boolean — collision occurred |
| `MinTTC` | Minimum time-to-collision in seconds (9999 = no conflict) |
| `algo_safety` | 1 if MinTTC < 0.5s OR Collision is True, else 0 |

The `feature_*` column naming and `algo_safety` flag are required by all analysis scripts.

---

## Running the Analysis

All scripts are run from the `evaluation/` directory. Each script auto-creates its output folder under `results/`.

### RQ1 — Criticality (Table III, Figure 3)

Computes collision rate and TTC-critical rate for all methods across all scenarios and both ADS.

```bash
python scripts/criticality.py --results "simulation results" --output results/rq1/rq1_raw_data.xlsx
```

**Outputs:**

| File | Contents |
|------|----------|
| `results/rq1/rq1_raw_data.xlsx` | Per-method metrics, one sheet per ADS |
| `results/rq1/rq1_summary.xlsx` | Pivot table matching Table III |
| `results/rq1/rq1_collision_rate.png` | Grouped bar chart — collision rate |
| `results/rq1/rq1_ttc_critical_rate.png` | Grouped bar chart — TTC-critical rate |
| `results/rq1/rq1_combined.png` | 2×2 paper-ready figure (Figure 3) |

---

### RQ2 — Physical Plausibility (Tables IV & V)

Two separate scripts — run both to fully reproduce RQ2.

**Physical violation rate** (Table IV — proportion of scenarios violating ≥1 physical constraint):

```bash
python scripts/physical_plausibility.py --results "simulation results" --output results/rq2/physical_plausibility.xlsx
```

**Clean Critical Rate** (Table V — proportion of critical scenarios that are also physically plausible):

```bash
python scripts/critical_plausibility.py --results "simulation results" --output results/rq2/critical_plausibility.xlsx
```

**Physical constraints implemented** (from Hao et al., arXiv:2311.10937):

| ID | Formal Rule | Physical Relationship |
|----|-------------|----------------------|
| C1 | P > 20 ⟹ D > 0 | Precipitation causes surface deposits |
| C2 | P > 20 ⟹ W > 0 | Precipitation causes road wetness |
| C3a | W < 40 ⟹ F ≤ 1 − W/200 | Wetness reduces friction (low regime) |
| C3b | W ≥ 40 ⟹ F ≤ 0.6 | Wetness reduces friction (high regime) |
| C4 | \|L − (100−G)\| ≤ 10 | Fog density determines fog distance |
| C5 | N ≥ 60 ⟹ G ≤ 40 | High wind disperses fog |
| C6 | P > 20 ⟹ C > 0 | Rain requires cloud cover |

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epsilon` | `10.0` | Tolerance ε for constraint C4 |
| `--precip-threshold` | `20.0` | Precipitation threshold for C1, C2, C6 |

---

### RQ3 — ODD Coverage (Table VI)

Computes TISA metrics (areaIS, areaBugs, covIS) via the MATLAB InstanceSpace toolbox. **Requires MATLAB.**

```bash
# Windows
python scripts/coverage_analysis.py ^
    --results  "simulation results" ^
    --tisa     tisa/tisa_matlab ^
    --output   results/rq3/tisa_coverage.xlsx ^
    --matlab   "C:\Program Files\MATLAB\R2025b\bin\matlab.exe"

# Linux / macOS
python scripts/coverage_analysis.py \
    --results  "simulation results" \
    --tisa     tisa/tisa_matlab \
    --output   results/rq3/tisa_coverage.xlsx \
    --matlab   /usr/local/MATLAB/R2023b/bin/matlab
```

**TISA metric mapping:**

| Paper metric | ISA-coverages.csv column |
|---|---|
| areaIS | `Footprint_area` |
| areaBugs | `Good_footprint_area` |
| covIS | `COV_prunnedBoundary` |

**Output:** `results/rq3/tisa_coverage.xlsx` — one sheet per ADS with per-run values, mean, and std for all three metrics across all methods and scenarios.

---

### RQ4 — Failure Characterization (Figure 4, Table VII)

Computes collision rate as a function of capability-variable degradation level (0 = clear, 5 = severe) for BayScen and BayScen-Common, across all SUT × scenario combinations.

```bash
python scripts/failure_characterization.py \
    --results  "simulation results" \
    --output   results/rq4/failure_characterization.xlsx
```

**Capability variables analyzed:**

| Variable | Paper symbol | Contributing features |
|----------|-------------|----------------------|
| Sensor_Perception | a_perc | FogDensity, FogDistance†, Cloudiness, Precipitation, TimeOfDay† |
| Surface_Traction | a_trac | RoadFriction†, Wetness, PrecipitationDeposits |
| Lateral_Stability | a_stab | WindIntensity |
| Conflict_Geometry | g | PathInteraction column (S1 & S2 only) |

† Inverted: higher raw value = less degradation.

To regenerate Figure 4 from the pre-computed Excel:

```bash
python results/rq4/generate_rq4_figure.py
```

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--suts` | both | Restrict to specific ADS (`Interfuser`, `Modular`) |
| `--scenarios` | all | Restrict to specific scenario numbers (e.g. `1 2`) |

---

## Dependencies

```bash
pip install pandas numpy matplotlib openpyxl tqdm
```

MATLAB is required for RQ3 only. All other scripts are pure Python.

---

## Notes

- Pre-computed results matching the paper are already present under `results/`. Re-running any script will overwrite them.
- The raw simulation results in `simulation results/` are pre-computed outputs from CARLA. Re-running the simulations requires a full CARLA installation and is not necessary to reproduce the paper's metrics — all scripts operate on the provided CSVs directly.
- Each script prints a console summary table in addition to writing the Excel output.
