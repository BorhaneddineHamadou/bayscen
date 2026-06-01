# Random Sampling Baseline

Uninformed lower bound: uniform random sampling over all valid parameter combinations, without any coverage optimisation or importance weighting.

---

## Overview

For each scenario, `run_random.py` draws N = 648 scenarios by independently sampling each parameter uniformly at random. Three seeds (42, 123, 7) produce the three evaluation runs reported in the paper.

### Parameter spaces

| Scenario | Structural axis | Environmental axes | Total variables |
|----------|----------------|--------------------|----------------|
| S1 Vehicle-Vehicle Junction | PathInteraction {c1,c2,c4} + route combo | 8 env (no TimeOfDay) | 10 |
| S2 Vehicle-Cyclist Junction | PathInteraction {c1,c2,c4} + route combo | 9 env (+ TimeOfDay) | 11 |
| S3 Vehicle-Vehicle Cut-In | Direction {left,right} | 9 env (+ TimeOfDay) | 10 |

Environmental parameters (all scenarios):
`Cloudiness, Precipitation, PrecipitationDeposits, WindIntensity, FogDensity, FogDistance, Wetness, RoadFriction`

S2 and S3 add:
`TimeOfDay` (sun altitude angle in degrees: −90, −60, −30, 0, 30, 60, 90)

---

## Usage

```bash
# Run in CARLA environment (requires Scenario Runner)
python run_random.py --scenario 1 --seed 42     # S1, run 1
python run_random.py --scenario 1 --seed 123    # S1, run 2
python run_random.py --scenario 1 --seed 7      # S1, run 3

python run_random.py --scenario 2 --seed 42
python run_random.py --scenario 3 --seed 42

# Dry run — generate and print N scenarios without calling CARLA
python run_random.py --scenario 1 --dry_run 10
python run_random.py --scenario 3 --dry_run 10
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | 1 | Scenario: 1=Vehicle-Vehicle, 2=Vehicle-Cyclist, 3=Cut-In |
| `--seed` | 42 | Random seed (42, 123, 7 for the three evaluation runs) |
| `--total` | 648 | Number of scenarios to generate |
| `--timeout` | 600 | Per-scenario CARLA timeout (seconds) |
| `--dry_run N` | 0 | Generate N scenarios and print without CARLA |

---

## CARLA Integration

For **S1/S2** (junction), each scenario calls:
```
effects_coverage.py --scenario IntersectionScenarioZ_11 [env params] [path params]
```

For **S3** (cut-in), each scenario calls:
```
cutin.py --direction left|right [env params]
```

Uncomment `--sync` in the command builder for the InterFuser model.

---

## Notes

- Random sampling serves as the uninformed lower bound in all comparisons.
- It uses the identical scenario budget (N = 648) as BayScen and AvFuzzer for fair comparison.
- Combinatorial methods (PICT, CTBC) use smaller budgets determined by their coverage criteria.
