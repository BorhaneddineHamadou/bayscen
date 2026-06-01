# AvFuzzer — Adversarial Genetic Search Baseline

Genetic adversarial search optimising directly for collision induction, adapted for all three BayScen scenarios.

**Original paper:**
```bibtex
@inproceedings{li2020av,
  title={AV-Fuzzer: Finding Safety Violations in Autonomous Driving Systems},
  author={Li, Guanpeng and Li, Yiran and Jha, Saurabh and Tsai, Timothy and
          Sullivan, Michael and Hari, Siva Kumar Sastry and Kalbarczyk, Zbigniew
          and Iyer, Ravishankar},
  booktitle={2020 IEEE 31st International Symposium on Software Reliability
             Engineering (ISSRE)},
  pages={25--36}, year={2020}, organization={IEEE}
}
```

**Official repository:** https://github.com/Dual-Star/AVFuzzer

---

## Overview

AvFuzzer treats scenario generation as a fitness-maximisation problem. A genetic algorithm evolves a population of scenario chromosomes to maximise collision probability (measured by TTC and collision flag). It is the strongest criticality baseline in the BayScen evaluation.

### GA Configuration (paper defaults)

| Parameter | Value |
|-----------|-------|
| Population size | 12 |
| Crossover rate | 0.4 |
| Mutation rate | 0.3 |
| Scenario budget | 648 (S1/S2) | 648 (S3) |
| Local fuzzer gens | 5 (paper Section IV-C) |
| Random restart patience | 5 stagnant generations (paper IV-D) |

### Chromosome structure

| Scenario | Structural gene(s) | Environmental genes |
|----------|-------------------|-------------------|
| S1 Vehicle-Vehicle Junction | PathInteraction {c1,c2,c4} + ComboIndex {0..3} | 8 discrete params |
| S2 Vehicle-Cyclist Junction | PathInteraction {c1,c2,c4} + ComboIndex {0..3} | 9 discrete params (+TimeOfDay) |
| S3 Vehicle-Vehicle Cut-In | Direction {left,right} | 9 discrete params (+TimeOfDay) |

### Fitness function (paper eq. 6 adapted)

```
fitness = 2.0              if collision occurred
          1.8              if min_ttc ≤ 0
          1 - min_ttc/10   otherwise (clipped to [0, 1])
```

### Adaptation from the original paper

- **Discrete gene space**: All environmental genes use the same 6-level grid as BayScen (not continuous), ensuring fair comparison.
- **S3 chromosome**: Direction replaces the junction PathInteraction axis; no route lookup needed.
- **Crash detection**: Stops immediately if CARLA output files are not updated; resumes from last checkpoint after CARLA restart.
- **Matching budget**: N = 648 scenarios for all scenarios, matching BayScen for a fair comparison.

---

## Usage

```bash
# S1 — three runs
python run_avfuzzer.py --scenario 1 --seed 42
python run_avfuzzer.py --scenario 1 --seed 123
python run_avfuzzer.py --scenario 1 --seed 7

# S2
python run_avfuzzer.py --scenario 2 --seed 42

# S3
python run_avfuzzer.py --scenario 3 --seed 42

# Resume a crashed/interrupted run
python run_avfuzzer.py --scenario 1 --resume

# Dry run — GA mechanics only, no CARLA calls
python run_avfuzzer.py --scenario 1 --dry_run 24
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | 1 | 1=Vehicle-Vehicle, 2=Vehicle-Cyclist, 3=Cut-In |
| `--seed` | 42 | Random seed (42, 123, 7 for three evaluation runs) |
| `--total` | 648 | Scenario budget |
| `--resume` | False | Resume from latest checkpoint |
| `--dry_run N` | 0 | Run N scenarios of GA mechanics without CARLA |
| `--min_ttc_path` | "" | Path to `min_ttc_log.json` written by Scenario Runner |
| `--results_path` | "" | Path to `run_results.json` written by Scenario Runner |

---

## CARLA Integration

Set `--min_ttc_path` and `--results_path` to the JSON output files written by the Scenario Runner after each scenario execution:

```bash
python run_avfuzzer.py --scenario 1 --seed 42 \
    --min_ttc_path /path/to/outputs/avfuzzer/run1/min_ttc_log.json \
    --results_path /path/to/outputs/avfuzzer/run1/run_results.json
```

For **S1/S2**: the runner script calls `effects_coverage.py`  
For **S3**: the runner script calls `cutin.py` (with `--sync` for InterFuser)

---

## Outputs

All outputs are written to `./avfuzzer_logs/`:

| File | Description |
|------|-------------|
| `checkpoint_s{N}.json` | Auto-saved state after every scenario; used for `--resume` |
| `avfuzzer_s{N}_{timestamp}.xlsx` | Full scenario log (All_Scenarios, Violations, GA_Convergence sheets) |
| `violations_s{N}_{timestamp}.json` | All collision-inducing scenarios |

---

## References

- Li et al. (2020). "AV-Fuzzer: Finding Safety Violations in Autonomous Driving Systems." *ISSRE 2020*, pp. 25–36. IEEE.
- Official AvFuzzer repository: https://github.com/Dual-Star/AVFuzzer
