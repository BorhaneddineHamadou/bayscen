# SitCov — Situation Coverage Baseline

Coverage-weighted sampling that drives near-uniform distribution across the situation hyperspace without hard constraints.

**Original paper:**
```bibtex
@inproceedings{tahir2021intersection,
  title={Intersection focused situation coverage-based verification and validation framework for autonomous vehicles implemented in CARLA},
  author={Tahir, Zaid and Alexander, Rob},
  booktitle={International Conference on Modelling and Simulation for Autonomous Systems},
  pages={191--212}, year={2021}, organization={Springer}
}
```

**Official repository:** https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA

---

## Overview

### Mechanism (Tahir & Alexander 2021)

Each bin of each situation element maintains a usage counter. At every scenario selection step:

1. `softmax(counts)` → probability distribution  
2. Invert: `p_inv = 1 − p`  
3. Normalise inverted probs → weights  
4. Weighted-random choice → **less-used bins selected more often**

This steers sampling toward under-covered parameter values, achieving near-uniform situation coverage over the full run.

### Implementation per scenario

| Scenario | Implementation source |
|----------|----------------------|
| S1 Vehicle-Vehicle Junction | Adapted from the original SitCov repository integrated in CARLA Scenario Runner |
| S2 Vehicle-Cyclist Junction | Adapted from the original SitCov repository integrated in CARLA Scenario Runner |
| S3 Vehicle-Vehicle Cut-In | Standalone adaptation applying the same SitCov mechanism to the cut-in parameter space |

For S1/S2, we used the SitCov framework as implemented in the authors' CARLA-integrated codebase, running through the same Scenario Runner infrastructure as all other junction baselines. For S3, we implemented the identical SitCov mechanism applied to the cut-in hyperspace (Direction + 9 environmental axes).

---

## Usage

```bash
# Run in CARLA environment (requires Scenario Runner)
python run_sitcov.py --scenario 1 --seed 42     # S1, run 1
python run_sitcov.py --scenario 2 --seed 123    # S2, run 2
python run_sitcov.py --scenario 3 --seed 7      # S3, run 3

# Dry run — generate and print N scenarios without calling CARLA
python run_sitcov.py --scenario 3 --dry_run 10
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | 1 | Scenario: 1=Vehicle-Vehicle, 2=Vehicle-Cyclist, 3=Cut-In |
| `--seed` | 42 | Random seed (42, 123, 7 for three evaluation runs) |
| `--total` | 648 | Number of scenarios |
| `--timeout` | 600 | Per-scenario CARLA timeout (seconds) |
| `--dry_run N` | 0 | Generate N scenarios without CARLA |

---

## Situation Hyperspace

### S1 (Vehicle-Vehicle Junction) — 10 axes

| Axis | Values |
|------|--------|
| PathInteraction | c1, c2, c4 |
| ComboIndex | 0, 1, 2, 3 |
| Cloudiness | 0, 20, 40, 60, 80, 100 |
| Precipitation | 0, 20, 40, 60, 80, 100 |
| PrecipitationDeposits | 0, 20, 40, 60, 80, 100 |
| WindIntensity | 0, 20, 40, 60, 80, 100 |
| FogDensity | 0, 20, 40, 60, 80, 100 |
| FogDistance | 0, 20, 40, 60, 80, 100 |
| Wetness | 0, 20, 40, 60, 80, 100 |
| RoadFriction | 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 |

### S2 adds: TimeOfDay (−90, −60, −30, 0, 30, 60, 90)

### S3 (Vehicle-Vehicle Cut-In) — replaces PathInteraction/ComboIndex with Direction

| Axis | Values |
|------|--------|
| Direction | left, right |
| + all 9 environmental axes from S2 (including TimeOfDay) | |

---

## CARLA Integration

For **S1/S2**: calls `effects_coverage.py` (junction Scenario Runner script)  
For **S3**: calls `cutin.py` (cut-in Scenario Runner script)

---

## References

- Tahir, Z., & Alexander, R. (2021). "Intersection focused situation coverage-based verification and validation framework for autonomous vehicles implemented in CARLA." In *MESAS 2021* (pp. 191–212). Springer.
- Official SitCov repository: https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA
