"""
run_sitcov.py
=============
Situation Coverage (SitCov) baseline for BayScen evaluation.

Implements the SitCov generation mechanism from:
  Tahir & Alexander, "Intersection focused Situation Coverage-based V&V
  Framework for Autonomous Vehicles Implemented in CARLA", MESAS 2021.

For S1 and S2 (junction scenarios), we adapted the original SitCov framework
from the authors' CARLA-integrated implementation:
  https://github.com/zaidtahirbutt/Situation-Coverage-based-AV-Testing-Framework-in-CARLA

For S3 (cut-in), we implemented the same SitCov mechanism applied to the
cut-in parameter space (Direction + 9 environmental variables).

SITCOV MECHANISM  (faithful to Tahir & Alexander 2021)
-------------------------------------------------------
Each bin of each situation element maintains a usage counter. At every step:
  1. softmax(counts)          -> probability distribution
  2. invert: p_inv = 1 - p
  3. normalise inverted probs -> weights
  4. Weighted-random selection -> bins used LESS get HIGHER probability
This drives near-uniform coverage without hard constraints.

USAGE
-----
  # Run in CARLA environment (requires scenario runner)
  python run_sitcov.py --scenario 1 --seed 42
  python run_sitcov.py --scenario 2 --seed 123
  python run_sitcov.py --scenario 3 --seed 7

  # Dry run: generate N scenarios and print them without calling CARLA
  python run_sitcov.py --scenario 3 --dry_run 5
"""

import subprocess
import random
import argparse
import numpy as np
from scipy.special import softmax


# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

ENV_VALUES       = [0, 20, 40, 60, 80, 100]
FRICTION_VALUES  = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
TIMEOFDAY_VALUES = [-90, -60, -30, 0, 30, 60, 90]
PATH_KEYS        = ["c1", "c2", "c4"]
N_COMBOS         = 4
CUT_IN_DIRECTIONS = ["left", "right"]

PATH_MAP = {
    "c1": [
        {"StartEgo": "left",  "GoalEgo": "right", "StartOther": "base",  "GoalOther": "left"},
        {"StartEgo": "left",  "GoalEgo": "right", "StartOther": "base",  "GoalOther": "right"},
        {"StartEgo": "base",  "GoalEgo": "left",  "StartOther": "left",  "GoalOther": "right"},
        {"StartEgo": "base",  "GoalEgo": "right", "StartOther": "left",  "GoalOther": "right"},
    ],
    "c2": [
        {"StartEgo": "right", "GoalEgo": "left",  "StartOther": "base",  "GoalOther": "left"},
        {"StartEgo": "right", "GoalEgo": "base",  "StartOther": "base",  "GoalOther": "left"},
        {"StartEgo": "base",  "GoalEgo": "left",  "StartOther": "right", "GoalOther": "left"},
        {"StartEgo": "base",  "GoalEgo": "left",  "StartOther": "right", "GoalOther": "base"},
    ],
    "c4": [
        {"StartEgo": "left",  "GoalEgo": "right", "StartOther": "right", "GoalOther": "base"},
        {"StartEgo": "left",  "GoalEgo": "base",  "StartOther": "right", "GoalOther": "base"},
        {"StartEgo": "right", "GoalEgo": "base",  "StartOther": "left",  "GoalOther": "right"},
        {"StartEgo": "right", "GoalEgo": "base",  "StartOther": "left",  "GoalOther": "base"},
    ],
}


# =============================================================================
# CORE SITCOV MECHANISM  (Tahir & Alexander 2021)
# =============================================================================

def _make_dict(values):
    """Initialise a parameter dict: {bin_index: [value, counter=0]}."""
    return {i: [v, 0] for i, v in enumerate(values)}


def sitcov_select(param_dict):
    """
    Select a bin index using the SitCov weighted-random mechanism:
      1. Extract usage counts.
      2. softmax(counts)        -> probability distribution.
      3. invert:  p_inv = 1-p
      4. normalise inverted probs -> weights.
      5. Weighted-random choice  -> less-used bins selected more often.
    Updates the counter in-place and returns (bin_index, value).
    """
    keys   = list(param_dict.keys())
    counts = np.array([param_dict[k][1] for k in keys], dtype=float)
    p      = softmax(counts)
    p_inv  = 1.0 - p
    w      = p_inv / p_inv.sum()
    chosen = int(np.random.choice(keys, p=w))
    param_dict[chosen][1] += 1
    return chosen, param_dict[chosen][0]


# =============================================================================
# HYPERSPACE BUILDERS
# =============================================================================

def _build_env_dicts(include_timeofday=False):
    d = {
        "Cloudiness":            _make_dict(ENV_VALUES),
        "Precipitation":         _make_dict(ENV_VALUES),
        "PrecipitationDeposits": _make_dict(ENV_VALUES),
        "WindIntensity":         _make_dict(ENV_VALUES),
        "FogDensity":            _make_dict(ENV_VALUES),
        "FogDistance":           _make_dict(ENV_VALUES),
        "Wetness":               _make_dict(ENV_VALUES),
        "RoadFriction":          _make_dict(FRICTION_VALUES),
    }
    if include_timeofday:
        d["TimeOfDay"] = _make_dict(TIMEOFDAY_VALUES)
    return d


def _build_junction_dicts(include_timeofday=False):
    dicts = {
        "PathInteraction": _make_dict(PATH_KEYS),
        "ComboIndex":      _make_dict(list(range(N_COMBOS))),
    }
    dicts.update(_build_env_dicts(include_timeofday))
    return dicts


def _build_cutin_dicts():
    dicts = {"Direction": _make_dict(CUT_IN_DIRECTIONS)}
    dicts.update(_build_env_dicts(include_timeofday=True))
    return dicts


# =============================================================================
# SITUATION GENERATION
# =============================================================================

def generate_junction_situation(dicts):
    """Apply SitCov selection to each axis of the junction hyperspace."""
    _, pi    = sitcov_select(dicts["PathInteraction"])
    _, combo = sitcov_select(dicts["ComboIndex"])
    combo    = int(combo)
    route    = PATH_MAP[pi][combo]

    row = {
        "PathInteraction": pi,
        "ComboIndex":      combo,
        "StartEgo":        route["StartEgo"],
        "GoalEgo":         route["GoalEgo"],
        "StartOther":      route["StartOther"],
        "GoalOther":       route["GoalOther"],
    }
    for name in ["Cloudiness", "Precipitation", "PrecipitationDeposits",
                 "WindIntensity", "FogDensity", "FogDistance", "Wetness", "RoadFriction"]:
        _, val = sitcov_select(dicts[name])
        row[name] = val
    if "TimeOfDay" in dicts:
        _, val = sitcov_select(dicts["TimeOfDay"])
        row["TimeOfDay"] = val
    return row


def generate_cutin_situation(dicts):
    """Apply SitCov selection to each axis of the cut-in hyperspace."""
    _, direction = sitcov_select(dicts["Direction"])
    row = {"Direction": direction}
    for name in ["Cloudiness", "Precipitation", "PrecipitationDeposits",
                 "WindIntensity", "TimeOfDay", "FogDensity", "FogDistance",
                 "Wetness", "RoadFriction"]:
        _, val = sitcov_select(dicts[name])
        row[name] = val
    return row


# =============================================================================
# CARLA COMMAND BUILDERS
# =============================================================================

def build_command_s12(row, scenario):
    cmd = [
        "python", "effects_coverage.py",
        "--scenario", "IntersectionScenarioZ_11",
        "--not_visualize",
        "--Activate_IntersectionScenario_Seed",
        "--IntersectionScenario_Seed", "26",
        "--use_sit_cov",
        "--reloadWorld",
        "--output",
        "--cloudiness",             str(row["Cloudiness"]),
        "--precipitation",          str(row["Precipitation"]),
        "--precipitation_deposits", str(row["PrecipitationDeposits"]),
        "--wind_intensity",         str(row["WindIntensity"]),
        "--fog_density",            str(row["FogDensity"]),
        "--fog_distance",           str(row["FogDistance"]),
        "--wetness",                str(row["Wetness"]),
        "--friction",               str(row["RoadFriction"]),
        "--start_ego",              str(row["StartEgo"]),
        "--goal_ego",               str(row["GoalEgo"]),
        "--start_other",            str(row["StartOther"]),
        "--goal_other",             str(row["GoalOther"]),
        "--PathInteraction",        str(row["PathInteraction"]),
        # "--sync",  # Uncomment for InterFuser model
    ]
    if scenario == 2:
        cmd += ["--sun_altitude_angle", str(row["TimeOfDay"])]
    return cmd


def build_command_s3(row):
    return [
        "python", "cutin.py",
        "--not_visualize",
        "--reloadWorld",
        "--output",
        "--direction",              str(row["Direction"]),
        "--cloudiness",             str(row["Cloudiness"]),
        "--precipitation",          str(row["Precipitation"]),
        "--precipitation_deposits", str(row["PrecipitationDeposits"]),
        "--wind_intensity",         str(row["WindIntensity"]),
        "--sun_altitude_angle",     str(row["TimeOfDay"]),
        "--fog_density",            str(row["FogDensity"]),
        "--fog_distance",           str(row["FogDistance"]),
        "--wetness",                str(row["Wetness"]),
        "--friction",               str(row["RoadFriction"]),
        # "--sync",  # Uncomment for InterFuser model
    ]


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SitCov baseline for BayScen evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sitcov.py --scenario 1 --seed 42            # S1, run 1
  python run_sitcov.py --scenario 2 --seed 123           # S2, run 2
  python run_sitcov.py --scenario 3 --seed 7             # S3, run 3
  python run_sitcov.py --scenario 3 --dry_run 10         # print 10 scenarios
        """,
    )
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3],
                        help="Scenario: 1=Vehicle-Vehicle, 2=Vehicle-Cyclist, 3=Cut-In")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (42 / 123 / 7 for the three runs)")
    parser.add_argument("--total",    type=int, default=648,
                        help="Total number of scenarios (default: 648)")
    parser.add_argument("--timeout",  type=int, default=600,
                        help="Per-scenario CARLA timeout in seconds (default: 600)")
    parser.add_argument("--dry_run",  type=int, default=0, metavar="N",
                        help="Generate N scenarios and print without running CARLA")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    scenario_desc = {1: "Vehicle-Vehicle Junction", 2: "Vehicle-Cyclist Junction",
                     3: "Vehicle-Vehicle Cut-In"}
    total = args.dry_run if args.dry_run > 0 else args.total

    print(f"{'='*65}")
    print(f"  SITUATION COVERAGE (SITCOV) BASELINE")
    print(f"  Reference: Tahir & Alexander, MESAS 2021")
    print(f"  Scenario : S{args.scenario} — {scenario_desc[args.scenario]}")
    print(f"  Seed     : {args.seed}")
    print(f"  Total    : {total}")
    print(f"  Mode     : {'DRY RUN (no CARLA)' if args.dry_run else 'CARLA simulation'}")
    print(f"{'='*65}\n")

    # Build situation hyperspace dicts (counters start at 0)
    if args.scenario == 1:
        dicts = _build_junction_dicts(include_timeofday=False)
    elif args.scenario == 2:
        dicts = _build_junction_dicts(include_timeofday=True)
    else:
        dicts = _build_cutin_dicts()

    for index in range(total):
        if args.scenario in [1, 2]:
            row = generate_junction_situation(dicts)
        else:
            row = generate_cutin_situation(dicts)

        print(f"Scenario {index+1}/{total} | {row}")

        if args.dry_run:
            continue

        if args.scenario in [1, 2]:
            command = build_command_s12(row, args.scenario)
        else:
            command = build_command_s3(row)

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=args.timeout,
            )
            print(f"  -> return code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  -> TIMEOUT")
        except Exception as e:
            print(f"  -> ERROR: {e}")

        print("-" * 40)

    print(f"\nDone. Generated {total} situations.")


if __name__ == "__main__":
    main()
