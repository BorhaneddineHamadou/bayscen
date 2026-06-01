"""
run_random.py
=============
Random Sampling baseline for BayScen evaluation.

Generates N scenarios by uniform random sampling over all valid parameter
values, then executes each one through the CARLA Scenario Runner.

Supports all three NHTSA scenarios:
  --scenario 1  Vehicle-Vehicle Junction    (8 env params + PathInteraction)
  --scenario 2  Vehicle-Cyclist Junction    (9 env params + PathInteraction + TimeOfDay)
  --scenario 3  Vehicle-Vehicle Cut-In      (9 env params + Direction + TimeOfDay)

SEEDS (3 runs per scenario for reproducibility):
  Run 1: --seed 42   Run 2: --seed 123   Run 3: --seed 7

USAGE
-----
  # Run in CARLA environment (requires scenario runner)
  python run_random.py --scenario 1 --seed 42
  python run_random.py --scenario 2 --seed 123
  python run_random.py --scenario 3 --seed 7

  # Dry run: generate N scenarios and print them without calling CARLA
  python run_random.py --scenario 1 --dry_run 5
"""

import subprocess
import random
import argparse


# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

ENV_VALUES      = [0, 20, 40, 60, 80, 100]
FRICTION_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
TIMEOFDAY_VALUES = [-90, -60, -30, 0, 30, 60, 90]   # sun altitude angle (degrees)

# Junction path interaction — valid conflict geometry states (c1/c2/c4)
PATH_KEYS  = ["c1", "c2", "c4"]
N_COMBOS   = 4   # each PathInteraction has 4 valid route combos

# Route combinations per conflict geometry (ego/adversary start-goal pairs)
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

CUT_IN_DIRECTIONS = ["left", "right"]


# =============================================================================
# SCENARIO GENERATORS
# =============================================================================

def random_scenario_s1():
    """Random scenario for S1 (Vehicle-Vehicle Junction) — 8 env params."""
    pi    = random.choice(PATH_KEYS)
    combo = random.randint(0, N_COMBOS - 1)
    route = PATH_MAP[pi][combo]
    return {
        "PathInteraction":       pi,
        "ComboIndex":            combo,
        "StartEgo":              route["StartEgo"],
        "GoalEgo":               route["GoalEgo"],
        "StartOther":            route["StartOther"],
        "GoalOther":             route["GoalOther"],
        "Cloudiness":            random.choice(ENV_VALUES),
        "Precipitation":         random.choice(ENV_VALUES),
        "PrecipitationDeposits": random.choice(ENV_VALUES),
        "WindIntensity":         random.choice(ENV_VALUES),
        "FogDensity":            random.choice(ENV_VALUES),
        "FogDistance":           random.choice(ENV_VALUES),
        "Wetness":               random.choice(ENV_VALUES),
        "RoadFriction":          random.choice(FRICTION_VALUES),
    }


def random_scenario_s2():
    """Random scenario for S2 (Vehicle-Cyclist Junction) — 9 env params + TimeOfDay."""
    row = random_scenario_s1()
    row["TimeOfDay"] = random.choice(TIMEOFDAY_VALUES)
    return row


def random_scenario_s3():
    """Random scenario for S3 (Vehicle-Vehicle Cut-In) — Direction + 9 env params."""
    return {
        "Direction":             random.choice(CUT_IN_DIRECTIONS),
        "Cloudiness":            random.choice(ENV_VALUES),
        "Precipitation":         random.choice(ENV_VALUES),
        "PrecipitationDeposits": random.choice(ENV_VALUES),
        "WindIntensity":         random.choice(ENV_VALUES),
        "TimeOfDay":             random.choice(TIMEOFDAY_VALUES),
        "FogDensity":            random.choice(ENV_VALUES),
        "FogDistance":           random.choice(ENV_VALUES),
        "Wetness":               random.choice(ENV_VALUES),
        "RoadFriction":          random.choice(FRICTION_VALUES),
    }


GENERATORS = {1: random_scenario_s1, 2: random_scenario_s2, 3: random_scenario_s3}


# =============================================================================
# CARLA COMMAND BUILDERS
# =============================================================================

def build_command_s12(row, scenario):
    """Build subprocess command for S1/S2 junction scenarios."""
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
    """Build subprocess command for S3 cut-in scenario."""
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
        description="Random Sampling baseline for BayScen evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_random.py --scenario 1 --seed 42            # S1, run 1
  python run_random.py --scenario 2 --seed 123           # S2, run 2
  python run_random.py --scenario 3 --seed 7             # S3, run 3
  python run_random.py --scenario 1 --dry_run 5          # print 5 scenarios without CARLA
        """,
    )
    parser.add_argument("--scenario",  type=int, default=1, choices=[1, 2, 3],
                        help="Scenario: 1=Vehicle-Vehicle, 2=Vehicle-Cyclist, 3=Cut-In")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (42 / 123 / 7 for the three runs)")
    parser.add_argument("--total",     type=int, default=648,
                        help="Total number of scenarios to generate (default: 648)")
    parser.add_argument("--timeout",   type=int, default=600,
                        help="Per-scenario CARLA timeout in seconds (default: 600)")
    parser.add_argument("--dry_run",   type=int, default=0, metavar="N",
                        help="Generate N scenarios and print them without running CARLA")
    args = parser.parse_args()

    random.seed(args.seed)

    scenario_desc = {1: "Vehicle-Vehicle Junction", 2: "Vehicle-Cyclist Junction",
                     3: "Vehicle-Vehicle Cut-In"}
    total = args.dry_run if args.dry_run > 0 else args.total

    print(f"{'='*65}")
    print(f"  RANDOM SAMPLING BASELINE")
    print(f"  Scenario : S{args.scenario} — {scenario_desc[args.scenario]}")
    print(f"  Seed     : {args.seed}")
    print(f"  Total    : {total}")
    print(f"  Mode     : {'DRY RUN (no CARLA)' if args.dry_run else 'CARLA simulation'}")
    print(f"{'='*65}\n")

    generate = GENERATORS[args.scenario]

    for index in range(total):
        row = generate()
        print(f"Scenario {index + 1}/{total} | {row}")

        if args.dry_run:
            continue   # print only, no CARLA call

        # Build CARLA command
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

    print(f"\nDone. Generated {total} scenarios.")


if __name__ == "__main__":
    main()
