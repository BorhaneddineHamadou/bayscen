"""
BayScen Test Suite

Tests all core modules that do not require CARLA, trained model files,
or external simulation infrastructure.

Run from the repo root:
    python bayscen/test_bayscen.py
"""

import sys
import traceback
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "abstraction"))
sys.path.insert(0, str(ROOT / "generation"))
sys.path.insert(0, str(ROOT / "modeling"))

PASS = "✓"
FAIL = "✗"
results = []


def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS} {name}")
    except Exception as exc:
        results.append((FAIL, name))
        print(f"  {FAIL} {name}")
        traceback.print_exc()


# ============================================================================
# 1. abstract_variables
# ============================================================================
print("\n── abstract_variables ─────────────────────────────────────────────")

from abstract_variables import (
    SENSOR_PERCEPTION, SURFACE_TRACTION, LATERAL_STABILITY, CONFLICT_GEOMETRY,
    CAPABILITY_VARIABLES, LEAF_NODES, LEAF_NODES_S3, ConflictGeometry,
)


def test_variable_names():
    assert SENSOR_PERCEPTION.name  == 'Sensor_Perception'
    assert SURFACE_TRACTION.name   == 'Surface_Traction'
    assert LATERAL_STABILITY.name  == 'Lateral_Stability'
    assert CONFLICT_GEOMETRY.name  == 'Conflict_Geometry'


def test_leaf_nodes_counts():
    from itertools import product
    s12 = list(product(*LEAF_NODES.values()))
    s3  = list(product(*LEAF_NODES_S3.values()))
    assert len(s12) == 648, f"Expected 648 S1/S2 configs, got {len(s12)}"
    assert len(s3)  == 216, f"Expected 216 S3 configs, got {len(s3)}"


def test_uniform_weights():
    # Surface Traction: 3 parents, each 1/3
    for _, rel, w in SURFACE_TRACTION.parents:
        assert abs(w - 1/3) < 1e-9, f"Weight {w} != 1/3"
    # Lateral Stability: 1 parent, weight 1.0
    assert LATERAL_STABILITY.parents[0][2] == 1.0
    # Sensor Perception S1: 4 parents, each 0.25
    for _, rel, w in SENSOR_PERCEPTION.parents:
        assert abs(w - 0.25) < 1e-9, f"Weight {w} != 0.25"
    # Sensor Perception with sun: 5 parents, each 0.20
    for _, rel, w in SENSOR_PERCEPTION.parents_with_sun:
        assert abs(w - 0.20) < 1e-9, f"Weight {w} != 0.20"


def test_conflict_geometry_states():
    assert CONFLICT_GEOMETRY.values == ['g1', 'g2', 'g3']
    rules = ConflictGeometry.define_conflict_logic()
    conflict_count = sum(1 for v in rules.values() if v is not None)
    none_count     = sum(1 for v in rules.values() if v is None)
    assert conflict_count > 0, "No conflicting trajectories found"
    assert none_count     > 0, "No non-conflicting trajectories found"
    assert conflict_count + none_count == 81, f"Expected 81 total, got {conflict_count+none_count}"
    assert all(v in ['g1', 'g2', 'g3', None] for v in rules.values()), \
        "Unexpected geometry state"


def test_sensor_perception_parents_scenario():
    s1_parents = SENSOR_PERCEPTION.get_parents_for_scenario(
        ['Cloudiness', 'Wind_Intensity', 'Precipitation', 'Fog_Density', 'Road_Friction', 'Fog_Distance', 'Wetness', 'Precipitation_Deposits']
    )
    s2_parents = SENSOR_PERCEPTION.get_parents_for_scenario(
        ['Sun_Altitude_Angle', 'Cloudiness', 'Wind_Intensity', 'Precipitation',
         'Fog_Density', 'Road_Friction', 'Fog_Distance', 'Wetness', 'Precipitation_Deposits']
    )
    assert len(s1_parents) == 4, f"S1 should have 4 parents, got {len(s1_parents)}"
    assert len(s2_parents) == 5, f"S2 should have 5 parents, got {len(s2_parents)}"
    assert all(abs(w - 0.25) < 1e-9 for _, _, w in s1_parents)
    assert all(abs(w - 0.20) < 1e-9 for _, _, w in s2_parents)


test("Variable names are correct",           test_variable_names)
test("LEAF_NODES counts (648 / 216)",        test_leaf_nodes_counts)
test("Uniform weights in all capability vars", test_uniform_weights)
test("Conflict geometry states g1/g2/g3",   test_conflict_geometry_states)
test("Sensor perception adapts to scenario", test_sensor_perception_parents_scenario)


# ============================================================================
# 2. mapping_functions
# ============================================================================
print("\n── mapping_functions ──────────────────────────────────────────────")

from mapping_functions import (
    map_road_friction_to_standard,
    map_sun_altitude_angle_to_standard,
    map_time_of_day_to_standard,
    map_standard_to_road_friction,
    MAP_TO_STANDARD,
    validate_mappings,
    convert_to_standard,
    is_standard_scale,
    STANDARD_VALUES,
)


def test_road_friction_mapping():
    assert map_road_friction_to_standard(0.1) == 20
    assert map_road_friction_to_standard(0.4) == 60
    assert map_road_friction_to_standard(0.8) == 100
    assert map_road_friction_to_standard(1.0) == 100
    # Round-trip
    for orig in [0.1, 0.2, 0.4, 0.8, 1.0]:
        std = map_road_friction_to_standard(orig)
        rev = map_standard_to_road_friction(std)
        assert orig in rev


def test_sun_altitude_mapping():
    assert map_sun_altitude_angle_to_standard(-90) == 0
    assert map_sun_altitude_angle_to_standard(0)   == 60
    assert map_sun_altitude_angle_to_standard(60)  == 100
    # Backward-compat alias
    assert map_time_of_day_to_standard(-30) == 40


def test_mapping_registry_has_both_names():
    assert 'Road_Friction'      in MAP_TO_STANDARD
    assert 'Sun_Altitude_Angle' in MAP_TO_STANDARD
    assert 'Time_of_Day'        in MAP_TO_STANDARD  # backward compat


def test_standard_scale_detection():
    assert not is_standard_scale('Road_Friction')
    assert not is_standard_scale('Sun_Altitude_Angle')
    assert is_standard_scale('Cloudiness')
    assert is_standard_scale('Precipitation')


def test_validate_mappings():
    validate_mappings()  # should not raise


test("Road_Friction mapping & round-trip",    test_road_friction_mapping)
test("Sun_Altitude_Angle mapping",            test_sun_altitude_mapping)
test("Registry contains both naming variants", test_mapping_registry_has_both_names)
test("Standard scale detection",              test_standard_scale_detection)
test("validate_mappings() passes",            test_validate_mappings)


# ============================================================================
# 3. abstraction_cpd — uniform-weight aggregation
# ============================================================================
print("\n── abstraction_cpd ────────────────────────────────────────────────")

from abstraction_cpd import (
    compute_capability_cpd,
    create_conflict_geometry_cpd,
)


def _make_mock_model():
    """Build a minimal pgmpy BN to test CPD computation."""
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD

    model = BayesianNetwork([('Fog_Density', 'Fog_Distance')])

    # Fog_Density: uniform over [0,20,40,60,80,100]
    cpd_fd = TabularCPD(
        'Fog_Density', 6,
        values=[[1/6], [1/6], [1/6], [1/6], [1/6], [1/6]],
        state_names={'Fog_Density': [0, 20, 40, 60, 80, 100]},
    )

    # Fog_Distance | Fog_Density (6×6 uniform conditional)
    vals = np.full((6, 6), 1/6)
    cpd_dist = TabularCPD(
        'Fog_Distance', 6, vals,
        evidence=['Fog_Density'], evidence_card=[6],
        state_names={
            'Fog_Distance': [0, 20, 40, 60, 80, 100],
            'Fog_Density':  [0, 20, 40, 60, 80, 100],
        },
    )

    model.add_cpds(cpd_fd, cpd_dist)
    assert model.check_model()
    return model


def test_compute_capability_cpd_columns_sum_to_one():
    model   = _make_mock_model()
    parents = [
        ('Fog_Density',  'inverse', 0.5),
        ('Fog_Distance', 'normal',  0.5),
    ]
    cpd = compute_capability_cpd('Sensor_Perception', parents, model)
    # Every column must sum to 1
    assert np.allclose(cpd.values.reshape(6, -1).sum(axis=0), 1.0), \
        "CPD columns do not sum to 1"


def test_compute_capability_cpd_hard_assignment():
    """Each column should be a one-hot (deterministic)."""
    model = _make_mock_model()
    parents = [
        ('Fog_Density',  'inverse', 0.5),
        ('Fog_Distance', 'normal',  0.5),
    ]
    cpd = compute_capability_cpd('Sensor_Perception', parents, model)
    col_matrix = cpd.values.reshape(6, -1)
    # Each column should have exactly one non-zero entry
    for col_idx in range(col_matrix.shape[1]):
        nonzero = np.count_nonzero(col_matrix[:, col_idx])
        assert nonzero == 1, f"Column {col_idx} has {nonzero} non-zero entries (expected 1)"


def test_create_conflict_geometry_cpd():
    rules = ConflictGeometry.define_conflict_logic()
    cpd   = create_conflict_geometry_cpd(rules)
    col_matrix = cpd.values.reshape(4, -1)
    assert np.allclose(col_matrix.sum(axis=0), 1.0), "Conflict geometry CPD columns != 1"


test("compute_capability_cpd: columns sum to 1",    test_compute_capability_cpd_columns_sum_to_one)
test("compute_capability_cpd: hard assignment (1-hot)", test_compute_capability_cpd_hard_assignment)
test("create_conflict_geometry_cpd: columns sum to 1",  test_create_conflict_geometry_cpd)


# ============================================================================
# 4. evaluation_metrics — physical plausibility
# ============================================================================
print("\n── evaluation_metrics (physical plausibility) ─────────────────────")

from evaluation_metrics import (
    check_physical_plausibility,
    physical_plausibility_summary,
    clean_critical_rate,
)


def _make_demo_df():
    np.random.seed(0)
    n = 300
    return pd.DataFrame({
        'Fog_Density':            np.random.choice([0,20,40,60,80,100], n),
        'Fog_Distance':           np.random.choice([0,20,40,60,80,100], n),
        'Cloudiness':             np.random.choice([0,20,40,60,80,100], n),
        'Precipitation':          np.random.choice([0,20,40,60,80,100], n),
        'Precipitation_Deposits': np.random.choice([0,20,40,60,80,100], n),
        'Wetness':                np.random.choice([0,20,40,60,80,100], n),
        'Road_Friction':          np.random.choice([0.1,0.2,0.4,0.8,1.0], n),
        'Wind_Intensity':         np.random.choice([0,20,40,60,80,100], n),
    })


def test_check_physical_plausibility_columns():
    df     = _make_demo_df()
    checks = check_physical_plausibility(df)
    for col in ['C1','C2','C3a','C3b','C4','C5','C6','any_violation','physically_plausible']:
        assert col in checks.columns, f"Missing column '{col}'"
    assert len(checks) == len(df)


def test_plausibility_summary_keys():
    df  = _make_demo_df()
    s   = physical_plausibility_summary(df)
    for key in ['overall_violation_rate', 'physically_plausible_rate',
                'n_scenarios', 'n_violations', 'n_plausible',
                'C1_violation_rate', 'C4_violation_rate']:
        assert key in s, f"Missing key '{key}'"
    assert s['n_scenarios'] == len(df)
    assert s['n_violations'] + s['n_plausible'] == len(df)


def test_c4_hardcoded_scenario():
    # C4: |L − (100−G)| ≤ ε   where ε=10
    # Perfect: G=40, L=60  → |60 − 60| = 0  (plausible)
    # Violating: G=40, L=0 → |0  − 60| = 60 (violation)
    df = pd.DataFrame({
        'Fog_Density':  [40, 40],
        'Fog_Distance': [60,  0],
        'Cloudiness':   [0,   0],
        'Precipitation':[0,   0],
        'Precipitation_Deposits': [0, 0],
        'Wetness':      [0,   0],
        'Road_Friction':[1.0, 1.0],
        'Wind_Intensity':[0,  0],
    })
    checks = check_physical_plausibility(df, epsilon=10)
    assert not checks.iloc[0]['C4'], "Row 0 should be C4-plausible"
    assert     checks.iloc[1]['C4'], "Row 1 should violate C4"


def test_clean_critical_rate_calculation():
    collision = np.array([True, True, False, True, True], dtype=bool)
    plausible = np.array([True, False, True, True, True], dtype=bool)
    df        = pd.DataFrame({'x': range(5)})
    r = clean_critical_rate(df, collision, plausible)
    assert r['n_critical']      == 4, f"Expected 4 critical, got {r['n_critical']}"
    assert r['n_clean_critical'] == 3, f"Expected 3 clean critical, got {r['n_clean_critical']}"
    assert abs(r['clean_critical_rate'] - 75.0) < 1e-9


test("check_physical_plausibility returns all constraint columns", test_check_physical_plausibility_columns)
test("physical_plausibility_summary has required keys",           test_plausibility_summary_keys)
test("C4 constraint on hardcoded scenario",                       test_c4_hardcoded_scenario)
test("clean_critical_rate calculation",                           test_clean_critical_rate_calculation)


# ============================================================================
# 5. generation_utils — path assignment
# ============================================================================
print("\n── generation_utils ───────────────────────────────────────────────")

from generation_utils import (
    assign_junction_paths,
    validate_scenarios,
    split_by_conflict_geometry,
    export_for_carla,
    PATH_MAPPINGS,
)


def test_path_mappings_keys():
    assert set(PATH_MAPPINGS.keys()) == {'g1', 'g2', 'g3'}
    for geom, paths in PATH_MAPPINGS.items():
        assert len(paths) >= 1, f"{geom} has no path mappings"
        for path in paths:
            assert all(k in path for k in ['Start_Ego','Goal_Ego','Start_Other','Goal_Other'])


def test_assign_junction_paths():
    df = pd.DataFrame({'Conflict_Geometry': ['g1', 'g2', 'g3', 'g1', 'g2']})
    out = assign_junction_paths(df)
    for col in ['Start_Ego','Goal_Ego','Start_Other','Goal_Other']:
        assert col in out.columns, f"Missing '{col}' after path assignment"
    assert out.notnull().all().all(), "Null values found after path assignment"


def test_validate_scenarios_valid():
    df = pd.DataFrame({
        'Fog_Density': [0, 20, 40],
        'Road_Friction': [0.1, 0.8, 1.0],
        'Conflict_Geometry': ['g1', 'g2', 'g3'],
    })
    result = validate_scenarios(df, required_columns=['Fog_Density', 'Road_Friction'])
    assert result['is_valid'], f"Validation failed: {result['issues']}"


def test_split_by_conflict_geometry():
    df = pd.DataFrame({'Conflict_Geometry': ['g1','g1','g2','g3','g3']})
    split = split_by_conflict_geometry(df)
    assert len(split['g1']) == 2
    assert len(split['g2']) == 1
    assert len(split['g3']) == 2


def test_export_for_carla(tmp_path=Path('/tmp')):
    df = pd.DataFrame({
        'Fog_Density': [20, 40],
        'Cloudiness':  [60, 80],
        'Road_Friction':[0.8, 1.0],
        'Wind_Intensity':[20, 40],
        'Precipitation': [0, 20],
        'Precipitation_Deposits':[0, 10],
        'Wetness':[0, 20],
        'Fog_Distance':[80, 60],
        'Conflict_Geometry':['g1','g2'],
        'Start_Ego':['Left','Right'],
        'Goal_Ego':['Right','Left'],
        'Start_Other':['Base','Base'],
        'Goal_Other':['Left','Right'],
    })
    out_path = tmp_path / "test_carla_export.csv"
    export_for_carla(df, str(out_path))
    import pandas as _pd
    carla = _pd.read_csv(str(out_path))
    assert 'scenario_id' in carla.columns
    assert 'fog_density' in carla.columns    # renamed
    assert len(carla) == 2


test("PATH_MAPPINGS keys are g1/g2/g3",          test_path_mappings_keys)
test("assign_junction_paths populates columns",   test_assign_junction_paths)
test("validate_scenarios passes on valid data",   test_validate_scenarios_valid)
test("split_by_conflict_geometry splits correctly", test_split_by_conflict_geometry)
test("export_for_carla renames columns correctly", test_export_for_carla)


# ============================================================================
# 6. generate_scenarios — argument parsing & pipeline init
# ============================================================================
print("\n── generate_scenarios (pipeline init) ─────────────────────────────")

from generate_scenarios import ScenarioGenerationPipeline


def test_pipeline_init_s1():
    p = ScenarioGenerationPipeline(scenario=1, mode='rare')
    assert p.scenario   == 1
    assert p.mode       == 'rare'
    assert p.prefer_rare is True
    assert 'Sensor_Perception' in p.capability_leaf_nodes
    assert 'Conflict_Geometry' in p.capability_leaf_nodes
    assert 'Cut_In_Direction' not in p.concrete_variables


def test_pipeline_init_s2():
    p = ScenarioGenerationPipeline(scenario=2, mode='common')
    assert p.prefer_rare is False
    assert 'Sun_Altitude_Angle' in p.concrete_variables
    assert 'Conflict_Geometry' in p.capability_leaf_nodes


def test_pipeline_init_s3():
    p = ScenarioGenerationPipeline(scenario=3, mode='rare')
    assert 'Sun_Altitude_Angle' in p.concrete_variables
    assert 'Cut_In_Direction'   in p.concrete_variables
    # S3 uses LEAF_NODES_S3 — no Conflict_Geometry
    assert 'Conflict_Geometry'  not in p.capability_leaf_nodes
    # 6×6×6 = 216 configurations
    from itertools import product
    combos = list(product(*p.capability_leaf_nodes.values()))
    assert len(combos) == 216, f"Expected 216 S3 configs, got {len(combos)}"


def test_pipeline_invalid_scenario():
    try:
        ScenarioGenerationPipeline(scenario=4, mode='rare')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


test("Pipeline initializes for S1 (rare)", test_pipeline_init_s1)
test("Pipeline initializes for S2 (common)", test_pipeline_init_s2)
test("Pipeline initializes for S3 (216 configs)", test_pipeline_init_s3)
test("Pipeline rejects invalid scenario number", test_pipeline_invalid_scenario)


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 65)
n_pass = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
print(f"RESULTS: {n_pass}/{len(results)} tests passed, {n_fail} failed")
if n_fail == 0:
    print("All tests PASSED ✓")
else:
    print("Failed tests:")
    for status, name in results:
        if status == FAIL:
            print(f"  {FAIL} {name}")
print("=" * 65)

sys.exit(0 if n_fail == 0 else 1)
