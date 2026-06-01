"""
ISO 34503:2023-Grounded Capability Variables

Defines the three ODD-level capability variables used in BayScen, grounded in
ISO 34503:2023. Each variable groups concrete CARLA parameters by the ADS
capability they degrade, enabling exhaustive combinatorial testing over a
compact set of genuinely distinct degradation regimes.

References:
    Paper Section II-C: ODD-Grounded Capability Abstraction
    ISO 34503:2023 §8.2, §9.3.7, §10.2.3, §10.3, §10.4
"""

from typing import Dict, List, Tuple, Optional
from itertools import product


# ============================================================================
# CAPABILITY VARIABLE DEFINITIONS
# ============================================================================

class CapabilityVariable:
    """Base class for a capability variable (leaf node in the BN)."""

    def __init__(self, name: str, values: List, description: str, iso_clause: str):
        self.name = name
        self.values = values
        self.description = description
        self.iso_clause = iso_clause
        self.parents: List[Tuple] = []

    def __repr__(self):
        return f"CapabilityVariable('{self.name}', values={self.values})"


class SensorPerception(CapabilityVariable):
    """
    Sensor Perception (a_perc) — ISO §10.2.4, §10.3, §10.4

    Groups parameters that degrade optical perception and illumination.
    Uniform-weight aggregation over all contributing parents.

    Parents (Scenario 1 — Vehicle-Vehicle):
        Fog_Density     (inverse)   §10.3   atmospheric obscuration
        Fog_Distance    (normal)    §10.3   visibility range
        Cloudiness      (inverse)   §10.4c  illumination reduction
        Precipitation   (inverse)   §10.2.4 optical degradation

    Additional parent (Scenario 2 & 3):
        Sun_Altitude_Angle (normal) §10.4d  solar elevation / lighting

    Degradation scale: 0 (maximally degraded) → 100 (undegraded / clear)
    """

    def __init__(self):
        super().__init__(
            name='Sensor_Perception',
            values=[0, 20, 40, 60, 80, 100],
            description='Combined sensor perception capability (optical systems)',
            iso_clause='ISO 34503 §10.2.4, §10.3, §10.4'
        )
        # Base parents for Scenario 1 — equal weights = 1/4
        self.parents = [
            ('Fog_Density',   'inverse', 0.25),
            ('Fog_Distance',  'normal',  0.25),
            ('Cloudiness',    'inverse', 0.25),
            ('Precipitation', 'inverse', 0.25),
        ]
        # Sun_Altitude_Angle added for Scenarios 2 & 3 — equal weights = 1/5
        self.parents_with_sun = [
            ('Fog_Density',        'inverse', 0.20),
            ('Fog_Distance',       'normal',  0.20),
            ('Cloudiness',         'inverse', 0.20),
            ('Precipitation',      'inverse', 0.20),
            ('Sun_Altitude_Angle', 'normal',  0.20),
        ]
        self.cardinality = 6

    def get_parents_for_scenario(self, available_variables: List[str]) -> List[Tuple]:
        """Return parent list based on available variables (S2/S3 include Sun_Altitude_Angle)."""
        if 'Sun_Altitude_Angle' in available_variables:
            return self.parents_with_sun
        return self.parents


class SurfaceTraction(CapabilityVariable):
    """
    Surface Traction (a_trac) — ISO §9.3.7

    Groups parameters that induce surface conditions affecting vehicle-road traction.
    Uniform-weight aggregation over all contributing parents.

    Parents (all scenarios):
        Road_Friction           (normal)    §9.3.7   baseline friction coefficient
        Wetness                 (inverse)   §9.3.7   induced wet-surface state
        Precipitation_Deposits  (inverse)   §9.3.7   surface contamination

    Degradation scale: 0 (maximally degraded / ice) → 100 (undegraded / dry)
    """

    def __init__(self):
        super().__init__(
            name='Surface_Traction',
            values=[0, 20, 40, 60, 80, 100],
            description='Vehicle-road traction capability (braking and cornering)',
            iso_clause='ISO 34503 §9.3.7'
        )
        # Equal weights = 1/3
        self.parents = [
            ('Road_Friction',          'normal',  1/3),
            ('Wetness',                'inverse', 1/3),
            ('Precipitation_Deposits', 'inverse', 1/3),
        ]
        self.cardinality = 6


class LateralStability(CapabilityVariable):
    """
    Lateral Stability (a_stab) — ISO §10.2.3

    Maps wind intensity to lateral aerodynamic loading on the vehicle.
    Single parent: weight = 1.0.

    Parent:
        Wind_Intensity  (inverse)   §10.2.3  lateral aerodynamic loading

    Degradation scale: 0 (maximally degraded / storm) → 100 (undegraded / calm)
    """

    def __init__(self):
        super().__init__(
            name='Lateral_Stability',
            values=[0, 20, 40, 60, 80, 100],
            description='Vehicle lateral stability under aerodynamic loading',
            iso_clause='ISO 34503 §10.2.3'
        )
        self.parents = [
            ('Wind_Intensity', 'inverse', 1.0),
        ]
        self.cardinality = 6


class ConflictGeometry(CapabilityVariable):
    """
    Conflict Geometry (g) — scenario-level abstraction (junction scenarios only)

    Reduces the 81 ego × adversary start-goal combinations to 3 topologically
    distinct trajectory intersection states (g1, g2, g3).

    States:
        g1: right-lane conflict point
        g2: centre-junction conflict point
        g3: left-lane conflict point

    This variable is NOT used for S3 (cut-in); the binary cut-in direction
    (Left/Right) is retained as a concrete variable.
    """

    def __init__(self):
        super().__init__(
            name='Conflict_Geometry',
            values=['g1', 'g2', 'g3'],
            description='Trajectory intersection topology (junction scenarios)',
            iso_clause='Scenario-level geometric abstraction (not ISO-derived)'
        )
        # Deterministic parents
        self.parents = [
            ('Start_Ego',   'deterministic'),
            ('Goal_Ego',    'deterministic'),
            ('Start_Other', 'deterministic'),
            ('Goal_Other',  'deterministic'),
        ]
        self.cardinality = 3

    @staticmethod
    def define_conflict_logic() -> Dict[Tuple, Optional[str]]:
        """
        Build the deterministic mapping from trajectory combinations to conflict
        geometry states.

        Junction layout (viewed from above):
                Base (north arm)
                    |
            Left ---+--- Right

        Each vehicle follows a path from start → goal. The conflict geometry is
        determined by which collision point the two paths share:
            g1: right lane  (Right side of Left–Right road)
            g2: centre       (junction centre, Base meets main road)
            g3: left lane   (Left side of Left–Right road)
        """
        locations = ['Left', 'Right', 'Base']

        # Which geometry points each path traverses
        path_geometries = {
            ('Left',  'Right'): ['g1', 'g3'],
            ('Right', 'Left'):  ['g2'],
            ('Base',  'Left'):  ['g1', 'g2'],
            ('Base',  'Right'): ['g1'],
            ('Left',  'Base'):  ['g3'],
            ('Right', 'Base'):  ['g2', 'g3'],
        }

        conflict_rules: Dict[Tuple, Optional[str]] = {}

        for start_ego, goal_ego, start_other, goal_other in product(locations, repeat=4):
            # Invalid: vehicle doesn't move or both start at same location
            if start_ego == goal_ego or start_other == goal_other or start_ego == start_other:
                conflict_rules[(start_ego, goal_ego, start_other, goal_other)] = None
                continue

            ego_pts   = path_geometries.get((start_ego,   goal_ego),   [])
            other_pts = path_geometries.get((start_other, goal_other), [])

            common = set(ego_pts) & set(other_pts)
            if common:
                conflict_rules[(start_ego, goal_ego, start_other, goal_other)] = sorted(common)[0]
            else:
                conflict_rules[(start_ego, goal_ego, start_other, goal_other)] = None

        return conflict_rules


# ============================================================================
# CAPABILITY VARIABLE REGISTRY
# ============================================================================

SENSOR_PERCEPTION  = SensorPerception()
SURFACE_TRACTION   = SurfaceTraction()
LATERAL_STABILITY  = LateralStability()
CONFLICT_GEOMETRY  = ConflictGeometry()

CAPABILITY_VARIABLES = {
    'Sensor_Perception': SENSOR_PERCEPTION,
    'Surface_Traction':  SURFACE_TRACTION,
    'Lateral_Stability': LATERAL_STABILITY,
    'Conflict_Geometry': CONFLICT_GEOMETRY,
}

# Leaf nodes for the BayScen generator — junction scenarios (S1, S2)
# Combinatorial space: 6 × 6 × 6 × 3 = 648 configurations
LEAF_NODES = {
    'Conflict_Geometry': ['g1', 'g2', 'g3'],
    'Sensor_Perception': [0, 20, 40, 60, 80, 100],
    'Surface_Traction':  [0, 20, 40, 60, 80, 100],
    'Lateral_Stability': [0, 20, 40, 60, 80, 100],
}

# Leaf nodes for S3 (cut-in) — no Conflict_Geometry
# Combinatorial space: 6 × 6 × 6 = 216 configurations
LEAF_NODES_S3 = {
    'Sensor_Perception': [0, 20, 40, 60, 80, 100],
    'Surface_Traction':  [0, 20, 40, 60, 80, 100],
    'Lateral_Stability': [0, 20, 40, 60, 80, 100],
}


# ============================================================================
# UTILITY
# ============================================================================

def print_capability_variable_info():
    """Print a human-readable summary of all capability variables."""
    print("=" * 70)
    print("BAYSCEN CAPABILITY VARIABLES (ISO 34503:2023-grounded)")
    print("=" * 70)

    for var in [SENSOR_PERCEPTION, SURFACE_TRACTION, LATERAL_STABILITY, CONFLICT_GEOMETRY]:
        print(f"\n{var.name}")
        print(f"  ISO grounding : {var.iso_clause}")
        print(f"  States        : {var.values}  (K={var.cardinality})")
        print(f"  Description   : {var.description}")
        print(f"  Parents:")
        for p in var.parents:
            if len(p) == 3:
                name, rel, w = p
                print(f"    {name:30s}  ({rel:8s}  weight={w:.4f})")
            else:
                name, rel = p
                print(f"    {name:30s}  ({rel})")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    print_capability_variable_info()

    print("\n\nScenario Combinatorial Spaces:")
    print("=" * 70)
    from itertools import product as iproduct
    s12_combos = list(iproduct(*LEAF_NODES.values()))
    s3_combos  = list(iproduct(*LEAF_NODES_S3.values()))
    print(f"  S1/S2 (junction): {len(s12_combos)} configurations "
          f"({' × '.join(str(len(v)) for v in LEAF_NODES.values())})")
    print(f"  S3  (cut-in)    : {len(s3_combos)} configurations "
          f"({' × '.join(str(len(v)) for v in LEAF_NODES_S3.values())})")

    print("\n\nConflict geometry mapping (sample):")
    print("=" * 70)
    rules = ConflictGeometry.define_conflict_logic()
    conflicts = {k: v for k, v in rules.items() if v is not None}
    print(f"  Conflicting trajectory combinations : {len(conflicts)}")
    print(f"  Non-conflicting (None)              : {sum(1 for v in rules.values() if v is None)}")
    for geom in ['g1', 'g2', 'g3']:
        count = sum(1 for v in rules.values() if v == geom)
        print(f"  Conflict geometry {geom}              : {count} combinations")
