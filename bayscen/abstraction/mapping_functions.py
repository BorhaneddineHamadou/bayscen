"""
Mapping Functions for Non-Standard Variable Scales

Converts variable values that do not use the standard [0, 20, 40, 60, 80, 100]
scale to that scale, as required when computing capability variable CPDs.

Variables requiring mapping:
    Road_Friction      : [0.1, 0.2, 0.4, 0.8, 1.0] → [0, 20, 40, 60, 80, 100]
    Sun_Altitude_Angle : [-90, -60, -30, 0, 30, 60, 90] → [0, 20, 40, 60, 80, 100]
        (sun altitude angle in degrees; -90 = midnight, 90 = solar noon)
        Also aliased as 'Time_of_Day' for backward compatibility with existing data.
"""

from typing import Optional, List


# ============================================================================
# STANDARD SCALE
# ============================================================================

STANDARD_VALUES = [0, 20, 40, 60, 80, 100]


# ============================================================================
# ROAD FRICTION MAPPINGS
# ============================================================================

def map_road_friction_to_standard(value: float) -> Optional[int]:
    """
    Map Road_Friction coefficient to standard 0–100 scale.

    CARLA values → standard scale:
        0.1 → 20  (near-zero friction, black ice)
        0.2 → 40  (poor friction, heavy snow/standing water)
        0.4 → 60  (moderate friction, wet asphalt)
        0.8 → 100 (good friction, damp/dry)
        1.0 → 100 (maximum friction, dry asphalt)

    Note: 0.8 and 1.0 both map to 100 — reverse mapping is ambiguous.
    """
    mapping = {
        0.1: 20,
        0.2: 40,
        0.4: 60,
        0.8: 100,
        1.0: 100,
    }
    return mapping.get(round(value, 4), None)


def map_standard_to_road_friction(value: int) -> Optional[List[float]]:
    """
    Map standard scale back to Road_Friction values (returns list for ambiguity).

    100 maps to both 0.8 and 1.0.
    """
    reverse = {
        20:  [0.1],
        40:  [0.2],
        60:  [0.4],
        100: [0.8, 1.0],
    }
    return reverse.get(value, None)


# ============================================================================
# SUN ALTITUDE ANGLE MAPPINGS  (ISO §10.4(d); also called Time_of_Day)
# ============================================================================

def map_sun_altitude_angle_to_standard(value: int) -> Optional[int]:
    """
    Map sun altitude angle (degrees) to standard 0–100 scale.

    Physical interpretation (lighting quality for optical sensors):
        -90° → 0   midnight (darkest — worst perception)
        -60° → 20  deep night
        -30° → 40  pre-dawn / post-dusk
          0° → 60  sunrise / sunset
         30° → 80  morning / afternoon sun
         60° → 100 midday (brightest — best perception)
         90° → 100 solar zenith (aliased to 60°)

    Note: 60° and 90° both map to 100 — reverse mapping is ambiguous.
    """
    mapping = {
        -90: 0,
        -60: 20,
        -30: 40,
          0: 60,
         30: 80,
         60: 100,
         90: 100,
    }
    return mapping.get(value, None)


# Backward-compatibility alias
map_time_of_day_to_standard = map_sun_altitude_angle_to_standard


def map_standard_to_sun_altitude_angle(value: int) -> Optional[List[int]]:
    """
    Map standard scale back to sun altitude angle in degrees.

    100 maps to both 60° and 90°.
    """
    reverse = {
          0: [-90],
         20: [-60],
         40: [-30],
         60: [0],
         80: [30],
        100: [60, 90],
    }
    return reverse.get(value, None)


# Backward-compatibility alias
map_standard_to_time_of_day = map_standard_to_sun_altitude_angle


# ============================================================================
# MAPPING REGISTRY
# ============================================================================

# Forward: original scale → standard [0, 20, 40, 60, 80, 100]
MAP_TO_STANDARD = {
    'Road_Friction':      map_road_friction_to_standard,
    'Sun_Altitude_Angle': map_sun_altitude_angle_to_standard,
    'Time_of_Day':        map_sun_altitude_angle_to_standard,  # backward compat
}

# Reverse: standard → original scale (list to handle ambiguous cases)
MAP_TO_ORIGINAL = {
    'Road_Friction':      map_standard_to_road_friction,
    'Sun_Altitude_Angle': map_standard_to_sun_altitude_angle,
    'Time_of_Day':        map_standard_to_sun_altitude_angle,  # backward compat
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_standard_scale(variable_name: str) -> bool:
    """Return True if the variable uses the standard 0–100 scale (no mapping needed)."""
    return variable_name not in MAP_TO_STANDARD


def convert_to_standard(variable_name: str, value) -> Optional[int]:
    """Convert a value to standard scale; returns original if no mapping required."""
    fn = MAP_TO_STANDARD.get(variable_name)
    return fn(value) if fn else value


def convert_from_standard(variable_name: str, value: int) -> List:
    """Convert standard scale back to original; returns [value] if no mapping."""
    fn = MAP_TO_ORIGINAL.get(variable_name)
    if fn:
        result = fn(value)
        return result if result is not None else [value]
    return [value]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_mappings():
    """Validate forward–inverse round-trip consistency for all mappings."""
    print("Validating mapping functions...")

    print("\n  Road_Friction:")
    for orig in [0.1, 0.2, 0.4, 0.8, 1.0]:
        std = map_road_friction_to_standard(orig)
        rev = map_standard_to_road_friction(std)
        assert orig in rev, f"Round-trip failed for Road_Friction={orig}"
        print(f"    {orig} → {std} → {rev}")

    print("\n  Sun_Altitude_Angle:")
    for orig in [-90, -60, -30, 0, 30, 60, 90]:
        std = map_sun_altitude_angle_to_standard(orig)
        rev = map_standard_to_sun_altitude_angle(std)
        assert orig in rev, f"Round-trip failed for Sun_Altitude_Angle={orig}"
        print(f"    {orig:4d}° → {std:3d} → {rev}°")

    print("\n✓ All mappings validated successfully")


if __name__ == "__main__":
    validate_mappings()
