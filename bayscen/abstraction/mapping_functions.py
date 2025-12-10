"""
Mapping Functions for Non-Standard Variables

This module provides mapping functions to convert non-standard parameter scales
to the standard [0, 20, 40, 60, 80, 100] scale used by abstracted variables.

Required for:
- Road_Friction: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0] → [0, 20, 40, 60, 80, 100, 100]
- Time_of_Day: [-90, -60, -30, 0, 30, 60, 90] → [0, 20, 40, 60, 80, 100, 100]

Note: Some mappings are ambiguous (e.g., 0.8 and 1.0 both map to 100).
Reverse mappings return lists to handle ambiguity.

Author: BayScen Team
Date: December 2025
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
    Map Road_Friction values to standard 0-100 scale.
    
    Physical interpretation:
    - 0.0 → 0: No friction (ice)
    - 0.1 → 20: Minimal friction (black ice)
    - 0.2 → 40: Poor friction (heavy snow/water)
    - 0.4 → 60: Moderate friction (wet surface)
    - 0.6 → 80: Good friction (damp)
    - 0.8, 1.0 → 100: Excellent friction (dry)
    
    Args:
        value: Road friction coefficient [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    Returns:
        Standard scale value [0, 20, 40, 60, 80, 100], or None if invalid
    
    Example:
        >>> map_road_friction_to_standard(0.2)
        40
        >>> map_road_friction_to_standard(1.0)
        100
    """
    mapping = {
        0.0: 0,
        0.1: 20,
        0.2: 40,
        0.4: 60,
        0.6: 80,
        0.8: 100,
        1.0: 100
    }
    return mapping.get(value, None)


def map_standard_to_road_friction(value: int) -> Optional[List[float]]:
    """
    Map standard 0-100 scale back to Road_Friction values.
    
    Returns a list because the mapping is not one-to-one:
    - 100 can map to either 0.8 or 1.0
    
    Args:
        value: Standard scale value [0, 20, 40, 60, 80, 100]
    
    Returns:
        List of possible Road_Friction values, or None if invalid
    
    Example:
        >>> map_standard_to_road_friction(40)
        [0.2]
        >>> map_standard_to_road_friction(100)
        [0.8, 1.0]
    """
    reverse_mapping = {
        0: [0.0],
        20: [0.1],
        40: [0.2],
        60: [0.4],
        80: [0.6],
        100: [0.8, 1.0]  # Ambiguous: both map to 100
    }
    return reverse_mapping.get(value, None)


# ============================================================================
# TIME OF DAY MAPPINGS (for Scenario 2)
# ============================================================================

def map_time_of_day_to_standard(value: int) -> Optional[int]:
    """
    Map Time_of_Day (sun altitude angle) to standard 0-100 scale.
    
    Physical interpretation:
    - -90° → 0: Midnight (darkest)
    - -60° → 20: Deep night
    - -30° → 40: Pre-dawn/post-dusk
    - 0° → 60: Sunrise/sunset
    - 30° → 80: Morning/afternoon
    - 60°, 90° → 100: Midday (brightest)
    
    Args:
        value: Sun altitude angle in degrees [-90, -60, -30, 0, 30, 60, 90]
    
    Returns:
        Standard scale value [0, 20, 40, 60, 80, 100], or None if invalid
    
    Example:
        >>> map_time_of_day_to_standard(-30)
        40
        >>> map_time_of_day_to_standard(60)
        100
    """
    mapping = {
        -90: 0,
        -60: 20,
        -30: 40,
        0: 60,
        30: 80,
        60: 100,
        90: 100
    }
    return mapping.get(value, None)


def map_standard_to_time_of_day(value: int) -> Optional[List[int]]:
    """
    Map standard 0-100 scale back to Time_of_Day values.
    
    Returns a list because the mapping is not one-to-one:
    - 100 can map to either 60° or 90°
    
    Args:
        value: Standard scale value [0, 20, 40, 60, 80, 100]
    
    Returns:
        List of possible Time_of_Day values (degrees), or None if invalid
    
    Example:
        >>> map_standard_to_time_of_day(40)
        [-30]
        >>> map_standard_to_time_of_day(100)
        [60, 90]
    """
    reverse_mapping = {
        0: [-90],
        20: [-60],
        40: [-30],
        60: [0],
        80: [30],
        100: [60, 90]  # Ambiguous: both map to 100
    }
    return reverse_mapping.get(value, None)


# ============================================================================
# MAPPING REGISTRY
# ============================================================================

# Forward mappings: original → standard
MAP_TO_STANDARD = {
    "Road_Friction": map_road_friction_to_standard,
    "Time_of_Day": map_time_of_day_to_standard
}

# Reverse mappings: standard → original (returns list for ambiguous cases)
MAP_TO_ORIGINAL = {
    "Road_Friction": map_standard_to_road_friction,
    "Time_of_Day": map_standard_to_time_of_day
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_standard_scale(variable_name: str) -> bool:
    """
    Check if a variable uses the standard 0-100 scale.
    
    Args:
        variable_name: Name of the variable
    
    Returns:
        True if variable uses standard scale, False if mapping required
    
    Example:
        >>> is_standard_scale('Precipitation')
        True
        >>> is_standard_scale('Road_Friction')
        False
    """
    return variable_name not in MAP_TO_STANDARD


def convert_to_standard(variable_name: str, value) -> Optional[int]:
    """
    Convert a value to standard scale if mapping exists.
    
    Args:
        variable_name: Name of the variable
        value: Original value
    
    Returns:
        Standard scale value, or original value if no mapping needed
    
    Example:
        >>> convert_to_standard('Road_Friction', 0.8)
        100
        >>> convert_to_standard('Precipitation', 40)
        40
    """
    map_func = MAP_TO_STANDARD.get(variable_name)
    if map_func:
        return map_func(value)
    return value


def convert_from_standard(variable_name: str, value: int) -> List:
    """
    Convert a standard scale value back to original scale.
    
    Returns a list to handle ambiguous mappings.
    
    Args:
        variable_name: Name of the variable
        value: Standard scale value
    
    Returns:
        List of possible original values, or [value] if no mapping needed
    
    Example:
        >>> convert_from_standard('Road_Friction', 100)
        [0.8, 1.0]
        >>> convert_from_standard('Precipitation', 40)
        [40]
    """
    map_func = MAP_TO_ORIGINAL.get(variable_name)
    if map_func:
        result = map_func(value)
        return result if result is not None else [value]
    return [value]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_mappings():
    """
    Validate that all mappings are consistent and complete.
    
    Checks:
    1. All forward mappings have corresponding reverse mappings
    2. Reverse mappings cover all standard values
    3. Round-trip consistency (where not ambiguous)
    """
    print("Validating mapping functions...")
    
    # Check Road_Friction
    print("\n Road_Friction:")
    for orig in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        std = map_road_friction_to_standard(orig)
        reverse = map_standard_to_road_friction(std)
        assert orig in reverse, f"Round-trip failed for {orig}"
        print(f"  {orig} → {std} → {reverse}")
    
    # Check Time_of_Day
    print("\nTime_of_Day:")
    for orig in [-90, -60, -30, 0, 30, 60, 90]:
        std = map_time_of_day_to_standard(orig)
        reverse = map_standard_to_time_of_day(std)
        assert orig in reverse, f"Round-trip failed for {orig}"
        print(f"  {orig}° → {std} → {reverse}°")
    
    print("\n✓ All mappings validated successfully")


if __name__ == "__main__":
    # Run validation
    validate_mappings()