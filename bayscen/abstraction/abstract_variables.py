"""
Abstract Variables Definition

This module defines the four abstract variables used in BayScen for
effect-based parameter grouping and dimension reduction.

References:
    Paper Section IV-A-3: Abstracted Variable Definition
"""

from typing import Dict, List, Tuple
from itertools import product


# ============================================================================
# ABSTRACT VARIABLE DEFINITIONS
# ============================================================================

class AbstractVariable:
    """Base class for abstract variable definition."""
    
    def __init__(self, name: str, values: List, description: str, effect: str):
        self.name = name
        self.values = values
        self.description = description
        self.effect = effect
        self.parents = []  # Will be set in subclasses
        
    def __repr__(self):
        return f"AbstractVariable('{self.name}', values={self.values})"


class CollisionPoint(AbstractVariable):
    """
    Collision Point - Geometric conflict point where paths intersect.
    
    Consolidates 4 intersection parameters (81 combinations → 3 states):
    - StartEgo: {Base, Left, Right}
    - GoalEgo: {Base, Left, Right}  
    - StartOther: {Base, Left, Right}
    - GoalOther: {Base, Left, Right}
    
    Physical Meaning:
    - c1: Head-on or T-bone collision point
    - c2: Side-swipe or angle collision point
    - c3: Rear-end or merge collision point
    """
    
    def __init__(self):
        super().__init__(
            name='Collision_Point',
            values=['c1', 'c2', 'c3'],
            description='Geometric conflict point from trajectory intersection',
            effect='Determines collision geometry and severity'
        )
        self.parents = [
            ('StartEgo', 'deterministic'),
            ('GoalEgo', 'deterministic'),
            ('StartOther', 'deterministic'),
            ('GoalOther', 'deterministic')
        ]
        self.cardinality = 3

    def define_collision_logic():
        """
        Define the collision point logic based on start and goal positions.
        Vehicles drive on the right side of the road.
        
        T-junction layout (viewed from above):
            Base (top/north arm)
                |
        Left ---+--- Right
        
        Collision points (as shown in the image):
        - c1: Right side of the horizontal road (right lane of Left-Right road)
        - c2: Center of junction (where Base meets the main road)
        - c3: Left side of the horizontal road (left lane of Left-Right road)
        
        Some paths pass through multiple collision points.
        """
        
        collision_rules = {}
        
        locations = ['Left', 'Right', 'Base']
        
        # Define which collision point(s) each path uses (some paths pass through multiple points)
        path_collision_points = {
            ('Left', 'Right'): ['c1', 'c3'],    # Going Left→Right, passes through c1 and c3
            ('Right', 'Left'): ['c2'],          # Going Right→Left, passes through c2
            ('Base', 'Left'): ['c1', 'c2'],     # From Base turning to Left, passes through c1 and c2
            ('Base', 'Right'): ['c1'],          # From Base turning to Right, passes through c1
            ('Left', 'Base'): ['c3'],           # From Left turning to Base, passes through c3
            ('Right', 'Base'): ['c2', 'c3'],    # From Right turning to Base, passes through c2 and c3
        }
        
        for start_ego, goal_ego, start_other, goal_other in product(locations, repeat=4):
            # Skip invalid scenarios where start == goal
            if start_ego == goal_ego or start_other == goal_other or start_ego == start_other:
                collision_rules[(start_ego, goal_ego, start_other, goal_other)] = None
                continue
            
            # Get collision points for each vehicle's path
            ego_collision_points = path_collision_points.get((start_ego, goal_ego), [])
            other_collision_points = path_collision_points.get((start_other, goal_other), [])
            
            # Find intersection of collision points (if any)
            common_collision_points = set(ego_collision_points) & set(other_collision_points)
            
            if common_collision_points:
                # They collide at one or more points
                # If multiple common points, choose the first one encountered (arbitrary but consistent)
                collision_point = sorted(common_collision_points)[0]
                collision_rules[(start_ego, goal_ego, start_other, goal_other)] = collision_point
            else:
                # Paths don't intersect
                collision_rules[(start_ego, goal_ego, start_other, goal_other)] = None
        
        return collision_rules


class Visibility(AbstractVariable):
    """
    Visibility - Combined visibility degradation affecting perception.
    
    Consolidates 3-4 parameters depending on scenario:
    - Fog_Density, Fog_Distance, Precipitation (always)
    - Time_of_Day (Scenario 2 only - lighting conditions)
    
    Aggregation (Scenario 1):
        Visibility = 0.45 * Fog_Distance + 0.45 * (100 - Fog_Density) + 0.10 * (100 - Precipitation)
    
    Aggregation (Scenario 2 with Time_of_Day):
        Visibility = 0.40 * Fog_Distance + 0.40 * (100 - Fog_Density) + 0.10 * (100 - Precipitation) + 0.10 * Time_of_Day
    """
    
    def __init__(self):
        super().__init__(
            name='Visibility',
            values=[0, 20, 40, 60, 80, 100],
            description='Sensor detection range and object recognition capability',
            effect='Affects perception system: camera, lidar, radar range'
        )
        # Base parents (always present) - weights for Scenario 1
        self.parents = [
            ('Fog_Density', 'inverse', 0.45),
            ('Fog_Distance', 'normal', 0.45),
            ('Precipitation', 'inverse', 0.10)
        ]
        # Optional parent (added for Scenario 2)
        self.optional_parents = [
            ('Time_of_Day', 'normal', 0.10)  # Lighting conditions
        ]
        self.cardinality = 6
    
    def get_parents_for_scenario(self, available_variables):
        """
        Get parents adjusted for available variables.
        
        Args:
            available_variables: List of variables in the base model
            
        Returns:
            List of (parent, relationship, weight) tuples
        """
        # Check if Time_of_Day is available
        has_time_of_day = 'Time_of_Day' in available_variables
        
        if has_time_of_day:
            # Scenario 2: Include Time_of_Day, adjust fog weights
            return [
                ('Fog_Density', 'inverse', 0.40),
                ('Fog_Distance', 'normal', 0.40),
                ('Precipitation', 'inverse', 0.10),
                ('Time_of_Day', 'normal', 0.10)
            ]
        else:
            # Scenario 1: Original weights
            return self.parents


class RoadSurface(AbstractVariable):
    """
    Road Surface - Combined traction effect from multiple factors.
    
    Consolidates 3 surface parameters:
    - Road_Friction: {0.1, 0.2, 0.4, 0.8, 1.0}
    - Wetness: {0, 20, 40, 60, 80, 100}
    - Precipitation_Deposits: {0, 20, 40, 60, 80, 100}
    
    Aggregation:
        Road_Surface = 0.60 * normalize(Road_Friction) + 
                       0.20 * (100 - Wetness) + 
                       0.20 * (100 - Precipitation_Deposits)
    
    Physical Meaning:
    - 0: Ice/black ice (no traction)
    - 20: Heavy snow/deep water
    - 40: Slush or wet leaves
    - 60: Wet asphalt
    - 80: Damp but good traction
    - 100: Dry asphalt (optimal)
    """
    
    def __init__(self):
        super().__init__(
            name='Road_Surface',
            values=[0, 20, 40, 60, 80, 100],
            description='Vehicle-road traction dynamics and braking capability',
            effect='Affects control system: maximum braking force, cornering limits'
        )
        self.parents = [
            ('Road_Friction', 'normal', 0.60),              # Higher friction → better surface
            ('Wetness', 'inverse', 0.20),                   # Water → worse surface
            ('Precipitation_Deposits', 'inverse', 0.20)     # Deposits → worse surface
        ]
        self.cardinality = 6


class VehicleStability(AbstractVariable):
    """
    Vehicle Stability - External destabilizing forces.
    
    Consolidates 2 parameters:
    - Wind_Intensity: {0, 20, 40, 60, 80, 100}
    - Road_Friction: {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}
    
    Aggregation:
        Vehicle_Stability = 0.20 * (100 - Wind_Intensity) + 0.80 * normalize(Road_Friction)
    
    Physical Meaning:
    - 0: Extreme (hurricane-force winds)
    - 20: Severe crosswinds
    - 40: Strong wind buffeting
    - 60: Moderate wind effects
    - 80: Light wind
    - 100: Calm (no wind effects)
    """
    
    def __init__(self):
        super().__init__(
            name='Vehicle_Stability',
            values=[0, 20, 40, 60, 80, 100],
            description='External destabilizing forces on vehicle trajectory',
            effect='Affects control system: trajectory deviation, stability margins'
        )
        self.parents = [
            ('Wind_Intensity', 'inverse', 0.2),  # Wind reduces stability
            ('Road_Friction', 'normal', 0.8) 
        ]
        self.cardinality = 6


# ============================================================================
# ABSTRACT VARIABLE REGISTRY
# ============================================================================

# Instantiate all abstract variables
COLLISION_POINT = CollisionPoint()
VISIBILITY = Visibility()
ROAD_SURFACE = RoadSurface()
VEHICLE_STABILITY = VehicleStability()

# Registry for easy access
ABSTRACT_VARIABLES = {
    'Collision_Point': COLLISION_POINT,
    'Visibility': VISIBILITY,
    'Road_Surface': ROAD_SURFACE,
    'Vehicle_Stability': VEHICLE_STABILITY
}

# Leaf nodes configuration for BayScen generator
LEAF_NODES = {
    'Collision_Point': ['c1', 'c2', 'c3'],
    'Visibility': [0, 20, 40, 60, 80, 100],
    'Road_Surface': [0, 20, 40, 60, 80, 100],
    'Vehicle_Stability': [0, 20, 40, 60, 80, 100]
}


def print_abstract_variable_info():
    """Print information about all abstract variables."""
    print("=" * 70)
    print("ABSTRACT VARIABLES - BAYSCEN")
    print("=" * 70)
    
    for var in ABSTRACT_VARIABLES.values():
        print(f"\n{var.name}")
        print("-" * 70)
        print(f"Values: {var.values} (cardinality: {var.cardinality})")
        print(f"Description: {var.description}")
        print(f"Effect: {var.effect}")
        print(f"\nParents:")
        for parent in var.parents:
            if len(parent) == 3:  # Has weight
                name, relationship, weight = parent
                print(f"  - {name} ({relationship}, weight={weight})")
            else:  # No weight (deterministic)
                name, relationship = parent
                print(f"  - {name} ({relationship})")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Print all abstract variable information
    print_abstract_variable_info()
    
    # Example: Access specific variable
    print("\n\nExample: Accessing Visibility variable")
    print("=" * 70)
    visibility = ABSTRACT_VARIABLES['Visibility']
    print(f"Name: {visibility.name}")
    print(f"Values: {visibility.values}")
    print(f"Parents: {[p[0] for p in visibility.parents]}")
    print(f"Weights: {[p[2] for p in visibility.parents if len(p) == 3]}")
    
    # Example: Get leaf nodes for generator
    print("\n\nExample: Leaf nodes for BayScen generator")
    print("=" * 70)
    print("LEAF_NODES = {")
    for name, values in LEAF_NODES.items():
        print(f"    '{name}': {values},")
    print("}")
    
    # Example: Calculate number of abstract combinations
    print("\n\nExample: Abstract combinations")
    print("=" * 70)
    from itertools import product
    combinations = list(product(*LEAF_NODES.values()))
    print(f"Total abstract configurations: {len(combinations)}")
    print(f"First 5 combinations:")
    for i, combo in enumerate(combinations[:5]):
        config = dict(zip(LEAF_NODES.keys(), combo))
        print(f"  {i+1}. {config}")