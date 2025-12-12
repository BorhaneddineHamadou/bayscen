#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side using Interfuser."""

from __future__ import print_function


# ==============================================================================
## Google API Import
# ==============================================================================
import os
import pathlib

if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  print("git clone the model")
  #!git clone --depth 1 https://github.com/tensorflow/models


#import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#from IPython.display import display
#from grabscreen import grab_screen
#import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

try: 
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_q

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE)# or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
        # ADDED

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import collections
import datetime
import glob
import math
import random
import re
import sys
import weakref


import carla

from examples.manual_control import (World, HUD, CameraManager, CollisionSensor, LaneInvasionSensor, GnssSensor, IMUSensor)  
# keyboardcontrol class removed
                
from carla import ColorConverter as cc
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# sys append path if compiler doesn't find these imports! (already included in env variables)
from agents.navigation.behavior_agent_z2 import BehaviorAgentZ2  # pylint: disable=import-error
from agents.navigation.roaming_agent_z import RoamingAgentZ  # pylint: disable=import-error
from agents.navigation.basic_agent_z import BasicAgentZ  # pylint: disable=import-error

from agents.navigation.basic_agent import BasicAgent

from team_code.interfuser_agent import InterfuserAgent
from team_code.interfuser_config import GlobalConfig
from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.envs.sensor_interface import SensorInterface


import os
import argparse
import logging
import time
import pygame
import traceback
import numpy as np
import cv2

from srunner.scenariomanager.timer import GameTime



# *********** START from Srunner Manual_control.py ********************************* 

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class WorldSR(World):  # World class inherited here, so the self.args to WORLDSR object are used by the init method of parent class World cx there is no init method here I presume. Yeah I was right Alhmd.

    restarted = False

    def restart(self):  # overwritten parent restart() method

        if self.restarted:
            return
        self.restarted = True

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Image width and height
        self.im_width = 800
        self.im_height = 450
        self.SHOW_CAM = False #True

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")  # yep this is entered.
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')  # all of em.
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":  # set_attribute to hero in carla_data_provider.py, so that role_name attribute has been changed on the server side even!
                    print("Ego vehicle found")
                    self.player = vehicle  # here we got our ego. Spawn RGB sensor/camera here as well
                    break
            self.world.tick()  # tick the world
        
        self.player_name = self.player.type_id  # The identifier of the blueprint this actor was based on, e.g. vehicle.ford.mustang.

        # one camera sensor here
        self.blueprint_library = self.world.get_blueprint_library()
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"100")

        transform = carla.Transform(carla.Location(x=2.5, z=1.0 ))
        self.sensor_cam = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.player)
        self.sensor_cam.listen(lambda data: self.process_img(data))


        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)  # 'Gamma correction of the camera (default: 2.2)'
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def process_img(self, image):
        i = np.array(image.raw_data)

        i2 = i.reshape(self.im_height, self.im_width, 4)  # (1440000,)
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3  #800x450 image

    def tick(self, clock):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            return False

        self.hud.tick(self, clock)
        return True

# *********** END from Srunner Manual_control.py ********************************* 

# Ego Control Agent class

class EgoControlAgent:

    agent = None  # will assign the control agent to this
    visualize = False
    collision_warning = False

    def __init__(self, scenario=1, test_method="test", run_number=3):

        scenario_name = f"scenario{scenario}_interfuser"
        run_number = f"run{run_number}"

        # --- PATH CONFIGURATION ---
        base_path = os.path.dirname(os.path.abspath(__file__))

        # e.g., .../scenario_runner-0.9.10/scenario1_interfuser/outputs/bayscen/run3/
        self.output_dir = os.path.join(base_path, scenario_name, "outputs", test_method, run_number)

        # Create the directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

        # Define file paths dynamically
        self.last_location_file_path = os.path.join(self.output_dir, "last_location.json")
        self.results_file_path = os.path.join(self.output_dir, "run_results.json")
        # --------------------------

        self.args = ArgsOverwrite()
        pygame.init()
        pygame.font.init()
        self.world = None
        self.controller = None
        self.vehicle_states = []

        # Initialize Interfuser components
        self.interfuser_config_path = os.path.join(os.path.dirname(__file__), "config_interfuser.py")
        self.interfuser = InterfuserAgent(self.interfuser_config_path)
        self.sensor_objects = {}
        self.sensor_interface = SensorInterface()

        
    def game_loop_init(self, goal_carla_location, visualize=False):
        self.visualize = visualize

        print("Goal location of EGO vehicle is", goal_carla_location)
        self.goal = goal_carla_location

        try:
            client = carla.Client(self.args.host, self.args.port)
            client.set_timeout(60.0) # It was 4.0

            if self.visualize:
                display = pygame.display.set_mode(
                    (self.args.width, self.args.height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)

            hud = HUD(self.args.width, self.args.height)
            self.world = WorldSR(client.get_world(), hud, self.args) 
            
            # Initialize Interfuser
            if self.interfuser_config_path:
                # self.interfuser.setup(self.interfuser_config_path)
                self.interfuser.setup(self.interfuser_config_path, self.world.player)
                self._setup_interfuser_sensors()

                # Create and set the global plan
                global_plan_gps, global_plan_world_coord = self.create_global_plan(goal_carla_location)
                
                self.interfuser.set_global_plan(global_plan_gps, global_plan_world_coord)

            self.init_last_location(self.last_location_file_path)  # Initialize the last location file

            self.start_system_time = time.time()
        except Exception as e:
            traceback.print_exc()
            print("Could not setup EgoControlAgent due to {}".format(e))

    def create_global_plan(self, goal_location):
        """
        Create a global plan in both GPS and world coordinates
        
        Args:
            goal_location: Carla.Location object representing the target destination
            
        Returns:
            Tuple containing (global_plan_gps, global_plan_world_coord)
        """
        # Get the map
        carla_map = self.world.world.get_map()
        
        # Get start waypoint - ensure we're getting the closest waypoint on a lane
        start_location = self.world.player.get_location()
        start_waypoint = carla_map.get_waypoint(
            start_location,
            project_to_road=True,  # Important: Project to nearest road
            lane_type=carla.LaneType.Driving  # Ensure we're on a driving lane
        )
        
        # Get end waypoint - ensure it's also on a valid road
        end_waypoint = carla_map.get_waypoint(
            goal_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        # Create route planner with smaller resolution for more precise routing
        dao = GlobalRoutePlannerDAO(carla_map, 1.0)  # Reduced resolution to 1.0
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        
        try:
            # Get the route between waypoints
            route = grp.trace_route(
                start_waypoint.transform.location,
                end_waypoint.transform.location
            )
            
            # Convert route to global plan format
            global_plan_gps = []
            global_plan_world_coord = []
            
            for wp, rd in route:
                # Ensure waypoint is valid
                if wp is None:
                    continue
                    
                # For GPS coordinates
                gps_coord = carla_map.transform_to_geolocation(wp.transform.location)
                gps_dict = {
                    "lat": float(gps_coord.latitude),
                    "lon": float(gps_coord.longitude)
                }
                global_plan_gps.append((gps_dict, rd))
                
                # For world coordinates
                global_plan_world_coord.append((wp.transform, rd))
            
            # Verify we have a valid route
            if not global_plan_gps or not global_plan_world_coord:
                raise ValueError("Generated route is empty")
                
            return global_plan_gps, global_plan_world_coord
            
        except Exception as e:
            print(f"Error creating global plan: {str(e)}")
            return None, None

    def _setup_interfuser_sensors(self):
        """Setup the sensors required by Interfuser"""
        # Get sensor specifications from Interfuser
        sensor_specs = self.interfuser.sensors()
        
        # Initialize each sensor based on specifications
        for sensor_spec in sensor_specs:
            sensor_type = sensor_spec["type"]
            sensor_id = sensor_spec["id"]
            
            # Create sensor blueprint
            
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = OpenDriveMapReader(self.world.player, sensor_spec['reading_frequency'])
                self.sensor_objects[sensor_id] = sensor
                

            elif sensor_spec['type'].startswith('sensor.speedometer'):
                print("Setting up speedometer")
                delta_time = 0.05  # it was 0.05 ==> 20 FPS, adjust as needed
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(self.world.player, frame_rate)
                print("Speedometer created:", sensor)
                self.sensor_objects[sensor_id] = sensor
                

            # These are the sensors spawned on the carla world
            else:
                bp = self.world.world.get_blueprint_library().find(sensor_type)
                if sensor_spec['type'].startswith('sensor.camera.semantic_segmentation'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.camera.depth'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar.ray_cast_semantic'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 for old lidar models
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(600000))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 to generate 360 degree LiDAR point cloud
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(600000))
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    bp.set_attribute('dropoff_general_rate', str(0.45))
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])

                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            
                # Spawn and attach sensor
                sensor = self.world.world.spawn_actor(bp, sensor_transform, attach_to=self.world.player)
                self.sensor_objects[sensor_id] = sensor

            if sensor_id not in self.sensor_interface._sensors_objects:
                # Register in SensorInterface with a callback
                callback = CallBack(sensor_id, sensor_type, sensor, self.sensor_interface)
                sensor.listen(callback)

        # Tick once to spawn the sensors
        CarlaDataProvider.get_world().tick()

    def _collect_sensor_data(self):
        """Collect the latest data from all sensors"""
        return self.sensor_interface.get_data()

    def game_loop_step(self):

        input_data = self._collect_sensor_data()
        
        timestamp = GameTime.get_time()

        control = self.interfuser.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        vehicle_state = self.retrieve_vehicle_state()
        self.vehicle_states.append(vehicle_state)

        self.update_last_location(self.last_location_file_path)  # Update the last location file

        self.world.player.apply_control(control) 

    def init_last_location(self, file_path):
        """
        Initialize a JSON file with the current location of self.world.player.
        If file exists, append a new entry.
        If not, create the file and write the entry.
        """
        # Extract current location as a dict
        loc = self.world.player.get_transform().location
        loc_dict = {"x": loc.x, "y": loc.y, "z": loc.z}
        
        entry = {"last_location": loc_dict}
        
        data = []
        
        # Check if file exists and read existing data
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Ensure data is a list
                    if not isinstance(data, list):
                        data = []
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read existing file {file_path}: {e}")
                data = []
        
        # Append new entry
        data.append(entry)
        
        # Write data to file (creates file if it doesn't exist)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Location saved to {file_path}")
        except IOError as e:
            print(f"Error: Could not write to file {file_path}: {e}")
            raise

    def update_last_location(self, file_path):
        """
        Update the last location entry in the JSON file with the current location of self.world.player.
        It replaces the last element's 'last_location' field.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist. Call init_last_location first.")

        loc = self.world.player.get_location()
        loc_dict = {"x": loc.x, "y": loc.y, "z": loc.z}

        with open(file_path, 'r') as f:
            data = json.load(f)

        if not data:
            # If empty list, append first entry
            data.append({"last_location": loc_dict})
        else:
            # Update last element's last_location
            data[-1]["last_location"] = loc_dict

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
      


    def check_collision_hazards(self):
        ego_vehicle = self.world.player
        ego_location = ego_vehicle.get_location()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = 3.6 * np.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)  # km/h

        # Adjust safety distance based on speed
        safety_distance = max(12.0, ego_speed * 0.5)  # Higher safety distance at higher speeds
        
        for vehicle in self.world.world.get_actors().filter('vehicle.*'):
            if vehicle.id != ego_vehicle.id:
                vehicle_location = vehicle.get_location()
                distance = ego_location.distance(vehicle_location)
                
                if distance < safety_distance:
                   
                    self.collision_warning = True
                    return True
        
        self.collision_warning = False
        return False

    def set_route_to_destination(self, destination):
        """
        Set a route for the ego vehicle to the destination using CARLA's GlobalRoutePlanner.
        """
        try:
            # Access the map
            carla_map = self.world.world.get_map()

            # Initialize the GlobalRoutePlannerDAO with a sampling resolution
            dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution=2.0)

            # Create and set up the GlobalRoutePlanner
            grp = GlobalRoutePlanner(dao)
            grp.setup()

            # Retrieve the start and destination locations
            start_location = self.world.player.get_location()

            # Trace the route
            waypoints = grp.trace_route(start_location, destination)

            if not waypoints or len(waypoints) == 0:
                print("Route planning failed: No waypoints generated.")
            else:
                print("Route successfully planned with", len(waypoints), "waypoints.")

            # Visualize the route with debug markers
            for waypoint, road_option in waypoints:
                self.world.world.debug.draw_string(
                    waypoint.transform.location,
                    "O",  # Marker
                    draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0),  # Red
                    life_time=10.0,
                )
            print("Route successfully traced to destination.")
            
        except Exception as e:
            print(f"Error setting route: {e}")

    

    @staticmethod
    def emergency_stop():
        """
        Send an emergency stop command to the vehicle

            :return: control for braking
        """
        
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = True

        return control


    def retrieve_vehicle_state(self):
        """
        Retrieves the current state of the vehicle, including trajectory, orientation, velocity, 
        acceleration, and control inputs.

        :return: A dictionary containing the vehicle state.
        """
        vehicle = self.world.player  # Reference to the ego vehicle

        # Get the vehicle's location (trajectory)
        location = vehicle.get_location()
        trajectory = {'x': location.x, 'y': location.y, 'z': location.z}

        # Get the vehicle's orientation (yaw, pitch, roll)
        rotation = vehicle.get_transform().rotation
        orientation = {'yaw': rotation.yaw, 'pitch': rotation.pitch, 'roll': rotation.roll}

        # Get the vehicle's velocity
        velocity = vehicle.get_velocity()
        linear_velocity = {'x': velocity.x, 'y': velocity.y, 'z': velocity.z}

        # Get the vehicle's angular velocity
        angular_velocity = vehicle.get_angular_velocity()
        angular_velocity_data = {'x': angular_velocity.x, 'y': angular_velocity.y, 'z': angular_velocity.z}

        # Get the vehicle's acceleration
        acceleration = vehicle.get_acceleration()
        linear_acceleration = {'x': acceleration.x, 'y': acceleration.y, 'z': acceleration.z}

        # Get control inputs (throttle, brake, steering)
        control = vehicle.get_control()
        control_inputs = {
            'throttle': control.throttle,
            'brake': control.brake,
            'steer': control.steer,
            'hand_brake': control.hand_brake
        }

        # Combine all data into a single dictionary
        vehicle_state = {
            'trajectory': trajectory,
            'collision_warning': self.collision_warning
        }

        return vehicle_state

    def append_vehicle_states_to_json(self, file_name, collision_occurred, run_duration):
        """
        Append the collected vehicle states to an existing JSON file, 
        creating a list of vehicle states over multiple scenarios.

        :param file_name: The name of the file to append data to.
        :param collision_occurred: Boolean indicating whether a collision occurred in this scenario.
        :param run_duration: Total duration of the scenario run.
        """
        try:
            # Check if the file exists
            if os.path.exists(file_name):
                # If file exists, read the existing data
                with open(file_name, 'r') as json_file:
                    data = json.load(json_file)
            else:
                # If the file does not exist, initialize an empty list
                data = []

            # Create a structured dictionary for this run
            scenario_data = {
                "collision_occurred": collision_occurred,  # Collision status
                "run_duration": run_duration  # Total duration of the scenario
            }

            # Append the scenario data to the list
            data.append(scenario_data)

            # Save the updated data back to the JSON file
            with open(file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            print(f"Vehicle states appended to {file_name}")

        except Exception as e:
            print(f"Error appending vehicle states to JSON: {e}")

    def game_loop_end(self, failed=False):

        try:
            self.world.sensor_cam.destroy()  # Destroy the spawned RGB camera
            self.world.destroy()  # World object returned by WorldSR class
            if hasattr(self, 'interfuser'):
                self.interfuser.destroy()
            for sensor in self.sensor_objects.values():
                sensor.destroy()

            pygame.quit()

            run_duration = time.time() - self.start_system_time  # Total duration of the scenario run

            # Use the dynamic path defined in __init__
            file_path = self.results_file_path
            
            self.append_vehicle_states_to_json(file_path, failed, run_duration)

        except Exception as e: 
            traceback.print_exc()
            print(e)

    def load_model3(self, model_name):
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = model_name + '.tar.gz'
        
        # Get the current script directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Construct path: .../scenario_runner-0.9.10/models/research/object_detection/models/
        dir_loc = os.path.join(base_path, 'models', 'research', 'object_detection', 'models')
        
        # Ensure the trailing slash exists for get_file or join it properly
        if not os.path.exists(dir_loc):
            os.makedirs(dir_loc)

        model_dir = tf.keras.utils.get_file(
            fname=model_name + '.tar.gz', 
            origin=base_url + model_file,
            untar=True,
            cache_dir=dir_loc)
        
        # Clean up pathing to find the actual unzipped folder
        extracted_dir_path = pathlib.Path(dir_loc) / "datasets" / model_name / "saved_model"
        
        model_dir = pathlib.Path(model_dir).parent / model_name / "saved_model"
        
        model = tf.saved_model.load(str(model_dir))

        return model

    def run_inference_for_single_image_new(self, model, image):
        """
        Run inference on a single image using YOLOv8 model.
        
        Args:
            model: YOLO model instance
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Dictionary containing detection results with keys:
            - detection_boxes: normalized coordinates [y1, x1, y2, x2]
            - detection_classes: class IDs
            - detection_scores: confidence scores
            - num_detections: number of detections
        """
        # Run inference
        results = model(image, verbose=False)[0]  # verbose=False to suppress printing
        
        # Initialize output dictionary
        output_dict = {}
        
        if len(results.boxes) > 0:
            # Get boxes, convert to numpy
            boxes = results.boxes.xyxy.cpu().numpy()  # xyxy format
            
            # Convert boxes from [x1,y1,x2,y2] to [y1,x1,y2,x2] format and normalize
            image_height, image_width = image.shape[:2]
            normalized_boxes = np.column_stack([
                boxes[:, 1] / image_height,  # y1
                boxes[:, 0] / image_width,   # x1
                boxes[:, 3] / image_height,  # y2
                boxes[:, 2] / image_width    # x2
            ])
            
            # Get classes and scores
            classes = results.boxes.cls.cpu().numpy().astype(np.int64)
            scores = results.boxes.conf.cpu().numpy()
            
            output_dict.update({
                'detection_boxes': normalized_boxes,
                'detection_classes': classes,
                'detection_scores': scores,
                'num_detections': len(boxes)
            })
        else:
            # No detections
            output_dict.update({
                'detection_boxes': np.array([]),
                'detection_classes': np.array([]),
                'detection_scores': np.array([]),
                'num_detections': 0
            })
        
        # If segmentation masks are available
        if hasattr(results, 'masks') and results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            output_dict['detection_masks_reframed'] = masks
        
        return output_dict


    def run_inference_for_single_image(self, model, image):
      image = np.asarray(image)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis,...]

      # Run inference
      with tf.device('/cpu:0'):
          model_fn = model.signatures['serving_default']
          output_dict = model_fn(input_tensor)

      num_detections = int(output_dict.pop('num_detections'))
      output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
      output_dict['num_detections'] = num_detections

      # detection_classes should be ints.
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
       
      # Handle models with masks:
      if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
      return output_dict

    
class ArgsOverwrite:

    rolename = 'hero'
    gamma = 2.2 
    width = 1280
    height = 720
    host = '127.0.0.1'
    port = 2000
    behavior = "cautious"
    agent = "Behavior"
    seed = None
    verbose = False
    autopilot = False

    def __init__ (self):
    #filter turns blue if I write it as it is without self
        self.filter = "vehicle.*"   # Needed for CARLA version
