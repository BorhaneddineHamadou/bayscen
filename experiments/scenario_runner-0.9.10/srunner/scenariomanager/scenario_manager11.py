#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import sys
import time

import os
import json

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer2 import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from automatic_control_interfuser import * # EgoControlAgent() imported from here, this is for interfuser mode


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze_scenario()
    5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, debug_mode=False, sync_mode=False, timeout=2.0, 
                 scenario=1, test_method="test", run_number=3):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        # --- PATH CONFIGURATION ---
        # Save these to pass to EgoControlAgent later
        self.scenario_number = scenario
        self.run_num = run_number
        self.scenario_name = f"scenario{scenario}_interfuser"
        self.test_method = test_method
        self.run_number = f"run{run_number}"

        # Calculate root path:
        # Current file: .../srunner/scenariomanager/scenario_manager11.py
        # Up 1: .../srunner/scenariomanager
        # Up 2: .../srunner
        # Up 3: .../scenario_runner-0.9.10/ (Target Root)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))

        self.output_dir = os.path.join(root_dir, self.scenario_name, "outputs", self.test_method, self.run_number)

        # Create directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.min_ttc_file_path = os.path.join(self.output_dir, "min_ttc_log.json")
        # --------------------------

        self._debug_mode = debug_mode
        self._agent = None
        self._sync_mode = sync_mode
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = timeout
        self._watchdog = Watchdog(float(self._timeout))

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.collision_counts = None
        self.collision_test_result = None

    def _reset(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def cleanup(self):
        """
        This function triggers a proper termination of a scenario
        """

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        CarlaDataProvider.cleanup()

    def load_scenario(self, scenario, ego_goal_location, other_veh_agentZ_og, ego_visualize=False, agent=None):
            """
            Load a new scenario
            """
            self._reset()

            # Initializing our custom ego with the dynamic arguments stored in __init__
            self.ego_agentZ = EgoControlAgent(
                scenario=self.scenario_number, 
                test_method=self.test_method, 
                run_number=self.run_num
            )

            # load the scenario settings in the ego driving agent 
            self.ego_agentZ.game_loop_init(ego_goal_location, ego_visualize)
            
            # default code of agent not used
            self._agent = AgentWrapper(agent) if agent else None  # No agent right now  # one agent returned
            if self._agent is not None:
                self._sync_mode = True
    
            ################################ Initialize Other Vehicle Agent here!!!!!!!!!!!!! ##############################
            self.other_veh_agentZ_og = other_veh_agentZ_og


            self.scenario_class = scenario 
            self.scenario = scenario.scenario 

            self.scenario_tree = self.scenario.scenario_tree
            
            self.ego_vehicles = scenario.ego_vehicles
            self.other_actors = scenario.other_actors

            # Use the dynamic instance variable created in __init__
            self.init_min_ttc_file(self.min_ttc_file_path)  

            if self._agent is not None:
                self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        while self._running:  # is equal to false when the scenario tree is finished running as seen below in self._tick_scenario(timestamp)
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()  # snapshot of what's happening in the world?
                '''
                This snapshot comprises all the information for every actor on scene at a certain moment of time. It creates and gives acces to a data structure containing a series of carla.ActorSnapshot. The client recieves a new snapshot on every tick that cannot be stored.

                '''


                if snapshot:
                    timestamp = snapshot.timestamp
                    #print(timestamp)
            if timestamp:  # meaning carla worl is running i.e., the scenario in that world

                # So the scenario is ticked in a while loop which is pretty fast :3 
                # So do I need to declare the veh driving agent here or independently? Let's see.
                # Maybe I should call the ego and other veh agents here :3 
                # Run the run_step method here of the automatic_control_agent_z here :3

                self.ego_agentZ.game_loop_step()  # making new plan again for each step, just an experiment.
                
                self._tick_scenario(timestamp)  # Run next tick of scenario and the agent.

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            self.ego_agentZ.game_loop_end(failed=True)
        else:
            self.ego_agentZ.game_loop_end()
        
        self.other_veh_agentZ_og.game_loop_end()  # ENDING IT HERE AL!

        self._watchdog.stop()

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")  # cool
                #sys.exit("fhawk")
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()  # update all actors velocity, location, transform

            if self._agent is not None:
                ego_action = self._agent()  # doesn't enter this condition for NoSignalJunctionCrossing so let's ignore it for now
            if self._agent is not None:
                self.ego_vehicles[0].apply_control(ego_action)  # will see what is this ego_action

            # I added this to compute the TTC and log it
            self.check_ttc(self.min_ttc_file_path)  # Check and log the minimum TTC

            # Tick scenario
            self.scenario_tree.tick_once()  # look at this, moving through the sequences in a tree I suppose!!!!!~~~!! vip!
            #sys.exit("fhawk")

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False  # meaning scenario finished  # When last child/node of the pytree i.e., the root = pytree sequence() is executed and in a non running state

        if self._sync_mode and self._running and self._watchdog.get_status():
            CarlaDataProvider.get_world().tick()

    def get_running_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

    def analyze_scenario(self, stdout, filename, junit):
        """
        This function is intended to be called from outside and provide
        the final statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            print("Nothing to analyze, this scenario has no criteria")
            return True

        for criterion in self.scenario.get_criteria():
            if (not criterion.optional and
                    criterion.test_status != "SUCCESS" and
                    criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        output = ResultOutputProvider(self, result, stdout, filename, junit)  # self is the scenario manager class object
        output.write()

        self.collision_counts = output.collision_counts
        self.collision_test_result = output.collision_test_result
        


        return failure or timeout

    def compute_TTC(self):
        # Get transforms and velocities
        ego_tf = self.ego_agentZ.world.player.get_transform()

        other_tf = self.other_veh_agentZ_og.agent.vehicle.get_transform()

        ego_vel = self.ego_agentZ.world.player.get_velocity()

        other_vel = self.other_veh_agentZ_og.agent.vehicle.get_velocity()

        # Convert to numpy arrays for vector math
        pos_ego = np.array([ego_tf.location.x, ego_tf.location.y])
        pos_other = np.array([other_tf.location.x, other_tf.location.y])

        vel_ego = np.array([ego_vel.x, ego_vel.y])
        vel_other = np.array([other_vel.x, other_vel.y])

        # Relative position and velocity
        rel_pos = pos_other - pos_ego
        rel_vel = vel_other - vel_ego

        rel_pos_norm = np.linalg.norm(rel_pos)
        if rel_pos_norm < 0.001:
            return 0.0  # Already overlapping

        closing_speed = np.dot(rel_vel, rel_pos / rel_pos_norm)

        # If closing speed <= 0, vehicles are not approaching each other
        if closing_speed <= 0:
            return float('inf')

        ttc = rel_pos_norm / closing_speed
        return ttc

    def init_min_ttc_file(self, file_path):
        """
        Initialize a JSON file with a min_ttc entry.
        If file doesn't exist, create it with initial value.
        If file exists, append a new min_ttc entry.
        """
        initial_value = 9999.0
        entry = {"min_ttc": initial_value}
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                # Create new file with initial entry
                with open(file_path, 'w') as f:
                    json.dump([entry], f, indent=2)
                print(f"Created new min_ttc file: {file_path}")
            else:
                # File exists - read existing data
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Ensure data is a list, if not, reset it
                    if not isinstance(data, list):
                        print(f"Warning: File {file_path} contains invalid format. Resetting to list.")
                        data = []
                    
                    # Append new entry
                    data.append(entry)
                    
                    # Write back to file
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Appended min_ttc entry to existing file: {file_path}")
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON in {file_path}: {e}")
                    print("Creating new file with initial entry.")
                    with open(file_path, 'w') as f:
                        json.dump([entry], f, indent=2)
                        
        except IOError as e:
            print(f"Error: Could not access file {file_path}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error while processing {file_path}: {e}")
            raise

    def check_ttc(self, file_path):
        current_ttc = self.compute_TTC()

        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            return

        # Load the existing min_ttc log
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            print(f"Invalid or empty JSON format in {file_path}.")
            return

        # Compare and update last min_ttc if needed
        last_entry = data[-1]
        if "min_ttc" in last_entry and current_ttc < last_entry["min_ttc"]:
            last_entry["min_ttc"] = current_ttc
            print(f"Updated min_ttc to {current_ttc:.3f}")

            # Write the updated data back
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
