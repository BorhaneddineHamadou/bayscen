import os

class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 5
    collision_buffer = [2.5, 1.2]
    
    # --- DYNAMIC PATH CALCULATION ---
    # 1. Get the folder containing this config file (.../scenario_runner-0.9.10)
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to the main root (.../WindowsNoEditor)
    _root_dir = os.path.dirname(_current_dir)
    
    # 3. Construct the path to the model in the InterFuser folder
    model_path = os.path.join(_root_dir, "InterFuser", "leaderboard", "team_code", "interfuser.pth.tar")
    # --------------------------------

    momentum = 0
    skip_frames = 1
    detect_threshold = 0.04

    model = "interfuser_baseline"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # specific check to ensure path exists, helpful for debugging
        if not os.path.exists(self.model_path):
            print(f"WARNING: Model path not found at: {self.model_path}")