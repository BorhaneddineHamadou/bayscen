import os
import subprocess

# Set environment variables
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ";leaderboard;leaderboard\\team_code"
os.environ["LEADERBOARD_ROOT"] = "leaderboard"
os.environ["CHALLENGE_TRACK_CODENAME"] = "SENSORS"
os.environ["PORT"] = "2000"  # Same as CARLA server port
os.environ["TM_PORT"] = "2500"  # Traffic Manager port
os.environ["DEBUG_CHALLENGE"] = "0"
os.environ["REPETITIONS"] = "1"  # Number of evaluation runs
os.environ["ROUTES"] = "leaderboard\\data\\training_routes\\routes_town03_long.xml"
os.environ["TEAM_AGENT"] = "leaderboard\\team_code\\interfuser_agent.py"
os.environ["TEAM_CONFIG"] = "leaderboard\\team_code\\interfuser_config.py"
os.environ["CHECKPOINT_ENDPOINT"] = "results\\sample_result.json"
os.environ["SCENARIOS"] = "leaderboard\\data\\scenarios\\town03_all_scenarios.json"
os.environ["SAVE_PATH"] = "data\\eval"
os.environ["RESUME"] = "True"


# Run the Python evaluation script
command = [
    "python", "leaderboard\\leaderboard\\leaderboard_evaluator.py",
    "--scenarios", os.environ["SCENARIOS"],
    "--routes", os.environ["ROUTES"],
    "--repetitions", os.environ["REPETITIONS"],
    "--track", os.environ["CHALLENGE_TRACK_CODENAME"],
    "--checkpoint", os.environ["CHECKPOINT_ENDPOINT"],
    "--agent", os.environ["TEAM_AGENT"],
    "--agent-config", os.environ["TEAM_CONFIG"],
    "--debug", os.environ["DEBUG_CHALLENGE"],
    "--resume", os.environ["RESUME"],
    "--port", os.environ["PORT"],
    "--trafficManagerPort", os.environ["TM_PORT"]
]

# Execute the command
subprocess.run(command)
