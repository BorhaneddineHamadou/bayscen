import subprocess

# Define the command as a list of arguments
command = [
    "python",
    "situationcoverage_AV_VV_Framework.py",
    "--scenario", "IntersectionScenarioZ_11",
    "--not_visualize",
    "--Activate_IntersectionScenario_Seed",
    "--IntersectionScenario_Seed", "26", # 67676 
    "--use_sit_cov",
    "--reloadWorld",
    "--output",
    "--sync", # Uncomment this only for interfuser model
]

# Run the command 100 times (Now it's 54 times)
for i in range(54-37):
    print(f"Running iteration {i + 1}/54...")
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Iteration {i + 1} finished with return code {result.returncode}")