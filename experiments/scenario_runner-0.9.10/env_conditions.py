import subprocess

# Define the command as a list of arguments
command_template = [
    "python",
    "situationcoverage_AV_VV_Framework.py",
    "--scenario", "IntersectionScenarioZ_11",
    "--not_visualize",
    "--Activate_IntersectionScenario_Seed",
    "--IntersectionScenario_Seed", "", # 67676 
    "--use_sit_cov",
    "--reloadWorld",
    "--output"
]

# Define the seed sequence and total runs
seed_sequence = ["67676", "11"] # seed_67676_11
total_runs = 10

# Run the command for the total number of runs
for i in range(total_runs):
    # Get the current seed from the sequence
    current_seed = seed_sequence[i % len(seed_sequence)]
    
    # Update the command with the current seed
    command_template[7] = current_seed  # Replace the placeholder with the seed
    
    print(f"command_template: {command_template}")
    print(f"Running iteration {i + 1}/{total_runs} with seed {current_seed}...")
    result = subprocess.run(command_template, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Iteration {i + 1} finished with return code {result.returncode}")