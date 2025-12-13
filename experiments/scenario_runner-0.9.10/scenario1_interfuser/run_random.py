import subprocess
import pandas as pd
import os
import argparse
import sys
import xml.etree.ElementTree as ET

def update_scenario_xml():
    """
    Parses the XML config file and HARDCODES the 'model' attribute 
    to 'vehicle.tesla.model3' (Scenario 1).
    """
    # 1. Get the directory where this script is located
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go one level up to find the root that contains 'srunner'
    # Structure: root/srunner vs root/this_script_folder/runner_script.py
    root_dir = os.path.dirname(current_script_dir)
    
    # 3. Build path to XML
    xml_file_path = os.path.join(root_dir, "srunner", "examples", "IntersectionScenarioZ11.xml")
    
    if not os.path.exists(xml_file_path):
        print(f"Error: Could not find XML config at {xml_file_path}")
        return False

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        scenario = root.find("scenario")
        if scenario is None:
            print("Error: No <scenario> tag found in XML.")
            return False

        other_actor = scenario.find("other_actor")
        if other_actor is None:
            print("Error: No <other_actor> tag found in XML.")
            return False

        # --- HARDCODED FOR SCENARIO 1 ---
        new_model = "vehicle.tesla.model3"
        # print(f"XML Update: Setting other_actor to CAR ({new_model})")

        # Update the attribute
        other_actor.set("model", new_model)

        # Save the changes back to the file
        tree.write(xml_file_path)
        return True

    except Exception as e:
        print(f"Error updating XML: {e}")
        return False


def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Run Random baseline (generation and simulation)")
    
    parser.add_argument('--run_number', type=int, required=True, 
                        help='The run identifier integer (e.g., 3).')

    # REMOVED --scenario_number argument since it is strictly 1 here.
    args = parser.parse_args()
    print(f"Output Configuration: Method=Random | Run={args.run_number} | Scenario=1 (Vehicle-Vehicle)")
    print("Output JSON files will be saved in the 'output/random' folder.")

    args = parser.parse_args()

    for index in range(648):
        
        print(f"Generating Scenario {index + 1}")

        # --- FORCE XML TO SCENARIO 1 (Tesla) ---
        success = update_scenario_xml()
        if not success:
            print("Aborting this iteration due to XML error.")
            continue
        # ---------------------------------------
        
        command = [
            "python",
            "sitcov.py",
            "--test_method", "random",
            "--run_number", str(args.run_number),
            "--scenario_number", "1",
            "--scenario", "IntersectionScenarioZ_11",
            "--not_visualize",
            "--Activate_IntersectionScenario_Seed",
            "--IntersectionScenario_Seed", "26",
            "--reloadWorld",
            "--output",
            "--sync", 
        ]
        
        try:
            # result = subprocess.run(command)
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200)
            print(f"Row {index + 1} finished with return code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"Row {index + 1} TIMEOUT - moving to next scenario")
        except Exception as e:
            print(f"Row {index + 1} ERROR: {e}")

        print("-" * 40)

if __name__ == "__main__":
    main()