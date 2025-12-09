# PICT Baseline Method

This folder contains the setup and execution instructions for generating test scenarios using **PICT (Pairwise Independent Combinatorial Testing)**.

## About PICT

PICT is a combinatorial testing tool developed by Microsoft that generates test cases based on pairwise (or higher-order) parameter interactions. It systematically covers all combinations of parameter values at a specified interaction strength.

**Official Repository:** [https://github.com/microsoft/pict](https://github.com/microsoft/pict)

## Files in This Folder

- **`parameters_scenario1.txt`**: Parameter definitions and constraints for Scenario 1 (Vehicle-Vehicle)
- **`parameters_scenario2.txt`**: Parameter definitions and constraints for Scenario 2 (Vehicle-Cyclist)
- **`Analysis.ipynb`**: Jupyter notebook for converting PICT-generated CSV files to Excel format for consistency with other baseline methods

## Setup

### 1. Install PICT

Download and install PICT from the [official GitHub repository](https://github.com/microsoft/pict).

### 2. Add PICT to System Path

Add the PICT installation directory to your system PATH environment variable:

```bash
# Example path (adjust according to your installation):
export PATH=$PATH:/path/to/BayScen/Baselines/PICT
```

Or on Windows:
```cmd
set PATH=%PATH%;C:\path\to\BayScen\Baselines\PICT
```

Verify installation:
```bash
pict
```

## Usage

### Generate Test Scenarios

#### For 2-way (Pairwise) Testing:

```bash
# Scenario 1 (Vehicle-Vehicle)
pict parameters_scenario1.txt /o:2 > scenarios_scenario1_2w.csv

# Scenario 2 (Vehicle-Cyclist)
pict parameters_scenario2.txt /o:2 > scenarios_scenario2_2w.csv
```

#### For 3-way Testing:

```bash
# Scenario 1 (Vehicle-Vehicle)
pict parameters_scenario1.txt /o:3 > scenarios_scenario1_3w.csv

# Scenario 2 (Vehicle-Cyclist)
pict parameters_scenario2.txt /o:3 > scenarios_scenario2_3w.csv
```

### View Generation Statistics

To see statistics about the generation process (total combinations, generated tests, generation time):

```bash
# For Scenario 1
pict parameters_scenario1.txt /o:3 /s

# For Scenario 2
pict parameters_scenario2.txt /o:3 /s
```

## Command-Line Options

- **`/o:N`** - Specifies the order of combinations:
  - `/o:2` - Pairwise testing (default) - covers all 2-way parameter interactions
  - `/o:3` - 3-way testing - covers all 3-way parameter interactions (more thorough)
  
- **`/s`** - Show statistics:
  - Total possible combinations
  - Number of generated test cases
  - Generation time

- **`> filename.csv`** - Redirects output to a CSV file

## Post-Processing

After generating the CSV files with PICT, use the provided Jupyter notebook to convert them to Excel format:

1. Open `Analysis.ipynb`
2. Run all cells to convert:
   - `scenarios_scenario1_2w.csv` → `PICT_2w_scenario1.xlsx`
   - `scenarios_scenario1_3w.csv` → `PICT_3w_scenario1.xlsx`
   - `scenarios_scenario2_2w.csv` → `PICT_2w_scenario2.xlsx`
   - `scenarios_scenario2_3w.csv` → `PICT_3w_scenario2.xlsx`

This ensures consistency with other baseline method outputs.

## Parameter Files Format

The parameter files (`parameters_scenario1.txt` and `parameters_scenario2.txt`) define:

1. **Parameters and their values**: Each parameter is listed with its possible discrete values
2. **Constraints**: IF-THEN rules that enforce valid combinations (e.g., collision point constraints)

Example:
```
PathInteraction: c1, c2, c4
RoadFriction: 0.1, 0.2, 0.4, 0.6, 0.8, 1

IF [PathInteraction] = "c1" THEN 
    ([StartEgo] = "Left" AND [GoalEgo] = "Right" AND [StartOther] = "Base" AND [GoalOther] = "Left") OR
    ...
```

## Expected Outputs

For the BayScen paper experiments:

| Scenario | Testing Level | Generated Tests |
|----------|--------------|-----------------|
| Scenario 1 (V-V) | 2-way | 61 |
| Scenario 1 (V-V) | 3-way | 456 |
| Scenario 2 (V-C) | 2-way | 68 |
| Scenario 2 (V-C) | 3-way | 525 |

## References

- **PICT Official Documentation**: [https://github.com/microsoft/pict/blob/main/doc/pict.md](https://github.com/microsoft/pict/blob/main/doc/pict.md)
- **Czerwonka, J.** (2006). "Pairwise testing in real world." *24th Pacific Northwest Software Quality Conference*, 200, 1-12.
- **Kuhn, D. R., Lei, Y., & Kacker, R.** (2008). "Practical combinatorial testing: Beyond pairwise." *IT Professional*, 10(3), 19-23.

## Citation

If you use PICT in your research, please cite:

```bibtex
@inproceedings{czerwonka2006pairwise,
  title={Pairwise testing in real world},
  author={Czerwonka, Jacek},
  booktitle={24th Pacific Northwest Software Quality Conference},
  volume={200},
  pages={1--12},
  year={2006}
}
```