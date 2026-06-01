# PICT — Pairwise Independent Combinatorial Testing

Microsoft PICT applied to autonomous vehicle junction and cut-in scenarios, generating pairwise (2-way) and three-way (3-way) combinatorial test suites.

**Official repository:** https://github.com/microsoft/pict

**References:**
```bibtex
@inproceedings{czerwonka2006pairwise,
  title={Pairwise testing in real world},
  author={Czerwonka, Jacek},
  booktitle={24th Pacific Northwest Software Quality Conference},
  volume={200}, pages={1--12}, year={2006}
}

@article{kuhn2008practical,
  title={Practical combinatorial testing: Beyond pairwise},
  author={Kuhn, D. Richard and Lei, Yu and Kacker, Raghu},
  journal={IT Professional}, volume={10}, number={3}, pages={19--23}, year={2008}
}
```

---

## Folder Structure

```
PICT/
├── pict.exe                       # PICT binary (Windows)
├── parameters_scenario1.txt       # Parameter spec + constraints for S1
├── parameters_scenario2.txt       # Parameter spec + constraints for S2
├── parameters_scenario3.txt       # Parameter spec + constraints for S3
├── scenarios_scenario{1,2,3}_2w.csv  # 2-way output (pre-generated)
├── scenarios_scenario{1,2,3}_3w.csv  # 3-way output (pre-generated)
├── PICT_2w_scenario{1,2,3}.xlsx   # 2-way (processed, ready to execute)
├── PICT_3w_scenario{1,2,3}.xlsx   # 3-way (processed, ready to execute)
├── Analysis.ipynb                 # CSV → Excel conversion notebook
└── README.md
```

---

## Scenarios

| ID | Description | Parameter file | 2-way tests | 3-way tests |
|----|-------------|---------------|------------|------------|
| S1 | Vehicle-Vehicle Junction | `parameters_scenario1.txt` | 61 | 456 |
| S2 | Vehicle-Cyclist Junction | `parameters_scenario2.txt` | 68 | 525 |
| S3 | Vehicle-Vehicle Cut-In | `parameters_scenario3.txt` | see xlsx | see xlsx |

S3 replaces the junction `PathInteraction` axis (c1/c2/c4) with a binary `Direction` (left/right) and adds `TimeOfDay` (sun altitude angle), matching the cut-in parameter space used by all S3 baselines.

---

## Setup

Install PICT from the [official GitHub repository](https://github.com/microsoft/pict) and add it to your PATH:

```bash
# Linux/macOS
export PATH=$PATH:/path/to/pict

# Windows (cmd)
set PATH=%PATH%;C:\path\to\pict
```

Verify: `pict` should print usage help.

A Windows `pict.exe` is already included in this folder.

---

## Usage

### Generate test scenarios

```bash
# Scenario 1 — 2-way
pict parameters_scenario1.txt /o:2 > scenarios_scenario1_2w.csv

# Scenario 1 — 3-way
pict parameters_scenario1.txt /o:3 > scenarios_scenario1_3w.csv

# Scenario 2 — 2-way
pict parameters_scenario2.txt /o:2 > scenarios_scenario2_2w.csv

# Scenario 2 — 3-way
pict parameters_scenario2.txt /o:3 > scenarios_scenario2_3w.csv

# Scenario 3 — 2-way
pict parameters_scenario3.txt /o:2 > scenarios_scenario3_2w.csv

# Scenario 3 — 3-way
pict parameters_scenario3.txt /o:3 > scenarios_scenario3_3w.csv
```

Add `/s` to print generation statistics (total combinations, generated tests, time).

### Post-process to Excel

Open `Analysis.ipynb` and run all cells to convert each CSV to an xlsx file consistent with the other baseline methods.

---

## Parameter File Format

Each `.txt` file defines parameters with their discrete values and optional constraints.

**Junction scenarios (S1/S2) — PathInteraction with route constraints:**

```
PathInteraction: c1, c2, c4
RoadFriction: 0.1, 0.2, 0.4, 0.6, 0.8, 1
...
IF [PathInteraction] = "c1" THEN
    ([StartEgo] = "Left" AND [GoalEgo] = "Right" ...) OR ...
```

**Cut-in scenario (S3) — Direction replaces PathInteraction:**

```
Direction: left, right
RoadFriction: 0.1, 0.2, 0.4, 0.6, 0.8, 1
TimeOfDay: -90, -60, -30, 0, 30, 60, 90
...
```

---

## References

- Microsoft PICT documentation: https://github.com/microsoft/pict/blob/main/doc/pict.md
- Czerwonka, J. (2006). "Pairwise testing in real world." *24th Pacific NW Software Quality Conference*.
- Kuhn, D. R., Lei, Y., & Kacker, R. (2008). "Practical combinatorial testing: Beyond pairwise." *IT Professional*, 10(3), 19–23.
