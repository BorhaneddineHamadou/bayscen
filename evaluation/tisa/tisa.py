"""
tisa.py
───────
Runs TISA metrics (area_IS, area_bugs, cov_IS) on a validation dataset
by calling the MATLAB buildIS.m code headlessly, reading back the output
CSVs, and saving an Excel results file.

This script is a direct translation of the notebook cells, in order:
  Cell 1  → OPTS dict + path setup
  Cell 2  → write options.json
  Cell 2b → patch autoNormalize.m for MATLAB R2019b+
  Cell 3  → write run_buildIS.m launcher, call MATLAB, stream output
  Cell 4  → read ISA-coverages.csv, extract three metrics
  (new)   → save Excel with Parameters + Metrics sheets

Metric mapping (from notebook Cell 4 markdown):
  area_IS   ← Footprint_area        in ISA-coverages.csv
  area_bugs ← Good_footprint_area   in ISA-coverages.csv
  cov_IS    ← COV_prunnedBoundary   in ISA-coverages.csv   (MATLAB typo kept)

Expected folder layout inside methods/tisa/
  methods/tisa/
  ├── tisa.py                    ← this script
  └── tisa_matlab/
      ├── ISA-code/
      │   ├── ISA-coverages.csv
      │   └── InstanceSpace/     ← all .m files (buildIS.m, PILOT.m, …)
      └── <dataset_run_folders created at runtime>

Usage
─────
  python methods/tisa/tisa.py \\
      --data    data/deepscenario_validation.csv \\
      --tisa    methods/tisa/tisa_matlab \\
      --output  results/deepscenario_tisa.xlsx \\
      --matlab  "C:\\Program Files\\MATLAB\\R2025b\\bin\\matlab.exe"
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Cell 1 equivalent — ISA / buildIS options (exact copy from notebook OPTS)
# ──────────────────────────────────────────────────────────────────────────────
OPTS = {
    "parallel": {"flag": False, "ncores": 1},

    # Performance binarisation
    # MaxPerf=False → lower is better (cost).  AbsPerf=True → absolute threshold.
    # algo_safety ∈ {0,1}; 0=safe, 1=bug.
    # epsilon=0.5 → "good" if value <= 0.5 → catches safe scenarios (value=0).
    # So Good_footprint_area = area covered by safe scenarios = area_bugs (per notebook mapping).
    "perf": {"MaxPerf": False, "AbsPerf": True, "epsilon": 0.50, "betaThreshold": 0.55},

    # auto.preproc and norm.flag trigger Box-Cox (autoNormalize.m).
    # Requires Statistics and Machine Learning Toolbox.
    # Set both to False if you get 'Undefined function boxcox'.
    "auto":    {"preproc": True},
    "bound":   {"flag": True},
    "norm":    {"flag": True},

    # Feature selection (SIFTED)
    "sifted":  {"flag": False, "rho": 0.1, "K": 10,
                "NTREES": 50, "MaxIter": 1000, "Replicates": 100},

    # 2-D projection (PILOT) — analytic=False uses BFGS (slower, better)
    "pilot":   {"analytic": False, "ntries": 5},

    # Boundary estimation (CLOISTER)
    "cloister": {"pval": 0.05, "cthres": 0.7},

    # Footprint calculation (TRACE)
    # usesim=False → use experimental data (Ybin/P), not SVM predictions.
    # usesim=True would crash because PYTHIA is commented out in buildIS.m.
    "trace":   {"usesim": False, "PI": 0.55},

    "selvars": {"smallscaleflag": False, "smallscale": 0.50,
                "fileidxflag": False, "fileidx": "",
                "densityflag": False, "mindistance": 0.1},

    # csv=True so Python can read results back; web=False (Matilda platform only)
    "outputs": {"csv": True, "web": False, "png": True},
}


# ──────────────────────────────────────────────────────────────────────────────
# Cell 2b equivalent — patched autoNormalize.m (R2019b+ compatible)
# ──────────────────────────────────────────────────────────────────────────────
# Original [y, lam] = boxcox(x) two-output form was removed in R2019b.
# Replacement: call boxcox(x) for lambda, apply transform manually.
AUTONORMALIZE_PATCH = """\
function [X,Y,out] = autoNormalize(X, Y)
% Patched version: compatible with MATLAB R2019b+ where [y,lam]=boxcox(x)
% was removed. Lambda is estimated with boxcox(x) and applied manually.

nfeats = size(X,2);
nalgos = size(Y,2);
disp("-> Auto-normalizing the data using Box-Cox and Z transformations.");
out.minX = min(X,[],1);
X = bsxfun(@minus,X,out.minX)+1;
out.lambdaX = zeros(1,nfeats);
out.muX     = zeros(1,nfeats);
out.sigmaX  = zeros(1,nfeats);
for i=1:nfeats
    aux = X(:,i);
    idx = isnan(aux);
    lam = boxcox(aux(~idx));
    out.lambdaX(i) = lam;
    aux(~idx) = applyBoxCox(aux(~idx), lam);
    [aux, out.muX(i), out.sigmaX(i)] = zscore(aux);
    X(~idx,i) = aux(~idx);
end

out.minY = min(Y(:));
Y = (Y - out.minY) + eps;
out.lambdaY = zeros(1,nalgos);
out.muY     = zeros(1,nalgos);
out.sigmaY  = zeros(1,nalgos);
for i=1:nalgos
    aux = Y(:,i);
    idx = isnan(aux);
    lam = boxcox(aux(~idx));
    out.lambdaY(i) = lam;
    aux(~idx) = applyBoxCox(aux(~idx), lam);
    [aux, out.muY(i), out.sigmaY(i)] = zscore(aux);
    Y(~idx,i) = aux(~idx);
end

end

function y = applyBoxCox(x, lam)
if abs(lam) < 1e-8
    y = log(x);
else
    y = (x .^ lam - 1) ./ lam;
end
end
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_validation_file(data_path: Path) -> pd.DataFrame:
    """Load CSV or Excel. Validates required TISA columns are present."""
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    df.columns = df.columns.str.strip()

    # Validate required columns
    missing = {"Instances", "algo_safety"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Validation file is missing required TISA columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        raise ValueError(
            "No 'feature_*' columns found. Validation file must follow TISA format."
        )

    n_bugs = int((df["algo_safety"] == 1).sum())
    n_safe = int((df["algo_safety"] == 0).sum())
    log.info(
        "Loaded %d scenarios | %d feature columns | %d safe | %d bugs",
        len(df), len(feature_cols), n_safe, n_bugs,
    )
    return df


def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode any feature_* column that contains non-numeric (string/object)
    values. MATLAB's buildIS.m requires all feature columns to be numeric.

    Encoding is deterministic (sorted unique values → 0, 1, 2, …) so results
    are reproducible across runs.

    Returns
    -------
    df       : copy of the dataframe with encoded columns
    mappings : { column_name → { original_value → integer_code } }
               saved to the Parameters sheet for traceability
    """
    df      = df.copy()
    mappings = {}

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in feature_cols:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            unique_vals = sorted(df[col].dropna().unique(), key=str)
            code_map    = {v: i for i, v in enumerate(unique_vals)}
            df[col]     = df[col].map(code_map)
            mappings[col] = code_map
            log.info(
                "  Encoded '%s': %d categories → integers %s",
                col, len(code_map),
                {str(k): v for k, v in list(code_map.items())[:6]},
            )

    if mappings:
        log.info("Label-encoded %d categorical feature column(s).", len(mappings))
    else:
        log.info("All feature columns are already numeric — no encoding needed.")

    return df, mappings


def write_options_json(dataset_dir: Path) -> None:
    """Cell 2: write a fresh options.json — delete stale copy first."""
    options_path = dataset_dir / "options.json"
    if options_path.exists():
        options_path.unlink()
        log.info("Deleted stale options.json")
    with open(options_path, "w") as fh:
        json.dump(OPTS, fh, indent=2)
    log.info("Written options.json → %s", options_path)


def patch_autonormalize(matlab_code_dir: Path) -> None:
    """Cell 2b: replace autoNormalize.m with R2019b+ compatible version."""
    target = matlab_code_dir / "autoNormalize.m"
    backup = matlab_code_dir / "autoNormalize.m.bak"

    if not target.exists():
        log.warning("autoNormalize.m not found at %s — skipping patch", target)
        return

    # Back up original only once
    if not backup.exists():
        shutil.copy(target, backup)
        log.info("Backed up original → %s", backup)

    target.write_text(AUTONORMALIZE_PATCH, encoding="utf-8")
    log.info("Patched autoNormalize.m (R2019b+ compatible)")


def write_launcher(dataset_dir: Path, matlab_code_dir: Path, coverages_dir: Path) -> Path:
    """
    Cell 3 (first part): write run_buildIS.m into dataset_dir.
    Using a .m file avoids all shell-quoting headaches on Windows.
    Note: EOF:SUCCESS is printed *after* buildIS returns so we detect completion.
    """
    launcher_path = dataset_dir / "run_buildIS.m"

    data_dir_m = dataset_dir.resolve().as_posix() + "/"
    code_dir_m = matlab_code_dir.resolve().as_posix()
    cov_dir_m  = coverages_dir.resolve().as_posix() + "/"

    launcher_path.write_text(
        f"addpath('{code_dir_m}');\n"
        f"try\n"
        f"    buildIS('{data_dir_m}', '{cov_dir_m}');\n"
        f"    disp('EOF:SUCCESS');\n"
        f"catch ME\n"
        f"    disp('EOF:ERROR');\n"
        f"    disp(ME.message);\n"
        f"end\n"
        f"exit;\n",
        encoding="utf-8",
    )
    log.info("Written launcher → %s", launcher_path)
    return launcher_path


def run_matlab(matlab_bin: str, launcher_path: Path, dataset_dir: Path) -> tuple[bool, float]:
    """
    Cell 3 (second part): launch MATLAB headlessly, mirror notebook logic exactly.

    Windows : -batch <launcher_stem>  with cwd=dataset_dir
    Linux   : -nodisplay -nosplash -nodesktop -r "run('<launcher_path>')"

    Returns (success, elapsed_seconds).
    """
    is_windows = sys.platform.startswith("win")
    log_file   = dataset_dir / "matlab_run.log"

    if is_windows:
        # launcher_path.stem = 'run_buildIS' (no .m extension for -batch)
        cmd = [matlab_bin, "-nosplash", "-nodesktop", "-wait",
               "-batch", launcher_path.stem]
        cwd = str(dataset_dir.resolve())
    else:
        cmd = [matlab_bin, "-nodisplay", "-nosplash", "-nodesktop",
               "-r", f"run('{launcher_path.as_posix()}')"]
        cwd = None

    log.info("Command: %s", " ".join(cmd))
    log.info("Starting MATLAB — this can take several minutes for large datasets…")
    print("-" * 60)

    t0      = time.perf_counter()
    success = False

    if is_windows:
        # -batch captures stdout; -wait blocks until MATLAB exits
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        log_file.write_text(output, encoding="utf-8")
        print(output)
        success = "EOF:SUCCESS" in output
    else:
        lines = []
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as proc:
            for line in proc.stdout:
                line = line.rstrip()
                print(line)
                lines.append(line)
                if "EOF:SUCCESS" in line:
                    success = True
                if "EOF:ERROR" in line:
                    break
        log_file.write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print("-" * 60)

    if success:
        log.info("✅ MATLAB finished successfully in %.1f s", elapsed)
    else:
        log.error("⚠️  EOF:SUCCESS not detected. Check log: %s", log_file)

    return success, elapsed


def extract_metrics(dataset_dir: Path, coverages_csv: Path) -> dict:
    """
    Cell 4: read ISA-coverages.csv and extract three TISA metrics.

    Match by Rootdir (normalise separators); fall back to last row if no match.
    """
    if not coverages_csv.exists():
        raise FileNotFoundError(
            f"ISA-coverages.csv not found: {coverages_csv}\n"
            "MATLAB may have failed — check matlab_run.log in the dataset folder."
        )

    cov_df = pd.read_csv(coverages_csv)

    def norm_path(s: str) -> str:
        return str(s).replace("\\", "/").rstrip("/")

    target = norm_path(dataset_dir.resolve())
    match  = cov_df[cov_df["Rootdir"].apply(norm_path) == target]
    latest = match if not match.empty else cov_df.tail(1)

    if match.empty:
        log.warning("Rootdir not matched exactly — using last row of ISA-coverages.csv")

    # Take the most recent matching row (in case of multiple runs on same folder)
    row = latest.iloc[-1]

    metrics = {
        # Three TISA metrics
        "area_IS":              float(row["Footprint_area"]),
        "area_bugs":            float(row["Good_footprint_area"]),
        "cov_IS":               float(row["COV_prunnedBoundary"]),   # note: MATLAB typo
        # Auxiliary
        "boundary_area":        float(row["Boundary_area"]),
        "pruned_boundary_area": float(row["prunedBoundaryArea"]),
        "footprint_density":    float(row["Footprint_density"]),
    }

    print("\n" + "=" * 52)
    print("  TISA Metrics")
    print("=" * 52)
    print(f"  area_IS   = {metrics['area_IS']:.4f}")
    print(f"             (diversity of the whole test suite)")
    print(f"  area_bugs = {metrics['area_bugs']:.4f}")
    print(f"             (diversity of the bug-revealing scenarios)")
    print(f"  cov_IS    = {metrics['cov_IS']:.2f} %")
    print(f"             (coverage of the feasible feature space)")
    print("-" * 52)
    print(f"  [aux] Boundary area (full)   = {metrics['boundary_area']:.4f}")
    print(f"  [aux] Boundary area (pruned) = {metrics['pruned_boundary_area']:.4f}")
    print("=" * 52 + "\n")

    return metrics


def save_excel(
    metrics: dict,
    elapsed_s: float,
    output_path: Path,
    dataset_name: str,
    data_path: Path,
    tisa_root: Path,
    matlab_bin: str,
    n_scenarios: int,
    n_features: int,
    n_bugs: int,
    encodings: dict,
) -> None:
    """
    Write a two-sheet Excel file:
      Sheet "Parameters" — every setting used (for reproducibility),
                           including label-encoding mappings for categorical features
      Sheet "Metrics"    — the three TISA metrics + timing + auxiliary values
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ── Parameters ────────────────────────────────────────────────────────
        param_rows = [
            ("dataset",                 dataset_name),
            ("data_path",               str(data_path)),
            ("tisa_root",               str(tisa_root)),
            ("matlab_binary",           matlab_bin),
            ("n_scenarios",             n_scenarios),
            ("n_feature_columns",       n_features),
            ("n_bug_scenarios",         n_bugs),
            ("n_safe_scenarios",        n_scenarios - n_bugs),
            # ISA options
            ("perf.MaxPerf",            OPTS["perf"]["MaxPerf"]),
            ("perf.AbsPerf",            OPTS["perf"]["AbsPerf"]),
            ("perf.epsilon",            OPTS["perf"]["epsilon"]),
            ("perf.betaThreshold",      OPTS["perf"]["betaThreshold"]),
            ("auto.preproc",            OPTS["auto"]["preproc"]),
            ("norm.flag",               OPTS["norm"]["flag"]),
            ("sifted.flag",             OPTS["sifted"]["flag"]),
            ("sifted.rho",              OPTS["sifted"]["rho"]),
            ("sifted.K",                OPTS["sifted"]["K"]),
            ("sifted.NTREES",           OPTS["sifted"]["NTREES"]),
            ("pilot.analytic",          OPTS["pilot"]["analytic"]),
            ("pilot.ntries",            OPTS["pilot"]["ntries"]),
            ("cloister.pval",           OPTS["cloister"]["pval"]),
            ("cloister.cthres",         OPTS["cloister"]["cthres"]),
            ("trace.usesim",            OPTS["trace"]["usesim"]),
            ("trace.PI",                OPTS["trace"]["PI"]),
        ]
        # Append label-encoding mappings so the reader knows what each integer means
        if encodings:
            param_rows.append(("--- categorical encodings ---", ""))
            for col, mapping in encodings.items():
                for original, code in mapping.items():
                    param_rows.append((f"encoding.{col}", f"{original} → {code}"))

        pd.DataFrame(param_rows, columns=["parameter", "value"]).to_excel(
            writer, sheet_name="Parameters", index=False
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        metric_row = {
            "dataset":               dataset_name,
            # Three TISA metrics
            "area_IS":               round(metrics["area_IS"],   4),
            "area_bugs":             round(metrics["area_bugs"], 4),
            "cov_IS_pct":            round(metrics["cov_IS"],    4),
            # Timing
            "computation_time_s":    round(elapsed_s,            2),
            # Auxiliary
            "boundary_area":         round(metrics["boundary_area"],        4),
            "pruned_boundary_area":  round(metrics["pruned_boundary_area"], 4),
            "footprint_density":     round(metrics["footprint_density"],    4),
            # Dataset info
            "n_scenarios":           n_scenarios,
            "n_feature_columns":     n_features,
            "n_bugs":                n_bugs,
            "n_safe":                n_scenarios - n_bugs,
        }
        pd.DataFrame([metric_row]).to_excel(writer, sheet_name="Metrics", index=False)

    log.info("Results saved → %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TISA metrics on a validation dataset via MATLAB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", type=Path, required=True,
        help=(
            "Validation file (CSV or Excel). Must already be in TISA format: "
            "columns Instances, feature_*, algo_safety."
        ),
    )
    p.add_argument(
        "--tisa", type=Path, required=True,
        help=(
            "Path to the tisa_matlab folder, i.e. the folder that contains "
            "ISA-code/InstanceSpace/ and ISA-code/ISA-coverages.csv."
        ),
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output Excel file (e.g. results/deepscenario_tisa.xlsx).",
    )
    p.add_argument(
        "--matlab",
        type=str,
        default=shutil.which("matlab") or "matlab",
        help=(
            "Full path to the MATLAB executable. "
            "Examples: "
            r'"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"  (Windows) | '
            '"/usr/local/MATLAB/R2023b/bin/matlab"  (Linux)'
        ),
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Derived paths (mirrors notebook variables)
    tisa_root       = args.tisa                               # tisa_matlab/
    matlab_code_dir = tisa_root / "ISA-code" / "InstanceSpace"
    coverages_csv   = tisa_root / "ISA-code" / "ISA-coverages.csv"
    coverages_dir   = tisa_root / "ISA-code"

    # Dataset name from output stem (e.g. "deepscenario_tisa" → "deepscenario")
    dataset_name = args.output.stem.replace("_tisa", "")

    # Runtime dataset folder inside tisa_matlab (mirrors DATA_DIR in notebook)
    dataset_dir = tisa_root / f"{dataset_name}_run"

    log.info("=" * 60)
    log.info("TISA — dataset: %s", dataset_name.upper())
    log.info("  Data:      %s", args.data)
    log.info("  TISA root: %s", tisa_root)
    log.info("  MATLAB:    %s", args.matlab)
    log.info("  Output:    %s", args.output)
    log.info("=" * 60)

    # ── Guard: MATLAB exists ──────────────────────────────────────────────────
    matlab_path = Path(args.matlab)
    if not matlab_path.exists() and not shutil.which(args.matlab):
        log.error(
            "MATLAB not found at '%s'.\n"
            "Set --matlab to the full path of your matlab executable.", args.matlab
        )
        sys.exit(1)

    # ── Guard: InstanceSpace folder exists ───────────────────────────────────
    if not matlab_code_dir.exists():
        log.error(
            "InstanceSpace folder not found: %s\n"
            "Check --tisa points to the tisa_matlab folder.", matlab_code_dir
        )
        sys.exit(1)

    # ── Load & validate data ──────────────────────────────────────────────────
    df           = load_validation_file(args.data)
    n_scenarios  = len(df)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    n_features   = len(feature_cols)
    n_bugs       = int((df["algo_safety"] == 1).sum())

    # ── Encode categorical features → integers (required by MATLAB buildIS) ──
    df, encodings = encode_categorical_features(df)

    # ── Prepare dataset folder ────────────────────────────────────────────────
    dataset_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_dir / "metadata.csv", index=False)
    log.info("Written metadata.csv → %s", dataset_dir / "metadata.csv")

    # ── Cell 2: options.json ──────────────────────────────────────────────────
    write_options_json(dataset_dir)

    # ── Cell 2b: patch autoNormalize.m ────────────────────────────────────────
    patch_autonormalize(matlab_code_dir)

    # ── Cell 3: write launcher + run MATLAB ───────────────────────────────────
    launcher = write_launcher(dataset_dir, matlab_code_dir, coverages_dir)
    success, elapsed_s = run_matlab(args.matlab, launcher, dataset_dir)

    if not success:
        log.error(
            "MATLAB did not complete successfully.\n"
            "Check: %s", dataset_dir / "matlab_run.log"
        )
        sys.exit(1)

    # ── Cell 4: extract metrics ───────────────────────────────────────────────
    metrics = extract_metrics(dataset_dir, coverages_csv)

    # ── Save Excel ────────────────────────────────────────────────────────────
    save_excel(
        metrics      = metrics,
        elapsed_s    = elapsed_s,
        output_path  = args.output,
        dataset_name = dataset_name,
        data_path    = args.data,
        tisa_root    = tisa_root,
        matlab_bin   = args.matlab,
        n_scenarios  = n_scenarios,
        n_features   = n_features,
        n_bugs       = n_bugs,
        encodings    = encodings,
    )


if __name__ == "__main__":
    main()