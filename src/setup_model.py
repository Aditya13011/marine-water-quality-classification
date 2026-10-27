"""
src/setup_model.py
──────────────────────────────────────────────────────────────────────────────
Cloud startup script — called by app.py if model artifacts are missing.

On Streamlit Community Cloud, model .joblib files are not committed to git
(they are large binaries excluded by .gitignore). This script regenerates
them from the committed raw dataset (data/marine_water_quality.csv) at
first startup.

Execution path:
    marine_water_quality.csv (committed)
        → data_preprocessing.py  → X_train_A.csv, pipeline_scenario_A.joblib …
        → train_model.py          → selected_model_scenario_A.joblib

This runs ONCE and is cached by Streamlit. Subsequent reruns skip it.

Estimated runtime: 90–180 seconds on cloud (6-algorithm comparison + XGBoost pipeline).
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import subprocess

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MODEL_A_PATH = os.path.join(ROOT, "models", "selected_model_scenario_A.joblib")
PREPROC_SCRIPT = os.path.join(ROOT, "src", "data_preprocessing.py")
TRAIN_SCRIPT   = os.path.join(ROOT, "src", "train_model_enhanced.py")


def model_exists() -> bool:
    """Return True if the production Scenario A model artifact is present."""
    return os.path.isfile(MODEL_A_PATH)


def run_script(path: str, label: str) -> None:
    """Run a Python script as a subprocess and stream output."""
    print(f"\n[setup_model] Running {label}…")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} exited with code {result.returncode}. "
            "Check the Streamlit Cloud logs for details."
        )
    print(f"[setup_model] {label} complete.")


def ensure_model_ready() -> None:
    """
    If the Scenario A model does not exist, regenerate it from the raw CSV.
    Called once at app startup via st.cache_resource.
    """
    if model_exists():
        print(f"[setup_model] Model already present: {MODEL_A_PATH}")
        return

    print("[setup_model] Model artifact not found — regenerating from raw CSV.")
    print(f"[setup_model] This runs ONCE at first deployment (≈60–120 seconds).")

    # Verify the raw dataset is present (committed to git)
    csv_path = os.path.join(ROOT, "data", "marine_water_quality.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. "
            "Ensure data/marine_water_quality.csv is committed to the repository."
        )

    # Step 1 — preprocessing: split + scale + save pipeline joblibs + split CSVs
    run_script(PREPROC_SCRIPT, "data_preprocessing.py")

    # Step 2 — enhanced training: 6-algorithm CV comparison, saves selected_model_scenario_A.joblib
    run_script(TRAIN_SCRIPT, "train_model_enhanced.py")

    # Final check
    if not model_exists():
        raise RuntimeError(
            "Training completed but model file is still missing: "
            f"{MODEL_A_PATH}"
        )
    print(f"[setup_model] ✔ Model ready: {MODEL_A_PATH}")
