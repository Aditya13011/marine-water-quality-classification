"""
src/prediction.py
─────────────────────────────────────────────────────────────────────────────
Shared prediction utility for the Streamlit dashboard.
Loads the Scenario A pipeline ONCE and provides a validated predict function.

Pipeline structure: imputer → StandardScaler → [best selected classifier]
Best classifier  : Determined by 5-fold CV F1 on training data (Phase 2).
                   May be RandomForest, XGBoost, ExtraTrees, LinearSVC, etc.
Features (order) : Temperature, pH, Salinity, Turbidity
Target mapping   : 0 = Unsafe, 1 = Safe

IMPORTANT
  The pipeline handles its own imputation and scaling.
  Do NOT pre-scale inputs before calling predict().
  Always pass a pd.DataFrame with correct column names to avoid sklearn warnings.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import joblib
import pandas as pd
import numpy  as np

# ── constants ─────────────────────────────────────────────────────────────────
_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH  = os.path.join(_ROOT, "models", "selected_model_scenario_A.joblib")

FEATURES    = ["Temperature", "pH", "Salinity", "Turbidity"]  # exact order

# Real dataset ranges (from marine_water_quality.csv, verified Phase 4)
FEATURE_RANGES = {
    "Temperature": (10.0,  32.0),
    "pH":          ( 6.5,   8.8),
    "Salinity":    (30.0,  40.0),
    "Turbidity":   ( 0.5,  50.0),
}
FEATURE_UNITS = {
    "Temperature": "°C",
    "pH":          "(dimensionless)",
    "Salinity":    "PSU",
    "Turbidity":   "NTU",
}
CLASS_LABELS = {0: "Unsafe", 1: "Safe"}

# ── lazy singleton loader ─────────────────────────────────────────────────────
_pipeline = None

def load_pipeline():
    """Load the saved pipeline once; cache in module-level variable."""
    global _pipeline
    if _pipeline is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Run src/train_model_enhanced.py (or src/train_model.py) to regenerate it."
            )
        _pipeline = joblib.load(MODEL_PATH)
        # Validate pipeline structure (any classifier is acceptable in the model step)
        steps = [n for n, _ in _pipeline.steps]
        assert steps == ["imputer", "scaler", "model"], \
            f"Unexpected pipeline steps: {steps}"
    return _pipeline


def validate_inputs(values: dict) -> tuple[bool, str]:
    """
    Validate that all four feature values are within the observed dataset range.
    Returns (is_valid: bool, error_message: str).
    """
    for feat in FEATURES:
        if feat not in values:
            return False, f"Missing feature: {feat}"
        val = values[feat]
        lo, hi = FEATURE_RANGES[feat]
        if not (lo <= val <= hi):
            return False, (
                f"{feat} = {val} is outside the observed dataset range "
                f"[{lo}, {hi}] {FEATURE_UNITS[feat]}"
            )
    return True, ""


def predict(temperature: float, ph: float, salinity: float,
            turbidity: float) -> dict:
    """
    Run a prediction from raw (unscaled) feature values.

    Returns a dict with:
        prediction   : int  (0 or 1)
        label        : str  ("Unsafe" or "Safe")
        prob_unsafe  : float
        prob_safe    : float
        valid        : bool
        error        : str or None
    """
    values = {
        "Temperature": temperature,
        "pH":          ph,
        "Salinity":    salinity,
        "Turbidity":   turbidity,
    }

    valid, err = validate_inputs(values)
    if not valid:
        return {"prediction": None, "label": None,
                "prob_unsafe": None, "prob_safe": None,
                "valid": False, "error": err}

    # Build named DataFrame — required to preserve feature names through the pipeline
    X = pd.DataFrame([values])[FEATURES]

    pipe = load_pipeline()
    # Suppress the sklearn feature-name warning emitted when StandardScaler converts
    # the named DataFrame to a numpy array before passing to RF (cosmetic only).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        pred  = int(pipe.predict(X)[0])
        probs = pipe.predict_proba(X)[0]          # [P(Unsafe), P(Safe)]

    return {
        "prediction":  pred,
        "label":       CLASS_LABELS[pred],
        "prob_unsafe": round(float(probs[0]), 4),
        "prob_safe":   round(float(probs[1]), 4),
        "valid":       True,
        "error":       None,
    }


def get_pipeline_info() -> dict:
    """Return metadata about the loaded pipeline for display purposes."""
    pipe = load_pipeline()
    return {
        "steps":          [n for n, _ in pipe.steps],
        "model_type":     type(pipe.named_steps["model"]).__name__,
        "n_estimators": getattr(pipe.named_steps["model"], "n_estimators", None),
        "features":       FEATURES,
        "scaler_mean":    dict(zip(FEATURES,
                                  pipe.named_steps["scaler"].mean_.round(3))),
        "scaler_scale":   dict(zip(FEATURES,
                                  pipe.named_steps["scaler"].scale_.round(3))),
    }
