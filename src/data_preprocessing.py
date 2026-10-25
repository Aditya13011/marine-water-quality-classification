"""
src/data_preprocessing.py
──────────────────────────────────────────────────────────────────────────────
Phase 2 — Feature Engineering and Leakage-Aware Preprocessing
Project : AI-Based Water Quality Prediction and Potability Analysis System
Dataset : Marine Water Quality Classification (Zenodo record 19506986)

EXPERIMENT DESIGN RATIONALE
────────────────────────────
The target column `is_safe` is a **deterministic** function of three columns:
    Safe (1) ⟺  Dissolved_Oxygen > 8  AND  Nitrate < 15  AND  Chlorophyll_a < 60

Including those three columns as input features would allow a trivial
threshold-checking classifier to achieve ~100 % accuracy — no actual
learning would occur.

This module therefore implements TWO feature sets so that both scenarios
can be compared honestly during Phase 3 model training:

  Scenario A — "Leakage-Excluded" (PRIMARY / RECOMMENDED)
      Features : Temperature, pH, Salinity, Turbidity
      Rationale: Tests genuine predictive power of features that are NOT
                 used to construct the target.  Results will be lower than
                 leakage scenario but are scientifically honest.

  Scenario B — "All Features" (SECONDARY / REFERENCE ONLY)
      Features : Temperature, pH, Salinity, Turbidity,
                 Dissolved_Oxygen, Nitrate, Chlorophyll_a
      Rationale: Retained solely to quantify the leakage effect.
                 Accuracy will be near-perfect and should NOT be presented
                 as genuine model performance.

NOTE: This module does NOT claim to eliminate every possible leakage source.
Correlations between Temperature and Dissolved Oxygen (temperature-dependent
dissolution) may still carry indirect signal.  This is documented below.

WHAT THIS MODULE DOES
──────────────────────
1. Loads the raw CSV (never modifies it).
2. Drops Sample_ID (identifier, no predictive value).
3. Defines feature sets (Scenario A and B).
4. Performs train/test split BEFORE any fitting (stratified, seed=42).
5. Builds a Scikit-learn Pipeline (SimpleImputer → StandardScaler) and
   fits it on training data ONLY — test data is only transformed, never seen.
6. Saves:
      data/X_train_A.csv, X_test_A.csv, y_train.csv, y_test.csv  (Scenario A)
      data/X_train_B.csv, X_test_B.csv                           (Scenario B)
      data/split_metadata.json   (reproducibility record)
7. Saves the fitted pipelines to:
      models/pipeline_scenario_A.joblib
      models/pipeline_scenario_B.joblib
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH   = os.path.join(ROOT, "data", "marine_water_quality.csv")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
TEST_SIZE      = 0.20          # 80 / 20 split
TARGET         = "is_safe"
ID_COL         = "Sample_ID"

# ⚠ These three columns DEFINE the target.  Excluding them in Scenario A.
THRESHOLD_COLS = ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]

# Primary features — do NOT include threshold-defining columns
FEATURES_A = ["Temperature", "pH", "Salinity", "Turbidity"]

# All numeric features (includes threshold cols) — reference only
FEATURES_B = FEATURES_A + THRESHOLD_COLS

# ──────────────────────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Returns a preprocessing pipeline:
        Step 1 — SimpleImputer (median strategy)
            Handles any future missing values without data snooping.
            Fitted on training split only.
        Step 2 — StandardScaler
            Zero-mean, unit-variance normalisation.
            Fitted on training split only.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def run_preprocessing():
    print("=" * 70)
    print("PHASE 2 — DATA PREPROCESSING & SPLIT")
    print("=" * 70)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if not os.path.isfile(CSV_PATH):
        sys.exit(f"ERROR: dataset not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"\n[1] Loaded raw CSV : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"     Columns        : {list(df.columns)}")

    # Verify expected shape
    assert df.shape == (10_000, 9), \
        f"Unexpected shape {df.shape} — expected (10000, 9)"
    assert df.isnull().sum().sum() == 0, "Unexpected missing values in raw CSV"
    assert df.duplicated().sum() == 0,   "Unexpected duplicate rows in raw CSV"
    print("  ✔ Shape, missing values, and duplicates verified.")

    # ── 2. Drop identifier ────────────────────────────────────────────────────
    df = df.drop(columns=[ID_COL])
    print(f"\n[2] Dropped identifier column '{ID_COL}'.")

    # ── 3. Separate features and target ───────────────────────────────────────
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    print(f"\n[3] Target column  : '{TARGET}'")
    print(f"     Class balance  : {dict(y.value_counts().sort_index())}")

    # ── 4. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,              # preserve 50/50 balance in both splits
    )

    print(f"\n[4] TRAIN / TEST SPLIT  (seed={RANDOM_SEED}, stratified)")
    print(f"     Training rows  : {len(X_train):,}  ({len(X_train)/len(X)*100:.0f}%)")
    print(f"     Test rows      : {len(X_test):,}   ({len(X_test)/len(X)*100:.0f}%)")
    print(f"     Train target   : {dict(y_train.value_counts().sort_index())}")
    print(f"     Test  target   : {dict(y_test.value_counts().sort_index())}")

    # ── 5. Build and fit pipelines ────────────────────────────────────────────
    print("\n[5] BUILDING PIPELINES")

    results = {}
    for scenario, feat_cols in [("A", FEATURES_A), ("B", FEATURES_B)]:
        lbl = "(PRIMARY — leakage excluded)" if scenario == "A" \
              else "(SECONDARY — all features, reference only)"
        print(f"\n  Scenario {scenario} {lbl}")
        print(f"    Features ({len(feat_cols)}): {feat_cols}")

        # Verify all expected features exist in the DataFrame
        missing_cols = [c for c in feat_cols if c not in X.columns]
        if missing_cols:
            sys.exit(f"ERROR: expected columns not found in dataset: {missing_cols}")
        print(f"    ✔ All {len(feat_cols)} feature columns verified in dataset.")

        X_tr = X_train[feat_cols]
        X_te = X_test[feat_cols]

        # Fit pipeline on TRAINING data only
        pipe = build_pipeline()
        X_tr_processed = pipe.fit_transform(X_tr)   # fit + transform train
        X_te_processed = pipe.transform(X_te)        # transform test only (no fit)

        print(f"    ✔ Pipeline fitted on training data only.")

        # Convert to DataFrames with original column names
        X_tr_df = pd.DataFrame(X_tr_processed, columns=feat_cols,
                               index=X_train.index)
        X_te_df = pd.DataFrame(X_te_processed, columns=feat_cols,
                               index=X_test.index)

        # Verify scaled stats (train should be ~N(0,1))
        tr_means = X_tr_df.mean().round(4)
        tr_stds  = X_tr_df.std().round(4)
        print(f"    Post-scaling train means (should be ≈0): {dict(tr_means)}")
        print(f"    Post-scaling train stds  (should be ≈1): {dict(tr_stds)}")

        # Save processed CSVs
        train_out = os.path.join(DATA_DIR, f"X_train_{scenario}.csv")
        test_out  = os.path.join(DATA_DIR, f"X_test_{scenario}.csv")
        X_tr_df.to_csv(train_out, index=True)
        X_te_df.to_csv(test_out,  index=True)
        print(f"    ✔ Saved: {os.path.basename(train_out)}")
        print(f"    ✔ Saved: {os.path.basename(test_out)}")

        # Save fitted pipeline
        pipe_out = os.path.join(MODELS_DIR, f"pipeline_scenario_{scenario}.joblib")
        joblib.dump(pipe, pipe_out)
        print(f"    ✔ Saved: models/pipeline_scenario_{scenario}.joblib")

        results[scenario] = {
            "features":          feat_cols,
            "n_features":        len(feat_cols),
            "n_train":           len(X_tr_df),
            "n_test":            len(X_te_df),
            "train_scaled_mean": {k: float(v) for k, v in tr_means.items()},
            "train_scaled_std":  {k: float(v) for k, v in tr_stds.items()},
        }

    # ── 6. Save y_train / y_test (shared across scenarios) ───────────────────
    y_train_path = os.path.join(DATA_DIR, "y_train.csv")
    y_test_path  = os.path.join(DATA_DIR, "y_test.csv")
    y_train.to_csv(y_train_path, header=True)
    y_test.to_csv(y_test_path,  header=True)
    print(f"\n[6] Saved: y_train.csv  ({len(y_train):,} records)")
    print(f"     Saved: y_test.csv   ({len(y_test):,} records)")

    # ── 7. Save metadata JSON ─────────────────────────────────────────────────
    metadata = {
        "phase":             "Phase 2 — Data Preprocessing",
        "dataset_source":    "https://zenodo.org/records/19506986",
        "dataset_provenance": (
            "Mathematically derived / synthetic.  Not real laboratory data.  "
            "Target `is_safe` is a deterministic function of Dissolved_Oxygen, "
            "Nitrate, and Chlorophyll_a thresholds."
        ),
        "random_seed":       RANDOM_SEED,
        "test_size_fraction":TEST_SIZE,
        "stratified":        True,
        "total_records":     int(len(y)),
        "n_train":           int(len(y_train)),
        "n_test":            int(len(y_test)),
        "target_column":     TARGET,
        "target_classes":    {
            "0": "Unsafe",
            "1": "Safe",
        },
        "train_class_dist": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "test_class_dist": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
        },
        "preprocessing_pipeline": [
            {
                "step": "SimpleImputer",
                "strategy": "median",
                "note": "Fitted on training data only."
            },
            {
                "step": "StandardScaler",
                "note": "Zero-mean unit-variance. Fitted on training data only."
            }
        ],
        "leakage_exclusion": {
            "threshold_defining_columns": THRESHOLD_COLS,
            "explanation": (
                "Dissolved_Oxygen, Nitrate, and Chlorophyll_a directly define "
                "the target via a hard threshold rule.  Including them would allow "
                "trivial rule-matching rather than genuine learning.  They are "
                "EXCLUDED from Scenario A (primary model)."
            ),
            "residual_leakage_note": (
                "Temperature is physically correlated with Dissolved Oxygen "
                "(temperature-dependent gas solubility).  This indirect correlation "
                "may carry weak residual signal. This is acknowledged but not "
                "eliminated — documentation is provided rather than removal, "
                "as Temperature is a legitimate marine quality indicator."
            )
        },
        "scenarios":         results,
        "output_files": {
            "X_train_A.csv":                     "Scenario A train features (scaled)",
            "X_test_A.csv":                      "Scenario A test features (scaled)",
            "X_train_B.csv":                     "Scenario B train features (scaled)",
            "X_test_B.csv":                      "Scenario B test features (scaled)",
            "y_train.csv":                       "Training labels",
            "y_test.csv":                        "Test labels",
            "pipeline_scenario_A.joblib":        "Fitted preprocessing pipeline — A",
            "pipeline_scenario_B.joblib":        "Fitted preprocessing pipeline — B",
        }
    }

    meta_path = os.path.join(DATA_DIR, "split_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"     Saved: split_metadata.json")

    # ── 8. Final verification printout ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 PREPROCESSING — FINAL VERIFICATION")
    print("=" * 70)

    print(f"\n  Total records    : {len(y):,}")
    print(f"  Training set     : {len(y_train):,} rows")
    print(f"  Test set         : {len(y_test):,} rows")
    print(f"  Random seed      : {RANDOM_SEED}")
    print(f"  Stratified       : Yes")

    print(f"\n  SCENARIO A — PRIMARY (leakage excluded)")
    print(f"    Features ({results['A']['n_features']}): {results['A']['features']}")
    print(f"    Train shape: ({results['A']['n_train']}, {results['A']['n_features']})")
    print(f"    Test  shape: ({results['A']['n_test']}, {results['A']['n_features']})")

    print(f"\n  SCENARIO B — REFERENCE ONLY (all features, leakage present)")
    print(f"    Features ({results['B']['n_features']}): {results['B']['features']}")
    print(f"    Train shape: ({results['B']['n_train']}, {results['B']['n_features']})")
    print(f"    Test  shape: ({results['B']['n_test']}, {results['B']['n_features']})")

    print(f"\n  Files saved to data/:")
    for fname, desc in metadata["output_files"].items():
        path = os.path.join(DATA_DIR, fname) if not fname.endswith(".joblib") \
               else os.path.join(MODELS_DIR, fname)
        exists = "✔" if os.path.isfile(path) else "✘ MISSING"
        print(f"    {exists} {fname:<45} {desc}")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE — No models trained yet.")
    print("Proceed to Phase 3: Model Training and Evaluation.")
    print("=" * 70)


if __name__ == "__main__":
    run_preprocessing()
