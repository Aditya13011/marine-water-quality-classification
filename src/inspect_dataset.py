"""
Phase 1 – Dataset Inspection Script
Project : AI-Based Water Quality Prediction and Potability Analysis System
Dataset : Marine Water Quality Classification (Zenodo record 19506986)
Author  : Phase 1 automated inspection
Date    : 2026-10-25

NOTE ON DATA PROVENANCE
-----------------------
This dataset contains exactly 10,000 mathematically derived records.
It is NOT real-world laboratory data.  The measurements are synthesised
using biogeochemical model equations documented in the Zenodo record.
The classification target is deterministic — an instance is labelled
'Safe' (1) if and only if ALL three criteria hold:
    • Dissolved Oxygen (DO) > 8 mg/L
    • Nitrate            < 15 mg/L
    • Chlorophyll-a      < 60 µg/L
All other instances are labelled 'Unsafe' (0).
This deterministic relationship may cause data-leakage during model
evaluation if DO, Nitrate, or Chlorophyll-a are used as direct features,
and should be taken into account in Phase 2.
"""

import sys
import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "marine_water_quality.csv")
CSV_PATH = os.path.normpath(CSV_PATH)

print("=" * 70)
print("PHASE 1 – MARINE WATER QUALITY DATASET INSPECTION")
print("=" * 70)

# ── 1. File existence check ───────────────────────────────────────────────────
print(f"\n[1] File path : {CSV_PATH}")
if not os.path.isfile(CSV_PATH):
    print("ERROR: File does not exist. Download may have failed.")
    sys.exit(1)
print(f"     File size : {os.path.getsize(CSV_PATH):,} bytes")

# ── 2. Load CSV ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"\n[2] Successfully loaded with pandas.")

# ── 3. Shape ──────────────────────────────────────────────────────────────────
rows, cols = df.shape
print(f"\n[3] SHAPE")
print(f"     Rows    : {rows:,}")
print(f"     Columns : {cols}")
if rows != 10_000:
    print(f"  ⚠  WARNING: Expected 10,000 rows but found {rows:,}. Stopping for review.")
    sys.exit(1)
else:
    print("  ✔  Row count matches expected 10,000.")

# ── 4. Column names ───────────────────────────────────────────────────────────
print(f"\n[4] COLUMN NAMES ({len(df.columns)} total)")
for i, col in enumerate(df.columns, 1):
    print(f"     {i:>2}. {col}")

# ── 5. Data types ─────────────────────────────────────────────────────────────
print(f"\n[5] DATA TYPES")
for col, dtype in df.dtypes.items():
    print(f"     {col:<35} {str(dtype)}")

# ── 6. Missing values ─────────────────────────────────────────────────────────
print(f"\n[6] MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
any_missing = False
for col in df.columns:
    m = missing[col]
    p = missing_pct[col]
    flag = "  ⚠" if m > 0 else "  ✔"
    print(f"  {flag} {col:<35} {m:>6} missing  ({p:.2f}%)")
    if m > 0:
        any_missing = True
if not any_missing:
    print("     → No missing values found.")

# ── 7. Duplicate records ──────────────────────────────────────────────────────
dupes = df.duplicated().sum()
print(f"\n[7] DUPLICATE RECORDS : {dupes:,}")
if dupes == 0:
    print("  ✔  No duplicate rows.")
else:
    print(f"  ⚠  {dupes} duplicate rows detected.")

# ── 8. Summary statistics ─────────────────────────────────────────────────────
print(f"\n[8] SUMMARY STATISTICS (numeric columns)")
desc = df.describe().T
desc_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
available = [c for c in desc_cols if c in desc.columns]
print(desc[available].to_string())

# ── 9. Target column identification ──────────────────────────────────────────
print(f"\n[9] TARGET COLUMN IDENTIFICATION")
# Look for columns with binary-like distributions
candidate_targets = []
for col in df.columns:
    unique_vals = sorted(df[col].dropna().unique().tolist())
    if unique_vals in [[0, 1], [0.0, 1.0]]:
        candidate_targets.append(col)
    elif set(unique_vals) == {"Safe", "Unsafe"} or set(str(v) for v in unique_vals) == {"0", "1"}:
        candidate_targets.append(col)

if candidate_targets:
    print(f"  Binary-valued column(s) found (likely target): {candidate_targets}")
    target_col = candidate_targets[-1]          # take the last one (typically rightmost)
else:
    # Try common names
    common = ["Quality", "quality", "Target", "target", "Label", "label",
              "Potability", "potability", "Class", "class"]
    target_col = None
    for c in common:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]
        print(f"  No obvious target found. Using last column: '{target_col}'")
    else:
        print(f"  Target column found by name: '{target_col}'")

print(f"\n[10] TARGET CLASS DISTRIBUTION  (column = '{target_col}')")
vc = df[target_col].value_counts().sort_index()
for label, count in vc.items():
    pct = count / len(df) * 100
    print(f"      Label {repr(label):>10} : {count:>6,} records  ({pct:.2f}%)")

# ── 11. Verify label scheme matches documentation ─────────────────────────────
print(f"\n[11] LABEL SCHEME VERIFICATION")
actual_labels = set(df[target_col].dropna().unique())
expected_labels = {0, 1}
if actual_labels == expected_labels or actual_labels == {0.0, 1.0}:
    print("  ✔  Labels match documented scheme: 0 = Unsafe, 1 = Safe")
else:
    print(f"  ⚠  Actual labels: {actual_labels}  (differ from expected {{0, 1}})")

# ── 12. Check deterministic threshold columns ─────────────────────────────────
print(f"\n[12] DETERMINISTIC THRESHOLD COLUMN CHECK")
key_cols = {
    "Dissolved_Oxygen": ("DO", ">", 8),
    "Dissolved Oxygen": ("DO", ">", 8),
    "DO": ("DO", ">", 8),
    "Nitrate": ("NO3", "<", 15),
    "Chlorophyll_a": ("Chl-a", "<", 60),
    "Chlorophyll-a": ("Chl-a", "<", 60),
    "Chlorophyll": ("Chl-a", "<", 60),
}
found_keys = {col: info for col, info in key_cols.items() if col in df.columns}
if found_keys:
    for col, (alias, op, thresh) in found_keys.items():
        print(f"  ✔  Found '{col}' (threshold {op} {thresh})")
else:
    print("  ℹ  Could not directly match threshold column names.")
    print("     Columns present:", list(df.columns))

# ── 13. Feature columns ───────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != target_col]
print(f"\n[13] FEATURE COLUMNS ({len(feature_cols)} total)")
for fc in feature_cols:
    print(f"      • {fc}")

print("\n" + "=" * 70)
print("PHASE 1 INSPECTION COMPLETE — No models trained.")
print("=" * 70)
