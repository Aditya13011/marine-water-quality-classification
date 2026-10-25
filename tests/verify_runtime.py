"""Runtime logic verification for all 5 dashboard pages (no Streamlit server needed)."""
import sys, os, json

ROOT = r"D:\water-quality-prediction"
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
import numpy as np

errors = []

# ── Home page ─────────────────────────────────────────────────────────────────
print("--- Home (app.py) data loading ---")
try:
    df   = pd.read_csv(os.path.join(ROOT, "data", "marine_water_quality.csv"))
    meta = json.load(open(os.path.join(ROOT, "reports", "model_metrics.json")))
    rf   = meta["scenario_A"]["results"]["RandomForest"]
    assert len(df) == 10000
    assert int((df["is_safe"] == 1).sum()) == 5000
    assert int((df["is_safe"] == 0).sum()) == 5000
    assert abs(rf["accuracy"] - 0.5135) < 0.001
    print(f"  Rows: {len(df):,}  Safe: 5000  Unsafe: 5000  RF accuracy: {rf['accuracy']*100:.2f}%")
    print("  PASS")
except Exception as e:
    errors.append(f"Home: {e}")
    print(f"  FAIL: {e}")

# ── Prediction page ───────────────────────────────────────────────────────────
print("\n--- Prediction page logic ---")
try:
    from prediction import predict, validate_inputs, FEATURES, FEATURE_RANGES
    r = predict(22.5, 7.8, 35.0, 15.0)
    assert r["valid"] is True
    assert r["prediction"] in [0, 1]
    assert 0.0 <= r["prob_safe"] <= 1.0
    r2 = predict(9.0, 7.8, 35.0, 15.0)   # Temp below 10.0 min
    assert r2["valid"] is False
    r3 = predict(22.5, 9.5, 35.0, 15.0)  # pH above 8.8 max
    assert r3["valid"] is False
    print(f"  predict(22.5,7.8,35,15) -> {r['label']}  p_safe={r['prob_safe']}")
    print("  Boundary validation: Temp<10 -> invalid OK | pH>8.8 -> invalid OK")
    print("  PASS")
except Exception as e:
    errors.append(f"Prediction: {e}")
    print(f"  FAIL: {e}")

# ── Data Analytics page ───────────────────────────────────────────────────────
print("\n--- Data Analytics page computations ---")
try:
    TARGET         = "is_safe"
    THRESHOLD_COLS = ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]
    SAFE_FEATS     = ["Temperature", "pH", "Salinity", "Turbidity"]
    ALL_NUMERIC    = [c for c in df.columns
                      if c not in [TARGET, "Sample_ID"]
                      and pd.api.types.is_numeric_dtype(df[c])]
    corr_df = df[ALL_NUMERIC + [TARGET]].corr().round(3)
    mv      = df.isnull().sum()
    stats   = df[SAFE_FEATS].describe().T.round(3)
    assert corr_df.shape == (8, 8), f"Expected 8x8 corr matrix, got {corr_df.shape}"
    assert mv.sum() == 0, "Missing values found!"
    corr_safe_feats = {f: round(df[f].corr(df[TARGET]), 4) for f in SAFE_FEATS}
    for f, r_val in corr_safe_feats.items():
        assert abs(r_val) < 0.05, f"{f} has r={r_val} — unexpectedly high!"
    print(f"  Correlation matrix: {corr_df.shape}  Missing: {mv.sum()}")
    print(f"  Safe-feature correlations: {corr_safe_feats}")
    print("  PASS")
except Exception as e:
    errors.append(f"Analytics: {e}")
    print(f"  FAIL: {e}")

# ── Model Performance page ────────────────────────────────────────────────────
print("\n--- Model Performance page data ---")
try:
    comp_df   = pd.read_csv(os.path.join(ROOT, "reports", "model_comparison.csv"))
    results_A = meta["scenario_A"]["results"]
    results_B = meta["scenario_B"]["results"]
    shap_ok   = meta.get("shap", {}).get("succeeded", False)
    assert len(comp_df) == 6, f"Expected 6 rows, got {len(comp_df)}"
    assert set(results_A.keys()) == {"LogisticRegression", "DecisionTree", "RandomForest"}
    assert set(results_B.keys()) == {"LogisticRegression", "DecisionTree", "RandomForest"}
    assert shap_ok is True
    assert meta["scenario_A"]["best_model"] == "RandomForest"
    assert meta["scenario_B"]["best_model"] == "LogisticRegression"
    # Verify SHAP values parse correctly
    raw_shap = meta["shap"]["mean_abs_shap"]
    for feat, val in raw_shap.items():
        if isinstance(val, str):
            nums = [float(x) for x in val.strip("[]").split()]
            assert all(v > 0 for v in nums)
    print(f"  Comp CSV: {len(comp_df)} rows | SHAP OK | best_A=RandomForest | best_B=LogisticRegression")
    print("  PASS")
except Exception as e:
    errors.append(f"ModelPerf: {e}")
    print(f"  FAIL: {e}")

# ── About page ────────────────────────────────────────────────────────────────
print("\n--- About page (static content only) ---")
try:
    # About page only reads ROOT-relative paths — verify key constants it uses
    assert os.path.isfile(os.path.join(ROOT, "data", "marine_water_quality.csv"))
    assert os.path.isfile(os.path.join(ROOT, "reports", "model_metrics.json"))
    print("  Static page — confirmed data files accessible")
    print("  PASS")
except Exception as e:
    errors.append(f"About: {e}")
    print(f"  FAIL: {e}")

# ── Report images ─────────────────────────────────────────────────────────────
print("\n--- Report images ---")
imgs = ["shap_summary.png", "confusion_matrices.png", "model_comparison.png",
        "feature_importance.png", "04_safe_features_boxplots.png",
        "05_threshold_leakage_analysis.png", "06_pairplot_safe_features.png"]
for img in imgs:
    p = os.path.join(ROOT, "reports", img)
    ok = os.path.isfile(p)
    if not ok:
        errors.append(f"Missing image: {img}")
    print(f"  {'OK' if ok else 'MISSING'}: {img}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  ✘ {e}")
    sys.exit(1)
else:
    print("ALL RUNTIME CHECKS PASSED ✔")
    print("=" * 60)
