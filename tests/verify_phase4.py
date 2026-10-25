"""Phase 4 verification script — run before declaring dashboard complete."""
import sys, os, ast, pathlib, json

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

errors = []

# 1. imports
print("--- Checking core imports ---")
try:
    import pandas as pd;    print("  pandas     OK")
    import numpy as np;     print("  numpy      OK")
    import joblib;          print("  joblib     OK")
    import sklearn;         print("  sklearn    OK")
    import plotly;          print("  plotly     OK")
    import streamlit;       print("  streamlit  OK")
    import shap;            print("  shap       OK")
except ImportError as e:
    errors.append(f"Import error: {e}")
    print(f"  ERROR: {e}")

# 2. prediction module
print("\n--- Checking src/prediction.py ---")
try:
    from prediction import load_pipeline, predict, get_pipeline_info
    info = get_pipeline_info()
    print(f"  Pipeline steps : {info['steps']}")
    print(f"  Model type     : {info['model_type']}")
    print(f"  Features       : {info['features']}")
    assert info["features"] == ["Temperature", "pH", "Salinity", "Turbidity"]
    print("  Feature order  : CORRECT")
except Exception as e:
    errors.append(f"prediction module: {e}")
    print(f"  ERROR: {e}")

# 3. smoke-test predict
print("\n--- Smoke-test prediction ---")
try:
    result = predict(22.5, 7.8, 35.0, 15.0)
    assert result["valid"] is True, "Validation failed"
    assert result["prediction"] in [0, 1], "Bad prediction value"
    assert 0.0 <= result["prob_safe"] <= 1.0
    print(f"  Input : Temp=22.5  pH=7.8  Salinity=35.0  Turbidity=15.0")
    print(f"  Result: label={result['label']}  prob_safe={result['prob_safe']}  prob_unsafe={result['prob_unsafe']}")
    print("  Prediction OK")
except Exception as e:
    errors.append(f"predict(): {e}")
    print(f"  ERROR: {e}")

# 4. model files
print("\n--- Checking model files ---")
model_files = [
    "models/selected_model_scenario_A.joblib",
    "models/selected_model_scenario_B_reference.joblib",
    "models/pipeline_scenario_A.joblib",
    "models/pipeline_scenario_B.joblib",
]
for mf in model_files:
    path = os.path.join(ROOT, mf)
    status = "OK" if os.path.isfile(path) else "MISSING"
    if status == "MISSING":
        errors.append(f"Missing: {mf}")
    print(f"  {mf}: {status}")

# 5. data files
print("\n--- Checking data files ---")
data_files = [
    "data/marine_water_quality.csv",
    "data/X_train_A.csv",
    "data/X_test_A.csv",
    "data/y_train.csv",
    "data/y_test.csv",
    "data/split_metadata.json",
]
for df in data_files:
    path = os.path.join(ROOT, df)
    status = "OK" if os.path.isfile(path) else "MISSING"
    if status == "MISSING":
        errors.append(f"Missing: {df}")
    print(f"  {df}: {status}")

# 6. report files
print("\n--- Checking report files ---")
report_files = [
    "reports/model_metrics.json",
    "reports/model_comparison.csv",
    "reports/shap_summary.png",
    "reports/confusion_matrices.png",
    "reports/model_comparison.png",
    "reports/feature_importance.png",
    "reports/01_class_distribution.png",
    "reports/02_feature_distributions.png",
    "reports/03_correlation_heatmap.png",
]
for rf in report_files:
    path = os.path.join(ROOT, rf)
    status = "OK" if os.path.isfile(path) else "MISSING"
    if status == "MISSING":
        errors.append(f"Missing: {rf}")
    print(f"  {rf}: {status}")

# 7. metrics sanity check from JSON
print("\n--- Verifying saved metrics ---")
try:
    with open(os.path.join(ROOT, "reports", "model_metrics.json")) as f:
        meta = json.load(f)
    rf = meta["scenario_A"]["results"]["RandomForest"]
    assert abs(rf["accuracy"]  - 0.5135) < 0.001, f"Accuracy mismatch: {rf['accuracy']}"
    assert abs(rf["f1_score"]  - 0.5083) < 0.001, f"F1 mismatch: {rf['f1_score']}"
    assert abs(rf["roc_auc"]   - 0.5141) < 0.001, f"AUC mismatch: {rf['roc_auc']}"
    assert meta["scenario_A"]["best_model"] == "RandomForest"
    assert meta["shap"]["succeeded"] is True
    print("  RandomForest Scenario A metrics verified: OK")
    print(f"  Accuracy={rf['accuracy']}  F1={rf['f1_score']}  AUC={rf['roc_auc']}")
except Exception as e:
    errors.append(f"Metrics check: {e}")
    print(f"  ERROR: {e}")

# 8. Syntax check pages
print("\n--- Syntax check: app.py + all pages ---")
pages_dir  = pathlib.Path(os.path.join(ROOT, "pages"))
check_files = list(pages_dir.glob("*.py")) + [pathlib.Path(os.path.join(ROOT, "app.py"))]
for p in sorted(check_files):
    try:
        ast.parse(p.read_text(encoding="utf-8"))
        print(f"  {p.name}: OK")
    except SyntaxError as e:
        errors.append(f"SyntaxError in {p.name} line {e.lineno}: {e.msg}")
        print(f"  {p.name}: SYNTAX ERROR line {e.lineno} — {e.msg}")

# 9. prediction utility: test boundary values
print("\n--- Boundary validation tests ---")
tests = [
    (22.5, 7.8, 35.0, 15.0,  True,  "valid mid-range"),
    (9.9,  7.8, 35.0, 15.0,  False, "Temp below min"),
    (10.0, 7.8, 35.0, 15.0,  True,  "Temp at min"),
    (32.0, 8.8, 40.0, 50.0, True,   "all at max"),
    (22.5, 9.0, 35.0, 15.0,  False, "pH above max"),
]
for temp, ph, sal, turb, expected_valid, desc in tests:
    r = predict(temp, ph, sal, turb)
    ok = r["valid"] == expected_valid
    marker = "OK" if ok else "FAIL"
    if not ok:
        errors.append(f"Boundary test '{desc}': expected valid={expected_valid}, got {r['valid']}")
    print(f"  [{marker}] {desc}: valid={r['valid']}")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"VERIFICATION FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  ✘ {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED ✔")
    print("Dashboard is ready to run:")
    print("  .venv\\Scripts\\streamlit.exe run app.py")
print("=" * 60)
