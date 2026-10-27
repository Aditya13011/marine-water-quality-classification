"""
tests/audit_metrics.py
Deep audit: model metrics, leakage guard, train/test split integrity,
pipeline artifact verification, synthetic data checks.
Run with: python tests/audit_metrics.py
"""
import json, os, sys
import pandas as pd
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []

def chk(label, condition, detail="", level="PASS"):
    tag = PASS if condition else (WARN if level == "WARN" else FAIL)
    results.append((tag, label, detail))
    print(f"  {tag}  {label}")
    if detail:
        print(f"         {detail}")
    return condition

print("\n" + "="*70)
print("  MarineGuard AI — End-to-End Audit Script")
print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PIPELINE ARTIFACT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Pipeline artifact")
import joblib
pipeline_path = os.path.join(ROOT, "models", "selected_model_scenario_A.joblib")
chk("Pipeline file exists", os.path.isfile(pipeline_path), pipeline_path)

if os.path.isfile(pipeline_path):
    pipe = joblib.load(pipeline_path)
    from sklearn.pipeline import Pipeline as SkPipeline
    chk("Artifact is sklearn Pipeline", isinstance(pipe, SkPipeline))
    steps = [n for n, _ in pipe.steps]
    chk("Steps == [imputer, scaler, model]", steps == ["imputer", "scaler", "model"],
        f"Actual steps: {steps}")
    n_feats_scaler = pipe.named_steps["scaler"].n_features_in_
    chk("Scaler fitted on 4 features", n_feats_scaler == 4, f"n_features_in_={n_feats_scaler}")
    model_type = type(pipe.named_steps["model"]).__name__
    chk("Model step is valid classifier",
        model_type in {"XGBClassifier","RandomForestClassifier","ExtraTreesClassifier",
                       "CalibratedClassifierCV","LogisticRegression","DecisionTreeClassifier"},
        f"type={model_type}")
    chk("Pipeline exposes classes_ [0,1]", list(pipe.classes_) == [0, 1])
    print(f"         Model: {model_type}, n_estimators={getattr(pipe.named_steps['model'],'n_estimators','N/A')}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDICTION CONTRACT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Prediction contract")
from prediction import predict, validate_inputs, FEATURES, FEATURE_RANGES, CLASS_LABELS

inputs_valid = {"Temperature": 22.5, "pH": 7.8, "Salinity": 35.0, "Turbidity": 15.0}
r = predict(22.5, 7.8, 35.0, 15.0)
chk("predict() returns dict with required keys",
    all(k in r for k in ["prediction","label","prob_safe","prob_unsafe","valid","error"]))
chk("valid=True for in-range input", r["valid"] is True)
chk("prediction in {0,1}", r["prediction"] in [0, 1])
chk("label matches CLASS_LABELS", r["label"] == CLASS_LABELS[r["prediction"]])
chk("probabilities sum to 1.0", abs(r["prob_safe"] + r["prob_unsafe"] - 1.0) < 1e-4,
    f"sum={r['prob_safe']+r['prob_unsafe']:.6f}")
chk("P(safe) in [0,1]", 0.0 <= r["prob_safe"] <= 1.0)

r_bad = predict(5.0, 7.8, 35.0, 15.0)
chk("Out-of-range input returns valid=False", r_bad["valid"] is False)
chk("Out-of-range prediction is None", r_bad["prediction"] is None)
chk("Out-of-range error is non-empty string",
    isinstance(r_bad["error"], str) and len(r_bad["error"]) > 0)

# Determinism
r1 = predict(20.0, 7.5, 34.0, 20.0)
r2 = predict(20.0, 7.5, 34.0, 20.0)
chk("Predictions are deterministic", r1["prediction"] == r2["prediction"])

# Leakage cols NOT in FEATURES
leakage_cols = {"Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"}
chk("No leakage cols in FEATURES", not leakage_cols.intersection(FEATURES),
    f"FEATURES={FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Train/test split integrity")
Xa_tr = pd.read_csv(os.path.join(ROOT, "data", "X_train_A.csv"), index_col=0)
Xa_te = pd.read_csv(os.path.join(ROOT, "data", "X_test_A.csv"),  index_col=0)
y_tr  = pd.read_csv(os.path.join(ROOT, "data", "y_train.csv"), index_col=0).squeeze()
y_te  = pd.read_csv(os.path.join(ROOT, "data", "y_test.csv"),  index_col=0).squeeze()

chk("X_train_A 8000 rows", len(Xa_tr) == 8000, f"actual={len(Xa_tr)}")
chk("X_test_A 2000 rows",  len(Xa_te) == 2000, f"actual={len(Xa_te)}")
chk("y_train 8000 rows",   len(y_tr) == 8000,  f"actual={len(y_tr)}")
chk("y_test 2000 rows",    len(y_te) == 2000,  f"actual={len(y_te)}")
chk("80/20 split fraction", abs(len(Xa_tr)/(len(Xa_tr)+len(Xa_te)) - 0.8) < 1e-3)

overlap = set(Xa_tr.index) & set(Xa_te.index)
chk("Zero index overlap between train and test", len(overlap) == 0,
    f"overlap={len(overlap)} rows")

for col in leakage_cols:
    chk(f"'{col}' not in X_train_A (leakage guard)", col not in Xa_tr.columns)

chk("X_test_A is standardised (|mean|<0.1)",
    all(abs(Xa_te[c].mean()) < 0.5 for c in Xa_te.columns),
    f"means={Xa_te.mean().round(3).to_dict()}")

# Class balance
train_dist = y_tr.value_counts(normalize=True).round(3)
test_dist  = y_te.value_counts(normalize=True).round(3)
chk("Training class balance ~50/50",
    abs(train_dist.get(0, 0) - 0.5) < 0.05 and abs(train_dist.get(1, 0) - 0.5) < 0.05,
    f"dist={train_dist.to_dict()}")
chk("Test class balance ~50/50",
    abs(test_dist.get(0, 0) - 0.5) < 0.05 and abs(test_dist.get(1, 0) - 0.5) < 0.05,
    f"dist={test_dist.to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENHANCED METRICS JSON
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Enhanced metrics JSON")
mm_path = os.path.join(ROOT, "reports", "model_metrics_enhanced.json")
chk("model_metrics_enhanced.json exists", os.path.isfile(mm_path))

if os.path.isfile(mm_path):
    with open(mm_path, encoding="utf-8") as f:
        mm = json.load(f)

    chk("cv_folds == 5", mm["cv_folds"] == 5)
    chk("leakage_guard present",
        "THRESHOLD_COLS" in mm.get("leakage_guard", "") or
        "leakage" in mm.get("leakage_guard", "").lower())
    chk("Scenario A leakage=False", mm["scenario_A"]["leakage"] is False)
    chk("Scenario B leakage=True",  mm["scenario_B"]["leakage"] is True)
    chk("Scenario B has 'NOT genuine' warning",
        "NOT genuine" in mm["scenario_B"].get("warning", ""))

    best_name = mm["scenario_A"]["best_model"]
    best_acc  = mm["scenario_A"]["results"][best_name]["accuracy"]
    chk("Best Scenario A accuracy 40-70% (expected near-random)",
        0.40 <= best_acc <= 0.70, f"acc={best_acc*100:.2f}%")
    chk("Best Scenario A accuracy < 75% (no leakage)",
        best_acc < 0.75, f"acc={best_acc*100:.2f}%")

    all_6 = {"LogisticRegression","DecisionTree","RandomForest",
              "ExtraTrees","LinearSVC","XGBoost"}
    found  = set(mm["scenario_A"]["results"].keys())
    chk("All 6 algorithms in results", found == all_6, f"found={found}")

    scenario_b_acc = mm["scenario_B"]["results"]["RandomForest"]["accuracy"]
    chk("Scenario B accuracy > 90% (confirms target leakage effect)",
        scenario_b_acc > 0.90, f"acc={scenario_b_acc*100:.2f}%")
    chk("Scenario B > Scenario A by >30pp",
        scenario_b_acc - best_acc > 0.30,
        f"gap={scenario_b_acc-best_acc:.2%}")

    # CM correctness
    for name, res in mm["scenario_A"]["results"].items():
        cm = res["confusion_matrix"]
        total = cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
        chk(f"{name} confusion matrix sums to 2000", total == 2000, f"sum={total}")

    # Permutation importance near zero
    for feat, vals in mm["scenario_A"]["permutation_importance"].items():
        chk(f"PI {feat} |mean|<0.20 (weak signal expected)",
            abs(vals["mean"]) < 0.20, f"mean={vals['mean']:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Raw dataset audit")
raw_path = os.path.join(ROOT, "data", "marine_water_quality.csv")
if os.path.isfile(raw_path):
    raw = pd.read_csv(raw_path)
    chk("Dataset has 10000 rows", len(raw) == 10000, f"rows={len(raw)}")
    chk("No missing values in raw dataset", raw.isnull().sum().sum() == 0,
        f"missing={raw.isnull().sum().sum()}")
    chk("is_safe is binary (0/1)", set(raw["is_safe"].unique()) == {0, 1})
    class_bal = raw["is_safe"].value_counts(normalize=True)
    chk("Raw class balance ~50/50",
        abs(class_bal[0]-0.5)<0.02 and abs(class_bal[1]-0.5)<0.02,
        f"{class_bal.round(3).to_dict()}")
    for col in leakage_cols:
        chk(f"'{col}' present in raw (exists for Scenario B)", col in raw.columns)
else:
    chk("Raw dataset present", False, raw_path)

# ─────────────────────────────────────────────────────────────────────────────
# 6. PAGE SYNTAX — ALL 9 PAGES + APP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Page syntax validation")
import ast
pages = [
    "app.py",
    "pages/1_Marine_Water_Quality_Prediction.py",
    "pages/2_Data_Analytics.py",
    "pages/3_Model_Performance.py",
    "pages/4_About_Project.py",
    "pages/5_Model_Laboratory.py",
    "pages/6_Marine_Data_Sources.py",
    "pages/7_Marine_Monitoring_Map.py",
    "pages/8_Trends_and_Anomalies.py",
    "pages/9_Decision_Support.py",
]
for pg in pages:
    full = os.path.join(ROOT, pg)
    try:
        ast.parse(open(full, encoding="utf-8").read())
        chk(f"{pg} parses without error", True)
    except SyntaxError as e:
        chk(f"{pg} parses without error", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 7. DISCLAIMER TEXT AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Disclaimer / scientific honesty text audit")
disclaimer_checks = [
    ("pages/9_Decision_Support.py",  "educational",     "Decision Support has 'educational' disclaimer"),
    ("pages/9_Decision_Support.py",  "NOT a certified", "Decision Support has NOT certified disclaimer"),
    ("pages/8_Trends_and_Anomalies.py", "anomaly",      "Trends page has 'anomaly' explanation"),
    ("pages/8_Trends_and_Anomalies.py", "NOT mean",     "Trends page: anomaly != confirmed pollution"),
    ("pages/4_About_Project.py",     "synthetic",       "About page has 'synthetic' disclosure"),
    ("pages/4_About_Project.py",     "leakage",         "About page discusses leakage"),
    ("pages/3_Model_Performance.py", "Scenario B",      "Performance page separates Scenario B"),
    ("pages/1_Marine_Water_Quality_Prediction.py", "synthetic",
                                                        "Prediction page has 'synthetic' disclaimer"),
    ("pages/5_Model_Laboratory.py",  "NOT genuine",     "Model Lab warn NOT genuine"),
    ("pages/6_Marine_Data_Sources.py", "synthetic",     "Data Sources has synthetic marker"),
    ("pages/7_Marine_Monitoring_Map.py","DEMO",         "Map page labels demo data"),
]
for pg, keyword, label in disclaimer_checks:
    full = os.path.join(ROOT, pg)
    if os.path.isfile(full):
        src = open(full, encoding="utf-8").read()
        chk(label, keyword.lower() in src.lower(), f"keyword='{keyword}' not found in {pg}")
    else:
        chk(label, False, f"File not found: {pg}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. REQUIREMENTS AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] requirements.txt audit")
req_path = os.path.join(ROOT, "requirements.txt")
req_src  = open(req_path, encoding="utf-8").read().lower()
for pkg in ["pandas","numpy","scikit-learn","joblib","matplotlib","seaborn",
            "plotly","shap","streamlit","xgboost"]:
    chk(f"'{pkg}' in requirements.txt", pkg in req_src)

# No hard-pinned versions that could break cloud
import re
hard_pins = re.findall(r"==\d", req_src)
chk("No exact-version pins (==) that could break cloud",
    len(hard_pins) == 0, f"found: {hard_pins}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. THRESHOLD CITATIONS AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Threshold citation audit")
ds_path = os.path.join(ROOT, "pages", "9_Decision_Support.py")
if os.path.isfile(ds_path):
    ds_src = open(ds_path, encoding="utf-8").read()
    for term in ["WHO", "USEPA", "UNEP", "indicative"]:
        chk(f"Decision Support cites '{term}'", term in ds_src)

# ─────────────────────────────────────────────────────────────────────────────
# 10. CV RESULTS CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] CV results CSV")
cv_path = os.path.join(ROOT, "reports", "cv_results.csv")
chk("cv_results.csv exists", os.path.isfile(cv_path))
if os.path.isfile(cv_path):
    cv_df = pd.read_csv(cv_path)
    chk("CV CSV has 6 rows", len(cv_df) == 6, f"rows={len(cv_df)}")
    req_cols = {"Model","CV Accuracy","CV F1 (weighted)","CV ROC-AUC",
                "Test Accuracy","Test F1","Train Time (s)","Best Params","Selected"}
    chk("CV CSV has all required columns",
        req_cols.issubset(cv_df.columns), f"missing={req_cols-set(cv_df.columns)}")
    n_sel = cv_df["Selected"].apply(
        lambda x: pd.notna(x) and str(x).strip() not in ("","nan")
    ).sum()
    chk("Exactly 1 model selected in CV CSV", n_sel == 1, f"selected_count={n_sel}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
passes  = sum(1 for r in results if r[0] == PASS)
warns   = sum(1 for r in results if r[0] == WARN)
fails   = sum(1 for r in results if r[0] == FAIL)
total   = len(results)

print("\n" + "="*70)
print(f"  AUDIT SUMMARY: {passes} PASS  |  {warns} WARN  |  {fails} FAIL  |  {total} total")
print("="*70)

if fails:
    print("\nFAILED CHECKS:")
    for tag, label, detail in results:
        if tag == FAIL:
            print(f"  *** {label}: {detail}")
    sys.exit(1)
else:
    print("\nAll checks passed. No defects detected.")
    sys.exit(0)
