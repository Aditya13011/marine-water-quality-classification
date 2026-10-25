"""
src/train_model.py
──────────────────────────────────────────────────────────────────────────────
Phase 3 — Model Training, Evaluation, and SHAP Interpretability
Project : AI-Based Water Quality Prediction and Potability Analysis System
Dataset : Marine Water Quality Classification (Zenodo record 19506986)

PRE-CONDITIONS (verified before this script was written)
─────────────────────────────────────────────────────────
• data/X_train_A.csv and data/X_test_A.csv  → ALREADY SCALED
  (StandardScaler fitted on training data in Phase 2, means ≈ 0, stds ≈ 1)
• data/X_train_B.csv and data/X_test_B.csv  → ALREADY SCALED
• models/pipeline_scenario_A.joblib          → Imputer + Scaler ONLY (no model)
• models/pipeline_scenario_B.joblib          → Imputer + Scaler ONLY (no model)
• Split: 8 000 train / 2 000 test, seed=42, stratified, 50/50 class balance

DO NOT apply a second scaler.  Feature CSVs are loaded directly.

EXPERIMENT DESIGN
──────────────────
  Scenario A (PRIMARY)    → 4 features: Temperature, pH, Salinity, Turbidity
      Threshold-defining columns (DO, Nitrate, Chl_a) are EXCLUDED.
      Expected performance: moderate — features have near-zero correlation
      with target. Results are honest.

  Scenario B (REFERENCE)  → 7 features: A + Dissolved_Oxygen, Nitrate, Chl_a
      Included ONLY to demonstrate the effect of direct target leakage.
      Near-perfect accuracy is expected and is NOT a genuine result.
      Results are clearly labeled as a leakage reference.

ALGORITHMS
───────────
  • Logistic Regression (LR)
  • Decision Tree Classifier (DT) — max_depth=6 to avoid overfitting
  • Random Forest Classifier (RF) — 200 trees

OUTPUTS
────────
  models/selected_model_scenario_A.joblib
  models/selected_model_scenario_B_reference.joblib
  reports/model_comparison.csv
  reports/model_metrics.json
  reports/confusion_matrices.png
  reports/model_comparison.png
  reports/shap_summary.png            (SHAP on best Scenario A model)
  reports/feature_importance.png      (fallback if SHAP fails)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RPT_DIR    = os.path.join(ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RPT_DIR,    exist_ok=True)

RANDOM_SEED = 42
TARGET      = "is_safe"
PLT_STYLE   = "seaborn-v0_8-whitegrid"
plt.style.use(PLT_STYLE)
sns.set_theme(style="whitegrid", font_scale=1.1)

FEAT_A = ["Temperature", "pH", "Salinity", "Turbidity"]
FEAT_B = ["Temperature", "pH", "Salinity", "Turbidity",
          "Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]

CLASS_NAMES = ["Unsafe (0)", "Safe (1)"]
PAL = {"Unsafe (0)": "#e05c5c", "Safe (1)": "#4c9be8"}

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load pre-scaled data (DO NOT re-scale)
# ──────────────────────────────────────────────────────────────────────────────
def load_data(scenario: str):
    """Load already-scaled feature matrices and targets for a given scenario."""
    X_tr = pd.read_csv(os.path.join(DATA_DIR, f"X_train_{scenario}.csv"), index_col=0)
    X_te = pd.read_csv(os.path.join(DATA_DIR, f"X_test_{scenario}.csv"),  index_col=0)
    y_tr = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"), index_col=0).squeeze()
    y_te = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"),  index_col=0).squeeze()

    expected = FEAT_A if scenario == "A" else FEAT_B
    assert list(X_tr.columns) == expected, \
        f"Column mismatch Scenario {scenario}: {list(X_tr.columns)} ≠ {expected}"

    # Sanity-check: already scaled
    for col in X_tr.columns:
        m = X_tr[col].mean()
        s = X_tr[col].std()
        assert abs(m) < 0.01, f"Column {col} mean={m:.4f} — data may NOT be pre-scaled!"
        assert 0.99 < s < 1.02, f"Column {col} std={s:.4f} — data may NOT be pre-scaled!"

    print(f"  ✔ Scenario {scenario} data loaded — X_train{scenario}: {X_tr.shape}, "
          f"X_test{scenario}: {X_te.shape}, leakage_cols_present: "
          f"{any(c in X_tr.columns for c in ['Dissolved_Oxygen','Nitrate','Chlorophyll_a'])}")
    return X_tr, X_te, y_tr, y_te

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Metric calculation
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(name: str, model, X_te, y_te) -> dict:
    """Evaluate a trained model on the held-out test set only."""
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob) if y_prob is not None else None
    cm   = confusion_matrix(y_te, y_pred)

    metrics = {
        "model":     name,
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_score":  round(float(f1),   4),
        "roc_auc":   round(float(auc),  4) if auc is not None else None,
        "cm":        cm.tolist(),
        "report":    classification_report(y_te, y_pred,
                                           target_names=CLASS_NAMES,
                                           output_dict=True),
    }
    return metrics

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_scenario(scenario: str, label: str):
    print(f"\n{'='*70}")
    print(f"SCENARIO {scenario} — {label}")
    print(f"{'='*70}")

    X_tr, X_te, y_tr, y_te = load_data(scenario)

    # Guard: ensure Scenario A never sees leakage columns
    if scenario == "A":
        leakage_cols = ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]
        found = [c for c in leakage_cols if c in X_tr.columns]
        assert not found, f"LEAKAGE DETECTED in Scenario A columns: {found}"
        print("  ✔ LEAKAGE GUARD: threshold-defining columns confirmed absent from Scenario A")

    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs",
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=10,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }

    results = {}
    trained_models = {}

    for clf_name, clf in classifiers.items():
        print(f"\n  ── Training {clf_name}…")
        clf.fit(X_tr, y_tr)
        m = compute_metrics(clf_name, clf, X_te, y_te)
        results[clf_name] = m
        trained_models[clf_name] = clf

        print(f"     Accuracy  : {m['accuracy']:.4f}")
        print(f"     Precision : {m['precision']:.4f}")
        print(f"     Recall    : {m['recall']:.4f}")
        print(f"     F1-score  : {m['f1_score']:.4f}")
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        print(f"     ROC-AUC   : {auc_str}")
        print(f"     Conf Mat  :\n{np.array(m['cm'])}")

    return results, trained_models, X_tr, X_te, y_tr, y_te

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Select best model by F1-score (Scenario A)
# ──────────────────────────────────────────────────────────────────────────────
def select_best_model(results: dict, trained_models: dict, scenario: str):
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_m    = results[best_name]
    best_clf  = trained_models[best_name]

    print(f"\n  ── Best model (Scenario {scenario} by F1): {best_name}")
    print(f"     F1-score  = {best_m['f1_score']:.4f}")
    print(f"     Accuracy  = {best_m['accuracy']:.4f}")
    print(f"     ROC-AUC   = {best_m['roc_auc']}")
    return best_name, best_clf

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Save model (preprocessor pipeline + estimator as a single pipeline)
# ──────────────────────────────────────────────────────────────────────────────
def save_full_pipeline(scenario: str, clf_name: str, clf, suffix: str):
    """
    Combine the Phase-2 preprocessor with the trained estimator into a single
    sklearn Pipeline so that inference from raw (unscaled) input works correctly.

    The saved pipeline has steps:
        imputer  → SimpleImputer (fitted in Phase 2)
        scaler   → StandardScaler (fitted in Phase 2)
        model    → trained classifier
    """
    preproc = joblib.load(os.path.join(MODELS_DIR,
                          f"pipeline_scenario_{scenario}.joblib"))
    full_pipe = Pipeline(
        preproc.steps + [("model", clf)],
        memory=None
    )
    out_path = os.path.join(MODELS_DIR, f"selected_model_scenario_{suffix}.joblib")
    joblib.dump(full_pipe, out_path)
    print(f"\n  ✔ Full pipeline saved: models/selected_model_scenario_{suffix}.joblib")
    print(f"     Pipeline steps: {[n for n, _ in full_pipe.steps]}")
    return full_pipe, out_path

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Confusion-matrix grid chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(results_A: dict, results_B: dict):
    clf_names = list(results_A.keys())
    n         = len(clf_names)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9))

    for row, (res, scenario, lbl) in enumerate([
        (results_A, "A", "Scenario A — PRIMARY (leakage excluded)"),
        (results_B, "B", "Scenario B — REFERENCE (leakage present)"),
    ]):
        for col, clf_name in enumerate(clf_names):
            ax = axes[row, col]
            cm = np.array(res[clf_name]["cm"])
            disp = ConfusionMatrixDisplay(cm, display_labels=["Unsafe\n(0)", "Safe\n(1)"])
            disp.plot(ax=ax, colorbar=False,
                      cmap="Blues" if scenario == "A" else "Reds")
            ax.set_title(
                f"{'[PRIMARY] ' if scenario=='A' else '[REF ONLY] '}{clf_name}\n"
                f"Acc={res[clf_name]['accuracy']:.3f}  F1={res[clf_name]['f1_score']:.3f}",
                fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(lbl, fontsize=10, fontweight="bold")

    fig.suptitle("Confusion Matrices — Scenario A (Primary) vs Scenario B (Reference)\n"
                 "⚠ Scenario B includes target-defining features — results are NOT genuine",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RPT_DIR, "confusion_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✔ Saved: reports/confusion_matrices.png")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Model comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results_A: dict, results_B: dict):
    metrics   = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    clf_names = list(results_A.keys())
    n_clfs    = len(clf_names)
    n_metrics = len(metrics)
    bar_w     = 0.25
    x         = np.arange(n_metrics)

    scenario_colors_A = ["#3a86ff", "#06d6a0", "#ffbe0b"]
    scenario_colors_B = ["#e05c5c", "#f4845f", "#f7c59f"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax_idx, (res, colors, stitle) in enumerate([
        (results_A, scenario_colors_A,
         "Scenario A — PRIMARY\n(Temperature, pH, Salinity, Turbidity)"),
        (results_B, scenario_colors_B,
         "Scenario B — REFERENCE ONLY\n⚠ includes leakage features"),
    ]):
        ax = axes[ax_idx]
        for i, (clf_name, color) in enumerate(zip(clf_names, colors)):
            vals = [res[clf_name][m] if res[clf_name][m] is not None else 0
                    for m in metrics]
            bars = ax.bar(x + i * bar_w, vals, bar_w - 0.03,
                          label=clf_name, color=color,
                          edgecolor="white", linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x + bar_w)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(stitle, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        if ax_idx == 1:
            ax.text(0.5, 0.97,
                    "⚠ REFERENCE ONLY — Not genuine performance",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ffe8e8", ec="#c0392b"))

    fig.suptitle("Model Comparison — All Metrics on Held-Out Test Set (n=2,000)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RPT_DIR, "model_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved: reports/model_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SHAP interpretability (Scenario A best model)
# ──────────────────────────────────────────────────────────────────────────────
def run_shap(best_clf, clf_name: str, X_tr: pd.DataFrame, X_te: pd.DataFrame,
             features: list):
    """Compute and plot SHAP summary for the best Scenario A model."""
    try:
        import shap
        print(f"\n  ── SHAP analysis for {clf_name} (Scenario A)…")

        # Use a 500-sample background for speed; test sample of 200
        np.random.seed(RANDOM_SEED)
        background_idx = np.random.choice(len(X_tr), size=500, replace=False)
        X_bg   = X_tr.iloc[background_idx].values
        X_samp = X_te.iloc[:200].values

        # Select explainer based on model type
        if isinstance(best_clf, RandomForestClassifier):
            explainer = shap.TreeExplainer(best_clf)
            shap_values = explainer.shap_values(X_samp)
            # shap_values for binary classification: list of [class0, class1]
            # We want class-1 (Safe) values
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        elif isinstance(best_clf, DecisionTreeClassifier):
            explainer  = shap.TreeExplainer(best_clf)
            shap_values = explainer.shap_values(X_samp)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            # LogisticRegression — use LinearExplainer
            explainer   = shap.LinearExplainer(best_clf, X_bg,
                                                feature_perturbation="correlation_dependent")
            shap_values = explainer.shap_values(X_samp)
            sv = shap_values if not isinstance(shap_values, list) else shap_values[1]

        X_samp_df = pd.DataFrame(X_samp, columns=features)

        # Beeswarm summary plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(sv, X_samp_df, feature_names=features,
                          plot_type="dot", show=False, max_display=len(features),
                          color_bar_label="Feature value (scaled)")
        plt.title(
            f"SHAP Summary — Scenario A | {clf_name}\n"
            f"(Effect on predicting 'Safe'=1, n=200 test samples)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        out = os.path.join(RPT_DIR, "shap_summary.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  ✔ Saved: reports/shap_summary.png")

        # Mean absolute SHAP values
        mean_abs = np.abs(sv).mean(axis=0)
        shap_importance = dict(zip(features, mean_abs.round(5)))
        print(f"  Mean |SHAP| per feature: {shap_importance}")
        return shap_importance, True

    except Exception as e:
        print(f"  ⚠  SHAP failed: {e}")
        print("     Falling back to feature_importances_ / coefficient-based plot.")
        return {}, False

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Feature importance fallback chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(best_clf, clf_name: str, features: list):
    """
    Coefficient magnitude (LR) or feature_importances_ (Tree/RF).
    Saved even when SHAP succeeds — useful supplemental chart.
    """
    try:
        if hasattr(best_clf, "feature_importances_"):
            importances = best_clf.feature_importances_
            title_sfx   = "Gini Importance"
        elif hasattr(best_clf, "coef_"):
            importances = np.abs(best_clf.coef_[0])
            title_sfx   = "|Coefficient|"
        else:
            print("  ℹ  No feature importance attribute found — skipping plot.")
            return

        imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
                       color="#3a86ff", edgecolor="white")
        for bar, val in zip(bars, imp_df["Importance"]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel(title_sfx, fontsize=10)
        ax.set_title(f"Feature Importance — Scenario A | {clf_name}\n({title_sfx})",
                     fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = os.path.join(RPT_DIR, "feature_importance.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✔ Saved: reports/feature_importance.png")
    except Exception as e:
        print(f"  ⚠  Feature importance plot failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Save CSV and JSON reports
# ──────────────────────────────────────────────────────────────────────────────
def save_reports(results_A: dict, results_B: dict,
                 best_A: str, best_B: str,
                 shap_importance: dict, shap_ok: bool):

    # ── Comparison CSV (A only — primary experiment)
    rows = []
    for clf_name, m in results_A.items():
        rows.append({
            "scenario":  "A (primary)",
            "model":     clf_name,
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1_score":  m["f1_score"],
            "roc_auc":   m["roc_auc"],
        })
    for clf_name, m in results_B.items():
        rows.append({
            "scenario":  "B (reference — leakage)",
            "model":     clf_name,
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1_score":  m["f1_score"],
            "roc_auc":   m["roc_auc"],
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(RPT_DIR, "model_comparison.csv"), index=False)
    print(f"\n  ✔ Saved: reports/model_comparison.csv")

    # ── Metrics JSON
    output = {
        "phase":          "Phase 3 — Model Training and Evaluation",
        "dataset":        "Marine Water Quality Classification (Zenodo 19506986)",
        "provenance":     "Synthetic / mathematically derived dataset.",
        "split":          {"n_train": 8000, "n_test": 2000, "seed": 42, "stratified": True},
        "scenario_A": {
            "description":    "PRIMARY — 4 leakage-free features",
            "features":       FEAT_A,
            "leakage_note":   "Dissolved_Oxygen, Nitrate, Chlorophyll_a excluded. "
                              "Residual indirect correlation via Temperature acknowledged.",
            "best_model":     best_A,
            "results":        {
                k: {kk: vv for kk, vv in v.items() if kk != "cm"}
                for k, v in results_A.items()
            },
        },
        "scenario_B": {
            "description":    "REFERENCE ONLY — 7 features including target-defining columns",
            "features":       FEAT_B,
            "leakage_warning":"Results are NOT genuine. Target is a deterministic function "
                              "of Dissolved_Oxygen, Nitrate, and Chlorophyll_a.",
            "best_model":     best_B,
            "results":        {
                k: {kk: vv for kk, vv in v.items() if kk != "cm"}
                for k, v in results_B.items()
            },
        },
        "shap": {
            "succeeded":       shap_ok,
            "model_explained": best_A if shap_ok else None,
            "mean_abs_shap":   shap_importance if shap_ok else None,
        },
    }

    with open(os.path.join(RPT_DIR, "model_metrics.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  ✔ Saved: reports/model_metrics.json")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Post-save load verification
# ──────────────────────────────────────────────────────────────────────────────
def verify_saved_models(feat_A: list, feat_B: list):
    """Load each saved pipeline and run a 3-row smoke-test inference."""
    print(f"\n{'─'*50}")
    print("VERIFICATION — Loading saved models and running inference")
    print(f"{'─'*50}")

    for suffix, feats in [("A", feat_A), ("B_reference", feat_B)]:
        path = os.path.join(MODELS_DIR, f"selected_model_scenario_{suffix}.joblib")
        if not os.path.isfile(path):
            print(f"  ✘ MISSING: {path}")
            continue

        pipe = joblib.load(path)
        steps = [n for n, _ in pipe.steps]
        print(f"\n  Scenario {suffix}")
        print(f"    Path  : {path}")
        print(f"    Steps : {steps}")

        # Smoke test — 3 synthetic raw (unscaled) samples
        rng      = np.random.default_rng(0)
        raw_vals = {
            "Temperature":    rng.uniform(10, 32, 3),
            "pH":             rng.uniform(6.5, 8.8, 3),
            "Salinity":       rng.uniform(30, 40, 3),
            "Turbidity":      rng.uniform(0.5, 50, 3),
        }
        if "B" in suffix:
            raw_vals.update({
                "Dissolved_Oxygen": rng.uniform(0, 16, 3),
                "Nitrate":          rng.uniform(0, 25, 3),
                "Chlorophyll_a":    rng.uniform(10, 100, 3),
            })
        raw_df   = pd.DataFrame({c: raw_vals[c] for c in feats})
        preds    = pipe.predict(raw_df)
        probs    = pipe.predict_proba(raw_df) if hasattr(pipe, "predict_proba") else None
        print(f"    Sample raw input (unscaled):\n{raw_df.to_string(index=False)}")
        print(f"    Predictions: {preds}  (0=Unsafe, 1=Safe)")
        if probs is not None:
            print(f"    Probabilities:\n{np.round(probs, 3)}")
        print(f"    ✔ Inference successful from raw (unscaled) input — pipeline handles scaling internally")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 70)
    print("PHASE 3 — MODEL TRAINING, EVALUATION, SHAP INTERPRETABILITY")
    print("=" * 70)

    # ── Train both scenarios ──────────────────────────────────────────────────
    results_A, models_A, X_tr_A, X_te_A, y_tr, y_te = train_scenario(
        "A", "PRIMARY — 4 leakage-free features"
    )
    results_B, models_B, X_tr_B, X_te_B, _,    _    = train_scenario(
        "B", "REFERENCE ONLY — all 7 features (leakage present)"
    )

    # ── Select best models by F1 ──────────────────────────────────────────────
    best_name_A, best_clf_A = select_best_model(results_A, models_A, "A")
    best_name_B, best_clf_B = select_best_model(results_B, models_B, "B")

    # ── Save full pipelines (preprocessor + model) ────────────────────────────
    print(f"\n{'─'*50}")
    print("SAVING FULL PIPELINES (preprocessor + estimator)")
    print(f"{'─'*50}")
    save_full_pipeline("A", best_name_A, best_clf_A, "A")
    save_full_pipeline("B", best_name_B, best_clf_B, "B_reference")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"SHAP INTERPRETABILITY — Scenario A ({best_name_A})")
    print(f"{'─'*50}")
    shap_importance, shap_ok = run_shap(
        best_clf_A, best_name_A, X_tr_A, X_te_A, FEAT_A
    )

    # Feature importance fallback (always produced)
    plot_feature_importance(best_clf_A, best_name_A, FEAT_A)

    # ── Charts ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("GENERATING CHARTS")
    print(f"{'─'*50}")
    plot_confusion_matrices(results_A, results_B)
    plot_model_comparison(results_A, results_B)

    # ── Reports ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("SAVING REPORTS")
    print(f"{'─'*50}")
    save_reports(results_A, results_B, best_name_A, best_name_B,
                 shap_importance, shap_ok)

    # ── Print final comparison table ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY — SCENARIO A (PRIMARY, LEAKAGE-EXCLUDED)")
    print(f"{'='*70}")
    hdr = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    print(hdr)
    print("─" * len(hdr))
    for clf_name, m in results_A.items():
        tag   = " ← SELECTED" if clf_name == best_name_A else ""
        auc_s = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "   N/A"
        print(f"  {clf_name:<20} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {auc_s:>9}{tag}")

    print(f"\n{'='*70}")
    print("SCENARIO B — REFERENCE ONLY (leakage present — DO NOT USE AS GENUINE RESULT)")
    print(f"{'='*70}")
    for clf_name, m in results_B.items():
        tag   = " ← SELECTED (ref)" if clf_name == best_name_B else ""
        auc_s = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "   N/A"
        print(f"  {clf_name:<20} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {auc_s:>9}{tag}")

    # ── Model selection rationale ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MODEL SELECTION RATIONALE — Scenario A")
    print(f"{'='*70}")
    print(f"  Selected    : {best_name_A}")
    print(f"  Criterion   : Highest F1-score on held-out test set")
    print(f"  F1 chosen because class balance is 50/50, making it equivalent to")
    print(f"  accuracy but more stable across precision/recall trade-offs.")
    print(f"\n  Trade-off note:")
    print(f"    • Logistic Regression  → Fastest; interpretable coefficients;")
    print(f"                             assumes linear decision boundary.")
    print(f"    • Decision Tree        → Interpretable rules; may overfit leaf nodes.")
    print(f"    • Random Forest        → Robust ensemble; best SHAP compatibility;")
    print(f"                             slightly less interpretable individually.")
    print(f"\n  Important caveat:")
    print(f"    Scenario A features (Temp, pH, Salinity, Turbidity) have near-zero")
    print(f"    Pearson r with target (all |r| < 0.01). Any accuracy above 50 %")
    print(f"    reflects genuine non-linear or interaction effects, but should not")
    print(f"    be over-interpreted given the synthetic nature of the dataset.")

    # ── Load verification ─────────────────────────────────────────────────────
    feat_B_ordered = FEAT_B
    verify_saved_models(FEAT_A, feat_B_ordered)

    print(f"\n{'='*70}")
    print("PHASE 3 COMPLETE — No dashboard built yet.")
    print("Await confirmation before proceeding to Phase 4.")
    print(f"{'='*70}")
