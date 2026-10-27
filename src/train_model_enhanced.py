"""
src/train_model_enhanced.py
────────────────────────────────────────────────────────────────────────────
Phase 2 — Enhanced ML Pipeline
Project : AI-Based Marine Water Quality Classification (MarineGuard AI)

EXPERIMENT DESIGN
──────────────────
Scenario A (PRIMARY, production):
    Features : Temperature, pH, Salinity, Turbidity  (4 features)
    Leakage  : None — threshold-defining columns excluded
    Expected : ~51% accuracy (near-random; features have |r|<0.01 with target)

Scenario B (REFERENCE, educational):
    Features : All 7 numeric columns (incl. DO, Nitrate, Chl_a)
    Purpose  : Quantify target-leakage effect (~97% accuracy)
    Warning  : NOT genuine predictive performance — never cite as real result

ALGORITHMS COMPARED (Scenario A)
──────────────────────────────────
  LR   — Logistic Regression       (linear baseline)
  DT   — Decision Tree             (max_depth=6, min_samples_leaf=20)
  RF   — Random Forest             (tuned via RandomizedSearchCV)
  ET   — Extra Trees               (tuned via RandomizedSearchCV)
  XGB  — XGBoost                   (tuned via RandomizedSearchCV)
  LSVC — LinearSVC                 (fast linear SVM)

METHODOLOGY
────────────
  • Hyperparameter tuning: 5-fold StratifiedKFold CV on TRAINING set only
  • Model selection: best CV F1-score on training set
  • Final evaluation: ONE pass on held-out test set (2 000 samples)
  • Permutation importance: on test set with best model
  • Leakage guard: assert no threshold cols in Scenario A features before CV

OUTPUTS
────────
  reports/model_metrics_enhanced.json    Extended metrics for all 6 algorithms
  reports/cv_results.csv                 Cross-validation summary table
  reports/permutation_importance.png     Permutation importance chart
  reports/model_comparison_enhanced.png  6-algorithm comparison bar chart
  reports/roc_curves.png                 ROC curves for all algorithms
  models/selected_model_scenario_A.joblib   Overwritten if CV selects a better model
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm             import LinearSVC
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_validate,
)
from sklearn.inspection      import permutation_importance
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)
from sklearn.calibration     import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠  XGBoost not installed — skipping XGB.")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RPT_DIR    = os.path.join(ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RPT_DIR,    exist_ok=True)

SEED = 42

# Leakage guard
THRESHOLD_COLS = {"Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"}
FEATURES_A = ["Temperature", "pH", "Salinity", "Turbidity"]
FEATURES_B = FEATURES_A + ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]

# ── Colour palette ─────────────────────────────────────────────────────────────
CLF_COLORS = {
    "LogisticRegression": "#3a86ff",
    "DecisionTree":       "#06d6a0",
    "RandomForest":       "#ffbe0b",
    "ExtraTrees":         "#8338ec",
    "XGBoost":            "#ff006e",
    "LinearSVC":          "#fb5607",
}

# ──────────────────────────────────────────────────────────────────────────────
# 1  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    print("\n[1/7] Loading pre-processed train/test arrays …")
    Xa_tr = pd.read_csv(os.path.join(DATA_DIR, "X_train_A.csv"), index_col=0)
    Xa_te = pd.read_csv(os.path.join(DATA_DIR, "X_test_A.csv"),  index_col=0)
    Xb_tr = pd.read_csv(os.path.join(DATA_DIR, "X_train_B.csv"), index_col=0)
    Xb_te = pd.read_csv(os.path.join(DATA_DIR, "X_test_B.csv"),  index_col=0)
    # y CSVs contain a stray pandas index column — read with index_col=0
    y_tr_df = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"), index_col=0)
    y_te_df = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"),  index_col=0)
    y_tr    = y_tr_df["is_safe"] if "is_safe" in y_tr_df.columns else y_tr_df.squeeze()
    y_te    = y_te_df["is_safe"] if "is_safe" in y_te_df.columns else y_te_df.squeeze()

    # Leakage guard
    for col in THRESHOLD_COLS:
        assert col not in Xa_tr.columns, (
            f"LEAKAGE: '{col}' found in Scenario A training features!"
        )
    print(f"  Scenario A training : {Xa_tr.shape}  |  test : {Xa_te.shape}")
    print(f"  Scenario B training : {Xb_tr.shape}  |  test : {Xb_te.shape}")
    return Xa_tr, Xa_te, Xb_tr, Xb_te, y_tr, y_te


# ──────────────────────────────────────────────────────────────────────────────
# 2  BUILD CLASSIFIERS + SEARCH SPACES
# ──────────────────────────────────────────────────────────────────────────────
def build_classifiers():
    """Return list of (name, estimator, param_distributions) tuples."""
    clfs = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=SEED),
            {"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [10, 20, 40]},
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {
                "n_estimators":      [100, 200, 300],
                "max_depth":         [None, 5, 10, 20],
                "min_samples_leaf":  [1, 2, 5],
                "max_features":      ["sqrt", "log2"],
            },
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
            {
                "n_estimators":     [100, 200, 300],
                "max_depth":        [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5],
                "max_features":     ["sqrt", "log2"],
            },
        ),
        (
            "LinearSVC",
            CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)),
            {"estimator__C": [0.01, 0.1, 1.0, 10.0]},
        ),
    ]
    if XGB_AVAILABLE:
        clfs.append((
            "XGBoost",
            xgb.XGBClassifier(
                random_state=SEED, eval_metric="logloss",
                verbosity=0, use_label_encoder=False,
            ),
            {
                "n_estimators":   [100, 200],
                "max_depth":      [3, 5, 7],
                "learning_rate":  [0.05, 0.1, 0.2],
                "subsample":      [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ))
    return clfs


# ──────────────────────────────────────────────────────────────────────────────
# 3  CROSS-VALIDATE + TUNE (training data only)
# ──────────────────────────────────────────────────────────────────────────────
def tune_and_evaluate(name, estimator, param_dist, X_train, y_train, cv):
    """
    Perform RandomizedSearchCV on training data.
    Returns (best_estimator, cv_summary_dict).
    """
    n_iter = min(20, len(param_dist) * 4) if param_dist else 1

    t0 = time.time()
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_est = search.best_estimator_

    # 5-fold cross-validation scores on training set (for reporting)
    cv_results = cross_validate(
        best_est, X_train, y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1":       "f1_weighted",
            "roc_auc":  "roc_auc",
        },
        return_train_score=False,
        n_jobs=-1,
    )

    summary = {
        "best_params":      search.best_params_,
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std":  float(np.std(cv_results["test_accuracy"])),
        "cv_f1_mean":       float(np.mean(cv_results["test_f1"])),
        "cv_f1_std":        float(np.std(cv_results["test_f1"])),
        "cv_roc_auc_mean":  float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std":   float(np.std(cv_results["test_roc_auc"])),
        "training_time_s":  round(elapsed, 2),
    }
    return best_est, summary


# ──────────────────────────────────────────────────────────────────────────────
# 4  TEST-SET EVALUATION (called once per estimator after tuning)
# ──────────────────────────────────────────────────────────────────────────────
def test_set_metrics(estimator, X_test, y_test, label_names=None):
    """Evaluate on held-out test set. Called once per model."""
    y_pred      = estimator.predict(X_test)
    y_prob      = estimator.predict_proba(X_test)[:, 1]
    label_names = label_names or ["Unsafe (0)", "Safe (1)"]

    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall":    float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score":  float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm,
        "report":    report,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5  PERMUTATION IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
def compute_permutation_importance(estimator, X_test, y_test, feature_names):
    print("  Computing permutation importance …")
    result = permutation_importance(
        estimator, X_test, y_test,
        n_repeats=20, random_state=SEED, n_jobs=-1, scoring="f1_weighted",
    )
    mean_imp = result.importances_mean
    std_imp  = result.importances_std
    order    = np.argsort(mean_imp)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1628")

    bars = ax.barh(
        [feature_names[i] for i in order],
        mean_imp[order],
        xerr=std_imp[order],
        color="#22d3ee", ecolor="#7dd3fc", capsize=5, alpha=0.85,
    )
    ax.set_xlabel("Mean decrease in F1-score (±1 std over 20 repeats)",
                  color="#94a3b8", fontsize=10)
    ax.set_title("Permutation Feature Importance — Scenario A (RF, test set)",
                 color="#22d3ee", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.xaxis.label.set_color("#94a3b8")
    fig.tight_layout()
    out_path = os.path.join(RPT_DIR, "permutation_importance.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d2137")
    plt.close(fig)
    print(f"  Saved → {out_path}")

    return {
        feat: {"mean": float(mean_imp[i]), "std": float(std_imp[i])}
        for i, feat in enumerate(feature_names)
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6  ROC CURVES
# ──────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for all trained Scenario A models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1628")

    ax.plot([0, 1], [0, 1], ":", color="#475569", lw=1, label="Random (AUC=0.50)")

    for clf_name, est in models_dict.items():
        if est is None:
            continue
        y_prob = est.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        color   = CLF_COLORS.get(clf_name, "#ffffff")
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{clf_name} (AUC={auc_val:.4f})")

    ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=10)
    ax.set_ylabel("True Positive Rate",  color="#94a3b8", fontsize=10)
    ax.set_title("ROC Curves — All Models, Scenario A (Leakage-Free)",
                 color="#22d3ee", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    legend = ax.legend(fontsize=8, loc="lower right")
    legend.get_frame().set_facecolor("#0f2744")
    for text in legend.get_texts():
        text.set_color("#cbd5e1")
    fig.tight_layout()
    out_path = os.path.join(RPT_DIR, "roc_curves.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d2137")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7  ENHANCED COMPARISON BAR CHART
# ──────────────────────────────────────────────────────────────────────────────
def plot_enhanced_comparison(enhanced_results):
    """Bar chart comparing all 6 algorithms across 5 metrics."""
    metrics_list  = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 (weighted)", "ROC-AUC"]

    clf_names = list(enhanced_results.keys())
    x = np.arange(len(metric_labels))
    width = 0.12
    offsets = np.linspace(-width * len(clf_names) / 2,
                           width * len(clf_names) / 2, len(clf_names))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1628")

    for i, clf_name in enumerate(clf_names):
        vals   = [enhanced_results[clf_name][m] * 100 for m in metrics_list]
        color  = CLF_COLORS.get(clf_name, "#ffffff")
        bars   = ax.bar(x + offsets[i], vals, width * 0.9, label=clf_name,
                        color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color="#e2f0ff")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Score (%)", color="#94a3b8", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("Enhanced Model Comparison — Scenario A (6 Algorithms, Leakage-Free)",
                 color="#22d3ee", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#cbd5e1")
    ax.axhline(50, color="#475569", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(metric_labels) - 0.5, 51.5, "50% baseline",
            color="#475569", fontsize=8, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    legend = ax.legend(fontsize=9, ncol=3, loc="upper right")
    legend.get_frame().set_facecolor("#0f2744")
    for text in legend.get_texts():
        text.set_color("#cbd5e1")
    fig.tight_layout()
    out_path = os.path.join(RPT_DIR, "model_comparison_enhanced.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d2137")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  MarineGuard AI — Enhanced ML Training (Phase 2)")
    print("  Scenario A: 4 leakage-free features | 6 algorithms | 5-fold CV")
    print("=" * 70)

    Xa_tr, Xa_te, Xb_tr, Xb_te, y_tr, y_te = load_data()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\n[2/7] Tuning classifiers on TRAINING SET only (5-fold CV) …")
    print("     ⚠  All tuning uses training data. Test set is NEVER seen until final eval.")

    classifiers = build_classifiers()

    scenario_A_results  = {}
    scenario_A_cv       = {}
    scenario_A_models   = {}

    for name, estimator, param_dist in classifiers:
        print(f"\n  → {name} …", flush=True)
        best_est, cv_summary = tune_and_evaluate(
            name, estimator, param_dist, Xa_tr, y_tr, cv
        )
        test_metrics = test_set_metrics(best_est, Xa_te, y_te)

        scenario_A_results[name] = test_metrics
        scenario_A_cv[name]      = cv_summary
        scenario_A_models[name]  = best_est

        print(f"     CV  F1={cv_summary['cv_f1_mean']:.4f} ± {cv_summary['cv_f1_std']:.4f}"
              f"  |  Test Acc={test_metrics['accuracy']*100:.2f}%"
              f"  |  Test F1={test_metrics['f1_score']:.4f}"
              f"  |  Time={cv_summary['training_time_s']}s")

    # ── Select best model by CV F1 ────────────────────────────────────────────
    print("\n[3/7] Selecting best Scenario A model …")
    best_name = max(scenario_A_results,
                    key=lambda n: scenario_A_cv[n]["cv_f1_mean"])
    best_model = scenario_A_models[best_name]
    print(f"  Best model: {best_name}"
          f"  (CV F1={scenario_A_cv[best_name]['cv_f1_mean']:.4f})")
    print(f"  Test accuracy : {scenario_A_results[best_name]['accuracy']*100:.2f}%")
    print(f"  Test F1       : {scenario_A_results[best_name]['f1_score']:.4f}")
    print(f"  Test ROC-AUC  : {scenario_A_results[best_name]['roc_auc']:.4f}")

    # Build and save a FULL sklearn Pipeline: imputer → scaler → best_model
    # This preserves the contract expected by prediction.py and the test suite.
    pre_path = os.path.join(MODELS_DIR, "pipeline_scenario_A.joblib")
    if os.path.isfile(pre_path):
        pre_pipeline = joblib.load(pre_path)
        # Replace the model step with the newly selected best classifier
        full_pipeline = Pipeline([
            ("imputer", pre_pipeline.named_steps["imputer"]),
            ("scaler",  pre_pipeline.named_steps["scaler"]),
            ("model",   best_model),
        ])
        out_path = os.path.join(MODELS_DIR, "selected_model_scenario_A.joblib")
        joblib.dump(full_pipeline, out_path)
        print(f"  Saved full pipeline ({best_name}) → {out_path}")
        # Also store the full_pipeline so permutation importance runs on scaled data
        best_model_for_perm = full_pipeline
    else:
        print("  ⚠  pipeline_scenario_A.joblib not found — saving bare classifier.")
        out_path = os.path.join(MODELS_DIR, "selected_model_scenario_A.joblib")
        joblib.dump(best_model, out_path)
        best_model_for_perm = best_model

    # ── Permutation importance on best bare classifier (Xa_te is pre-scaled) ──
    print("\n[4/7] Permutation importance on best Scenario A model …")
    perm_imp = compute_permutation_importance(
        best_model, Xa_te, y_te, feature_names=FEATURES_A
    )

    # ── ROC curves (all bare classifiers on pre-scaled Xa_te) ─────────────────
    print("\n[5/7] ROC curves …")
    plot_roc_curves(scenario_A_models, Xa_te, y_te)

    # ── Enhanced comparison chart ─────────────────────────────────────────────
    print("\n[6/7] Enhanced comparison bar chart …")
    plot_enhanced_comparison(scenario_A_results)

    # ── Scenario B (reference, quick — no tuning needed) ─────────────────────
    print("\n[6b/7] Scenario B reference evaluation (Random Forest only) …")
    rf_b = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf_b.fit(Xb_tr, y_tr)
    scenario_B_results = {"RandomForest": test_set_metrics(rf_b, Xb_te, y_te)}
    scenario_B_best = "RandomForest"
    print(f"  Scenario B RF accuracy: {scenario_B_results['RandomForest']['accuracy']*100:.2f}%"
          "  ⚠ LEAKAGE — not genuine performance")

    # ── CV results CSV ────────────────────────────────────────────────────────
    cv_rows = []
    for clf_name, cv_s in scenario_A_cv.items():
        cv_rows.append({
            "Model":           clf_name,
            "CV Accuracy":     f"{cv_s['cv_accuracy_mean']*100:.2f}% ± {cv_s['cv_accuracy_std']*100:.2f}%",
            "CV F1 (weighted)":f"{cv_s['cv_f1_mean']:.4f} ± {cv_s['cv_f1_std']:.4f}",
            "CV ROC-AUC":      f"{cv_s['cv_roc_auc_mean']:.4f} ± {cv_s['cv_roc_auc_std']:.4f}",
            "Test Accuracy":   f"{scenario_A_results[clf_name]['accuracy']*100:.2f}%",
            "Test F1":         f"{scenario_A_results[clf_name]['f1_score']:.4f}",
            "Test ROC-AUC":    f"{scenario_A_results[clf_name]['roc_auc']:.4f}",
            "Train Time (s)":  cv_s["training_time_s"],
            "Best Params":     str(cv_s["best_params"]),
            "Selected":        "✅" if clf_name == best_name else "",
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_path = os.path.join(RPT_DIR, "cv_results.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"\n  Saved → {cv_path}")

    # ── Save enhanced metrics JSON ────────────────────────────────────────────
    print("\n[7/7] Saving enhanced metrics JSON …")
    with open(os.path.join(DATA_DIR, "split_metadata.json")) as f:
        split_meta = json.load(f)

    enhanced_metrics = {
        "generated_by":  "train_model_enhanced.py",
        "phase":         "Phase 2 — Enhanced ML",
        "split":         split_meta.get("split", {}),
        "cv_folds":      5,
        "cv_strategy":   "StratifiedKFold — training data only",
        "tuning_method": "RandomizedSearchCV (n_iter=≤20)",
        "leakage_guard": "THRESHOLD_COLS excluded from Scenario A",
        "scenario_A": {
            "features":    FEATURES_A,
            "best_model":  best_name,
            "leakage":     False,
            "results":     scenario_A_results,
            "cv_results":  scenario_A_cv,
            "permutation_importance": perm_imp,
        },
        "scenario_B": {
            "features":    FEATURES_B,
            "best_model":  scenario_B_best,
            "leakage":     True,
            "warning":     "NOT genuine predictive performance — DO NOT cite as model quality",
            "results":     scenario_B_results,
        },
    }

    out_json = os.path.join(RPT_DIR, "model_metrics_enhanced.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(enhanced_metrics, f, indent=2, default=str)
    print(f"  Saved → {out_json}")

    print("\n" + "=" * 70)
    print("  Enhanced ML training complete.")
    print(f"  Best Scenario A model : {best_name}")
    print(f"  Test accuracy         : {scenario_A_results[best_name]['accuracy']*100:.2f}%")
    print(f"  Test F1               : {scenario_A_results[best_name]['f1_score']:.4f}")
    print(f"  Outputs in            : {RPT_DIR}")
    print("  NOTE: ~51% accuracy is expected (synthetic data, |r|<0.01).")
    print("=" * 70)


if __name__ == "__main__":
    main()
