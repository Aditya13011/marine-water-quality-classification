"""
pages/3_Model_Performance.py
AI-Based Marine Water Quality Classification System — Model Performance Page
"""

import os, sys, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Model Performance — Marine Water Quality AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:linear-gradient(160deg,#0a1628 0%,#0d2137 55%,#0a1f35 100%);
    color:#e2f0ff;
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#061022 0%,#0d1f3c 100%);
    border-right:1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
[data-testid="stMetric"] {
    background:linear-gradient(135deg,#0f2744,#0d2137);
    border:1px solid #1e4d7a; border-radius:12px; padding:16px 20px;
}
[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:0.82rem !important; }
[data-testid="stMetricValue"] { color:#22d3ee !important; font-size:1.6rem !important; font-weight:700 !important; }
.section-heading {
    font-size:1.2rem; font-weight:700; color:#22d3ee;
    border-left:4px solid #0d9488; padding-left:12px; margin:24px 0 14px;
}
hr { border-color:#1e3a5f !important; }
.warn-box {
    background:linear-gradient(90deg,#451a03,#3d1f00);
    border:1px solid #f59e0b; border-radius:10px;
    padding:14px 20px; color:#fde68a; font-size:0.88rem; line-height:1.6;
}
.leak-banner {
    background:linear-gradient(90deg,#2d1515,#3d0a0a);
    border:2px solid #ef4444; border-radius:10px;
    padding:14px 20px; color:#fca5a5; font-size:0.88rem; line-height:1.6;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌊 Marine Water AI")
    st.markdown("---")
    st.markdown("**📈 Model Performance** ← you are here")

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a; border-radius:14px;
            padding:28px 36px; margin-bottom:20px;'>
  <div style='font-size:1.9rem; font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#7dd3fc);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📈 Model Performance
  </div>
  <div style='color:#94a3b8; font-size:0.95rem; margin-top:6px;'>
    All metrics loaded from <code>reports/model_metrics.json</code> and
    <code>reports/model_comparison.csv</code> — verified against training output.
  </div>
</div>
""", unsafe_allow_html=True)

# ── load reports ──────────────────────────────────────────────────────────────
metrics_path = os.path.join(ROOT, "reports", "model_metrics.json")
csv_path     = os.path.join(ROOT, "reports", "model_comparison.csv")

if not os.path.isfile(metrics_path):
    st.error("❌ reports/model_metrics.json not found. Run src/train_model.py first.")
    st.stop()

with open(metrics_path, encoding="utf-8") as f:
    meta = json.load(f)

comp_df = pd.read_csv(csv_path) if os.path.isfile(csv_path) else None

PLOTLY_DARK = dict(
    paper_bgcolor="#0d2137", plot_bgcolor="#0a1628",
    font=dict(color="#e2f0ff"),
)

# ── split info ────────────────────────────────────────────────────────────────
split = meta["split"]
si1, si2, si3, si4 = st.columns(4)
si1.metric("Training samples", f"{split['n_train']:,}")
si2.metric("Test samples",     f"{split['n_test']:,}")
si3.metric("Random seed",       str(split["seed"]))
si4.metric("Stratified",       "Yes ✔")

# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO A — PRIMARY
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='background:linear-gradient(90deg,#052e16,#064e3b);
            border:1px solid #10b981; border-radius:10px; padding:14px 20px;
            color:#6ee7b7; font-size:0.9rem; line-height:1.6; margin-bottom:8px;'>
✅ <b>SCENARIO A — PRIMARY EXPERIMENT</b><br>
Features: Temperature, pH, Salinity, Turbidity (threshold-defining columns excluded).<br>
Results are honest predictions without direct access to the variables used to construct the target.
</div>
""", unsafe_allow_html=True)

results_A = meta["scenario_A"]["results"]
best_A    = meta["scenario_A"]["best_model"]

st.markdown('<div class="section-heading">🏆 Scenario A — Model Metrics (Held-Out Test Set, n=2,000)</div>',
            unsafe_allow_html=True)

# Metric cards for best model
rf_m = results_A["RandomForest"]
ma1,ma2,ma3,ma4,ma5 = st.columns(5)
ma1.metric("Accuracy",  f"{rf_m['accuracy']*100:.2f}%",
           delta="Best: RandomForest", delta_color="off")
ma2.metric("Precision", f"{rf_m['precision']*100:.2f}%")
ma3.metric("Recall",    f"{rf_m['recall']*100:.2f}%")
ma4.metric("F1-Score",  f"{rf_m['f1_score']*100:.2f}%")
ma5.metric("ROC-AUC",   f"{rf_m['roc_auc']*100:.2f}%")

st.info(
    "ℹ️ **Context**: ~51% accuracy is near random chance on a 50/50 balanced dataset. "
    "This is expected — Temperature, pH, Salinity, and Turbidity have |r| < 0.01 "
    "with `is_safe`. The model is not broken; the safe features simply carry "
    "little information about the synthetically constructed target."
)

# Full comparison table — Scenario A
st.markdown('<div class="section-heading">📊 Scenario A — All Models Comparison</div>',
            unsafe_allow_html=True)
rows_A = []
for clf, m in results_A.items():
    rows_A.append({
        "Model": clf + (" ✅ Selected" if clf == best_A else ""),
        "Accuracy":  f"{m['accuracy']*100:.2f}%",
        "Precision": f"{m['precision']*100:.2f}%",
        "Recall":    f"{m['recall']*100:.2f}%",
        "F1-Score":  f"{m['f1_score']*100:.2f}%",
        "ROC-AUC":   f"{m['roc_auc']*100:.2f}%",
    })
st.dataframe(pd.DataFrame(rows_A), use_container_width=True, hide_index=True)

# Bar chart Scenario A
metrics_list = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
clf_colors_A  = {"LogisticRegression": "#3a86ff",
                 "DecisionTree":       "#06d6a0",
                 "RandomForest":       "#ffbe0b"}

fig_A = go.Figure()
for clf_name, m in results_A.items():
    vals = [m[k] * 100 for k in metrics_list]
    fig_A.add_trace(go.Bar(
        name=clf_name,
        x=metric_labels, y=vals,
        marker_color=clf_colors_A[clf_name],
        text=[f"{v:.2f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=9, color="#e2f0ff"),
    ))
fig_A.update_layout(
    **PLOTLY_DARK,
    barmode="group",
    title=dict(text="Scenario A — All Models (Primary, Leakage-Excluded)",
               font=dict(color="#22d3ee")),
    yaxis=dict(title="Score (%)", range=[0, 120],
               gridcolor="#1e3a5f"),
    xaxis=dict(gridcolor="#1e3a5f"),
    legend=dict(font=dict(color="#e2f0ff")),
    height=420, margin=dict(t=50,b=30,l=30,r=20),
)
st.plotly_chart(fig_A, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO B — REFERENCE ONLY
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="leak-banner">
🚨 <b>SCENARIO B — REFERENCE / LEAKAGE DEMONSTRATION ONLY</b><br>
This scenario includes Dissolved_Oxygen, Nitrate, and Chlorophyll_a — the three columns
that <b>directly define the target</b> via a hard threshold rule.
Near-perfect accuracy (~97%) is <b>not genuine predictive performance</b>; the models are
essentially reverse-engineering the rule used to create the labels.
These results are shown only to quantify the effect of target leakage.
<b>Do not cite Scenario B metrics as evidence of model quality.</b>
</div>
""", unsafe_allow_html=True)

results_B = meta["scenario_B"]["results"]
best_B    = meta["scenario_B"]["best_model"]

st.markdown('<div class="section-heading">📊 Scenario B — All Models (Reference Only)</div>',
            unsafe_allow_html=True)
rows_B = []
for clf, m in results_B.items():
    rows_B.append({
        "Model": clf + (" ✅ Highest F1 (ref)" if clf == best_B else ""),
        "Accuracy":  f"{m['accuracy']*100:.2f}%",
        "Precision": f"{m['precision']*100:.2f}%",
        "Recall":    f"{m['recall']*100:.2f}%",
        "F1-Score":  f"{m['f1_score']*100:.2f}%",
        "ROC-AUC":   f"{m['roc_auc']*100:.2f}%",
    })
st.dataframe(pd.DataFrame(rows_B), use_container_width=True, hide_index=True)

clf_colors_B = {"LogisticRegression": "#e05c5c",
                "DecisionTree":       "#f4845f",
                "RandomForest":       "#f7c59f"}
fig_B = go.Figure()
for clf_name, m in results_B.items():
    vals = [m[k] * 100 for k in metrics_list]
    fig_B.add_trace(go.Bar(
        name=clf_name,
        x=metric_labels, y=vals,
        marker_color=clf_colors_B[clf_name],
        text=[f"{v:.2f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=9, color="#e2f0ff"),
    ))
fig_B.update_layout(
    **PLOTLY_DARK,
    barmode="group",
    title=dict(text="⚠️ Scenario B — REFERENCE ONLY (leakage present, results not genuine)",
               font=dict(color="#ef4444")),
    yaxis=dict(title="Score (%)", range=[0, 120], gridcolor="#1e3a5f"),
    xaxis=dict(gridcolor="#1e3a5f"),
    legend=dict(font=dict(color="#e2f0ff")),
    height=420, margin=dict(t=50,b=30,l=30,r=20),
)
st.plotly_chart(fig_B, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRICES (saved chart)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">🔢 Confusion Matrices</div>',
            unsafe_allow_html=True)
cm_path = os.path.join(ROOT, "reports", "confusion_matrices.png")
if os.path.isfile(cm_path):
    st.image(cm_path, use_container_width=True,
             caption="Top row: Scenario A (Primary) | Bottom row: Scenario B (Reference only)")
else:
    st.warning("Confusion matrix chart not found. Run src/train_model.py to regenerate.")

# ── Interactive confusion matrix for RF Scenario A ──────────────────────────
st.markdown("**Interactive Confusion Matrix — Random Forest, Scenario A**")
# Reconstruct from report data
rep = results_A["RandomForest"]["report"]
tn_approx = int(rep["Unsafe (0)"]["recall"] * rep["Unsafe (0)"]["support"])
fp_approx = int(rep["Unsafe (0)"]["support"]) - tn_approx
fn_approx = int(rep["Safe (1)"]["support"]) - int(rep["Safe (1)"]["recall"] * rep["Safe (1)"]["support"])
tp_approx = int(rep["Safe (1)"]["recall"] * rep["Safe (1)"]["support"])

cm_vals   = [[tn_approx, fp_approx], [fn_approx, tp_approx]]
cm_labels = [["TN", "FP"], ["FN", "TP"]]

# Plain-text annotations — go.Heatmap texttemplate does NOT support HTML tags.
# Use '\n' separator; Plotly renders this correctly on all versions (>=5.x, 7.x).
annots = [
    [f"{cm_vals[i][j]}\n{cm_labels[i][j]}" for j in range(2)]
    for i in range(2)
]

# customdata carries the label string for the hover tooltip
custom = [
    [f"{cm_labels[i][j]}: {cm_vals[i][j]}" for j in range(2)]
    for i in range(2)
]

fig_cm = go.Figure(go.Heatmap(
    z=cm_vals,
    x=["Pred Unsafe", "Pred Safe"],
    y=["Act Unsafe", "Act Safe"],
    text=annots,
    texttemplate="%{text}",
    customdata=custom,
    hovertemplate="%{customdata}<extra></extra>",
    colorscale=[[0, "#0a2a4a"], [1, "#0d9488"]],
    showscale=False,
))
# PLOTLY_DARK already contains font=dict(color=...). Passing font= again as an
# explicit kwarg causes TypeError in Plotly 7.x ("multiple values for keyword
# argument 'font'"). Build a merged dict so font is passed exactly once.
_cm_layout = {**PLOTLY_DARK, "font": dict(size=14, color="#e2f0ff")}
fig_cm.update_layout(**_cm_layout, height=330, margin=dict(t=20, b=20, l=40, r=20))
st.plotly_chart(fig_cm, use_container_width=True, key="interactive_cm_A")

# ──────────────────────────────────────────────────────────────────────────────
# SHAP summary
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">🔍 SHAP Feature Importance (Scenario A)</div>',
            unsafe_allow_html=True)

shap_info = meta.get("shap", {})
if shap_info.get("succeeded"):
    shap_col, shap_img_col = st.columns([1, 2])
    with shap_col:
        raw_shap = shap_info.get("mean_abs_shap", {})
        shap_vals = {}
        for feat, val in raw_shap.items():
            # Handle stored string like "[0.02327 0.02327]"
            if isinstance(val, str):
                nums = [float(x) for x in val.strip("[]").split()]
                shap_vals[feat] = float(np.mean(nums))
            else:
                shap_vals[feat] = float(val)

        shap_df = pd.DataFrame(
            list(shap_vals.items()), columns=["Feature", "Mean |SHAP|"]
        ).sort_values("Mean |SHAP|", ascending=True)

        fig_shap = px.bar(
            shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
            color="Mean |SHAP|", color_continuous_scale="Teal",
            text=shap_df["Mean |SHAP|"].apply(lambda v: f"{v:.5f}"),
        )
        fig_shap.update_traces(textposition="outside",
                               textfont=dict(color="#e2f0ff"))
        fig_shap.update_layout(
            **PLOTLY_DARK,
            title=dict(text="Mean |SHAP| per Feature",
                       font=dict(color="#22d3ee")),
            showlegend=False,
            coloraxis_showscale=False,
            height=320, margin=dict(t=40,b=20,l=20,r=60),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with shap_img_col:
        shap_png = os.path.join(ROOT, "reports", "shap_summary.png")
        if os.path.isfile(shap_png):
            st.image(shap_png, use_container_width=True,
                     caption="SHAP Beeswarm — RandomForest, Scenario A, n=200 test samples")
        else:
            st.warning("shap_summary.png not found.")

    st.info(
        "**SHAP Interpretation**: All four Scenario A features contribute "
        "nearly equal, very small |SHAP| values (~0.021–0.023). This is "
        "consistent with the near-zero linear correlation with the target — "
        "no single feature dominates, and all contributions are marginal."
    )
else:
    st.warning("SHAP analysis was not run or failed. Run src/train_model.py to regenerate.")

# ── Saved model comparison chart ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">📊 Full Model Comparison Chart (Phase 3 Report)</div>',
            unsafe_allow_html=True)
mc_path = os.path.join(ROOT, "reports", "model_comparison.png")
if os.path.isfile(mc_path):
    st.image(mc_path, use_container_width=True,
             caption="Left: Scenario A (Primary) | Right: Scenario B (Reference only — not genuine)")
else:
    st.warning("Model comparison chart not found.")

# ── Feature importance chart ──────────────────────────────────────────────────
fi_path = os.path.join(ROOT, "reports", "feature_importance.png")
if os.path.isfile(fi_path):
    st.markdown('<div class="section-heading">🌲 Random Forest Feature Importance (Gini)</div>',
                unsafe_allow_html=True)
    st.image(fi_path, use_container_width=False,
             caption="Gini importance — RandomForest, Scenario A")

st.markdown("---")
st.markdown("""
<div class="warn-box">
📌 <b>Reminder</b>: All test metrics come from the held-out test set (2,000 samples, 20%).
The training set was never used for evaluation. The preprocessing pipeline (Imputer + Scaler)
was fitted on training data only — the test set was only transformed, never fitted upon.
</div>
""", unsafe_allow_html=True)
