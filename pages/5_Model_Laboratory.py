"""
pages/5_Model_Laboratory.py
MarineGuard AI — Model Laboratory
Enhanced ML comparison: 6 algorithms, 5-fold CV, hyperparameter tuning,
permutation importance, ROC curves.
"""

import os, sys, json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

st.set_page_config(
    page_title="Model Laboratory — MarineGuard AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{--navy:#0a1628;--teal:#0d9488;--cyan:#22d3ee;--sky:#7dd3fc;}
[data-testid="stAppViewContainer"]{
    background:linear-gradient(160deg,#0a1628 0%,#0d2137 55%,#0a1f35 100%);
    color:#e2f0ff;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#061022 0%,#0d1f3c 100%);
    border-right:1px solid #1e3a5f;
}
[data-testid="stSidebar"] *{color:#cbd5e1 !important;}
[data-testid="stMetric"]{
    background:linear-gradient(135deg,#0f2744,#0d2137);
    border:1px solid #1e4d7a;border-radius:12px;padding:16px 20px;
}
[data-testid="stMetricLabel"]{color:#94a3b8 !important;font-size:0.82rem !important;}
[data-testid="stMetricValue"]{color:#22d3ee !important;font-size:1.6rem !important;font-weight:700 !important;}
.section-heading{
    font-size:1.2rem;font-weight:700;color:#22d3ee;
    border-left:4px solid #0d9488;padding-left:12px;margin:24px 0 14px;
}
.card{background:linear-gradient(135deg,#0f2744,#0a2a4a);
      border:1px solid #1e4d7a;border-radius:12px;padding:20px 24px;margin-bottom:16px;}
.warn-box{background:linear-gradient(90deg,#451a03,#3d1f00);
          border:1px solid #f59e0b;border-radius:10px;
          padding:14px 20px;color:#fde68a;font-size:0.88rem;line-height:1.6;}
.best-badge{display:inline-block;background:linear-gradient(90deg,#0d9488,#0891b2);
            border-radius:6px;padding:3px 10px;font-size:0.72rem;font-weight:700;
            color:white;letter-spacing:0.5px;}
hr{border-color:#1e3a5f !important;}
[data-testid="stPlotlyChart"]{border-radius:12px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    paper_bgcolor="#0d2137", plot_bgcolor="#0a1628",
    font=dict(color="#e2f0ff"),
)

CLF_COLORS = {
    "LogisticRegression": "#3a86ff",
    "DecisionTree":       "#06d6a0",
    "RandomForest":       "#ffbe0b",
    "ExtraTrees":         "#8338ec",
    "XGBoost":            "#ff006e",
    "LinearSVC":          "#fb5607",
}

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 MarineGuard AI")
    st.markdown("---")
    st.markdown("**🧪 Model Laboratory** ← you are here")
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem;color:#64748b;line-height:1.7;'>
<b>Scenario A</b>: 4 features (no leakage)<br>
<b>Scenario B</b>: 7 features (leakage ref)<br>
<b>CV</b>: 5-fold StratifiedKFold<br>
<b>Tuning</b>: RandomizedSearchCV<br>
<b>Models</b>: 6 algorithms compared
</div>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a;border-radius:14px;
            padding:28px 36px;margin-bottom:20px;'>
  <div style='font-size:1.9rem;font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#8338ec);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    🧪 Model Laboratory
  </div>
  <div style='color:#94a3b8;font-size:0.95rem;margin-top:6px;'>
    Six ML algorithms · 5-fold cross-validation (training data only) ·
    RandomizedSearchCV · Permutation importance · ROC curves
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
⚠️ <b>Scientific Honesty Notice</b><br>
All Scenario A models operate on 4 features (Temperature, pH, Salinity, Turbidity) that have
near-zero correlation (|r|&lt;0.01) with the synthetic target. Accuracies near 50% are
<b>expected and correct</b> — they reflect the dataset design, not a modelling failure.
Scenario B (~97%) is shown purely to quantify the leakage effect and
<b>must not be cited as real predictive performance</b>.
</div>
""", unsafe_allow_html=True)

# ── load enhanced metrics ─────────────────────────────────────────────────────
enhanced_path = os.path.join(ROOT, "reports", "model_metrics_enhanced.json")
cv_csv_path   = os.path.join(ROOT, "reports", "cv_results.csv")

if not os.path.isfile(enhanced_path):
    st.error("⚠️ Enhanced metrics not found. Run: `python src/train_model_enhanced.py`")
    st.stop()

with open(enhanced_path, encoding="utf-8") as f:
    meta = json.load(f)

results_A = meta["scenario_A"]["results"]
cv_data   = meta["scenario_A"]["cv_results"]
perm_imp  = meta["scenario_A"]["permutation_importance"]
best_name = meta["scenario_A"]["best_model"]

clf_names = list(results_A.keys())

# ── 1. KPI cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📊 Best Model at a Glance</div>',
            unsafe_allow_html=True)
best = results_A[best_name]
best_cv = cv_data[best_name]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Best Model",     best_name.replace("Regression","Reg."))
c2.metric("Test Accuracy",  f"{best['accuracy']*100:.2f}%")
c3.metric("Test F1",        f"{best['f1_score']:.4f}")
c4.metric("Test ROC-AUC",   f"{best['roc_auc']:.4f}")
c5.metric("CV F1 (mean)",   f"{best_cv['cv_f1_mean']:.4f}")
c6.metric("CV F1 (±std)",   f"±{best_cv['cv_f1_std']:.4f}")

st.info(
    "ℹ️ Best model selected by **cross-validated F1 on training data**. "
    "Test set evaluated exactly once, after selection."
)

# ── 2. Comparison bar chart ───────────────────────────────────────────────────
st.markdown('<div class="section-heading">📈 6-Algorithm Comparison (Test Set)</div>',
            unsafe_allow_html=True)

metrics_map = {
    "Accuracy":  "accuracy",
    "Precision": "precision",
    "Recall":    "recall",
    "F1 (weighted)": "f1_score",
    "ROC-AUC":   "roc_auc",
}
metric_choice = st.radio(
    "Metric to display",
    list(metrics_map.keys()),
    horizontal=True,
    key="lab_metric_radio",
)
m_key = metrics_map[metric_choice]

vals   = [results_A[n][m_key] * 100 for n in clf_names]
colors = [CLF_COLORS.get(n, "#ffffff") for n in clf_names]

fig_bar = go.Figure(go.Bar(
    x=clf_names,
    y=vals,
    marker_color=colors,
    marker_line_color="#1e4d7a",
    marker_line_width=1,
    text=[f"{v:.2f}%" for v in vals],
    textposition="outside",
    textfont=dict(color="#e2f0ff", size=11),
))
fig_bar.add_hline(y=50, line_dash="dot", line_color="#475569",
                  annotation_text="50% baseline",
                  annotation_font_color="#64748b")
fig_bar.update_layout(
    **PLOTLY_DARK,
    height=380,
    title=dict(text=f"{metric_choice} — All 6 Algorithms, Scenario A",
               font=dict(color="#22d3ee", size=13)),
    xaxis_title="Algorithm",
    yaxis_title=f"{metric_choice} (%)",
    yaxis=dict(range=[0, 105]),
    margin=dict(t=60, b=20, l=30, r=20),
)
st.plotly_chart(fig_bar, use_container_width=True, key="lab_bar_chart")

# ── 3. Cross-validation summary table ────────────────────────────────────────
st.markdown('<div class="section-heading">🔁 5-Fold Cross-Validation Results (Training Set)</div>',
            unsafe_allow_html=True)
st.caption("All CV scores computed on training data only. Test set never used during selection.")

cv_rows = []
for n in clf_names:
    cv_s = cv_data[n]
    res  = results_A[n]
    cv_rows.append({
        "Model":          n,
        "CV Acc":         f"{cv_s['cv_accuracy_mean']*100:.2f}% ± {cv_s['cv_accuracy_std']*100:.2f}%",
        "CV F1":          f"{cv_s['cv_f1_mean']:.4f} ± {cv_s['cv_f1_std']:.4f}",
        "CV ROC-AUC":     f"{cv_s['cv_roc_auc_mean']:.4f} ± {cv_s['cv_roc_auc_std']:.4f}",
        "Test Acc":       f"{res['accuracy']*100:.2f}%",
        "Test F1":        f"{res['f1_score']:.4f}",
        "Time (s)":       f"{cv_s['training_time_s']:.1f}",
        "Selected":       "✅ Best" if n == best_name else "",
    })

cv_df = pd.DataFrame(cv_rows)

def _row_style(row):
    if row["Selected"]:
        return ["background-color:#0d2f1f; color:#10b981; font-weight:600"] * len(row)
    return [""] * len(row)

st.dataframe(
    cv_df.style.apply(_row_style, axis=1),
    use_container_width=True,
    hide_index=True,
    key="cv_table",
)

# ── 4. CV error bars ──────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📉 CV F1 Scores with Uncertainty (±1 std)</div>',
            unsafe_allow_html=True)

cv_means = [cv_data[n]["cv_f1_mean"] for n in clf_names]
cv_stds  = [cv_data[n]["cv_f1_std"]  for n in clf_names]
colors_c  = [CLF_COLORS.get(n, "#fff") for n in clf_names]

fig_cv = go.Figure()
for i, n in enumerate(clf_names):
    fig_cv.add_trace(go.Bar(
        name=n,
        x=[n],
        y=[cv_means[i]],
        error_y=dict(type="data", array=[cv_stds[i]], visible=True, color="#e2f0ff"),
        marker_color=colors_c[i],
        marker_line_color="#1e4d7a",
        marker_line_width=1,
        text=[f"{cv_means[i]:.4f}"],
        textposition="outside",
        textfont=dict(color="#e2f0ff", size=10),
    ))
fig_cv.add_hline(y=0.5, line_dash="dot", line_color="#475569")
fig_cv.update_layout(
    **PLOTLY_DARK,
    height=350,
    showlegend=False,
    title=dict(text="5-Fold CV F1 (weighted) — Training Data Only",
               font=dict(color="#22d3ee", size=13)),
    yaxis=dict(range=[0.40, 0.60], title="CV F1 (weighted)"),
    margin=dict(t=60, b=20, l=40, r=20),
)
st.plotly_chart(fig_cv, use_container_width=True, key="cv_f1_bars")

# ── 5. Permutation importance ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">🔍 Permutation Feature Importance (Best Model, Test Set)</div>',
            unsafe_allow_html=True)
st.caption(f"Model: {best_name} · 20 repeats · metric: F1 (weighted) · error bars = ±1 std")

feat_names = list(perm_imp.keys())
pi_means   = [perm_imp[f]["mean"] for f in feat_names]
pi_stds    = [perm_imp[f]["std"]  for f in feat_names]
order      = sorted(range(len(pi_means)), key=lambda i: pi_means[i])

fig_perm = go.Figure(go.Bar(
    x=[pi_means[i] for i in order],
    y=[feat_names[i] for i in order],
    orientation="h",
    error_x=dict(
        type="data",
        array=[pi_stds[i] for i in order],
        visible=True, color="#7dd3fc",
    ),
    marker_color="#22d3ee",
    marker_line_color="#1e4d7a",
    marker_line_width=1,
))
fig_perm.update_layout(
    **PLOTLY_DARK,
    height=280,
    title=dict(
        text="Mean decrease in F1 when feature shuffled",
        font=dict(color="#22d3ee", size=13),
    ),
    xaxis_title="Δ F1 (lower = less important)",
    margin=dict(t=50, b=20, l=120, r=20),
)
st.plotly_chart(fig_perm, use_container_width=True, key="perm_imp_chart")
st.info(
    "📌 All permutation importance scores are near zero — confirming that none of the "
    "four Scenario A features carry strong predictive signal for the synthetic target. "
    "This is consistent with the ~51% accuracy ceiling."
)

# ── 6. Scenario B leakage reference ──────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">⚠️ Scenario B — Target Leakage Reference</div>',
            unsafe_allow_html=True)

res_b = meta["scenario_B"]["results"]["RandomForest"]
ba, bf, br = res_b["accuracy"], res_b["f1_score"], res_b["roc_auc"]

col_a, col_b = st.columns([1, 2])
with col_a:
    st.markdown(f"""
<div class="card" style="border-color:#7f1d1d;">
  <div style="color:#ef4444;font-size:1.1rem;font-weight:700;">⚠️ LEAKAGE SCENARIO</div>
  <div style="margin-top:12px;color:#94a3b8;font-size:0.85rem;">
    <b style="color:#fca5a5;">Accuracy</b>: {ba*100:.2f}%<br>
    <b style="color:#fca5a5;">F1-Score</b>: {bf:.4f}<br>
    <b style="color:#fca5a5;">ROC-AUC</b>: {br:.4f}
  </div>
</div>
""", unsafe_allow_html=True)
with col_b:
    st.markdown("""
<div class="warn-box">
<b>What Scenario B shows:</b><br>
Adding Dissolved Oxygen, Nitrate, and Chlorophyll-a causes accuracy to jump from
~51% to ~97%. These three features <i>directly define</i> the target via a hard
threshold rule. Including them lets the model reverse-engineer the label rather than
learn from water quality signals. This is textbook target leakage.<br><br>
<b>This result must NOT be presented as genuine model performance.</b>
</div>
""", unsafe_allow_html=True)

# ── 7. Best params expander ───────────────────────────────────────────────────
st.markdown("---")
with st.expander("⚙️ Best Hyperparameters per Model (from RandomizedSearchCV)"):
    param_rows = []
    for n in clf_names:
        params = cv_data[n].get("best_params", {})
        param_rows.append({
            "Model":  n,
            "Best Parameters": str(params),
            "CV F1":  f"{cv_data[n]['cv_f1_mean']:.4f}",
        })
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

# ── 8. Methodology note ───────────────────────────────────────────────────────
with st.expander("📖 Methodology — How Models Were Compared"):
    st.markdown("""
### Evaluation Protocol

1. **Data split**: 8,000 train / 2,000 test — stratified, seed 42, fixed before any model sees it.
2. **Cross-validation**: 5-fold `StratifiedKFold` on training set only (1,600 samples per fold held out).
3. **Hyperparameter search**: `RandomizedSearchCV` (≤20 iterations) scored by weighted F1 on CV folds.
4. **Model selection**: Best model chosen by mean CV F1 across 5 folds — test set not consulted.
5. **Final evaluation**: Each tuned model evaluated once on 2,000-sample test set.
6. **Permutation importance**: 20 shuffle repeats on test set with winning model.
7. **Leakage guard**: Assert that Dissolved_Oxygen, Nitrate, Chlorophyll_a are absent from Scenario A features before training.

### Why all accuracies are ~51%
The four Scenario A features (Temperature, pH, Salinity, Turbidity) have Pearson |r|<0.01
with the target `is_safe`. No algorithm — linear or ensemble — can exceed ~51% when
the features contain almost no information about the target. This is honest and expected.

### Reproducibility
- Random seed: `42` throughout
- Training command: `python src/train_model_enhanced.py`
- Output: `reports/model_metrics_enhanced.json`, `reports/cv_results.csv`
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.78rem;'>"
    "MarineGuard AI · Model Laboratory · Dataset: Zenodo 19506986 (Synthetic)"
    "</div>",
    unsafe_allow_html=True,
)
