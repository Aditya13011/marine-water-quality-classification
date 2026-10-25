"""
app.py — Home Dashboard
AI-Based Marine Water Quality Classification System
"""

import os, sys, json
import streamlit as st
import pandas as pd

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Water Quality AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared path helper ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base palette ── */
:root {
    --navy:   #0a1628;
    --deep:   #0d2137;
    --teal:   #0d9488;
    --cyan:   #22d3ee;
    --sky:    #7dd3fc;
    --white:  #f0f9ff;
    --muted:  #94a3b8;
    --warn:   #f59e0b;
    --danger: #ef4444;
}

/* ── Background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a1628 0%, #0d2137 55%, #0a1f35 100%);
    color: #e2f0ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061022 0%, #0d1f3c 100%);
    border-right: 1px solid #1e3a5f;
}

/* ── Sidebar text ── */
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .st-emotion-cache-1cypcdb { color: #22d3ee !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f2744 0%, #0d2137 100%);
    border: 1px solid #1e4d7a;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"]  { color: #22d3ee !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* ── Info / warning / success banners ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Hero card ── */
.hero-card {
    background: linear-gradient(135deg, #0f2744 0%, #0a2a4a 50%, #083344 100%);
    border: 1px solid #1e4d7a;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(34,211,238,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #22d3ee, #7dd3fc, #a5f3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 0;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #22d3ee;
    border-left: 4px solid #0d9488;
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* ── Info chip ── */
.chip {
    display: inline-block;
    background: #0f2744;
    border: 1px solid #1e4d7a;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #7dd3fc;
    margin: 3px;
}

/* ── Synthetic data warning banner ── */
.synthetic-banner {
    background: linear-gradient(90deg, #451a03, #3d1f00);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0;
    color: #fde68a;
    font-size: 0.9rem;
    line-height: 1.55;
}

/* ── Phase badge ── */
.phase-badge {
    display: inline-block;
    background: linear-gradient(90deg, #0d9488, #0891b2);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    color: white;
    letter-spacing: 0.5px;
    margin-right: 6px;
}

/* ── Divider ── */
hr { border-color: #1e3a5f !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Plotly/Altair chart containers ── */
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1628; }
::-webkit-scrollbar-thumb { background: #1e4d7a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Marine Water AI")
    st.markdown("---")
    st.markdown("""
**Navigation**
- 🏠 **Home** ← you are here
- 🔬 **Prediction**
- 📊 **Data Analytics**
- 📈 **Model Performance**
- ℹ️ **About Project**

Use the page list above ↑ to navigate.
""")
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem; color:#64748b; line-height:1.6;'>
<b>Dataset</b><br>Zenodo 19506986<br>
<b>Records</b>: 10,000 (synthetic)<br>
<b>Model</b>: RandomForest (Scenario A)<br>
<b>Phase</b>: 4 — Dashboard
</div>
""", unsafe_allow_html=True)

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <div class="hero-title">🌊 AI-Based Marine Water Quality<br>Classification System</div>
  <div class="hero-sub">
    Machine-learning classification of marine water samples as <b>Safe</b> or <b>Unsafe</b>
    using physicochemical indicators — with full transparency about model limitations.
  </div>
</div>
""", unsafe_allow_html=True)

# ── synthetic data notice ─────────────────────────────────────────────────────
st.markdown("""
<div class="synthetic-banner">
⚠️ <b>Dataset Provenance Notice</b><br>
This project uses a <b>mathematically derived (synthetic) dataset</b>
(Zenodo record 19506986). The 10,000 records were generated using
established biogeochemical model equations — they are <b>not</b> real
laboratory or field observations. Model results should not be
generalised to real marine environments without independent validation.
</div>
""", unsafe_allow_html=True)

# ── load live data ────────────────────────────────────────────────────────────
csv_path  = os.path.join(ROOT, "data", "marine_water_quality.csv")
meta_path = os.path.join(ROOT, "reports", "model_metrics.json")

df   = pd.read_csv(csv_path)   if os.path.isfile(csv_path)  else None
meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else None

# ── dataset overview metrics ──────────────────────────────────────────────────
st.markdown('<div class="section-heading">📋 Dataset Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
if df is not None:
    n_safe   = int((df["is_safe"] == 1).sum())
    n_unsafe = int((df["is_safe"] == 0).sum())
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("Features Used",  "4 (Scenario A)")
    c3.metric("✅ Safe Samples",   f"{n_safe:,}")
    c4.metric("🚫 Unsafe Samples", f"{n_unsafe:,}")
    c5.metric("Class Balance",   "50 / 50 %")
else:
    st.error("⚠️ Dataset file not found: data/marine_water_quality.csv")

# ── selected model metrics ────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🤖 Selected Model — Random Forest (Scenario A)</div>',
            unsafe_allow_html=True)

if meta:
    rf = meta["scenario_A"]["results"]["RandomForest"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{rf['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{rf['precision']*100:.2f}%")
    m3.metric("Recall",    f"{rf['recall']*100:.2f}%")
    m4.metric("F1-Score",  f"{rf['f1_score']*100:.2f}%")
    m5.metric("ROC-AUC",   f"{rf['roc_auc']*100:.2f}%")
else:
    st.warning("reports/model_metrics.json not found — run Phase 3 training first.")

st.info(
    "ℹ️ **Honest performance context:** Scenario A accuracy (~51%) is near random chance. "
    "The four input features (Temperature, pH, Salinity, Turbidity) have near-zero linear "
    "correlation with the target. This reflects a genuine limitation of the synthetic "
    "dataset design, not a model bug."
)

# ── feature overview ──────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🔬 Input Features (Scenario A)</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
| Feature | Unit | Dataset Range |
|---------|------|---------------|
| Temperature | °C | 10.0 – 32.0 |
| pH | (dimensionless) | 6.5 – 8.8 |
| Salinity | PSU | 30.0 – 40.0 |
| Turbidity | NTU | 0.5 – 50.0 |
""")
with col_b:
    st.markdown("""
**Why these four features?**

The target `is_safe` is defined by a deterministic threshold rule on
**Dissolved Oxygen**, **Nitrate**, and **Chlorophyll-a**.
Including those three columns would allow the model to trivially memorise
the rule (~97% accuracy) — which is misleading, not genuine learning.

Scenario A excludes those columns and tests whether the four remaining
physicochemical parameters carry predictive signal independently.
""")

# ── classification scheme ─────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🎯 Classification Scheme</div>',
            unsafe_allow_html=True)
st.markdown("""
A water sample is labelled **Safe (1)** in the dataset if and only if:

> Dissolved Oxygen > 8 mg/L  **AND**  Nitrate < 15 mg/L  **AND**  Chlorophyll-a < 60 μg/L

All other samples are **Unsafe (0)**.

*(None of these three columns are used as input features in Scenario A.)*
""")

# ── phase status ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📌 Project Phase Status</div>',
            unsafe_allow_html=True)
phases = [
    ("1", "Dataset Download & Inspection",              "✅ Complete"),
    ("2", "EDA, Feature Engineering & Preprocessing",  "✅ Complete"),
    ("3", "Model Training, Evaluation & SHAP",          "✅ Complete"),
    ("4", "Streamlit Dashboard",                        "✅ Complete"),
]
ph_cols = st.columns(4)
for i, (num, desc, status) in enumerate(phases):
    with ph_cols[i]:
        st.markdown(f"""
<div style='background:#0f2744; border:1px solid #1e4d7a; border-radius:10px;
            padding:16px; text-align:center;'>
  <div style='font-size:1.6rem; font-weight:800; color:#22d3ee;'>Phase {num}</div>
  <div style='font-size:0.82rem; color:#94a3b8; margin:6px 0;'>{desc}</div>
  <div style='font-size:0.9rem; font-weight:600; color:#10b981;'>{status}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.78rem;'>"
    "AI-Based Marine Water Quality Classification System · "
    "Dataset: Zenodo 19506986 (Synthetic) · Python + Streamlit + Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
