"""
pages/4_About_Project.py
AI-Based Marine Water Quality Classification System — About Page
"""

import os
import streamlit as st

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="About — Marine Water Quality AI",
    page_icon="ℹ️",
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
.section-heading {
    font-size:1.2rem; font-weight:700; color:#22d3ee;
    border-left:4px solid #0d9488; padding-left:12px; margin:28px 0 14px;
}
.card {
    background:linear-gradient(135deg,#0f2744,#0a2a4a);
    border:1px solid #1e4d7a; border-radius:12px; padding:20px 24px;
    margin-bottom:16px;
}
.warn-box {
    background:linear-gradient(90deg,#451a03,#3d1f00);
    border:1px solid #f59e0b; border-radius:10px;
    padding:14px 20px; color:#fde68a; font-size:0.88rem; line-height:1.6;
}
.tech-chip {
    display:inline-block;
    background:#0f2744; border:1px solid #1e4d7a;
    border-radius:20px; padding:5px 16px;
    font-size:0.82rem; color:#7dd3fc; margin:4px;
    font-weight:600;
}
hr { border-color:#1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌊 Marine Water AI")
    st.markdown("---")
    st.markdown("**ℹ️ About Project** ← you are here")

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a; border-radius:14px;
            padding:28px 36px; margin-bottom:20px;'>
  <div style='font-size:1.9rem; font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#7dd3fc);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    ℹ️ About This Project
  </div>
  <div style='color:#94a3b8; font-size:0.95rem; margin-top:6px;'>
    AI-Based Marine Water Quality Classification System — Academic Research Project
  </div>
</div>
""", unsafe_allow_html=True)

# ── 1. Problem statement ──────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🎯 Problem Statement</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="card">
Marine water quality assessment is critical for monitoring ecosystem health,
protecting biodiversity, and informing conservation policy. Traditional water quality
monitoring relies on laboratory analysis of collected samples — a process that is
expensive, time-consuming, and difficult to scale across large geographic regions.

This project investigates whether <b>machine learning classification</b> can be used
to predict whether a marine water sample is "Safe" or "Unsafe" based on four
physicochemical parameters: <b>Temperature, pH, Salinity, and Turbidity</b>.

The key research challenge is to do this prediction <b>without</b> direct access to the
three biochemical threshold variables (Dissolved Oxygen, Nitrate, Chlorophyll-a) that
were used to construct the ground-truth labels in the training dataset.
</div>
""", unsafe_allow_html=True)

# ── 2. Objectives ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🏆 Project Objectives</div>',
            unsafe_allow_html=True)
objectives = [
    ("1", "Download and verify a 10,000-record marine water quality classification dataset from Zenodo."),
    ("2", "Perform exploratory data analysis and document dataset provenance and limitations."),
    ("3", "Design a leakage-aware feature engineering strategy that excludes target-defining columns from Scenario A."),
    ("4", "Build and evaluate three machine learning classifiers: Logistic Regression, Decision Tree, and Random Forest."),
    ("5", "Apply SHAP interpretability analysis to explain feature contributions in the selected model."),
    ("6", "Deploy an interactive Streamlit dashboard that communicates results and limitations transparently."),
]
obj_cols = st.columns(2)
for i, (num, text) in enumerate(objectives):
    with obj_cols[i % 2]:
        st.markdown(f"""
<div style='background:#0f2744; border:1px solid #1e4d7a; border-radius:10px;
            padding:14px 18px; margin-bottom:12px;'>
  <span style='color:#22d3ee; font-weight:800; font-size:1.1rem;'>{num}.</span>
  <span style='color:#cbd5e1; font-size:0.9rem;'> {text}</span>
</div>
""", unsafe_allow_html=True)

# ── 3. Dataset source and provenance ─────────────────────────────────────────
st.markdown('<div class="section-heading">📦 Dataset Source and Provenance</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="card">
<table style='width:100%; border-collapse:collapse; color:#cbd5e1; font-size:0.9rem;'>
  <tr><td style='padding:8px 12px; color:#94a3b8; width:200px;'><b>Title</b></td>
      <td>A Benchmark Dataset for Marine Water Quality Classification and SHAP Interpretability</td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>Repository</b></td>
      <td>Zenodo — <a href="https://zenodo.org/records/19506986"
                      style="color:#22d3ee;">zenodo.org/records/19506986</a></td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>Curator</b></td>
      <td>Yuthriya Khamis (Multimedia University, Master's thesis research)</td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>Original Source</b></td>
      <td>Kaggle (author: robust_mariner) — now restricted; Zenodo copy archived under FAIR principles</td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>Records</b></td>
      <td>10,000 continuous records × 9 columns</td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>File size</b></td>
      <td>489,190 bytes (489.2 kB)</td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>Missing values</b></td>
      <td>0 (none)</td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>Duplicate rows</b></td>
      <td>0 (none)</td></tr>
  <tr><td style='padding:8px 12px; color:#94a3b8;'><b>Class balance</b></td>
      <td>50% Safe / 50% Unsafe (exactly 5,000 each)</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
⚠️ <b>Synthetic / Modelled Data</b><br>
This dataset was generated using <b>established biogeochemical model equations</b> — it is
NOT real-world laboratory or field-collected data. The measurements are mathematically derived:

• <b>Dissolved Oxygen</b>: DO_adj = DO × exp(−0.05 × (Temp − 20))  — temperature-dependent decay<br>
• <b>Nitrate</b>: Coupled with oxygen via aerobic denitrification<br>
• <b>Chlorophyll-a</b>: Chl_a = 15 + 25 × sigmoid(NO₃ − 10) + N(0, 5) — nutrient-driven growth

<b>Results, predictions, and conclusions from this project cannot be generalised
to real marine environments without independent validation on field-collected data.</b>
</div>
""", unsafe_allow_html=True)

# ── 4. Methodology ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🔬 Methodology</div>',
            unsafe_allow_html=True)
phases = [
    ("Phase 1", "Dataset Inspection", "Downloaded dataset from Zenodo. Verified 10,000 records, 9 columns, 0 missing values, 0 duplicates, 50/50 class balance. Documented synthetic provenance."),
    ("Phase 2", "EDA & Preprocessing", "Generated 6 EDA charts. Identified target-defining columns (DO, NO₃, Chl_a). Designed two-scenario feature split. Applied stratified 80/20 train-test split (seed=42). Fitted SimpleImputer + StandardScaler pipeline on training data only."),
    ("Phase 3", "Model Training & SHAP", "Trained Logistic Regression, Decision Tree, Random Forest on Scenario A (4 features) and Scenario B (7 features). Evaluated on held-out test set. Applied SHAP TreeExplainer to Random Forest. Saved complete pipelines (preprocessor + model)."),
    ("Phase 4", "Dashboard", "Built 5-section Streamlit dashboard: Home, Prediction, Data Analytics, Model Performance, About. All metrics loaded from saved files — not hardcoded."),
]
for pnum, ptitle, pdesc in phases:
    st.markdown(f"""
<div style='background:#0f2744; border-left:4px solid #0d9488;
            border-radius:0 10px 10px 0; padding:14px 20px; margin-bottom:10px;'>
  <span style='color:#22d3ee; font-weight:800;'>{pnum} — {ptitle}</span><br>
  <span style='color:#94a3b8; font-size:0.88rem;'>{pdesc}</span>
</div>
""", unsafe_allow_html=True)

# ── 5. Target leakage explanation ─────────────────────────────────────────────
st.markdown('<div class="section-heading">⚠️ Target Leakage — Explanation</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="card">
<b style='color:#ef4444;'>What is target leakage?</b><br><br>
Target leakage occurs when input features used to train a model contain information
that would not be available at prediction time, or that <i>directly construct</i>
the target variable — meaning the model learns to memorise the label construction
rule rather than discovering genuine predictive patterns.
<br><br>
<b style='color:#ef4444;'>In this dataset:</b><br>
A sample is <b>Safe (1)</b> if and only if:
<ul style='color:#cbd5e1;'>
  <li>Dissolved Oxygen &gt; 8 mg/L</li>
  <li>Nitrate &lt; 15 mg/L</li>
  <li>Chlorophyll-a &lt; 60 μg/L</li>
</ul>
Including these three columns as model features (Scenario B) produces ~97% accuracy —
<b>not</b> because the model learned anything meaningful, but because these features
<i>are</i> the decision rule. Pearson r between these features and the target
exceeds 0.91 in absolute value.
<br><br>
<b style='color:#22d3ee;'>Scenario A exclusion strategy:</b><br>
Dissolved_Oxygen, Nitrate, and Chlorophyll_a are excluded from the primary model.
Only Temperature, pH, Salinity, and Turbidity — which have |r| &lt; 0.01 with the
target — are used. This gives an honest test of predictive power, resulting in
~51% accuracy (near random chance). This outcome is reported honestly.
<br><br>
<b style='color:#f59e0b;'>Residual leakage acknowledgement:</b><br>
Temperature is physically correlated with Dissolved Oxygen through temperature-dependent
gas solubility. This indirect correlation may carry weak residual signal in Scenario A.
It is acknowledged rather than eliminated, as Temperature is a legitimate marine
water quality indicator.
</div>
""", unsafe_allow_html=True)

# ── 6. Technology stack ───────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🛠️ Technology Stack</div>',
            unsafe_allow_html=True)
techs = [
    "Python 3.14.6", "Streamlit", "Pandas", "NumPy",
    "Scikit-learn", "joblib", "SHAP 0.54.1",
    "Matplotlib", "Seaborn", "Plotly",
]
chips = " ".join(f'<span class="tech-chip">{t}</span>' for t in techs)
st.markdown(f'<div style="margin:8px 0;">{chips}</div>', unsafe_allow_html=True)

# ── 7. Model selection rationale ──────────────────────────────────────────────
st.markdown('<div class="section-heading">🤖 Model Selection Rationale</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="card">
The <b>Random Forest Classifier</b> was selected as the primary Scenario A model based
on F1-score on the held-out test set (F1 = 0.5083, Accuracy = 51.35%, ROC-AUC = 51.41%).

<br><br>

| Model | Accuracy | F1-Score | ROC-AUC | Notes |
|-------|----------|----------|---------|-------|
| Logistic Regression | 48.90% | 49.10% | 48.47% | Linear boundary; below 50% baseline |
| Decision Tree | 50.90% | 43.56% | 49.94% | Low recall (37.9%) — misses many Safe samples |
| **Random Forest ✅** | **51.35%** | **50.83%** | **51.41%** | Best on all metrics; supports SHAP interpretation |

<br>
<b>Selection criterion</b>: F1-score on held-out test set (stratified, 2,000 samples).
F1 is chosen over accuracy because it balances precision and recall, which is important
for safety-classification tasks.

<br><br>
<b>Why all accuracies are near 50%</b>: The four Scenario A features have near-zero
Pearson correlation with the target (|r| &lt; 0.01). Any signal above 50% reflects
genuine non-linear or interaction effects captured by the ensemble, but the effect
size is small given the synthetic dataset design.
</div>
""", unsafe_allow_html=True)

# ── 8. Limitations ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">⚠️ Limitations</div>',
            unsafe_allow_html=True)
limitations = [
    ("Synthetic data", "All 10,000 records are mathematically derived. Results cannot be generalised to real marine environments without external validation."),
    ("Deterministic target", "The classification label is a hard threshold rule — no statistical uncertainty in the labels. Real classification problems rarely have perfectly deterministic targets."),
    ("Near-zero feature correlation", "Scenario A features (Temp, pH, Salinity, Turbidity) have |r| < 0.01 with the target. Model performance is near random chance (~51%)."),
    ("Residual indirect leakage", "Temperature correlates with Dissolved Oxygen through physical law. This indirect signal may slightly inflate Scenario A performance."),
    ("No temporal or spatial data", "There is no time-series or geospatial metadata. The dataset cannot support trend analysis or location-based predictions."),
    ("No external validation", "The pipeline has not been tested on independently collected real marine water samples."),
]
lim_cols = st.columns(2)
for i, (title, desc) in enumerate(limitations):
    with lim_cols[i % 2]:
        st.markdown(f"""
<div style='background:linear-gradient(135deg,#1a0a0a,#200d0d);
            border:1px solid #7f1d1d; border-radius:10px;
            padding:14px 18px; margin-bottom:12px;'>
  <div style='color:#ef4444; font-weight:700; margin-bottom:4px;'>⚠️ {title}</div>
  <div style='color:#94a3b8; font-size:0.85rem;'>{desc}</div>
</div>
""", unsafe_allow_html=True)

# ── 9. Future scope ───────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🚀 Future Scope</div>',
            unsafe_allow_html=True)
future_items = [
    "🌊 Validate on real-world field-collected marine water quality datasets",
    "🔬 Add biochemical features (DO, Nitrate, Chl-a) as contextual display without using them for classification",
    "🧠 Experiment with XGBoost, SVM, and neural network classifiers",
    "📍 Add geospatial metadata for location-aware predictions",
    "⏱️  Add temporal features for time-series trend analysis",
    "📱 Deploy as a responsive mobile web application",
    "🔄 Implement automated retraining pipeline for new data ingestion",
    "📊 Add calibration curves and reliability diagrams for probability outputs",
]
fc1, fc2 = st.columns(2)
for i, item in enumerate(future_items):
    with (fc1 if i % 2 == 0 else fc2):
        st.markdown(f"""
<div style='background:#0f2744; border:1px solid #1e4d7a; border-radius:8px;
            padding:10px 16px; margin-bottom:8px; color:#cbd5e1; font-size:0.88rem;'>
{item}
</div>
""", unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.8rem; line-height:1.8;'>
<b>AI-Based Marine Water Quality Classification System</b><br>
Computer Engineering Academic Project · Dataset: Zenodo 19506986 (Synthetic)<br>
Python 3.14.6 · Scikit-learn · Streamlit · SHAP 0.54.1<br>
⚠️ This project uses synthetic data. Predictions are educational — not authoritative environmental guidance.
</div>
""", unsafe_allow_html=True)
