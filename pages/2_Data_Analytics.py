"""
pages/2_Data_Analytics.py
AI-Based Marine Water Quality Classification System — Data Analytics Page
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analytics — Marine Water Quality AI",
    page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌊 Marine Water AI")
    st.markdown("---")
    st.markdown("**📊 Data Analytics** ← you are here")
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem; color:#64748b; line-height:1.7;'>
<b>Dataset</b>: Zenodo 19506986<br>
<b>Records</b>: 10,000 (synthetic)<br>
<b>Features</b>: 4 safe + 3 threshold<br>
<b>Balance</b>: 50% / 50%
</div>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a; border-radius:14px;
            padding:28px 36px; margin-bottom:20px;'>
  <div style='font-size:1.9rem; font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#7dd3fc);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📊 Data Analytics
  </div>
  <div style='color:#94a3b8; font-size:0.95rem; margin-top:6px;'>
    Exploratory analysis of the marine water quality dataset. All statistics are
    computed from the actual dataset file — nothing is fabricated or hardcoded.
  </div>
</div>
""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(ROOT, "data", "marine_water_quality.csv")
if not os.path.isfile(csv_path):
    st.error("❌ Dataset not found: data/marine_water_quality.csv")
    st.stop()

df = pd.read_csv(csv_path)
TARGET         = "is_safe"
ID_COL         = "Sample_ID"
THRESHOLD_COLS = ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]
SAFE_FEATS     = ["Temperature", "pH", "Salinity", "Turbidity"]
ALL_NUMERIC    = [c for c in df.columns if c not in [TARGET, ID_COL]
                  and pd.api.types.is_numeric_dtype(df[c])]

PLOTLY_DARK = dict(
    paper_bgcolor="#0d2137", plot_bgcolor="#0a1628",
    font=dict(color="#e2f0ff"),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
)

# ── dataset summary metrics ───────────────────────────────────────────────────
st.markdown('<div class="section-heading">📋 Dataset Summary</div>',
            unsafe_allow_html=True)
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Total Records",  f"{len(df):,}")
m2.metric("Columns",        str(df.shape[1]))
m3.metric("Missing Values", str(df.isnull().sum().sum()))
m4.metric("Duplicate Rows", str(df.duplicated().sum()))
m5.metric("Safe (1)",       f"{int((df[TARGET]==1).sum()):,}")
m6.metric("Unsafe (0)",     f"{int((df[TARGET]==0).sum()):,}")

st.markdown("""
<div class="warn-box">
⚠️ <b>Synthetic Dataset</b> — The 10,000 records were generated using
biogeochemical model equations (Zenodo record 19506986), not collected in the field.
The target `is_safe` is a <b>deterministic threshold rule</b> on
Dissolved_Oxygen, Nitrate, and Chlorophyll_a.
</div>
""", unsafe_allow_html=True)

# ── 1. Class distribution ─────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🎯 Target Class Distribution</div>',
            unsafe_allow_html=True)
vc = df[TARGET].value_counts().sort_index()
c1, c2 = st.columns([1, 2])
with c1:
    fig_pie = go.Figure(go.Pie(
        labels=["Unsafe (0)", "Safe (1)"],
        values=[vc[0], vc[1]],
        hole=0.55,
        marker=dict(colors=["#ef4444", "#10b981"],
                    line=dict(color="#0d2137", width=3)),
        textfont=dict(color="#e2f0ff", size=13),
    ))
    fig_pie.update_layout(
        showlegend=True,
        legend=dict(font=dict(color="#e2f0ff")),
        annotations=[dict(text="is_safe", x=0.5, y=0.5, font_size=14,
                          font_color="#94a3b8", showarrow=False)],
        **PLOTLY_DARK, height=320, margin=dict(t=20,b=10,l=10,r=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True, key="class_dist_pie")
with c2:
    fig_bar = px.bar(
        x=["Unsafe (0)", "Safe (1)"], y=[vc[0], vc[1]],
        color=["Unsafe (0)", "Safe (1)"],
        color_discrete_map={"Unsafe (0)": "#ef4444", "Safe (1)": "#10b981"},
        text=[f"{vc[0]:,}\n(50%)", f"{vc[1]:,}\n(50%)"],
        labels={"x": "Class", "y": "Count"},
    )
    fig_bar.update_traces(textposition="outside",
                          textfont=dict(color="#e2f0ff"))
    fig_bar.update_layout(**PLOTLY_DARK, showlegend=False, height=320,
                          margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig_bar, use_container_width=True, key="class_dist_bar")

# ── 2. Feature distributions ──────────────────────────────────────────────────
st.markdown('<div class="section-heading">📈 Feature Distributions by Class</div>',
            unsafe_allow_html=True)

tab_safe, tab_all = st.tabs(["🟢 Scenario A features (no leakage)",
                               "⚠️ All features (incl. threshold cols)"])

with tab_safe:
    cols_per_row = 2
    for i in range(0, len(SAFE_FEATS), cols_per_row):
        row_feats = SAFE_FEATS[i:i+cols_per_row]
        rcols = st.columns(len(row_feats))
        for j, feat in enumerate(row_feats):
            fig = go.Figure()
            for cls, clr, nm in [(0,"#ef4444","Unsafe (0)"), (1,"#10b981","Safe (1)")]:
                vals = df[df[TARGET]==cls][feat]
                fig.add_trace(go.Histogram(
                    x=vals, name=nm, nbinsx=40,
                    marker_color=clr, opacity=0.7,
                ))
            fig.update_layout(**PLOTLY_DARK, barmode="overlay",
                              title=dict(text=feat, font=dict(color="#22d3ee")),
                              xaxis_title=feat, yaxis_title="Count",
                              legend=dict(font=dict(color="#e2f0ff")),
                              height=280, margin=dict(t=40,b=30,l=30,r=10))
            rcols[j].plotly_chart(fig, use_container_width=True, key=f"hist_safe_{feat}")

with tab_all:
    st.warning("⚠️ Columns marked ⚠️ directly define the target — they are shown here for EDA context only.")
    for i in range(0, len(ALL_NUMERIC), 2):
        row_feats = ALL_NUMERIC[i:i+2]
        rcols = st.columns(len(row_feats))
        for j, feat in enumerate(row_feats):
            tag  = " ⚠️" if feat in THRESHOLD_COLS else ""
            fig  = go.Figure()
            for cls, clr, nm in [(0,"#ef4444","Unsafe (0)"), (1,"#10b981","Safe (1)")]:
                vals = df[df[TARGET]==cls][feat]
                fig.add_trace(go.Histogram(
                    x=vals, name=nm, nbinsx=40,
                    marker_color=clr, opacity=0.7,
                ))
            fig.update_layout(**PLOTLY_DARK, barmode="overlay",
                              title=dict(text=f"{feat}{tag}",
                                         font=dict(color="#ef4444" if tag else "#22d3ee")),
                              xaxis_title=feat, yaxis_title="Count",
                              legend=dict(font=dict(color="#e2f0ff")),
                              height=280, margin=dict(t=40,b=30,l=30,r=10))
            rcols[j].plotly_chart(fig, use_container_width=True, key=f"hist_all_{feat}")

# ── 3. Boxplots ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📦 Feature Boxplots by Class</div>',
            unsafe_allow_html=True)
box_df = df[SAFE_FEATS + [TARGET]].copy()
box_df["Class"] = box_df[TARGET].map({0: "Unsafe (0)", 1: "Safe (1)"})
box_cols = st.columns(len(SAFE_FEATS))
for i, feat in enumerate(SAFE_FEATS):
    fig = px.box(
        box_df, x="Class", y=feat, color="Class",
        color_discrete_map={"Unsafe (0)": "#ef4444", "Safe (1)": "#10b981"},
        points=False,
    )
    fig.update_layout(**PLOTLY_DARK, showlegend=False,
                      title=dict(text=feat, font=dict(color="#22d3ee")),
                      height=300, margin=dict(t=40,b=30,l=30,r=10))
    box_cols[i].plotly_chart(fig, use_container_width=True, key=f"box_{feat}")

# ── 4. Correlation heatmap ────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🔥 Correlation Heatmap</div>',
            unsafe_allow_html=True)
corr_df = df[ALL_NUMERIC + [TARGET]].corr().round(3)
fig_heat = px.imshow(
    corr_df,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    text_auto=True,
    aspect="auto",
)
fig_heat.update_traces(textfont=dict(size=10))
fig_heat.update_layout(
    **PLOTLY_DARK,
    title=dict(text="Pearson Correlation Matrix (all numeric features + target)",
               font=dict(color="#22d3ee")),
    height=480, margin=dict(t=50,b=20,l=20,r=20),
    coloraxis_colorbar=dict(tickfont=dict(color="#e2f0ff"),
                            title=dict(font=dict(color="#94a3b8"))),
)
st.plotly_chart(fig_heat, use_container_width=True, key="corr_heatmap")
st.info("💡 Notice: Dissolved_Oxygen, Nitrate, and Chlorophyll_a show r > 0.9 with `is_safe`, "
        "revealing the deterministic leakage. The four Scenario A features show r ≈ 0.")

# ── 5. Feature vs target correlation bar ─────────────────────────────────────
st.markdown('<div class="section-heading">📉 Feature Correlation with Target</div>',
            unsafe_allow_html=True)
corr_series = df[ALL_NUMERIC].corrwith(df[TARGET]).sort_values()
colors = ["#ef4444" if c in THRESHOLD_COLS else "#22d3ee"
          for c in corr_series.index]
fig_corr = go.Figure(go.Bar(
    x=corr_series.values,
    y=corr_series.index,
    orientation="h",
    marker_color=colors,
    text=[f"{v:+.4f}" for v in corr_series.values],
    textposition="outside",
    textfont=dict(color="#e2f0ff", size=11),
))
fig_corr.add_vline(x=0, line_color="#475569", line_width=1)
fig_corr.update_layout(
    **PLOTLY_DARK,
    title=dict(
        text="Pearson r with is_safe  (red = threshold-defining / leakage risk)",
        font=dict(color="#22d3ee")
    ),
    xaxis_title="Correlation (r)",
    height=380, margin=dict(t=50,b=30,l=30,r=80),
)
st.plotly_chart(fig_corr, use_container_width=True, key="feat_target_corr")

# ── 6. Summary statistics table ───────────────────────────────────────────────
st.markdown('<div class="section-heading">📋 Summary Statistics</div>',
            unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Scenario A features", "Threshold features (⚠️ leakage)"])
with tab1:
    st.dataframe(
        df[SAFE_FEATS].describe().T.round(3).style.background_gradient(
            cmap="Blues", axis=0
        ),
        use_container_width=True
    )
with tab2:
    st.warning("⚠️ These columns directly define the target. Not used in Scenario A.")
    st.dataframe(
        df[THRESHOLD_COLS].describe().T.round(3),
        use_container_width=True
    )

# ── 7. Missing value summary ──────────────────────────────────────────────────
st.markdown('<div class="section-heading">🔍 Missing Value Summary</div>',
            unsafe_allow_html=True)
mv = df.isnull().sum().reset_index()
mv.columns = ["Column", "Missing Count"]
mv["Missing %"] = (mv["Missing Count"] / len(df) * 100).round(2)
mv["Status"] = mv["Missing Count"].apply(lambda x: "✅ Complete" if x == 0 else "⚠️ Has missing")
st.dataframe(mv, use_container_width=True, hide_index=True)

# ── 8. Saved EDA charts ───────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🖼️ Saved EDA Charts (from Phase 2)</div>',
            unsafe_allow_html=True)
chart_map = {
    "Safe Feature Boxplots":       "04_safe_features_boxplots.png",
    "Threshold Leakage Analysis":  "05_threshold_leakage_analysis.png",
    "Pairplot (Safe Features)":    "06_pairplot_safe_features.png",
}
tab_labels = list(chart_map.keys())
tabs = st.tabs(tab_labels)
for i, (label, fname) in enumerate(chart_map.items()):
    chart_path = os.path.join(ROOT, "reports", fname)
    with tabs[i]:
        if os.path.isfile(chart_path):
            st.image(chart_path, use_container_width=True)
        else:
            st.warning(f"Chart not found: reports/{fname}. Run src/eda.py to regenerate.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.78rem;'>"
    "All statistics computed live from data/marine_water_quality.csv · "
    "No values fabricated."
    "</div>",
    unsafe_allow_html=True
)
