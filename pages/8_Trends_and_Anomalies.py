"""
pages/8_Trends_and_Anomalies.py
MarineGuard AI — Phase 5: Trend Analysis & Anomaly Detection
Isolation Forest with configurable contamination, CSV export of flagged records.
Trend analysis requires timestamp column; explains why forecasting is deferred.
"""

import os, sys, io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

st.set_page_config(
    page_title="Trends & Anomalies — MarineGuard AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background:linear-gradient(160deg,#0a1628 0%,#0d2137 55%,#0a1f35 100%);
    color:#e2f0ff;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#061022 0%,#0d1f3c 100%);
    border-right:1px solid #1e3a5f;
}
[data-testid="stSidebar"] *{color:#cbd5e1 !important;}
.section-heading{
    font-size:1.2rem;font-weight:700;color:#22d3ee;
    border-left:4px solid #0d9488;padding-left:12px;margin:24px 0 14px;
}
.warn-box{background:linear-gradient(90deg,#451a03,#3d1f00);
          border:1px solid #f59e0b;border-radius:10px;
          padding:14px 20px;color:#fde68a;font-size:0.88rem;line-height:1.6;}
.info-box{background:linear-gradient(90deg,#0c2340,#0a1f35);
          border:1px solid #1e4d7a;border-radius:10px;
          padding:14px 20px;color:#7dd3fc;font-size:0.88rem;line-height:1.6;}
hr{border-color:#1e3a5f !important;}
[data-testid="stPlotlyChart"]{border-radius:12px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(paper_bgcolor="#0d2137", plot_bgcolor="#0a1628", font=dict(color="#e2f0ff"))

with st.sidebar:
    st.markdown("## 🌊 MarineGuard AI")
    st.markdown("---")
    st.markdown("**📡 Trends & Anomalies** ← you are here")

st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a;border-radius:14px;padding:28px 36px;margin-bottom:20px;'>
  <div style='font-size:1.9rem;font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#ff006e);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    📡 Trends & Anomaly Detection
  </div>
  <div style='color:#94a3b8;font-size:0.95rem;margin-top:6px;'>
    Historical trend analysis · Isolation Forest anomaly detection ·
    Flagged record export · Forecasting deferred (see below)
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
tabs = st.tabs(["📈 Trend Analysis", "🚨 Anomaly Detection", "📊 About & Limitations"])

# ── Helper: load data (uploaded or project CSV) ───────────────────────────────
@st.cache_data(show_spinner=False)
def load_project_csv():
    path = os.path.join(ROOT, "data", "marine_water_quality.csv")
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None

def get_dataframe():
    """Return uploaded df or project df, or None."""
    if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
        return st.session_state["uploaded_df"].copy(), st.session_state.get("uploaded_name","uploaded")
    df = load_project_csv()
    if df is not None:
        return df, "marine_water_quality.csv (project dataset, synthetic)"
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-heading">📈 Trend Analysis</div>', unsafe_allow_html=True)

    df_base, df_name = get_dataframe()

    # Check for timestamp
    has_timestamp = False
    time_col = None

    if df_base is not None:
        for c in df_base.columns:
            if c.lower() in ("timestamp","date","time","datetime"):
                try:
                    df_base[c] = pd.to_datetime(df_base[c])
                    time_col = c
                    has_timestamp = True
                    break
                except Exception:
                    pass

    if not has_timestamp:
        st.markdown("""
<div class="warn-box">
⚠️ <b>No timestamp column detected in the current dataset.</b><br><br>
The project dataset (Zenodo 19506986) contains no temporal metadata —
it is a static cross-sectional synthetic dataset with no dates or time ordering.<br><br>
<b>To use trend analysis:</b>
<ul>
  <li>Upload a CSV with a <code>timestamp</code>, <code>date</code>, or <code>datetime</code> column via Marine Data Sources.</li>
  <li>Copernicus and NOAA products include timestamps suitable for trend analysis.</li>
</ul>
<b>Forecasting is also deferred</b> for the same reason: without a time-series dataset,
no valid forecast can be produced. Using a static dataset to train a forecasting model
would be methodologically unsound.
</div>
""", unsafe_allow_html=True)

        # Show distribution plots as a useful alternative
        if df_base is not None:
            st.markdown('<div class="section-heading">📊 Variable Distributions (Static Analysis)</div>',
                        unsafe_allow_html=True)
            num_cols = df_base.select_dtypes(include="number").columns.tolist()
            excl = {"is_safe","Unnamed: 0"}
            num_cols = [c for c in num_cols if c not in excl]

            if num_cols:
                feat_choice = st.selectbox("Select variable", num_cols, key="trend_feat")
                if "is_safe" in df_base.columns:
                    fig = px.histogram(
                        df_base, x=feat_choice, color=df_base["is_safe"].map({0:"Unsafe",1:"Safe"}),
                        barmode="overlay", nbins=50,
                        color_discrete_map={"Safe":"#10b981","Unsafe":"#ef4444"},
                    )
                else:
                    fig = px.histogram(df_base, x=feat_choice, nbins=50,
                                       color_discrete_sequence=["#22d3ee"])
                fig.update_layout(**PLOTLY_DARK, height=360,
                                  title=dict(text=f"{feat_choice} distribution",
                                             font=dict(color="#22d3ee")),
                                  margin=dict(t=50,b=20,l=30,r=20))
                st.plotly_chart(fig, use_container_width=True, key="trend_dist")
    else:
        num_cols_t = df_base.select_dtypes(include="number").columns.tolist()
        num_cols_t = [c for c in num_cols_t if c not in {"is_safe","Unnamed: 0"}]

        st.success(f"✅ Timestamp column found: `{time_col}` · Source: **{df_name}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            feat_t = st.selectbox("Variable", num_cols_t, key="trend_feat_ts")
        with col2:
            agg_period = st.selectbox("Aggregate by", ["Day","Week","Month"], index=2,
                                      key="trend_agg")
        with col3:
            if "location" in df_base.columns or "region" in df_base.columns:
                loc_col = "location" if "location" in df_base.columns else "region"
                locs = ["All"] + sorted(df_base[loc_col].dropna().unique().tolist())
                sel_loc = st.selectbox("Location", locs, key="trend_loc")
                if sel_loc != "All":
                    df_base = df_base[df_base[loc_col] == sel_loc]

        freq_map = {"Day":"D","Week":"W","Month":"ME"}
        freq = freq_map[agg_period]

        df_ts = df_base.set_index(time_col)[[feat_t]].dropna()
        df_agg = df_ts.resample(freq).agg(["mean","std","min","max"]).reset_index()
        df_agg.columns = [time_col,"mean","std","min","max"]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_agg[time_col], y=df_agg["max"],
            mode="lines", line=dict(width=0), name="Max",
            showlegend=False,
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_agg[time_col], y=df_agg["min"],
            mode="lines", fill="tonexty",
            fillcolor="rgba(34,211,238,0.08)",
            line=dict(width=0), name="Min–Max range",
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_agg[time_col], y=df_agg["mean"],
            mode="lines+markers",
            line=dict(color="#22d3ee", width=2),
            marker=dict(size=4),
            name=f"Mean {feat_t}",
        ))
        fig_trend.update_layout(
            **PLOTLY_DARK, height=400,
            title=dict(text=f"{feat_t} — {agg_period}ly trend",
                       font=dict(color="#22d3ee")),
            xaxis_title="Date", yaxis_title=feat_t,
            margin=dict(t=60,b=20,l=40,r=20),
            legend=dict(bgcolor="#0f2744"),
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="trend_ts_chart")

        # Missing data indicator
        n_missing = int(df_ts.isnull().sum().sum())
        if n_missing:
            st.info(f"ℹ️ {n_missing} missing values were excluded from aggregation.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-heading">🚨 Isolation Forest Anomaly Detection</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="warn-box">
⚠️ <b>Important terminology</b><br>
An <i>anomaly</i> here means a <b>statistically unusual observation</b> relative to the rest of the dataset.
It does NOT mean pollution, contamination, or safety risk without independent scientific verification.
Always investigate flagged records using domain knowledge before drawing conclusions.
</div>
""", unsafe_allow_html=True)

    df_anom, df_anom_name = get_dataframe()

    if df_anom is None:
        st.error("⚠️ No dataset available. Load data via Marine Data Sources.")
        st.stop()

    num_cols_all = df_anom.select_dtypes(include="number").columns.tolist()
    excluded_default = {"is_safe","Unnamed: 0"}
    num_cols_all = [c for c in num_cols_all if c not in excluded_default]

    if not num_cols_all:
        st.error("No numeric columns found in the dataset.")
        st.stop()

    st.caption(f"Dataset: **{df_anom_name}** — {len(df_anom):,} rows")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        selected_feats = st.multiselect(
            "Features for anomaly detection",
            options=num_cols_all,
            default=num_cols_all[:min(4, len(num_cols_all))],
            key="anom_feats",
        )
    with col_b:
        contamination = st.slider(
            "Contamination (%)",
            min_value=1, max_value=25, value=5,
            key="anom_contamination",
            help="Expected % of anomalous records. Higher values flag more points.",
        )
    with col_c:
        n_estimators = st.slider(
            "Trees (n_estimators)",
            min_value=50, max_value=300, value=100, step=50,
            key="anom_n_trees",
        )

    if not selected_feats:
        st.warning("Select at least one feature.")
        st.stop()

    if st.button("🔍 Run Anomaly Detection", key="run_anomaly", type="primary"):
        with st.spinner("Fitting Isolation Forest …"):
            X_anom = df_anom[selected_feats].dropna()
            idx_valid = X_anom.index

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_anom)

            iso = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination / 100.0,
                random_state=42,
                n_jobs=-1,
            )
            preds  = iso.fit_predict(X_scaled)
            scores = iso.decision_function(X_scaled)  # higher = more normal

            df_result = df_anom.loc[idx_valid].copy()
            df_result["anomaly_flag"]  = (preds == -1)
            df_result["anomaly_score"] = -scores        # invert: higher = more anomalous

            n_flagged = int(df_result["anomaly_flag"].sum())
            n_total   = len(df_result)

        st.success(f"✅ Done — **{n_flagged}** of {n_total:,} records flagged as statistically unusual "
                   f"({n_flagged/n_total*100:.1f}%)")

        # ── Scatter of anomaly score ──────────────────────────────────────────
        fig_scores = go.Figure()
        normal  = df_result[~df_result["anomaly_flag"]]
        anomaly = df_result[ df_result["anomaly_flag"]]

        fig_scores.add_trace(go.Scatter(
            x=normal.index, y=normal["anomaly_score"],
            mode="markers", name="Normal",
            marker=dict(color="#22d3ee", size=4, opacity=0.5),
        ))
        fig_scores.add_trace(go.Scatter(
            x=anomaly.index, y=anomaly["anomaly_score"],
            mode="markers", name="⚠️ Flagged",
            marker=dict(color="#ef4444", size=6, symbol="x"),
        ))
        fig_scores.update_layout(
            **PLOTLY_DARK, height=340,
            title=dict(text="Anomaly Score by Record Index",
                       font=dict(color="#22d3ee")),
            xaxis_title="Record index", yaxis_title="Anomaly score (higher = more unusual)",
            margin=dict(t=50,b=20,l=40,r=20),
            legend=dict(bgcolor="#0f2744"),
        )
        st.plotly_chart(fig_scores, use_container_width=True, key="anom_score_scatter")

        if len(selected_feats) >= 2:
            feat_x = selected_feats[0]
            feat_y = selected_feats[1]

            fig_feat = go.Figure()
            fig_feat.add_trace(go.Scatter(
                x=normal[feat_x], y=normal[feat_y],
                mode="markers", name="Normal",
                marker=dict(color="#22d3ee", size=4, opacity=0.4),
            ))
            fig_feat.add_trace(go.Scatter(
                x=anomaly[feat_x], y=anomaly[feat_y],
                mode="markers", name="⚠️ Flagged",
                marker=dict(color="#ef4444", size=7, symbol="x"),
            ))
            fig_feat.update_layout(
                **PLOTLY_DARK, height=360,
                title=dict(text=f"Flagged records: {feat_x} vs {feat_y}",
                           font=dict(color="#22d3ee")),
                xaxis_title=feat_x, yaxis_title=feat_y,
                margin=dict(t=50,b=20,l=40,r=20),
                legend=dict(bgcolor="#0f2744"),
            )
            st.plotly_chart(fig_feat, use_container_width=True, key="anom_feat_scatter")

        # ── Flagged records table ─────────────────────────────────────────────
        st.markdown('<div class="section-heading">📋 Flagged Records</div>',
                    unsafe_allow_html=True)
        flagged_df = df_result[df_result["anomaly_flag"]][
            selected_feats + ["anomaly_score"]
        ].sort_values("anomaly_score", ascending=False)

        st.dataframe(
            flagged_df.head(200).style.format(
                {c: "{:.4f}" for c in selected_feats + ["anomaly_score"]}
            ),
            use_container_width=True, hide_index=False,
        )

        # ── CSV export ────────────────────────────────────────────────────────
        csv_out = flagged_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label=f"⬇️ Export {n_flagged} flagged records to CSV",
            data=csv_out,
            file_name="anomalies_flagged.csv",
            mime="text/csv",
            key="anom_export",
        )

        st.markdown(f"""
<div class="info-box">
📌 <b>Isolation Forest settings used:</b><br>
Features: {', '.join(selected_feats)}<br>
Contamination: {contamination}% ({n_flagged} records flagged)<br>
Trees: {n_estimators}<br>
Preprocessing: StandardScaler (zero mean, unit variance)<br><br>
⚠️ Flagged records are <b>statistically unusual</b> within this dataset only.
They do not indicate confirmed pollution or safety risk.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT & LIMITATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-heading">📖 Methodology & Limitations</div>',
                unsafe_allow_html=True)

    st.markdown("""
### Isolation Forest — How it works
Isolation Forest is an unsupervised anomaly detection algorithm that isolates observations
by randomly partitioning the feature space. Points that require fewer partitions to isolate
are considered anomalous (they are "far" from the majority of the data in feature space).

**Algorithm**:
1. Build `n_estimators` random isolation trees on the training data
2. For each point, compute average path length across all trees
3. Points with short average path lengths receive high anomaly scores
4. Flag the top `contamination`% as anomalous

**Preprocessing**: Features are scaled to zero mean / unit variance before fitting,
so no single feature dominates due to scale differences.

### Why forecasting is deferred
The project dataset (Zenodo 19506986) is a static, cross-sectional dataset with:
- No timestamp column
- No time ordering
- No geographic coordinates
- No seasonal or cyclical structure

Fitting any time-series model (ARIMA, Prophet, LSTM) on this data would be scientifically
invalid. Forecasting will be enabled when a dataset with valid temporal structure is loaded
(e.g., from Copernicus Marine Service or NOAA CoastWatch).

### Quality criteria for trend analysis
For the trend analysis tab to produce valid results, the dataset must have:
- A parseable date/timestamp column
- At least 12 non-missing observations in the chosen variable
- Sufficient temporal coverage (daily/monthly observations)
- No major data gaps that would distort trend lines

### Anomaly interpretation
An Isolation Forest anomaly flag means:
> "This record is statistically unusual relative to the distribution of the selected features
> in this dataset."

It does NOT mean:
- The water is polluted
- The measurement is erroneous
- There is a safety risk

Always validate flagged records against:
- Instrument calibration records
- Field notes from the sampling event
- Reference values from independent monitoring programmes
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.78rem;'>"
    "MarineGuard AI · Trends & Anomalies · Anomaly ≠ Confirmed Pollution"
    "</div>",
    unsafe_allow_html=True,
)
