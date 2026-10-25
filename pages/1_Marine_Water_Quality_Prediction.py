"""
pages/1_Marine_Water_Quality_Prediction.py
AI-Based Marine Water Quality Classification System — Prediction Page
"""

import os, sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prediction import (predict, validate_inputs, FEATURES,
                         FEATURE_RANGES, FEATURE_UNITS, get_pipeline_info)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction — Marine Water Quality AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared CSS (duplicated across pages so each page is self-contained) ───────
st.markdown("""
<style>
:root { --navy:#0a1628; --teal:#0d9488; --cyan:#22d3ee; --sky:#7dd3fc; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg,#0a1628 0%,#0d2137 55%,#0a1f35 100%);
    color:#e2f0ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#061022 0%,#0d1f3c 100%);
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
.result-safe {
    background:linear-gradient(135deg,#052e16,#064e3b);
    border:2px solid #10b981; border-radius:16px; padding:28px 36px;
    text-align:center;
}
.result-unsafe {
    background:linear-gradient(135deg,#2d0a0a,#450a0a);
    border:2px solid #ef4444; border-radius:16px; padding:28px 36px;
    text-align:center;
}
.result-label { font-size:2.4rem; font-weight:800; margin-bottom:6px; }
.warn-box {
    background:linear-gradient(90deg,#451a03,#3d1f00);
    border:1px solid #f59e0b; border-radius:10px;
    padding:14px 20px; color:#fde68a; font-size:0.88rem; line-height:1.6;
}
hr { border-color:#1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Marine Water AI")
    st.markdown("---")
    st.markdown("**🔬 Prediction** ← you are here")
    st.markdown("---")
    try:
        info = get_pipeline_info()
        st.markdown(f"""
<div style='font-size:0.78rem; color:#64748b; line-height:1.7;'>
<b>Model</b>: {info['model_type']}<br>
<b>Trees</b>: {info['n_estimators']}<br>
<b>Features</b>: {len(info['features'])}<br>
<b>Pipeline</b>: {' → '.join(info['steps'])}
</div>
""", unsafe_allow_html=True)
    except Exception:
        st.warning("Model info unavailable.")

# ── page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a; border-radius:14px; padding:28px 36px; margin-bottom:20px;'>
  <div style='font-size:1.9rem; font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#7dd3fc);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🔬 Marine Water Quality Prediction
  </div>
  <div style='color:#94a3b8; font-size:0.95rem; margin-top:6px;'>
    Enter water measurements to classify a sample as Safe or Unsafe using the
    Scenario A Random Forest model. Features are internally scaled by the
    pipeline — enter raw values only.
  </div>
</div>
""", unsafe_allow_html=True)

# ── experimental model warning ────────────────────────────────────────────────
st.markdown("""
<div class="warn-box">
⚠️ <b>Experimental Model — Important Limitations</b><br>
• This model is trained on <b>synthetic (mathematically derived) data</b> and has
  not been validated on real marine water samples.<br>
• Scenario A accuracy is <b>~51%</b> (near random chance) because Temperature,
  pH, Salinity and Turbidity have near-zero correlation with the synthetic target.<br>
• This tool is for <b>educational and research purposes only</b>. It is
  <b>NOT</b> a validated marine environmental safety assessment instrument.<br>
• Predicted probabilities are model confidence values — not real-world safety probabilities.
</div>
""", unsafe_allow_html=True)

# ── input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📥 Enter Water Sample Measurements</div>',
            unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider(
            "🌡️ Temperature (°C)",
            min_value=10.0, max_value=32.0, value=21.0, step=0.1,
            help="Observed dataset range: 10.0 – 32.0 °C"
        )
        ph = st.slider(
            "⚗️ pH",
            min_value=6.5, max_value=8.8, value=7.7, step=0.01,
            help="Observed dataset range: 6.5 – 8.8 (dimensionless)"
        )

    with col2:
        salinity = st.slider(
            "🧂 Salinity (PSU)",
            min_value=30.0, max_value=40.0, value=35.0, step=0.1,
            help="Observed dataset range: 30.0 – 40.0 PSU"
        )
        turbidity = st.slider(
            "💧 Turbidity (NTU)",
            min_value=0.5, max_value=50.0, value=25.0, step=0.5,
            help="Observed dataset range: 0.5 – 50.0 NTU"
        )

    st.markdown("---")
    c_left, c_right = st.columns([1, 3])
    with c_left:
        submitted = st.form_submit_button(
            "🔍 Predict Water Quality",
            use_container_width=True,
            type="primary"
        )
    with c_right:
        st.markdown(
            "<div style='padding-top:10px; color:#64748b; font-size:0.82rem;'>"
            "Values are validated against observed dataset ranges before prediction."
            "</div>",
            unsafe_allow_html=True
        )

# ── prediction result ─────────────────────────────────────────────────────────
if submitted:
    result = predict(temp, ph, salinity, turbidity)

    if not result["valid"]:
        st.error(f"❌ Input validation failed: {result['error']}")
    else:
        pred  = result["prediction"]
        label = result["label"]
        p_safe   = result["prob_safe"]
        p_unsafe = result["prob_unsafe"]

        st.markdown('<div class="section-heading">📊 Prediction Result</div>',
                    unsafe_allow_html=True)

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
<div class="result-safe">
  <div class="result-label" style="color:#10b981;">✅ SAFE</div>
  <div style="color:#6ee7b7; font-size:1rem; margin-top:4px;">
    Model predicts this sample is likely safe.
  </div>
  <div style="color:#94a3b8; font-size:0.82rem; margin-top:12px;">
    Confidence (P Safe): <b style="color:#10b981;">{p_safe*100:.1f}%</b>
    &nbsp;|&nbsp;
    Confidence (P Unsafe): <b>{p_unsafe*100:.1f}%</b>
  </div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-unsafe">
  <div class="result-label" style="color:#ef4444;">🚫 UNSAFE</div>
  <div style="color:#fca5a5; font-size:1rem; margin-top:4px;">
    Model predicts this sample may be unsafe.
  </div>
  <div style="color:#94a3b8; font-size:0.82rem; margin-top:12px;">
    Confidence (P Unsafe): <b style="color:#ef4444;">{p_unsafe*100:.1f}%</b>
    &nbsp;|&nbsp;
    Confidence (P Safe): <b>{p_safe*100:.1f}%</b>
  </div>
</div>
""", unsafe_allow_html=True)

        with gauge_col:
            # Gauge chart for Safe probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(p_safe * 100, 1),
                title={"text": "Safe Probability (%)", "font": {"color": "#94a3b8", "size": 14}},
                number={"font": {"color": "#22d3ee", "size": 40}},
                gauge={
                    "axis":      {"range": [0, 100], "tickcolor": "#475569"},
                    "bar":       {"color": "#10b981" if pred == 1 else "#ef4444"},
                    "bgcolor":   "#0d2137",
                    "bordercolor": "#1e4d7a",
                    "steps": [
                        {"range": [0,  40], "color": "#2d0a0a"},
                        {"range": [40, 60], "color": "#1c1f2e"},
                        {"range": [60, 100], "color": "#052e16"},
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="#0d2137",
                font={"color": "#e2f0ff"},
                height=260,
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── input summary ──────────────────────────────────────────────────────
        st.markdown('<div class="section-heading">📋 Input Summary</div>',
                    unsafe_allow_html=True)
        inp_df = pd.DataFrame([{
            "Feature": "Temperature", "Value": f"{temp} °C",
            "Range":   "10.0 – 32.0 °C"
        }, {
            "Feature": "pH", "Value": str(round(ph, 2)),
            "Range":   "6.5 – 8.8"
        }, {
            "Feature": "Salinity", "Value": f"{salinity} PSU",
            "Range":   "30.0 – 40.0 PSU"
        }, {
            "Feature": "Turbidity", "Value": f"{turbidity} NTU",
            "Range":   "0.5 – 50.0 NTU"
        }])
        st.dataframe(inp_df, use_container_width=True, hide_index=True)

        # ── disclaimer ─────────────────────────────────────────────────────────
        st.markdown("""
<div class="warn-box" style="margin-top:16px;">
📌 <b>Result Disclaimer</b><br>
This prediction is generated by a Random Forest model (Scenario A) trained on synthetic data.
Overall test accuracy is 51.35%. The output should be treated as an experimental
machine-learning classification, <b>not</b> as authoritative environmental guidance.
</div>
""", unsafe_allow_html=True)

else:
    # Show default state with example values
    st.markdown("""
<div style="background:#0f2744; border:1px solid #1e4d7a; border-radius:12px;
            padding:24px; text-align:center; color:#64748b; margin-top:16px;">
  <div style="font-size:2.5rem; margin-bottom:8px;">🔬</div>
  <div style="font-size:1rem; color:#94a3b8;">
    Adjust the sliders above and click <b style="color:#22d3ee;">Predict Water Quality</b>
    to classify a sample.
  </div>
</div>
""", unsafe_allow_html=True)

# ── feature reference ─────────────────────────────────────────────────────────
with st.expander("ℹ️ About the input features and model"):
    st.markdown("""
### Feature Descriptions

| Feature | Description | Marine Significance |
|---------|-------------|---------------------|
| **Temperature** (°C) | Water surface/sample temperature | Affects oxygen solubility and biological activity |
| **pH** | Acidity/alkalinity | Ocean acidification indicator; affects marine organisms |
| **Salinity** (PSU) | Dissolved salt concentration | Determines habitat suitability for species |
| **Turbidity** (NTU) | Water clarity / particle suspension | Affects light penetration and photosynthesis |

### Why are Dissolved Oxygen, Nitrate, and Chlorophyll-a excluded?
The dataset target (`is_safe`) is a **deterministic threshold rule** on those three variables.
Using them as features would allow the model to reverse-engineer the rule, achieving ~97%
accuracy — but that is not genuine predictive modelling. Scenario A deliberately excludes
them to test independent predictive power.

### Model pipeline
```
Raw input → SimpleImputer (median) → StandardScaler → RandomForestClassifier (200 trees)
```
Both Imputer and Scaler are fitted on 8,000 training samples only.
    """)
