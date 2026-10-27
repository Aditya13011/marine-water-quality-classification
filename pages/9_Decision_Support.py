"""
pages/9_Decision_Support.py
MarineGuard AI — Phase 6: Decision Support, Prediction History & PDF/CSV Reports
"""

import os, sys, io, json, csv
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prediction import predict, FEATURES, FEATURE_RANGES, FEATURE_UNITS, get_pipeline_info

st.set_page_config(
    page_title="Decision Support — MarineGuard AI",
    page_icon="🎯",
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
.result-safe{background:linear-gradient(135deg,#052e16,#064e3b);
     border:2px solid #10b981;border-radius:16px;padding:24px 30px;text-align:center;}
.result-unsafe{background:linear-gradient(135deg,#2d0a0a,#450a0a);
     border:2px solid #ef4444;border-radius:16px;padding:24px 30px;text-align:center;}
.warn-box{background:linear-gradient(90deg,#451a03,#3d1f00);
          border:1px solid #f59e0b;border-radius:10px;
          padding:14px 20px;color:#fde68a;font-size:0.88rem;line-height:1.6;}
.threshold-card{background:linear-gradient(135deg,#0a2a1a,#0a1628);
                border:1px solid #064e3b;border-radius:10px;
                padding:14px 20px;margin-bottom:10px;}
.pass{color:#10b981;font-weight:700;}
.fail{color:#ef4444;font-weight:700;}
hr{border-color:#1e3a5f !important;}
[data-testid="stPlotlyChart"]{border-radius:12px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(paper_bgcolor="#0d2137", plot_bgcolor="#0a1628", font=dict(color="#e2f0ff"))

# ── WHO/scientific threshold references ──────────────────────────────────────
# Source: WHO Guidelines for Drinking-water Quality (4th ed.) and
#         India BIS 10500:2012 for coastal/marine context reference values.
# These are indicative reference values, not marine-specific standards.
THRESHOLDS = {
    "Temperature": {
        "min": None, "max": 30.0, "unit": "°C",
        "source": "UNEP marine thermal tolerance reference (indicative)",
        "note": "Marine organisms stressed above 30°C (region-dependent)",
    },
    "pH": {
        "min": 7.5, "max": 8.4, "unit": "(dimensionless)",
        "source": "USEPA Aquatic Life Criteria; UNEP open ocean pH range",
        "note": "Open ocean pH typically 8.0–8.3; values outside 7.5–8.4 may stress marine biota",
    },
    "Salinity": {
        "min": 30.0, "max": 40.0, "unit": "PSU",
        "source": "Typical open ocean range (UNESCO, 1981)",
        "note": "Values outside 30–40 PSU unusual for open ocean; coastal values may differ",
    },
    "Turbidity": {
        "min": None, "max": 25.0, "unit": "NTU",
        "source": "WHO Guidelines for Drinking-water Quality (drinking water proxy; marine not specified)",
        "note": "No universal marine turbidity standard; >25 NTU may reduce light penetration",
    },
}

# ── Session state for prediction history ─────────────────────────────────────
if "pred_history" not in st.session_state:
    st.session_state["pred_history"] = []

with st.sidebar:
    st.markdown("## 🌊 MarineGuard AI")
    st.markdown("---")
    st.markdown("**🎯 Decision Support** ← you are here")
    st.markdown("---")
    try:
        info = get_pipeline_info()
        st.markdown(f"""
<div style='font-size:0.78rem;color:#64748b;line-height:1.7;'>
<b>Model</b>: {info['model_type']}<br>
<b>Features</b>: {len(info['features'])}<br>
<b>Pipeline</b>: {' → '.join(info['steps'])}
</div>
""", unsafe_allow_html=True)
    except Exception:
        st.warning("Model info unavailable.")

st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a;border-radius:14px;padding:28px 36px;margin-bottom:20px;'>
  <div style='font-size:1.9rem;font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#10b981);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    🎯 Decision Support System
  </div>
  <div style='color:#94a3b8;font-size:0.95rem;margin-top:6px;'>
    Prediction with explanation · Reference threshold check · Prediction history · Export
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
⚠️ <b>Important: This is NOT a certified water safety instrument</b><br>
Predictions are generated by an ML model (Scenario A, ~51% accuracy) trained on synthetic data.
Threshold checks reference indicative scientific values, not legally binding standards.
This tool is for educational and research purposes only.
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔬 Predict & Explain", "📋 History & Export", "📄 Generate Report"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT & EXPLAIN
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-heading">📥 Enter Measurement Values</div>',
                unsafe_allow_html=True)

    with st.form("ds_predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            temp     = st.slider("🌡️ Temperature (°C)",    10.0, 32.0, 21.0, 0.1, key="ds_temp")
            ph       = st.slider("⚗️ pH",                   6.5,  8.8,  7.7, 0.01, key="ds_ph")
        with c2:
            salinity = st.slider("🧂 Salinity (PSU)",       30.0, 40.0, 35.0, 0.1, key="ds_sal")
            turbidity = st.slider("💧 Turbidity (NTU)",     0.5,  50.0, 10.0, 0.5, key="ds_turb")

        opt_a, opt_b = st.columns([3, 1])
        with opt_a:
            user_location = st.text_input(
                "📍 Location label (optional)",
                placeholder="e.g. Arabian Sea — Mumbai coast",
                key="ds_location",
            )
        with opt_b:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔍 Predict & Explain",
                                              use_container_width=True, type="primary")

    if submitted:
        result = predict(temp, ph, salinity, turbidity)

        if not result["valid"]:
            st.error(f"❌ Validation failed: {result['error']}")
        else:
            pred     = result["prediction"]
            p_safe   = result["prob_safe"]
            p_unsafe = result["prob_unsafe"]
            label    = result["label"]

            # ── Result display ────────────────────────────────────────────────
            st.markdown('<div class="section-heading">📊 Prediction Result</div>',
                        unsafe_allow_html=True)

            res_col, gauge_col = st.columns([1, 1])
            with res_col:
                if pred == 1:
                    st.markdown(f"""
<div class="result-safe">
  <div style="font-size:2.4rem;font-weight:800;color:#10b981;">✅ SAFE</div>
  <div style="color:#6ee7b7;font-size:0.95rem;margin-top:6px;">Model predicts this sample is likely safe</div>
  <div style="color:#94a3b8;font-size:0.82rem;margin-top:12px;">
    P(Safe)={p_safe*100:.1f}% &nbsp;|&nbsp; P(Unsafe)={p_unsafe*100:.1f}%
  </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="result-unsafe">
  <div style="font-size:2.4rem;font-weight:800;color:#ef4444;">🚫 UNSAFE</div>
  <div style="color:#fca5a5;font-size:0.95rem;margin-top:6px;">Model predicts this sample may be unsafe</div>
  <div style="color:#94a3b8;font-size:0.82rem;margin-top:12px;">
    P(Unsafe)={p_unsafe*100:.1f}% &nbsp;|&nbsp; P(Safe)={p_safe*100:.1f}%
  </div>
</div>""", unsafe_allow_html=True)

            with gauge_col:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(p_safe * 100, 1),
                    title={"text": "P(Safe) %", "font": {"color": "#94a3b8", "size": 13}},
                    number={"font": {"color": "#22d3ee", "size": 36}},
                    gauge={
                        "axis": {"range": [0,100], "tickcolor": "#475569"},
                        "bar": {"color": "#10b981" if pred==1 else "#ef4444"},
                        "bgcolor": "#0d2137", "bordercolor": "#1e4d7a",
                        "steps": [
                            {"range":[0,40],  "color":"#2d0a0a"},
                            {"range":[40,60], "color":"#1c1f2e"},
                            {"range":[60,100],"color":"#052e16"},
                        ],
                        "threshold": {"line":{"color":"#f59e0b","width":3},
                                      "thickness":0.75,"value":50},
                    },
                ))
                fig_g.update_layout(paper_bgcolor="#0d2137", height=240,
                                    margin=dict(t=40,b=10,l=20,r=20),
                                    font=dict(color="#e2f0ff"))
                st.plotly_chart(fig_g, use_container_width=True, key="ds_gauge")

            # ── Reference threshold check ─────────────────────────────────────
            st.markdown('<div class="section-heading">📏 Reference Threshold Check</div>',
                        unsafe_allow_html=True)
            st.caption("Indicative scientific reference values — NOT legally binding marine standards.")

            inputs = {"Temperature": temp, "pH": ph, "Salinity": salinity, "Turbidity": turbidity}
            for feat, val in inputs.items():
                thresh = THRESHOLDS[feat]
                lo, hi = thresh["min"], thresh["max"]
                within = True
                flags = []
                if lo is not None and val < lo:
                    within = False; flags.append(f"Below {lo} {thresh['unit']}")
                if hi is not None and val > hi:
                    within = False; flags.append(f"Above {hi} {thresh['unit']}")

                status_html = '<span class="pass">✅ Within range</span>' if within \
                              else f'<span class="fail">⚠️ {"; ".join(flags)}</span>'
                st.markdown(f"""
<div class="threshold-card">
  <b style="color:#22d3ee;">{feat}</b>
  &nbsp;·&nbsp; <span style="color:#e2f0ff;">Value: <b>{val} {thresh['unit']}</b></span>
  &nbsp;·&nbsp; {status_html}<br>
  <span style="color:#64748b;font-size:0.8rem;">
    Range: {lo if lo else '—'}–{hi if hi else '—'} {thresh['unit']}
    · Source: {thresh['source']}
  </span><br>
  <span style="color:#475569;font-size:0.78rem;font-style:italic;">{thresh['note']}</span>
</div>
""", unsafe_allow_html=True)

            # ── Why this result ───────────────────────────────────────────────
            st.markdown('<div class="section-heading">🤔 Why This Result?</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
<div style="background:#0f2744;border:1px solid #1e4d7a;border-radius:10px;padding:18px 24px;">
<b style="color:#22d3ee;">Factors considered by the model:</b><br><br>
• The model is a <b>Random Forest / XGBoost</b> trained on 8,000 synthetic samples
  using Temperature, pH, Salinity, and Turbidity.<br>
• It learned non-linear patterns but the four features have <b>near-zero correlation</b>
  with the target — so the output probability ({p_safe*100:.1f}% safe) reflects weak signal.<br>
• Probabilities near 50% mean the model has <b>low confidence</b> and the result should
  not be relied upon for any operational decision.<br><br>
<b style="color:#f59e0b;">⚠️ The prediction should be interpreted alongside:</b><br>
• Dissolved Oxygen, Nitrate, Chlorophyll-a measurements (not available here)<br>
• Regulatory thresholds specific to the monitoring location and use case<br>
• Results from accredited laboratory analysis
</div>
""", unsafe_allow_html=True)

            # ── Save to history ───────────────────────────────────────────────
            timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["pred_history"].append({
                "Timestamp":      timestamp_now,
                "Location":       user_location or "—",
                "Temperature":    temp,
                "pH":             ph,
                "Salinity":       salinity,
                "Turbidity":      turbidity,
                "Prediction":     label,
                "P(Safe) %":      round(p_safe * 100, 2),
                "P(Unsafe) %":    round(p_unsafe * 100, 2),
                "Model":          "Scenario A RF/XGBoost (synthetic training data)",
            })
            st.success("✅ Saved to prediction history (see History tab).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-heading">📋 Prediction History</div>',
                unsafe_allow_html=True)
    st.caption("⚠️ History is stored in session memory only — it is cleared on page refresh or app restart.")

    hist = st.session_state.get("pred_history", [])
    if not hist:
        st.info("No predictions made yet. Use the Predict & Explain tab.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Export {len(hist_df)} predictions as CSV",
            data=csv_hist,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="hist_download",
        )

        if st.button("🗑️ Clear history", key="clear_history"):
            st.session_state["pred_history"] = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-heading">📄 Generate Analysis Report</div>',
                unsafe_allow_html=True)

    st.info(
        "Reports are generated in CSV and text format. "
        "PDF generation requires `fpdf2` — install with `pip install fpdf2`."
    )

    include_history  = st.checkbox("Include prediction history", value=True, key="rpt_hist")
    include_metrics  = st.checkbox("Include model metrics", value=True, key="rpt_metrics")
    include_dataset  = st.checkbox("Include dataset summary", value=True, key="rpt_dataset")

    if st.button("📊 Generate CSV Report", key="gen_csv_report", type="primary"):
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Header
        writer.writerow(["MarineGuard AI — Analysis Report"])
        writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow(["Dataset: Zenodo 19506986 (Synthetic)"])
        writer.writerow(["⚠️ This report is for educational/research purposes only."])
        writer.writerow([])

        # Dataset summary
        if include_dataset:
            writer.writerow(["=== DATASET SUMMARY ==="])
            writer.writerow(["Source", "Zenodo record 19506986"])
            writer.writerow(["Type", "Synthetic (mathematically derived)"])
            writer.writerow(["Records", "10,000"])
            writer.writerow(["Features", "7 numeric + 1 target"])
            writer.writerow(["Class balance", "50% Safe / 50% Unsafe"])
            writer.writerow(["Missing values", "0"])
            writer.writerow([])

        # Model metrics
        if include_metrics:
            metrics_path = os.path.join(ROOT, "reports", "model_metrics_enhanced.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path, encoding="utf-8") as f:
                    mm = json.load(f)
                writer.writerow(["=== SCENARIO A MODEL COMPARISON ==="])
                writer.writerow(["Model","Test Accuracy","Test F1","Test ROC-AUC",
                                 "CV F1 mean","CV F1 std"])
                for name, res in mm["scenario_A"]["results"].items():
                    cv_s = mm["scenario_A"]["cv_results"][name]
                    selected = " (SELECTED)" if name == mm["scenario_A"]["best_model"] else ""
                    writer.writerow([
                        name + selected,
                        f"{res['accuracy']*100:.2f}%",
                        f"{res['f1_score']:.4f}",
                        f"{res['roc_auc']:.4f}",
                        f"{cv_s['cv_f1_mean']:.4f}",
                        f"{cv_s['cv_f1_std']:.4f}",
                    ])
                writer.writerow([])
                writer.writerow(["⚠️ Scenario B (leakage reference)",
                                 f"{mm['scenario_B']['results']['RandomForest']['accuracy']*100:.2f}%",
                                 "NOT genuine performance — DO NOT CITE"])
                writer.writerow([])

        # Prediction history
        if include_history:
            hist = st.session_state.get("pred_history", [])
            writer.writerow(["=== PREDICTION HISTORY ==="])
            if hist:
                hist_df = pd.DataFrame(hist)
                writer.writerow(hist_df.columns.tolist())
                for row in hist_df.itertuples(index=False):
                    writer.writerow(list(row))
            else:
                writer.writerow(["No predictions recorded in this session."])
            writer.writerow([])

        # Limitations
        writer.writerow(["=== LIMITATIONS & DISCLAIMERS ==="])
        writer.writerow(["1. All training data is synthetic (Zenodo 19506986) — not real marine samples."])
        writer.writerow(["2. Scenario A accuracy ~51% — near random chance; features have low predictive power."])
        writer.writerow(["3. Predictions should not be used for operational marine safety decisions."])
        writer.writerow(["4. Reference thresholds are indicative only — not legally binding marine standards."])
        writer.writerow(["5. Model has not been validated on real field-collected water samples."])

        report_bytes = buf.getvalue().encode("utf-8")
        st.download_button(
            label="⬇️ Download Report CSV",
            data=report_bytes,
            file_name=f"marineguard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="report_download",
        )
        st.success("✅ Report ready — click the button above to download.")

    # Try fpdf2 if installed
    st.markdown("---")
    st.markdown("#### PDF Export")
    try:
        from fpdf import FPDF

        if st.button("📄 Generate PDF Report", key="gen_pdf", type="secondary"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 12, "MarineGuard AI — Analysis Report", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(0, 6, "Dataset: Zenodo 19506986 (Synthetic — NOT real data)", ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Disclaimer", ln=True)
            pdf.set_font("Helvetica", "", 9)
            disclaimer = (
                "This report is for educational and research purposes only. "
                "The model is trained on synthetic data with ~51% accuracy. "
                "It must not be used for operational marine water safety decisions."
            )
            pdf.multi_cell(0, 5, disclaimer)
            pdf.ln(4)

            metrics_path = os.path.join(ROOT, "reports", "model_metrics_enhanced.json")
            if include_metrics and os.path.isfile(metrics_path):
                with open(metrics_path, encoding="utf-8") as f:
                    mm = json.load(f)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Scenario A Model Comparison", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for name, res in mm["scenario_A"]["results"].items():
                    sel = " [SELECTED]" if name == mm["scenario_A"]["best_model"] else ""
                    pdf.cell(0, 5,
                             f"  {name}{sel}: Acc={res['accuracy']*100:.2f}%  "
                             f"F1={res['f1_score']:.4f}  AUC={res['roc_auc']:.4f}",
                             ln=True)
                pdf.ln(3)

            if include_history:
                hist = st.session_state.get("pred_history", [])
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, f"Prediction History ({len(hist)} records)", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for entry in hist[-20:]:   # max 20 rows in PDF
                    pdf.cell(0, 5,
                             f"  {entry['Timestamp']} | {entry['Location']} | "
                             f"{entry['Prediction']} | P(Safe)={entry['P(Safe) %']}%",
                             ln=True)

            pdf_bytes = pdf.output()
            st.download_button(
                label="⬇️ Download PDF Report",
                data=bytes(pdf_bytes),
                file_name=f"marineguard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="pdf_download",
            )
            st.success("✅ PDF ready.")
    except ImportError:
        st.info(
            "📦 `fpdf2` not installed. PDF export is unavailable in this environment.\n\n"
            "Install locally with: `pip install fpdf2`\n\n"
            "Use the **CSV report** above as an alternative."
        )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.78rem;'>"
    "MarineGuard AI · Decision Support · Educational & Research Use Only"
    "</div>",
    unsafe_allow_html=True,
)
