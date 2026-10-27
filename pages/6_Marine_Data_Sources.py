"""
pages/6_Marine_Data_Sources.py
MarineGuard AI — Phase 3: Real Data Integration
Dataset catalogue, CSV upload/validation, data provenance,
and fallback sample demonstration.
"""

import os, sys, io
import streamlit as st
import pandas as pd
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Marine Data Sources — MarineGuard AI",
    page_icon="🌐",
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
.card{background:linear-gradient(135deg,#0f2744,#0a2a4a);
      border:1px solid #1e4d7a;border-radius:12px;padding:20px 24px;margin-bottom:14px;}
.source-card{background:linear-gradient(135deg,#0a2a1a,#0a1f2e);
             border:1px solid #064e3b;border-radius:12px;padding:20px 24px;margin-bottom:14px;}
.warn-box{background:linear-gradient(90deg,#451a03,#3d1f00);
          border:1px solid #f59e0b;border-radius:10px;
          padding:14px 20px;color:#fde68a;font-size:0.88rem;line-height:1.6;}
.badge{display:inline-block;border-radius:6px;padding:2px 10px;
       font-size:0.72rem;font-weight:700;letter-spacing:0.5px;margin-right:4px;}
.badge-free{background:#064e3b;color:#6ee7b7;}
.badge-auth{background:#451a03;color:#fbbf24;}
.badge-synth{background:#1e1b4b;color:#a5b4fc;}
hr{border-color:#1e3a5f !important;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌊 MarineGuard AI")
    st.markdown("---")
    st.markdown("**🌐 Marine Data Sources** ← you are here")

st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a;border-radius:14px;padding:28px 36px;margin-bottom:20px;'>
  <div style='font-size:1.9rem;font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#06d6a0);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    🌐 Marine Data Sources & Integration
  </div>
  <div style='color:#94a3b8;font-size:0.95rem;margin-top:6px;'>
    Dataset catalogue · CSV upload and validation · Data provenance ·
    How to obtain real marine water quality data
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📋 Dataset Catalogue", "📤 Upload Your Data", "🗺️ Sample Data Demo"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATASET CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-heading">🗂️ Documented Data Sources</div>',
                unsafe_allow_html=True)

    # Source 1 — Project dataset
    st.markdown("""
<div class="card">
<b style="color:#22d3ee;font-size:1.05rem;">📦 Project Dataset (Currently in Use)</b>
<br><br>
<table style='width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.88rem;'>
<tr><td style='padding:5px 10px;color:#94a3b8;width:180px;'><b>Name</b></td>
    <td>Marine Water Quality Classification Benchmark</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Provider</b></td>
    <td>Zenodo (archived from Kaggle)</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>URL</b></td>
    <td><a href="https://zenodo.org/records/19506986" style="color:#22d3ee;">zenodo.org/records/19506986</a></td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Type</b></td>
    <td><span class="badge badge-synth">SYNTHETIC</span> Mathematically derived via biogeochemical equations</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Variables</b></td>
    <td>Temperature (°C), pH, Salinity (PSU), Turbidity (NTU),
        Dissolved Oxygen (mg/L), Nitrate (mg/L), Chlorophyll-a (μg/L), is_safe</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Records</b></td><td>10,000</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Coverage</b></td><td>No geographic or temporal metadata</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>License</b></td><td>CC BY 4.0 (Zenodo)</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Limitation</b></td>
    <td>Not real field data; cannot generalise to actual marine environments</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    # Source 2 — Copernicus
    st.markdown("""
<div class="source-card">
<b style="color:#06d6a0;font-size:1.05rem;">🛰️ Copernicus Marine Service (CMEMS)</b>
<span class="badge badge-auth">REQUIRES FREE ACCOUNT</span>
<br><br>
<table style='width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.88rem;'>
<tr><td style='padding:5px 10px;color:#94a3b8;width:180px;'><b>Provider</b></td>
    <td>European Commission — Copernicus Programme</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>URL</b></td>
    <td><a href="https://data.marine.copernicus.eu" style="color:#06d6a0;">data.marine.copernicus.eu</a></td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Relevant Products</b></td>
    <td>
      GLOBAL_MULTIYEAR_PHY_001_030 — Sea surface temperature &amp; salinity<br>
      OCEANCOLOUR_GLO_BGC_L3_NRT_009_101 — Chlorophyll-a (satellite-derived)<br>
      GLOBAL_ANALYSISFORECAST_BGC_001_028 — Oxygen, nitrate (model output)
    </td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Variable types</b></td>
    <td>Mix of satellite-derived estimates and numerical model reanalysis</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Format</b></td><td>NetCDF</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Coverage</b></td>
    <td>Global; daily to monthly; ~4 km to ~25 km resolution (product-dependent)</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Indian Ocean</b></td>
    <td>All products cover Indian Ocean and Arabian Sea</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>License</b></td>
    <td>Free for scientific/educational use after registration (CMEMS User Licence)</td></tr>
</table>
<br>
<b style="color:#6ee7b7;">How to access:</b>
<ol style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">
  <li>Register at <a href="https://data.marine.copernicus.eu/register" style="color:#06d6a0;">data.marine.copernicus.eu/register</a> (free)</li>
  <li>Search for product ID (e.g., OCEANCOLOUR_GLO_BGC_L3_NRT_009_101)</li>
  <li>Use the Copernicus Marine Toolbox: <code>pip install copernicusmarine</code></li>
  <li>Download subset: <code>copernicusmarine get --dataset-id &lt;ID&gt; ...</code></li>
  <li>Convert NetCDF to CSV with xarray/pandas: <code>ds.to_dataframe().to_csv(...)</code></li>
</ol>
</div>
""", unsafe_allow_html=True)

    # Source 3 — NOAA CoastWatch
    st.markdown("""
<div class="source-card">
<b style="color:#06d6a0;font-size:1.05rem;">🌊 NOAA CoastWatch Ocean Color Products</b>
<span class="badge badge-free">FREE — NO ACCOUNT REQUIRED</span>
<br><br>
<table style='width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.88rem;'>
<tr><td style='padding:5px 10px;color:#94a3b8;width:180px;'><b>Provider</b></td>
    <td>NOAA CoastWatch / OceanWatch</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>URL</b></td>
    <td><a href="https://coastwatch.noaa.gov" style="color:#06d6a0;">coastwatch.noaa.gov</a></td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Relevant Products</b></td>
    <td>
      MODIS-Aqua OC (Chlorophyll-a, sea surface colour)<br>
      VIIRS OC (NPP / JPSS-1 Chlorophyll-a)<br>
      SST products (GOES, MODIS)<br>
      Harmful Algal Bloom (HAB) monitoring products
    </td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Variable type</b></td>
    <td>Satellite-derived estimates; cloud-free composites available</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Format</b></td><td>NetCDF, HDF, browser-based ERDDAP</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Coverage</b></td>
    <td>Global, coastal focus; Indian Ocean products available</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>License</b></td><td>Public domain (US Government)</td></tr>
</table>
<br>
<b style="color:#6ee7b7;">Programmatic access via ERDDAP:</b>
<div style="background:#0a1628;border-radius:8px;padding:12px;margin-top:6px;font-size:0.8rem;color:#a5b4fc;">
# Example: ERDDAP CSV download (no installation required)<br>
import pandas as pd<br>
url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "erdMH1chla8day.csv?chlorophyll[(2024-01-01)]"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "[(5.0):(25.0)][(65.0):(90.0)]")  # Arabian Sea box<br>
df = pd.read_csv(url, skiprows=[1])  # row 1 = units
</div>
</div>
""", unsafe_allow_html=True)

    # Source 4 — India CPCB
    st.markdown("""
<div class="card">
<b style="color:#22d3ee;font-size:1.05rem;">🇮🇳 India CPCB / NWMP Water Quality Data</b>
<span class="badge badge-free">FREE</span>
<br><br>
<table style='width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.88rem;'>
<tr><td style='padding:5px 10px;color:#94a3b8;width:180px;'><b>Provider</b></td>
    <td>Central Pollution Control Board (India)</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>URL</b></td>
    <td><a href="https://cpcb.nic.in/water-quality-data/" style="color:#22d3ee;">cpcb.nic.in/water-quality-data/</a></td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Variables</b></td>
    <td>pH, DO, BOD, COD, Nitrate, Fecal Coliform (monitoring stations)</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Coverage</b></td>
    <td>Rivers and coastal monitoring stations across India; annual/monthly</td></tr>
<tr><td style='padding:5px 10px;color:#94a3b8;'><b>Note</b></td>
    <td>Station data; not gridded; useful for coastal/estuarine analysis</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="warn-box">
⚠️ <b>Integration Status</b><br>
Live API integration for Copernicus and NOAA requires registered credentials and
network access to external APIs — unavailable in this classroom/cloud deployment.
The catalogue above documents the exact steps and product IDs needed to obtain real data.
Use the <b>Upload Your Data</b> tab to load downloaded datasets into this application.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CSV UPLOAD AND VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-heading">📤 Upload and Validate Your Dataset</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<b style="color:#22d3ee;">Accepted CSV Format</b><br><br>
Your CSV must contain at least <b>one numeric column</b>. Optional but recommended columns:

| Column | Unit | Notes |
|--------|------|-------|
| `Temperature` | °C | Sea-surface or sample temp |
| `pH` | — | 0–14 dimensionless |
| `Salinity` | PSU | Practical Salinity Units |
| `Turbidity` | NTU | Nephelometric Turbidity Units |
| `Dissolved_Oxygen` | mg/L | Optional |
| `Nitrate` | mg/L | Optional |
| `Chlorophyll_a` | μg/L | Optional |
| `latitude` | decimal degrees | For map view |
| `longitude` | decimal degrees | For map view |
| `timestamp` | ISO 8601 | For trend view |
</div>
""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="data_source_upload",
        help="Max 200 MB. Must be a plain comma-separated file with a header row.",
    )

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ Loaded: **{uploaded.name}** — {len(df_up):,} rows × {len(df_up.columns)} columns")

            # ── Basic validation ──────────────────────────────────────────────
            issues = []
            notes  = []

            num_cols = df_up.select_dtypes(include="number").columns.tolist()
            if len(num_cols) == 0:
                issues.append("No numeric columns found — nothing to analyse.")

            has_lat  = any(c.lower() in ("latitude","lat")        for c in df_up.columns)
            has_lon  = any(c.lower() in ("longitude","lon","long") for c in df_up.columns)
            has_time = any(c.lower() in ("timestamp","date","time","datetime") for c in df_up.columns)

            known = ["Temperature","pH","Salinity","Turbidity",
                     "Dissolved_Oxygen","Nitrate","Chlorophyll_a"]
            found_known = [c for c in known if c in df_up.columns]
            missing_known = [c for c in known if c not in df_up.columns]
            if found_known:
                notes.append(f"Recognised columns: {', '.join(found_known)}")
            if missing_known:
                notes.append(f"Missing optional columns: {', '.join(missing_known)}")
            if not has_lat or not has_lon:
                notes.append("No lat/lon detected — map view unavailable for this dataset.")
            if not has_time:
                notes.append("No timestamp column detected — trend view unavailable.")
            if df_up.isnull().any().any():
                n_missing = int(df_up.isnull().sum().sum())
                notes.append(f"{n_missing} missing values detected — will be imputed or dropped per page.")

            if issues:
                for iss in issues:
                    st.error(f"❌ {iss}")
            else:
                for note in notes:
                    st.info(f"ℹ️ {note}")

                # Save to session state
                st.session_state["uploaded_df"]   = df_up
                st.session_state["uploaded_name"] = uploaded.name

                # Preview
                st.markdown('<div class="section-heading">Preview (first 10 rows)</div>',
                            unsafe_allow_html=True)
                st.dataframe(df_up.head(10), use_container_width=True)

                # Summary stats
                st.markdown('<div class="section-heading">Summary Statistics</div>',
                            unsafe_allow_html=True)
                st.dataframe(df_up[num_cols].describe().T.style.format("{:.4f}"),
                             use_container_width=True)

                # Missing values heatmap
                if df_up.isnull().any().any():
                    st.markdown('<div class="section-heading">Missing Values</div>',
                                unsafe_allow_html=True)
                    miss_pct = (df_up.isnull().mean() * 100).reset_index()
                    miss_pct.columns = ["Column", "Missing %"]
                    miss_pct = miss_pct[miss_pct["Missing %"] > 0].sort_values("Missing %", ascending=False)
                    st.dataframe(miss_pct, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
    else:
        st.markdown("""
<div style='background:#0f2744;border:1px dashed #1e4d7a;border-radius:12px;
            padding:40px;text-align:center;color:#64748b;'>
  <div style='font-size:3rem;margin-bottom:12px;'>📂</div>
  <div style='font-size:1rem;color:#94a3b8;'>Drag and drop a CSV file above, or click to browse.</div>
  <div style='font-size:0.82rem;margin-top:8px;'>
    Supported: Copernicus CSV exports, NOAA ERDDAP downloads, CPCB water quality reports,
    or any CSV with numeric marine water quality measurements.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SAMPLE DATA DEMO
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("""
<div class="warn-box">
⚠️ <b>This tab shows a clearly labeled SYNTHETIC demonstration dataset.</b><br>
The sample measurements below are NOT from real monitoring stations.
They were generated using realistic ranges derived from published Indian Ocean
water quality surveys. They are provided solely to demonstrate application functionality.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🗺️ Sample Indian Coastline Demonstration Data</div>',
                unsafe_allow_html=True)

    np.random.seed(42)
    n_demo = 200

    lats_regions = {
        "Arabian Sea — Mumbai Coast":    (18.8, 72.8),
        "Arabian Sea — Goa Coast":       (15.4, 73.7),
        "Arabian Sea — Kochi":           (9.9,  76.2),
        "Bay of Bengal — Chennai Coast": (13.0, 80.3),
        "Bay of Bengal — Vizag Coast":   (17.7, 83.3),
        "Bay of Bengal — Kolkata":       (22.0, 88.0),
        "Indian Ocean — Andaman Is.":    (11.7, 92.7),
        "Indian Ocean — Lakshadweep":    (10.6, 72.6),
    }

    sample_rows = []
    for region, (base_lat, base_lon) in lats_regions.items():
        n_r = n_demo // len(lats_regions)
        for _ in range(n_r):
            sample_rows.append({
                "region":      region,
                "latitude":    round(base_lat  + np.random.uniform(-0.5, 0.5), 4),
                "longitude":   round(base_lon  + np.random.uniform(-0.5, 0.5), 4),
                "timestamp":   pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(np.random.randint(0, 365))),
                "Temperature": round(np.random.uniform(26.0, 32.0), 2),
                "pH":          round(np.random.uniform(7.8, 8.3), 2),
                "Salinity":    round(np.random.uniform(33.5, 36.5), 2),
                "Turbidity":   round(np.random.uniform(0.5, 15.0), 2),
                "data_source": "⚠️ SYNTHETIC DEMO",
            })

    demo_df = pd.DataFrame(sample_rows)
    demo_df["timestamp"] = pd.to_datetime(demo_df["timestamp"]).dt.date

    st.dataframe(demo_df.head(30), use_container_width=True, hide_index=True)

    # Download button
    csv_bytes = demo_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download full sample CSV (200 rows)",
        data=csv_bytes,
        file_name="marine_demo_sample_SYNTHETIC.csv",
        mime="text/csv",
        key="demo_download",
        help="This file contains synthetic demonstration data — NOT real observations.",
    )

    st.info(
        "💡 **Tip**: Download this CSV, then upload it in the **Upload Your Data** tab "
        "or the **Marine Monitoring Map** page to see all interactive features in action."
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.78rem;'>"
    "MarineGuard AI · Marine Data Sources · All real-data integration requires external credentials."
    "</div>",
    unsafe_allow_html=True,
)
