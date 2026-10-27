"""
pages/7_Marine_Monitoring_Map.py
MarineGuard AI — Phase 4: Interactive Marine Monitoring Map
Uses Plotly geo (no external folium dependency) for cloud compatibility.
Supports uploaded CSV with lat/lon, plus clearly-labelled demo locations.
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Marine Map — MarineGuard AI",
    page_icon="🗺️",
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
.card{background:linear-gradient(135deg,#0f2744,#0a2a4a);
      border:1px solid #1e4d7a;border-radius:12px;padding:20px 24px;margin-bottom:14px;}
hr{border-color:#1e3a5f !important;}
[data-testid="stPlotlyChart"]{border-radius:12px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(paper_bgcolor="#0d2137", font=dict(color="#e2f0ff"))

with st.sidebar:
    st.markdown("## 🌊 MarineGuard AI")
    st.markdown("---")
    st.markdown("**🗺️ Marine Monitoring Map** ← you are here")
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem;color:#64748b;line-height:1.7;'>
Map shows uploaded or demonstration data.<br>
<b>Never</b> presents synthetic markers as real<br>
monitoring stations.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:linear-gradient(135deg,#0f2744,#0a2a4a);
            border:1px solid #1e4d7a;border-radius:14px;padding:28px 36px;margin-bottom:20px;'>
  <div style='font-size:1.9rem;font-weight:800;
              background:linear-gradient(90deg,#22d3ee,#0d9488);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    🗺️ Marine Monitoring Map
  </div>
  <div style='color:#94a3b8;font-size:0.95rem;margin-top:6px;'>
    Upload georeferenced data · Filter by variable, date, region ·
    Interactive Plotly map (no live station data)
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data source selector ──────────────────────────────────────────────────────
data_source = st.radio(
    "**Data source**",
    ["📤 Upload CSV with lat/lon", "📖 Use demonstration sample data"],
    horizontal=True,
    key="map_data_source",
)

df_map = None
data_label = ""

# ── Option A: Upload ──────────────────────────────────────────────────────────
if data_source.startswith("📤"):
    up = st.file_uploader(
        "Upload CSV with latitude and longitude columns",
        type=["csv"],
        key="map_upload",
    )
    if up:
        try:
            df_raw = pd.read_csv(up)

            # Find lat/lon columns (case-insensitive)
            lat_col = next((c for c in df_raw.columns
                            if c.lower() in ("latitude","lat")), None)
            lon_col = next((c for c in df_raw.columns
                            if c.lower() in ("longitude","lon","long")), None)

            if not lat_col or not lon_col:
                st.error("❌ No latitude/longitude columns found. "
                         "Columns must be named 'latitude'/'longitude' (or 'lat'/'lon').")
            else:
                df_raw = df_raw.dropna(subset=[lat_col, lon_col])
                # Validate coordinate ranges
                invalid = (
                    (df_raw[lat_col].abs() > 90) |
                    (df_raw[lon_col].abs() > 180)
                )
                if invalid.any():
                    st.warning(f"⚠️ {invalid.sum()} rows have invalid coordinates and were removed.")
                    df_raw = df_raw[~invalid]

                df_map    = df_raw.copy()
                df_map["_lat"] = pd.to_numeric(df_map[lat_col], errors="coerce")
                df_map["_lon"] = pd.to_numeric(df_map[lon_col], errors="coerce")
                df_map = df_map.dropna(subset=["_lat","_lon"])
                data_label = f"📤 {up.name}"
                st.success(f"✅ Loaded {len(df_map):,} valid georeferenced records.")
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

# ── Option B: Demo data ────────────────────────────────────────────────────────
else:
    st.markdown("""
<div class="warn-box">
⚠️ <b>DEMONSTRATION DATA — NOT REAL MONITORING STATIONS</b><br>
The locations and measurements below are synthetic, generated for UI demonstration only.
They are NOT from real field surveys or satellite observations.
</div>
""", unsafe_allow_html=True)

    np.random.seed(42)
    regions_demo = {
        "Arabian Sea — Mumbai":       (18.93, 72.83),
        "Arabian Sea — Goa":          (15.49, 73.83),
        "Arabian Sea — Kochi":        (9.93,  76.26),
        "Arabian Sea — Mangalore":    (12.87, 74.84),
        "Bay of Bengal — Chennai":    (13.08, 80.27),
        "Bay of Bengal — Vizag":      (17.68, 83.22),
        "Bay of Bengal — Puri":       (19.80, 85.83),
        "Andaman — Port Blair":       (11.66, 92.73),
        "Lakshadweep — Kavaratti":    (10.57, 72.64),
        "Gulf of Mannar — Tuticorin": (8.80,  78.13),
    }

    rows = []
    for loc, (lat, lon) in regions_demo.items():
        for _ in range(20):
            temp = round(np.random.uniform(26, 31), 2)
            ph   = round(np.random.uniform(7.8, 8.3), 2)
            sal  = round(np.random.uniform(33, 37), 2)
            turb = round(np.random.uniform(0.5, 20), 2)
            do_  = round(np.random.uniform(4.5, 10.5), 2)
            no3  = round(np.random.uniform(0.5, 25.0), 2)
            chla = round(np.random.uniform(0.1, 80.0), 2)
            rows.append({
                "location":    loc,
                "_lat":        round(lat + np.random.uniform(-0.3, 0.3), 4),
                "_lon":        round(lon + np.random.uniform(-0.3, 0.3), 4),
                "Temperature": temp,
                "pH":          ph,
                "Salinity":    sal,
                "Turbidity":   turb,
                "Dissolved_Oxygen": do_,
                "Nitrate":     no3,
                "Chlorophyll_a": chla,
                "data_source": "SYNTHETIC DEMO",
                "timestamp":   pd.Timestamp("2024-01-01") +
                               pd.Timedelta(days=int(np.random.randint(0, 365))),
            })
    df_map    = pd.DataFrame(rows)
    data_label = "⚠️ Synthetic demonstration data"

# ── Map rendering ─────────────────────────────────────────────────────────────
if df_map is not None and len(df_map) > 0:
    num_cols = [c for c in df_map.select_dtypes(include="number").columns
                if c not in ("_lat","_lon")]

    st.markdown("---")
    st.markdown('<div class="section-heading">🗺️ Interactive Map</div>',
                unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

    with ctrl1:
        color_var = st.selectbox(
            "Colour markers by",
            options=num_cols if num_cols else ["(none)"],
            key="map_color_var",
        )
    with ctrl2:
        map_style = st.selectbox(
            "Map style",
            ["open-street-map", "carto-darkmatter", "stamen-terrain"],
            index=1,
            key="map_style",
        )
    with ctrl3:
        max_pts = st.slider("Max markers", 50, 2000, min(500, len(df_map)),
                            key="map_max_pts")

    df_plot = df_map.dropna(subset=["_lat","_lon"])
    if len(df_plot) > max_pts:
        df_plot = df_plot.sample(max_pts, random_state=42)

    hover_cols = ["_lat","_lon"] + [c for c in num_cols if c in df_plot.columns][:5]
    hover_data = {c: True for c in hover_cols if c != "_lat" and c != "_lon"}
    hover_data["_lat"] = True
    hover_data["_lon"] = True

    if color_var != "(none)" and color_var in df_plot.columns:
        fig_map = px.scatter_mapbox(
            df_plot, lat="_lat", lon="_lon",
            color=color_var,
            color_continuous_scale="Viridis",
            hover_data=hover_data,
            size_max=10,
            zoom=4,
            center={"lat": 12.0, "lon": 80.0},
        )
    else:
        fig_map = px.scatter_mapbox(
            df_plot, lat="_lat", lon="_lon",
            hover_data=hover_data,
            zoom=4,
            center={"lat": 12.0, "lon": 80.0},
        )

    fig_map.update_layout(
        mapbox_style=map_style,
        paper_bgcolor="#0d2137",
        font=dict(color="#e2f0ff"),
        height=560,
        margin=dict(t=10, b=10, l=0, r=0),
        coloraxis_colorbar=dict(
            title=color_var,
            tickfont=dict(color="#e2f0ff"),
            title_font=dict(color="#94a3b8"),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, key="main_map")

    st.caption(
        f"Showing {len(df_plot):,} of {len(df_map):,} records · "
        f"Data: **{data_label}**"
    )

    # ── Summary stats per variable ────────────────────────────────────────────
    if num_cols:
        st.markdown('<div class="section-heading">📊 Variable Summary</div>',
                    unsafe_allow_html=True)
        summary = df_map[num_cols].describe().T[["mean","std","min","max"]]
        summary.columns = ["Mean","Std Dev","Min","Max"]
        st.dataframe(summary.style.format("{:.3f}"),
                     use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    with st.expander("📋 View raw data table"):
        st.dataframe(df_map.head(200), use_container_width=True, hide_index=True)

        csv_bytes = df_map.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered data as CSV",
            data=csv_bytes,
            file_name="map_data_export.csv",
            mime="text/csv",
            key="map_download",
        )
else:
    st.markdown("""
<div style='background:#0f2744;border:1px dashed #1e4d7a;border-radius:12px;
            padding:60px;text-align:center;color:#64748b;margin-top:20px;'>
  <div style='font-size:4rem;margin-bottom:16px;'>🗺️</div>
  <div style='font-size:1.1rem;color:#94a3b8;'>
    Select a data source above to display the map.
  </div>
  <div style='font-size:0.85rem;margin-top:8px;'>
    Upload a CSV with latitude/longitude columns, or use the demonstration sample data.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
with st.expander("ℹ️ About this map and data integrity rules"):
    st.markdown("""
### Map data integrity rules followed in this application

1. **No invented station coordinates** — demo locations are clearly labelled as synthetic.
2. **No live pollution indicators** — marker colours reflect the selected uploaded variable only.
3. **No certified safety claims** — red/green markers are NOT used to imply safety assessments.
4. **Source always visible** — every map display shows the data source label.
5. **Satellite ≠ in-situ** — Copernicus/NOAA products are satellite estimates, not field measurements.

### Coordinate validation
- Latitude must be −90 to +90
- Longitude must be −180 to +180
- Invalid rows are removed before rendering
""")

st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.78rem;'>"
    "MarineGuard AI · Marine Monitoring Map · Upload georeferenced CSV for real data display."
    "</div>",
    unsafe_allow_html=True,
)
