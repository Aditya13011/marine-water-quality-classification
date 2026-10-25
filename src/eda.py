"""
src/eda.py
──────────────────────────────────────────────────────────────────────────────
Phase 2 — Exploratory Data Analysis
Project : AI-Based Water Quality Prediction and Potability Analysis System
Dataset : Marine Water Quality Classification (Zenodo record 19506986)

⚠ DATA PROVENANCE
  This dataset is mathematically derived (synthetic).  Measurements are NOT
  real-world laboratory observations.  The target `is_safe` is a deterministic
  function of Dissolved_Oxygen, Nitrate, and Chlorophyll_a thresholds:
      Safe (1)  ⟺  DO > 8  AND  NO₃ < 15  AND  Chl_a < 60
  Models trained with those three columns as features will trivially learn
  this rule and produce misleadingly high accuracy.

PURPOSE
  Generate exploratory charts saved to reports/ to guide feature selection
  and model design.  No models are trained here.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "data", "marine_water_quality.csv")
RPT_DIR  = os.path.join(ROOT, "reports")
os.makedirs(RPT_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
PAL      = {0: "#e05c5c", 1: "#4c9be8"}          # Unsafe=red, Safe=blue
PAL_LIST = [PAL[0], PAL[1]]
SAFE_LBL = {0: "Unsafe (0)", 1: "Safe (1)"}

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

# ── load ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

if not os.path.isfile(CSV_PATH):
    sys.exit(f"ERROR: dataset not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"\n✔ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# columns
TARGET        = "is_safe"
ID_COL        = "Sample_ID"
THRESHOLD_COLS = ["Dissolved_Oxygen", "Nitrate", "Chlorophyll_a"]       # ⚠ leakage
SAFE_FEATURES  = ["Temperature", "pH", "Salinity", "Turbidity"]          # primary
ALL_NUMERIC    = [c for c in df.columns if c not in [TARGET, ID_COL]
                  and df[c].dtype in ["float64", "int64"]]

print(f"\n  Target column        : {TARGET}")
print(f"  Threshold columns   : {THRESHOLD_COLS}  (⚠ leakage risk)")
print(f"  Safe feature cols   : {SAFE_FEATURES}")
print(f"  All numeric columns : {ALL_NUMERIC}")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CHART 1 — Target Class Distribution
# └─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
vc = df[TARGET].value_counts().sort_index()
bars = ax.bar(
    [SAFE_LBL[k] for k in vc.index],
    vc.values,
    color=[PAL[k] for k in vc.index],
    edgecolor="white", linewidth=1.5, width=0.5
)
for bar, val in zip(bars, vc.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Number of Records", fontsize=12)
ax.set_ylim(0, max(vc.values) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(RPT_DIR, "01_class_distribution.png"), dpi=150)
plt.close(fig)
print("\n✔ Saved: 01_class_distribution.png")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CHART 2 — Feature Distributions (ALL numeric, split by class)
# └─────────────────────────────────────────────────────────────────────────────
n_cols = 4
n_rows = int(np.ceil(len(ALL_NUMERIC) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(ALL_NUMERIC):
    ax = axes[i]
    for cls in [0, 1]:
        subset = df[df[TARGET] == cls][col]
        ax.hist(subset, bins=45, alpha=0.60,
                color=PAL[cls], label=SAFE_LBL[cls], edgecolor="none")
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if col in THRESHOLD_COLS:
        ax.set_title(f"{col} ⚠", fontsize=11, fontweight="bold", color="#c0392b")

# hide unused subplots
for j in range(len(ALL_NUMERIC), len(axes)):
    axes[j].set_visible(False)

handles = [mpatches.Patch(color=PAL[0], alpha=0.7, label="Unsafe (0)"),
           mpatches.Patch(color=PAL[1], alpha=0.7, label="Safe (1)")]
fig.legend(handles=handles, loc="lower right",
           fontsize=11, frameon=True, framealpha=0.9)
fig.suptitle("Feature Distributions by Class\n(⚠ = threshold-defining column, leakage risk)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(RPT_DIR, "02_feature_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("✔ Saved: 02_feature_distributions.png")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CHART 3 — Correlation Heatmap (numeric columns)
# └─────────────────────────────────────────────────────────────────────────────
corr = df[ALL_NUMERIC + [TARGET]].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only
sns.heatmap(corr, mask=~np.triu(np.ones_like(corr, dtype=bool)),
            annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0,
            linewidths=0.5, linecolor="white",
            ax=ax, annot_kws={"size": 9})
ax.set_title("Pearson Correlation Heatmap (all numeric + target)",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
fig.savefig(os.path.join(RPT_DIR, "03_correlation_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("✔ Saved: 03_correlation_heatmap.png")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CHART 4 — Boxplots by Class (safe features only — no leakage columns)
# └─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(SAFE_FEATURES),
                          figsize=(5 * len(SAFE_FEATURES), 5))
for i, col in enumerate(SAFE_FEATURES):
    ax = axes[i]
    data_grp = [df[df[TARGET] == cls][col].values for cls in [0, 1]]
    bp = ax.boxplot(data_grp, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2.5))
    for patch, cls in zip(bp["boxes"], [0, 1]):
        patch.set_facecolor(PAL[cls])
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set(color="#555555", linewidth=1)
    ax.set_xticklabels(["Unsafe (0)", "Safe (1)"], fontsize=10)
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_ylabel(col, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Safe Feature Distributions by Class (Boxplots)\n"
             "[Temperature, pH, Salinity, Turbidity — no threshold leakage]",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(RPT_DIR, "04_safe_features_boxplots.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("✔ Saved: 04_safe_features_boxplots.png")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CHART 5 — Threshold columns: DO, Nitrate, Chl_a with threshold lines
# └─────────────────────────────────────────────────────────────────────────────
threshold_info = {
    "Dissolved_Oxygen":  (">", 8,   "mg/L", "DO > 8 (Safe threshold)"),
    "Nitrate":           ("<", 15,  "mg/L", "NO₃ < 15 (Safe threshold)"),
    "Chlorophyll_a":     ("<", 60,  "µg/L", "Chl_a < 60 (Safe threshold)"),
}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (col, (op, thresh, unit, lbl)) in enumerate(threshold_info.items()):
    ax = axes[i]
    for cls in [0, 1]:
        subset = df[df[TARGET] == cls][col]
        ax.hist(subset, bins=50, alpha=0.60,
                color=PAL[cls], label=SAFE_LBL[cls], edgecolor="none")
    ax.axvline(thresh, color="#e67e22", linewidth=2.2,
               linestyle="--", label=lbl)
    ax.set_title(f"{col}\n(⚠ threshold-defining feature)", fontsize=11,
                 fontweight="bold", color="#c0392b")
    ax.set_xlabel(f"{col} ({unit})", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Threshold-Defining Columns — Leakage Risk Analysis\n"
             "Target `is_safe` is a deterministic function of these three columns",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(RPT_DIR, "05_threshold_leakage_analysis.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("✔ Saved: 05_threshold_leakage_analysis.png")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ CHART 6 — Pairplot of safe features (Temperature, pH, Salinity, Turbidity)
# └─────────────────────────────────────────────────────────────────────────────
pair_df = df[SAFE_FEATURES + [TARGET]].copy()
pair_df["Class"] = pair_df[TARGET].map({0: "Unsafe", 1: "Safe"})

g = sns.pairplot(pair_df, hue="Class", vars=SAFE_FEATURES,
                 palette={"Unsafe": PAL[0], "Safe": PAL[1]},
                 plot_kws={"alpha": 0.25, "s": 8},
                 diag_kind="kde", corner=True)
g.figure.suptitle(
    "Pairplot — Safe Features Only (Temperature, pH, Salinity, Turbidity)\n"
    "No threshold-defining columns included",
    y=1.02, fontsize=13, fontweight="bold"
)
g.figure.savefig(os.path.join(RPT_DIR, "06_pairplot_safe_features.png"),
                 dpi=130, bbox_inches="tight")
plt.close("all")
print("✔ Saved: 06_pairplot_safe_features.png")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PRINTED SUMMARY
# └─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EDA SUMMARY")
print("=" * 70)

print(f"\nDataset shape    : {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Missing values   : {df.isnull().sum().sum()}")
print(f"Duplicate rows   : {df.duplicated().sum()}")
print(f"\nTarget distribution:")
for cls, cnt in df[TARGET].value_counts().sort_index().items():
    print(f"  {cls} ({SAFE_LBL[cls]:<14}) : {cnt:,}  ({cnt/len(df)*100:.1f}%)")

print("\nSummary statistics — SAFE features (no leakage):")
print(df[SAFE_FEATURES].describe().T.to_string())

print("\nSummary statistics — THRESHOLD features (⚠ leakage risk):")
print(df[THRESHOLD_COLS].describe().T.to_string())

print("\nCorrelation of each feature with `is_safe`:")
corr_target = df[ALL_NUMERIC].corrwith(df[TARGET]).sort_values(ascending=False)
for col, r in corr_target.items():
    tag = " ⚠ LEAKAGE" if col in THRESHOLD_COLS else ""
    print(f"  {col:<25} r = {r:+.4f}{tag}")

print(f"\n✔ All charts saved to  : {RPT_DIR}")
print("=" * 70)
print("EDA COMPLETE — Proceed to data_preprocessing.py")
print("=" * 70)
