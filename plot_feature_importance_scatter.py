## -*- coding: utf-8 -*-
"""
Scatter/strip plot of RAW Random Forest feature importances across ALL models.
X-axis: individual features (HR, HR_baseline, flugerfahrung, ...)
Y-axis: importance values
Hue: number of features used in the model (1,2,3,4)

Reads: rf_feature_combos_importances*.csv (recursive)
Expected columns (robust):
- combo / feature_combo / features / selected_features  (model feature set)
- feature
- importance
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\49157\Desktop 2\Bachelorarbeit\Daten_real"
PATTERN = "rf_feature_combos_importances*.csv"

OUT_DIR = os.path.join(BASE_DIR, "feature_importance_stats_new")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "feature_importance_scatter_by_model_size.png")

# Set to False if you want to exclude flugerfahrung from the plot
INCLUDE_FLUGERFAHRUNG = True

# Optional: exclude any features containing these substrings
EXCLUDE_CONTAINS = []  # e.g. ["__missing_ratio"]

# =========================
# HELPERS
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def to_numeric_decimal_comma(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def count_features_in_combo(combo: str) -> int:
    """
    Count how many base features are in the model combo string.
    Supports delimiters: + , ; | whitespace
    Examples:
      "HR" -> 1
      "HR+EDA" -> 2
      "HR, EDA, Temp" -> 3
    """
    if combo is None or (isinstance(combo, float) and np.isnan(combo)):
        return np.nan
    s = str(combo).strip()
    if not s:
        return np.nan

    # normalize separators to "+"
    s = re.sub(r"[,\;\|]", "+", s)
    s = re.sub(r"\s+", "", s)

    parts = [p for p in s.split("+") if p]
    return len(parts)

# =========================
# LOAD RAW IMPORTANCES
# =========================
files = glob.glob(os.path.join(BASE_DIR, "**", PATTERN), recursive=True)
if not files:
    raise RuntimeError(f"No files found for pattern {PATTERN} under {BASE_DIR}")

COMBO_CANDS = ["combo", "feature_combo", "features", "features_used", "selected_features", "feature_set"]
FEAT_COL = "feature"
IMP_COL = "importance"

dfs = []
skipped = []

for f in files:
    try:
        df = pd.read_csv(f, sep=None, engine="python")
        df = clean_columns(df)

        combo_col = pick_col(df.columns, COMBO_CANDS)
        if combo_col is None:
            skipped.append((f, "missing combo/feature_combo column", list(df.columns)))
            continue
        if FEAT_COL not in df.columns or IMP_COL not in df.columns:
            skipped.append((f, "missing feature/importance columns", list(df.columns)))
            continue

        tmp = df[[combo_col, FEAT_COL, IMP_COL]].copy()
        tmp.rename(columns={combo_col: "combo"}, inplace=True)

        tmp["feature"] = tmp["feature"].astype(str).str.strip()
        tmp["importance"] = to_numeric_decimal_comma(tmp["importance"])

        tmp = tmp.dropna(subset=["feature", "importance", "combo"])

        tmp["n_features_in_model"] = tmp["combo"].apply(count_features_in_combo)

        tmp["source_file"] = os.path.basename(f)

        dfs.append(tmp)

    except Exception as e:
        skipped.append((f, f"exception: {e}", []))

if not dfs:
    print("First skipped reasons:")
    for s in skipped[:10]:
        print(" -", s[0], "=>", s[1])
    raise RuntimeError("No valid importance data loaded.")

data = pd.concat(dfs, ignore_index=True)

# =========================
# FILTERS
# =========================
if not INCLUDE_FLUGERFAHRUNG:
    data = data[data["feature"].str.lower() != "flugerfahrung"]

for sub in EXCLUDE_CONTAINS:
    data = data[~data["feature"].str.contains(sub, regex=False)]

# Keep only 1..4 groups (if you truly have only up to 4 base feature modalities)
data = data[data["n_features_in_model"].isin([1, 2, 3, 4])]

print(f"✅ Loaded {len(data)} importance rows from {len(files)} files.")
print("n_features_in_model counts:")
print(data["n_features_in_model"].value_counts().sort_index())

# =========================
# ORDER FEATURES (optional but makes plot readable)
# sort by mean importance (descending)
order = (
    data.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .index
)

# =========================
# PLOT: Points colored by model size
# =========================
plt.figure(figsize=(max(12, 0.35 * len(order)), 5))

sns.stripplot(
    data=data,
    x="feature",
    y="importance",
    order=order,
    hue="n_features_in_model",
    jitter=True,
    palette="bright",      # Kräftigere Farben
    alpha=0.8,             # Weniger transparent (vorher 0.55)              # Größere Punkte (vorher 4)
    linewidth=0.5,
    dodge=True,   # separate hues horizontally
    size=4
)

plt.title("Raw Random Forest feature importances across all models\nColored by number of features used in the model")
plt.xlabel("Feature")
plt.ylabel("Feature importance")
plt.xticks(rotation=35, ha="right")
plt.legend(title="#features in model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()

print(f"📊 Saved plot to:\n{OUT_PNG}")

# Optional: save the prepared long table used for plotting
out_long = os.path.join(OUT_DIR, "feature_importance_long_for_plot.csv")
data.to_csv(out_long, sep=";", index=False)
print(f"🧾 Saved plot data to:\n{out_long}")

if skipped:
    print(f"\n⚠️ Skipped {len(skipped)} files (showing up to 10):")
    for s in skipped[:10]:
        print(" -", s[0], "=>", s[1])
