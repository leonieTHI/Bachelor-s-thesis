# -*- coding: utf-8 -*-
"""
Two plots:
1) Random train/test only (variants: 30s, 30s_rounded, binary_30s, 60s)
2) LOSO only          (variants: loso_30s, loso_30s_rounded, loso_30s_binary, loso_60s)

Within each plot:
- x-axis: feature_combo
- hue: preprocessing variant (4 groups)
- violin + raw points
- mean ± 95% CI per (feature_combo, variant)

Input: test_acc_long.csv (variant, feature_combo, test_acc)
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\49157\Desktop 2\Bachelorarbeit\Daten_real\model_eval_stats"
IN_CSV = os.path.join(BASE_DIR, "test_acc_long.csv")

OUT_DIR = os.path.join(BASE_DIR, "accuracy_spread_plots_split")
os.makedirs(OUT_DIR, exist_ok=True)

# Exact variant sets (based on your examples)
RANDOM_VARIANTS = ["30s", "30s_rounded", "binary_30s", "60s"]
LOSO_VARIANTS   = ["loso_30s", "loso_30s_rounded", "loso_30s_binary", "loso_60s"]

# =========================
# LOAD
# =========================
df = pd.read_csv(IN_CSV, sep=";")
df["variant"] = df["variant"].astype(str).str.strip()
df["feature_combo"] = df["feature_combo"].astype(str).str.strip()
df["test_acc"] = pd.to_numeric(df["test_acc"].astype(str).str.replace(",", ".", regex=False),
                               errors="coerce")
df = df.dropna(subset=["variant", "feature_combo", "test_acc"])

df_random = df[df["variant"].isin(RANDOM_VARIANTS)].copy()
df_loso   = df[df["variant"].isin(LOSO_VARIANTS)].copy()

if df_random.empty:
    raise RuntimeError("No rows for RANDOM_VARIANTS found. Check names.")
if df_loso.empty:
    raise RuntimeError("No rows for LOSO_VARIANTS found. Check names.")

# =========================
# HELPERS
# =========================
def add_mean_ci_points(ax, sub_df, order, hue_order, dodge=0.8):
    """
    Add mean ± 95% CI for each (feature_combo, variant) on the existing categorical plot.
    We compute x positions manually, aligned with seaborn's dodge grouping.
    """
    grouped = (
        sub_df.groupby(["feature_combo", "variant"])["test_acc"]
              .agg(["mean", "std", "count"])
              .reset_index()
    )
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))

    # map categories to indices
    x_map = {fc: i for i, fc in enumerate(order)}
    h_map = {v: j for j, v in enumerate(hue_order)}

    # seaborn categorical "dodge" positions: spread points inside each x bin
    m = len(hue_order)
    if m == 1:
        offsets = {hue_order[0]: 0.0}
    else:
        # centers around 0, scaled by dodge
        base = np.linspace(-dodge/2, dodge/2, m)
        offsets = {hue_order[j]: base[j] for j in range(m)}

    xs, ys, yerr = [], [], []
    for _, r in grouped.iterrows():
        fc = r["feature_combo"]
        v  = r["variant"]
        if fc not in x_map or v not in offsets:
            continue
        xs.append(x_map[fc] + offsets[v])
        ys.append(r["mean"])
        yerr.append(r["ci95"])

    ax.errorbar(xs, ys, yerr=yerr, fmt="o", color="red", capsize=4, linewidth=1.5,
                label="Mean ± 95% CI")

def plot_split(sub_df, title, out_png, variants_order):
    # sort feature combos by overall mean (within this subset)
    order = (
        sub_df.groupby("feature_combo")["test_acc"]
              .mean()
              .sort_values(ascending=False)
              .index.tolist()
    )

    plt.figure(figsize=(14, 6))
    ax = sns.violinplot(
        data=sub_df,
        x="feature_combo",
        y="test_acc",
        hue="variant",
        order=order,
        hue_order=variants_order,
        cut=0,
        inner=None
    )

    sns.stripplot(
        data=sub_df,
        x="feature_combo",
        y="test_acc",
        hue="variant",
        order=order,
        hue_order=variants_order,
        dodge=True,
        alpha=0.55,
        size=3,
        linewidth=0
    )

    # Add mean±CI on top
    add_mean_ci_points(ax, sub_df, order=order, hue_order=variants_order, dodge=0.8)

    # Make legend clean (avoid duplicate legends from violin + strip + errorbar)
    handles, labels = ax.get_legend_handles_labels()
    # keep only unique by label order
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    ax.legend([h for h, _ in uniq], [l for _, l in uniq],
              title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("Feature combination")
    ax.set_ylabel("Test accuracy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, out_png)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Saved: {out_path}")

# =========================
# RUN
# =========================
plot_split(
    df_random,
    title="Accuracy distribution by preprocessing variant (Random train/test split)\n(violin, individual models, mean and 95% CI)",
    out_png="accuracy_violin_random_by_variant.png",
    variants_order=RANDOM_VARIANTS
)

plot_split(
    df_loso,
    title="Accuracy distribution by preprocessing variant (LOSO)\n(violin, individual models, mean and 95% CI)",
    out_png="accuracy_violin_loso_by_variant.png",
    variants_order=LOSO_VARIANTS
)

print("✅ Done. Outputs in:", OUT_DIR)
