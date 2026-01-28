# -*- coding: utf-8 -*-
"""
Plot accuracy spread across preprocessing variants
for ALL feature combinations in ONE plot (points).

INPUT:
- test_acc_long.csv

OUTPUT:
- accuracy_spread_all_features_points.png
- (optional) accuracy_spread_all_features_violin.png
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\49157\Desktop 2\Bachelorarbeit\Daten_real\model_eval_stats"
IN_CSV = os.path.join(BASE_DIR, "test_acc_long.csv")
OUT_DIR = os.path.join(BASE_DIR, "accuracy_spread_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(IN_CSV, sep=";")
df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
df = df.dropna(subset=["feature_combo", "variant", "test_acc"])

# Sort feature combos by mean accuracy (nicer ordering)
order = (
    df.groupby("feature_combo")["test_acc"]
      .mean()
      .sort_values(ascending=False)
      .index
)

# =========================
# 1) POINT / STRIP PLOT
# =========================
plt.figure(figsize=(12, 5))
sns.stripplot(
    data=df,
    x="feature_combo",
    y="test_acc",
    order=order,
    jitter=True,
    alpha=0.7,
    size=6
)

plt.title("Accuracy spread across preprocessing variants\n(all feature combinations)")
plt.ylabel("Test accuracy")
plt.xlabel("Feature combination")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

out_points = os.path.join(OUT_DIR, "accuracy_spread_all_features_points.png")
plt.savefig(out_points)
plt.close()

# =========================
# 2) OPTIONAL: VIOLIN + POINTS
# =========================
import numpy as np

# =========================
# 2) VIOLIN + POINTS + MEAN + CI
# =========================
plt.figure(figsize=(12, 5))

# Violin
sns.violinplot(
    data=df,
    x="feature_combo",
    y="test_acc",
    order=order,
    inner=None,
    color="lightgray",
    cut=0
)

# Raw points
sns.stripplot(
    data=df,
    x="feature_combo",
    y="test_acc",
    order=order,
    jitter=True,
    alpha=0.6,
    size=5,
    color="black"
)

# Compute mean and CI per feature combo
grouped = df.groupby("feature_combo")["test_acc"]
means = grouped.mean()
stds = grouped.std()
ns = grouped.count()

# 95% confidence interval
ci95 = 1.96 * stds / np.sqrt(ns)

# X positions
x_pos = np.arange(len(order))

# Mean as horizontal bar
plt.errorbar(
    x=x_pos,
    y=means[order],
    yerr=ci95[order],     # <-- CI
    fmt="o",
    color="red",
    capsize=5,
    linewidth=2,
    label="Mean ± 95% CI"
)

# OPTIONAL: use STD instead of CI
# plt.errorbar(
#     x=x_pos,
#     y=means[order],
#     yerr=stds[order],
#     fmt="o",
#     color="red",
#     capsize=5,
#     linewidth=2,
#     label="Mean ± SD"
# )

plt.title(
    "Accuracy distribution across preprocessing variants\n"
    "(violin, individual models, mean and 95% confidence interval)"
)
plt.ylabel("Test accuracy")
plt.xlabel("Feature combination")
plt.xticks(x_pos, order, rotation=30, ha="right")
plt.legend()
plt.tight_layout()

out_violin = os.path.join(OUT_DIR, "accuracy_spread_all_features_violin_mean_ci.png")
plt.savefig(out_violin)
plt.close()

print("✅ Violin plot with mean and confidence interval created.")


print("✅ Accuracy spread plots created.")
print(f"Saved to:\n{OUT_DIR}")
