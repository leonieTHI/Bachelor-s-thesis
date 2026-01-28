# -*- coding: utf-8 -*-
"""
Run global nonparametric tests on accuracy across preprocessing variants
SEPARATELY for:
- Random train/test split (non-LOSO)
- LOSO

Input format: test_acc_wide.csv
- first column: variant
- remaining columns: feature combinations
- rows: variants (should contain 8 total if both random + loso are included)

Outputs:
- global_effectsizes_random.csv
- global_effectsizes_loso.csv
- global_effectsizes_both.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal

# =========================
# CONFIG
# =========================
IN_CSV = r"C:\Users\49157\Desktop 2\Bachelorarbeit\Daten_real\model_eval_stats\test_acc_wide.csv"
OUT_DIR = os.path.join(os.path.dirname(IN_CSV), "accuracy_variation_results")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_RANDOM = os.path.join(OUT_DIR, "global_effectsizes_random.csv")
OUT_LOSO = os.path.join(OUT_DIR, "global_effectsizes_loso.csv")
OUT_BOTH = os.path.join(OUT_DIR, "global_effectsizes_both.csv")

# How we identify LOSO rows
LOSO_TOKEN = "loso"  # case-insensitive

# Optional: only keep these 4 variants per group (substring matching).
# Leave as None to just take all rows that match LOSO/non-LOSO.
KEEP_4_VARIANTS_SUBSTRINGS = ["30s", "30s_rounded", "30s_binary", "60s"]  # adjust if your names differ

# =========================
# HELPERS
# =========================
def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df

def interpret_w(w: float) -> str:
    if np.isnan(w):
        return "NA"
    if w < 0.10:
        return "negligible"
    if w < 0.30:
        return "small"
    if w < 0.50:
        return "moderate"
    return "large"

def interpret_eps2(e: float) -> str:
    if np.isnan(e):
        return "NA"
    if e < 0.01:
        return "negligible"
    if e < 0.06:
        return "small"
    if e < 0.14:
        return "moderate"
    return "large"

def filter_to_4_variants(df_sub: pd.DataFrame, variant_col: str) -> pd.DataFrame:
    if not KEEP_4_VARIANTS_SUBSTRINGS:
        return df_sub
    keep = []
    v = df_sub[variant_col].astype(str).str.lower()
    for s in KEEP_4_VARIANTS_SUBSTRINGS:
        keep.append(v.str.contains(str(s).lower(), regex=False))
    mask = np.logical_or.reduce(keep) if keep else np.ones(len(df_sub), dtype=bool)
    return df_sub.loc[mask].copy()

def run_global_tests(df_sub: pd.DataFrame, variant_col: str, label: str) -> pd.DataFrame:
    """
    Friedman (paired) + Kendall's W using only feature-combos complete across all variants in df_sub.
    Kruskal–Wallis (unpaired check) + epsilon^2 on the same paired-complete matrix (consistent base).
    """
    variants = df_sub[variant_col].astype(str).tolist()

    mat = df_sub.set_index(variant_col).T  # rows = combos, cols = variants
    mat = mat[variants]  # preserve order

    # paired complete rows
    mat_complete = mat.dropna(axis=0, how="any")
    n_blocks = mat_complete.shape[0]
    k = mat_complete.shape[1]

    if n_blocks < 2 or k < 2:
        raise RuntimeError(
            f"[{label}] Not enough complete data for tests: "
            f"need >=2 feature-combos with all variants. Got n_blocks={n_blocks}, k={k}."
        )

    # Friedman + Kendall's W
    arrays = [mat_complete[col].values for col in mat_complete.columns]
    chi2, p_f = friedmanchisquare(*arrays)
    W = chi2 / (n_blocks * (k - 1))

    # Kruskal–Wallis + epsilon^2 (unpaired check)
    groups = [mat_complete[col].values for col in mat_complete.columns]
    H, p_kw = kruskal(*groups)
    n_obs = mat_complete.size
    eps2 = (H - k + 1) / (n_obs - k) if (n_obs - k) > 0 else np.nan

    out = pd.DataFrame([
        {
            "set": label,
            "test": "Friedman",
            "statistic": "chi2",
            "statistic_value": float(chi2),
            "p_value": float(p_f),
            "effect_size_name": "Kendalls_W",
            "effect_size_value": float(W),
            "effect_interpretation": interpret_w(float(W)),
            "n_feature_combos_used": int(n_blocks),
            "n_variants_used": int(k),
            "variants_used": "|".join(map(str, variants)),
        },
        {
            "set": label,
            "test": "Kruskal-Wallis",
            "statistic": "H",
            "statistic_value": float(H),
            "p_value": float(p_kw),
            "effect_size_name": "Epsilon_squared",
            "effect_size_value": float(eps2),
            "effect_interpretation": interpret_eps2(float(eps2)),
            "n_feature_combos_used": int(n_blocks),
            "n_variants_used": int(k),
            "variants_used": "|".join(map(str, variants)),
        },
    ])
    return out

# =========================
# LOAD
# =========================
df = pd.read_csv(IN_CSV, sep=";")
variant_col = df.columns[0]
df[variant_col] = df[variant_col].astype(str).str.strip()
df = coerce_numeric_df(df)

# Split into random vs loso
is_loso = df[variant_col].str.lower().str.contains(LOSO_TOKEN, regex=False)
df_loso = df.loc[is_loso].copy()
df_random = df.loc[~is_loso].copy()

# Optionally keep only the 4 preprocessing variants in each subset
df_loso = filter_to_4_variants(df_loso, variant_col)
df_random = filter_to_4_variants(df_random, variant_col)

print("Random variants used:", df_random[variant_col].tolist())
print("LOSO variants used:", df_loso[variant_col].tolist())

# Run tests
res_random = run_global_tests(df_random, variant_col, "random_train_test")
res_loso = run_global_tests(df_loso, variant_col, "loso")

# Save
res_random.to_csv(OUT_RANDOM, sep=";", index=False)
res_loso.to_csv(OUT_LOSO, sep=";", index=False)

res_both = pd.concat([res_random, res_loso], ignore_index=True)
res_both.to_csv(OUT_BOTH, sep=";", index=False)

print(f"✅ Saved: {OUT_RANDOM}")
print(f"✅ Saved: {OUT_LOSO}")
print(f"✅ Saved: {OUT_BOTH}")
print("\nResults:")
print(res_both)
