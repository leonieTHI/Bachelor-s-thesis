
#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def coerce_numeric_locale(s: pd.Series) -> pd.Series:
    """
    Macht aus Strings mit Komma-Dezimaltrennzeichen richtige Floats.
    Beispiele:
        '0,123'      -> 0.123
        '10.165,0'   -> 10165.0
    """
    # erster Versuch: direkte Konvertierung
    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.notna().mean() >= 0.8:
        return s_num

    # zweite Stufe: Kommas und Tausenderpunkte behandeln
    s_str = s.astype(str)
    has_comma = s_str.str.contains(',', regex=False, na=False).mean() > 0.2
    if has_comma:
        s_fix = s_str.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        s_num2 = pd.to_numeric(s_fix, errors='coerce')
        if s_num2.notna().mean() > s_num.notna().mean():
            return s_num2
    return s_num


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge per-window features with baseline means and flight experience."
    )

    p.add_argument("--features-csv", required=True,
                   help="Feature-CSV (z.B. features_30s.csv)")
    p.add_argument("--baseline-csv", required=True,
                   help="Baseline-CSV (z.B. processed_timeseries.csv)")
    p.add_argument("--out", required=True,
                   help="Ausgabe-CSV")

    # Einlese-Parameter
    p.add_argument("--sep-feat", default=";", help="Separator in Feature-CSV")
    p.add_argument("--decimal-feat", default=",", help="Dezimalzeichen in Feature-CSV")
    p.add_argument("--sep-base", default=";", help="Separator in Baseline-CSV")
    p.add_argument("--decimal-base", default=",", help="Dezimalzeichen in Baseline-CSV")
    p.add_argument("--encoding", default="utf-8-sig", help="Datei-Encoding")

    # Welche Spalten aus der Baseline? (Originalnamen, OHNE __mean!)
    p.add_argument(
        "--baseline-cols",
        default="EDA,HR_bpm,RMSSD_ms,Temperature",
        help="Kommagetrennte Liste von Spaltennamen aus der Baseline-CSV, "
             "z.B. 'EDA,HR_bpm,RMSSD_ms,Temperature'"
    )

    # Flugerfahrung
    p.add_argument("--flugerfahrung", type=float, default=None,
                   help="Flugerfahrung in Stunden (wird als konstante Spalte angefügt)")

    # Excel-freundliche Ausgabe
    p.add_argument("--excel-friendly", action="store_true",
                   help="Nutze ; als Separator, , als Dezimalzeichen und utf-8-sig")

    return p.parse_args()


def ensure_all_present(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Diese Baseline-Spalten fehlen in der Datei: {missing}")


def main():
    args = parse_args()

    # --- Features einlesen ---
    df_feat = pd.read_csv(
        args.features_csv,
        sep=args.sep_feat,
        decimal=args.decimal_feat,
        encoding=args.encoding
    )

    # --- Baseline einlesen ---
    df_base = pd.read_csv(
        args.baseline_csv,
        sep=args.sep_base,
        decimal=args.decimal_base,
        encoding=args.encoding
    )

    # Baseline-Spaltenliste
    base_cols = [c.strip() for c in args.baseline_cols.split(",") if c.strip()]
    ensure_all_present(df_base, base_cols)

    # --- Mittelwerte der Baseline-Spalten robust berechnen ---
    baseline_means = {}
    for col in base_cols:
        col_num = coerce_numeric_locale(df_base[col])
        mean_val = float(col_num.mean(skipna=True))
        baseline_means[f"{col}__baseline_mean"] = mean_val

    print("Baseline-Mittelwerte:")
    for k, v in baseline_means.items():
        print(f"  {k}: {v:.4f}")

    # --- Output-DataFrame bauen ---
    df_out = df_feat.copy()

    # konstante Baseline-Mittelwerte an jede Zeile hängen
    for k, v in baseline_means.items():
        df_out[k] = v

    # Flugerfahrung, falls angegeben
    if args.flugerfahrung is not None:
        df_out["flugerfahrung"] = args.flugerfahrung

    # --- Ausgabe schreiben ---
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.excel_friendly:
        sep = ";"
        dec = ","
        enc = "utf-8-sig"
        lineterm = "\r\n"
    else:
        sep = args.sep_feat
        dec = args.decimal_feat
        enc = args.encoding
        lineterm = "\n"

    df_out.to_csv(
        args.out,
        index=False,
        sep=sep,
        decimal=dec,
        encoding=enc,
        lineterminator=lineterm
    )

    print(f"✅ Saved merged features + baseline to: {args.out}")


if __name__ == "__main__":
    main()
