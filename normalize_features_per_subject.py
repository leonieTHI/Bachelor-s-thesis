#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Z-Normalisierung der Mean-Features (ein Proband pro Datei) "
                    "+ zusätzliche gerundete Versionen."
    )

    p.add_argument("--input", required=True, help="Input CSV mit Features")
    p.add_argument("--output", required=True, help="Output CSV Pfad")

    # CSV I/O
    p.add_argument("--sep-in", default=",", help="Input Spaltentrenner")
    p.add_argument("--decimal-in", default=".", help="Input Dezimaltrennzeichen")
    p.add_argument("--sep-out", default=",", help="Output Spaltentrenner")
    p.add_argument("--decimal-out", default=".", help="Output Dezimaltrennzeichen")
    p.add_argument("--encoding", default="utf-8", help="Encoding")
    return p.parse_args()


def z_norm(series: pd.Series) -> pd.Series:
    """Einfache Z-Normalisierung einer Spalte: (x - mu) / sigma."""
    x = series.astype(float)
    mu = float(x.mean())
    sigma = float(x.std())

    if sigma == 0 or np.isnan(sigma):
        # falls Spalte konstant ist → alles 0
        return pd.Series(np.zeros(len(x)), index=series.index)

    return (x - mu) / sigma


def main():
    args = parse_args()

    # 1) Einlesen
    df = pd.read_csv(
        args.input,
        sep=args.sep_in,
        decimal=args.decimal_in,
        encoding=args.encoding
    )

    # 2) Relevante Spalten definieren
    feature_cols = {
        "EDA__mean": "EDA",
        "HR_bpm__mean": "HR",
        "RMSSD_ms__mean": "RMSSD",
        "Temperature__mean": "Temp",
    }

    # prüfen, was wirklich existiert
    existing = [c for c in feature_cols.keys() if c in df.columns]
    if not existing:
        raise RuntimeError(
            "Keine der erwarteten Mean-Spalten gefunden. "
            "Erwartet z.B.: EDA__mean, HR_bpm__mean, RMSSD_ms__mean, Temperature__mean"
        )

    print("[INFO] Gefundene Mean-Spalten:", existing)

    # 3) Z-Normalisierung: neue Spalten "<name>_z"
    for col in existing:
        z_col = col + "_z"
        df[z_col] = z_norm(df[col])

    # 4) Rundung: neue Spalten "<name>_rounded"
    #    (damit Originalwerte UNVERÄNDERT bleiben)
    if "HR_bpm__mean" in existing:
        df["HR_bpm__mean_rounded"] = df["HR_bpm__mean"].round(0)

    if "RMSSD_ms__mean" in existing:
        df["RMSSD_ms__mean_rounded"] = df["RMSSD_ms__mean"].round(0)

    if "Temperature__mean" in existing:
        df["Temperature__mean_rounded"] = df["Temperature__mean"].round(2)

    if "EDA__mean" in existing:
        df["EDA__mean_rounded"] = df["EDA__mean"].round(2)

    # 5) Kurze Kontrolle: Mittelwert & Std der Z-Spalten ausgeben
    print("\n[DEBUG] Z-Score-Kontrolle (sollte ~0 Mittelwert und ~1 Std haben):")
    for col in existing:
        z_col = col + "_z"
        mu = float(df[z_col].mean())
        sigma = float(df[z_col].std())
        print(f"  {z_col}: mean={mu:.4f}, std={sigma:.4f}")

    # 6) Speichern
    df.to_csv(
        args.output,
        index=False,
        sep=args.sep_out,
        decimal=args.decimal_out,
        encoding=args.encoding
    )

    print(f"\n[OK] Normalisierte Datei gespeichert unter: {args.output}")


if __name__ == "__main__":
    main()
