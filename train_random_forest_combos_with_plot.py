#!/usr/bin/env python3
"""
Train Random Forest für alle Feature-Kombinationen:

Basis-Features (immer inkl. Baseline + flugerfahrung):
    HR      -> HR_bpm__mean_rounded, HR_bpm__baseline_mean
    HRV     -> RMSSD_ms__mean_rounded, RMSSD_ms__baseline_mean
    EDA     -> EDA__mean_rounded, EDA__baseline_mean
    TEMP    -> Temperature__mean_rounded, Temperature__baseline_mean
    + flugerfahrung (immer dabei)

Es werden trainiert:
    - 4x 1er-Kombis     (HR, HRV, EDA, TEMP)
    - 6x 2er-Kombis
    - 4x 3er-Kombis
    - 1x 4er-Kombi
= 15 Modelle

Für jedes Modell:
    - Train Accuracy
    - Test Accuracy
    - F1 (macro)
    - Prozentuale Fehlklassifikationsmatrix
    - Feature Importances

Am Ende:
    - Balkendiagramm der Test-Accuracy über alle 15 Modelle
    - CSV: Modell-Metriken
    - CSV: Feature Importances
"""

import argparse
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Random Forest auf allen Feature-Kombinationen (HR, HRV, EDA, TEMP)"
    )
    p.add_argument(
        "--root", required=True,
        help="Root directory mit final_features_round.csv Dateien (alle Probanden)"
    )
    p.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test-Split (default: 0.2 = 20%% Test)"
    )
    p.add_argument(
        "--random-state", type=int, default=42,
        help="Zufallsseed für Reproduzierbarkeit"
    )
    p.add_argument(
        "--out-plot", default="rf_feature_combos_accuracy.png",
        help="Dateiname für Balkendiagramm der Test-Accuracy"
    )
    p.add_argument(
        "--out-metrics-csv", default="rf_feature_combos_metrics.csv",
        help="CSV-Datei für Modellmetriken"
    )
    p.add_argument(
        "--out-importances-csv", default="rf_feature_combos_importances.csv",
        help="CSV-Datei für Feature Importances"
    )
    return p.parse_args()


def find_feature_files(root: str):
    root_path = Path(root)
    files = list(root_path.rglob("final_features_round.csv"))
    return files



def print_percentage_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    cm_df = pd.DataFrame(cm_percent, index=class_names, columns=class_names)

    print("\n📊 Prozentuale Fehlklassifikationsmatrix (%):")
    print(cm_df.round(2))

    print("\n❌ Fehlklassifikationen nach Klasse (in %):")
    for i, cls in enumerate(class_names):
        wrong = 100 - cm_percent[i, i]
        print(f"  {cls:15s} → falsch klassifiziert: {wrong:.2f}%")


def load_all_feature_files(root: str) -> pd.DataFrame:
    files = find_feature_files(root)
    if not files:
        raise FileNotFoundError(f"Keine final_features_round.csv Dateien unter: {root}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=";", decimal=",", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(f)  # Fallback
        df["__source_file"] = str(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print("🔍 Lade Daten ...")
    print(f"✅ Gesamt: {len(df_all)} Samples aus {len(files)} Dateien")
    return df_all


def clean_labels_and_encode(df: pd.DataFrame):
    if "label" not in df.columns:
        raise KeyError("Spalte 'label' nicht gefunden.")

    labels_raw = df["label"].astype(str).str.strip()
    mask_valid = (labels_raw != "") & (~labels_raw.str.lower().isin(["nan", "none"]))

    n_before = len(labels_raw)
    n_after = mask_valid.sum()
    if n_after == 0:
        raise ValueError("Keine gültigen Labels nach Bereinigung.")

    if n_before != n_after:
        print(f"⚠️ Label-Bereinigung: {n_before - n_after} Zeilen entfernt, übrig: {n_after}")

    df_clean = df[mask_valid].reset_index(drop=True)
    labels_clean = labels_raw[mask_valid].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(labels_clean)

    return df_clean, y, le, labels_clean


def build_X_for_combo(df: pd.DataFrame, combo_keys):
    """
    combo_keys: Tuple z.B. ('HR',) oder ('HR', 'EDA', 'TEMP')

    Gibt zurück:
        X: ndarray [n_samples, n_features]
        feature_names: Liste der tatsächlichen Spaltennamen
    """
    #mapping = {
     #   "HR":   ["HR_bpm__mean", "HR_bpm__baseline_mean"],
      #  "HRV":  ["RMSSD_ms__mean", "RMSSD_ms__baseline_mean"],
       # "EDA":  ["EDA__mean", "EDA__baseline_mean"],
        #"TEMP": ["Temperature__mean", "Temperature__baseline_mean"],
    #}
    
    mapping = {
        "HR":   ["HR_bpm__mean_rounded", "HR_bpm__baseline_mean"],
        "HRV":  ["RMSSD_ms__mean_rounded", "RMSSD_ms__baseline_mean"],
        "EDA":  ["EDA__mean_rounded", "EDA__baseline_mean"],
        "TEMP": ["Temperature__mean_rounded", "Temperature__baseline_mean"],
    }

    feature_cols = []
    for key in combo_keys:
        if key not in mapping:
            raise KeyError(f"Unbekanntes Feature-Label: {key}")
        feature_cols.extend(mapping[key])

    # flugerfahrung immer hinzufügen
    feature_cols.append("flugerfahrung")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Diese Feature-Spalten fehlen in den Daten: {missing}")

    X = df[feature_cols].to_numpy(dtype=float)
    return X, feature_cols


def main():
    args = parse_args()

    # 1) Daten laden
    df_all = load_all_feature_files(args.root)

    # 2) Labels bereinigen & encoden
    df_all, y, le, labels_raw = clean_labels_and_encode(df_all)

    # 3) Basis-Feature-Schlüssel
    base_keys = ["HR", "HRV", "EDA", "TEMP"]

    results = []          # für Modell-Metriken
    importance_rows = []  # für Feature Importances

    print("\n🚀 Starte Training für alle Feature-Kombinationen...\n")

    # 4) Alle Kombinationen durchgehen: 1er, 2er, 3er, 4er
    for k in range(1, len(base_keys) + 1):
        for combo in itertools.combinations(base_keys, k):
            combo_name = "+".join(combo)

            print("=" * 60)
            print(f"🧪 Kombination: {combo_name}")
            print("=" * 60)

            # Feature-Matrix für diese Kombination
            X, feature_names = build_X_for_combo(df_all, combo)

            # NaN/Inf entfernen
            mask_good = np.isfinite(X).all(axis=1)
            if (~mask_good).sum() > 0:
                print(f"⚠️ Entferne {(~mask_good).sum()} Samples mit NaN/Inf in den Features.")
                X = X[mask_good]
                y_local = y[mask_good]
            else:
                y_local = y

            # Train/Test-Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_local,
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=y_local,
            )

            print(f"📊 Train/Test Split: {X_train.shape[0]} Train, {X_test.shape[0]} Test")

            # Random Forest
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=args.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            clf.fit(X_train, y_train)

            # Performance
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            class_names = list(le.classes_)
            print_percentage_confusion_matrix(y_test, y_test_pred, class_names)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1_macro = f1_score(y_test, y_test_pred, average="macro")

            print("\n🏁 Modell-Performance:")
            print(f"   Kombination: {combo_name}")
            print(f"   Train Accuracy: {train_acc:.3f}")
            print(f"   Test  Accuracy: {test_acc:.3f}")
            print(f"   F1 (macro):     {f1_macro:.3f}")

            # Feature Importances
            importances = clf.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print("\n📌 Feature Importances (absteigend):")
            for rank, idx in enumerate(sorted_idx, start=1):
                fname = feature_names[idx]
                imp = importances[idx]
                print(f"   {fname:30s} -> {imp:.3f}")
                importance_rows.append({
                    "combo": combo_name,
                    "k": k,
                    "feature": fname,
                    "importance": imp,
                    "rank": rank,
                })

            # Ergebnis speichern für Metrics-CSV
            results.append({
                "combo": combo_name,
                "k": k,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "f1_macro": f1_macro,
            })

    # 5) CSVs schreiben
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values(by=["k", "combo"])
    metrics_df.to_csv(args.out_metrics_csv, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"\n💾 Modell-Metriken gespeichert unter: {args.out_metrics_csv}")

    importances_df = pd.DataFrame(importance_rows)
    importances_df = importances_df.sort_values(by=["k", "combo", "rank"])
    importances_df.to_csv(args.out_importances_csv, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"💾 Feature Importances gespeichert unter: {args.out_importances_csv}")

    # 6) Balkendiagramm der Test-Accuracy
    print("\n📉 Erzeuge Balkendiagramm der Test-Accuracy über alle Kombinationen...")

    results_sorted = metrics_df.to_dict(orient="records")

    labels_plot = [r["combo"] for r in results_sorted]
    accs_plot = [r["test_acc"] for r in results_sorted]
    ks_plot = [r["k"] for r in results_sorted]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels_plot))

    # Farben nach Kombigröße
    color_map = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    bar_colors = [color_map[k] for k in ks_plot]

    bars = plt.bar(x, accs_plot, color=bar_colors)

    # Prozentwerte über die Balken schreiben
    for bar, acc in zip(bars, accs_plot):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{acc * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(x, labels_plot, rotation=45, ha="right")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Feature-Kombination")
    plt.title("Random Forest: Test Accuracy für alle Feature-Kombinationen")
    plt.ylim(0.0, 1.05)

    # Legend für Kombigrößen
    handles = []
    labels_legend = []
    for k in sorted(set(ks_plot)):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[k]))
        labels_legend.append(f"{k} Basis-Feature(s)")
    plt.legend(handles, labels_legend, title="Kombinationsgröße")

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    plt.close()

    print(f"✅ Balkendiagramm gespeichert unter: {args.out_plot}")
    print("Fertig.")


if __name__ == "__main__":
    main()
