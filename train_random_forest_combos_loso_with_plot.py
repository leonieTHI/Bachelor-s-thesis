#!/usr/bin/env python3
"""
Random Forest für alle Feature-Kombinationen mit subject-basiertem Split
(“leave-some-subjects-out”):

Basis-Features (immer inkl. Baseline + flugerfahrung):
    HR      -> HR_bpm__mean, HR_bpm__baseline_mean
    HRV     -> RMSSD_ms__mean, RMSSD_ms__baseline_mean
    EDA     -> EDA__mean, EDA__baseline_mean
    TEMP    -> Temperature__mean, Temperature__baseline_mean
    + flugerfahrung (immer dabei)

Kombinationen:
    - 4x 1er-Kombis     (HR, HRV, EDA, TEMP)
    - 6x 2er-Kombis
    - 4x 3er-Kombis
    - 1x 4er-Kombi
= 15 Modelle

Split:
    - subject-basiert:
      ca. 20% der Subjects -> Test
      rest -> Train

Für jedes Modell:
    - Train Acc
    - Test Acc
    - F1 (macro)
    - Prozentuale Fehlklassifikationsmatrix
    - Feature Importances

Am Ende:
    - Balkendiagramm der Test-Accuracy
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
from sklearn.preprocessing import LabelEncoder


def parse_args():
    p = argparse.ArgumentParser(
        description="Random Forest Feature-Kombinationen mit subject-basiertem Split"
    )
    p.add_argument(
        "--root", required=True,
        help="Root directory mit feature_final.csv Dateien (alle Probanden)"
    )
    p.add_argument(
        "--test-subject-frac", type=float, default=0.2,
        help="Anteil der Subjects im Test-Set (ca., default: 0.2)"
    )
    p.add_argument(
        "--random-state", type=int, default=42,
        help="Zufallsseed für Reproduzierbarkeit"
    )
    p.add_argument(
        "--out-plot", default="rf_feature_combos_accuracy_subjectsplit.png",
        help="Dateiname für Balkendiagramm der Test-Accuracy"
    )
    p.add_argument(
        "--out-metrics-csv", default="rf_feature_combos_metrics_loso.csv",
        help="CSV-Datei für Modellmetriken (subject-basiert)"
    )
    p.add_argument(
        "--out-importances-csv", default="rf_feature_combos_importances_loso.csv",
        help="CSV-Datei für Feature Importances (subject-basiert)"
    )
    return p.parse_args()


def find_feature_files(root: str):
    root_path = Path(root)
    return list(root_path.rglob("final_features_round.csv"))


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


def infer_subject_id(path: Path) -> str:
    """
    Versucht aus dem Dateipfad eine Subject-ID zu gewinnen.
    Heuristik:
      - gehe durch die Pfadteile von hinten
      - nimm den ersten Teil, der mit 'P' beginnt und eine Ziffer enthält
      - falls nichts gefunden, nimm den Parent-Ordnernamen
    """
    for part in reversed(path.parts):
        if part.startswith("P") and any(ch.isdigit() for ch in part):
            return part
    return path.parent.name


def load_all_feature_files(root: str) -> pd.DataFrame:
    files = find_feature_files(root)
    if not files:
        raise FileNotFoundError(f"Keine feature_final.csv Dateien unter: {root}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=";", decimal=",", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(f)  # Fallback
        df["__source_file"] = str(f)
        df["__subject_id"] = infer_subject_id(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print("🔍 Lade Daten ...")
    print(f"✅ Gesamt: {len(df_all)} Samples aus {len(files)} Dateien")
    print(f"👤 Subjects gefunden: {df_all['__subject_id'].nunique()}")
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

    feature_cols.append("flugerfahrung")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Diese Feature-Spalten fehlen in den Daten: {missing}")

    X = df[feature_cols].to_numpy(dtype=float)
    return X, feature_cols


def subject_based_split(subject_ids: np.ndarray, test_frac: float, random_state: int):
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    if n_subjects < 2:
        raise ValueError("Zu wenige Subjects für subject-basierten Split.")

    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(unique_subjects)

    n_test = max(1, int(round(test_frac * n_subjects)))
    n_test = min(n_test, n_subjects - 1)

    test_subjects = shuffled[:n_test]
    train_subjects = shuffled[n_test:]

    print(f"👤 Subject-Split: {n_subjects} gesamt | {len(train_subjects)} Train | {len(test_subjects)} Test")
    print(f"   Train-Subjects: {list(train_subjects)}")
    print(f"   Test-Subjects:  {list(test_subjects)}")

    mask_test = np.isin(subject_ids, test_subjects)
    mask_train = np.isin(subject_ids, train_subjects)

    return mask_train, mask_test


def main():
    args = parse_args()

    # 1) Daten laden
    df_all = load_all_feature_files(args.root)

    # 2) Labels bereinigen & encoden
    df_all, y, le, labels_raw = clean_labels_and_encode(df_all)
    subject_ids_all = df_all["__subject_id"].to_numpy()

    base_keys = ["HR", "HRV", "EDA", "TEMP"]

    results = []
    importance_rows = []

    print("\n🚀 Starte Training für alle Feature-Kombinationen (subject-basiert)...\n")

    for k in range(1, len(base_keys) + 1):
        for combo in itertools.combinations(base_keys, k):
            combo_name = "+".join(combo)

            print("=" * 70)
            print(f"🧪 Kombination: {combo_name}")
            print("=" * 70)

            X_all, feature_names = build_X_for_combo(df_all, combo)

            mask_good = np.isfinite(X_all).all(axis=1)
            if (~mask_good).sum() > 0:
                print(f"⚠️ Entferne {(~mask_good).sum()} Samples mit NaN/Inf in den Features.")
            X = X_all[mask_good]
            y_local = y[mask_good]
            subj_local = subject_ids_all[mask_good]

            mask_train, mask_test = subject_based_split(
                subj_local, test_frac=args.test_subject_frac, random_state=args.random_state
            )

            X_train = X[mask_train]
            y_train = y_local[mask_train]
            X_test = X[mask_test]
            y_test = y_local[mask_test]

            print(f"📊 Train/Test Samples: {X_train.shape[0]} Train, {X_test.shape[0]} Test")

            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=args.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            class_names = list(le.classes_)
            print_percentage_confusion_matrix(y_test, y_test_pred, class_names)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1_macro = f1_score(y_test, y_test_pred, average="macro")

            print("\n🏁 Modell-Performance:")
            print(f"   Kombination:  {combo_name}")
            print(f"   Train Accuracy: {train_acc:.3f}")
            print(f"   Test  Accuracy: {test_acc:.3f}")
            print(f"   F1 (macro):     {f1_macro:.3f}")

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
    print(f"\n💾 Modell-Metriken (subject-basiert) gespeichert unter: {args.out_metrics_csv}")

    importances_df = pd.DataFrame(importance_rows)
    importances_df = importances_df.sort_values(by=["k", "combo", "rank"])
    importances_df.to_csv(args.out_importances_csv, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"💾 Feature Importances (subject-basiert) gespeichert unter: {args.out_importances_csv}")

    # 6) Balkendiagramm der Test-Accuracy
    print("\n📉 Erzeuge Balkendiagramm der Test-Accuracy über alle Kombinationen...")

    results_sorted = metrics_df.to_dict(orient="records")

    labels_plot = [r["combo"] for r in results_sorted]
    accs_plot = [r["test_acc"] for r in results_sorted]
    ks_plot = [r["k"] for r in results_sorted]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels_plot))

    color_map = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    bar_colors = [color_map[k] for k in ks_plot]

    bars = plt.bar(x, accs_plot, color=bar_colors)

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
    plt.ylabel("Test Accuracy (subject-basiert)")
    plt.xlabel("Feature-Kombination")
    plt.title("Random Forest: Test Accuracy für alle Feature-Kombinationen\n(subject-basiertes Train/Test)")
    plt.ylim(0.0, 1.05)

    handles = []
    labels_legend = []
    for k_unique in sorted(set(ks_plot)):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[k_unique]))
        labels_legend.append(f"{k_unique} Basis-Feature(s)")
    plt.legend(handles, labels_legend, title="Kombinationsgröße")

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    plt.close()

    print(f"✅ Balkendiagramm gespeichert unter: {args.out_plot}")
    print("Fertig.")


if __name__ == "__main__":
    main()
