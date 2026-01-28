# save as: make_binary_labels.py

import argparse
import pandas as pd

def recode_label(val: object) -> object:
    """
    Mappt:
      - 'meditation', 'stress1', 'stress 1', 'questionnaire', '', NaN -> 'kein stress'
      - 'stress2', 'stress 2', 'stress3', 'stress 3' -> 'stress'
    Alles andere bleibt unverändert.
    """
    # Leere oder fehlende Werte -> kein stress
    if val is None or str(val).strip() == "" or pd.isna(val):
        return "kein stress"

    s = str(val).strip().lower()
    s_no_space = s.replace(" ", "")

    # Kein-Stress-Klasse
    if s == "meditation" or s_no_space in ("stress1", "questionnaire"):
        return "kein stress"

    # Stress-Klasse
    if s_no_space in ("stress2", "stress3"):
        return "stress"

    # Unverändert zurückgeben (z.B. "Baseline")
    return val

def main():
    parser = argparse.ArgumentParser(description="Binaere Stress-Klassifikation auf CSV anwenden.")
    parser.add_argument("--input", "-i", required=True, help="Pfad zur Eingabe-CSV")
    parser.add_argument("--output", "-o", required=True, help="Pfad zur Ausgabe-CSV")
    parser.add_argument("--label-col", "-c", default="label",
                        help="Name der Label-Spalte (Standard: 'label')")
    parser.add_argument("--sep", default=";", help="Spaltentrenner (Standard: ';')")

    args = parser.parse_args()

    # CSV einlesen
    df = pd.read_csv(args.input, sep=args.sep, dtype=str)

    if args.label_col not in df.columns:
        raise ValueError(f"Label-Spalte '{args.label_col}' nicht gefunden. "
                         f"Spalten: {list(df.columns)}")

    # Labels rekodieren
    df[args.label_col] = df[args.label_col].apply(recode_label)

    # CSV speichern
    df.to_csv(args.output, sep=args.sep, index=False)

if __name__ == "__main__":
    main()
