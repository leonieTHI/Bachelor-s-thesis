#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XDF -> CSV/XLSX (+ Marker-Labels + IBI/HR + merged_all_streams + UTC)
Python 3.9 kompatibel
---------------------------------------------------------------
Features:
- Per-Stream CSV (oder optional per Kanal) mit:
    timestamp (LSL s), utc_epoch, utc_iso, ...kanäle..., label
- Marker-Stream (string) separat + marker_event in merged_all_streams.csv
- IBI (s) + HR (bpm) aus gewähltem PPG/BVP-Stream (Red/IR/Green/…)
- UTC-Spalten via Offset:
    OFFSET = UTC_Start - LSL_Start (LSL_Start = min(time_stamps) aller Streams)
    UTC_Start: --utc-start (ISO) oder Dateimtime als Heuristik
- CSV-Output steuerbar: --sep, --decimal, --encoding, --bom, --lineterm
- Header-Säuberung: Kommas/Semikolons im Spaltennamen entfernen (--clean-headers)
- XLSX optional: --excel [--sheet-name]
- Spaltensteuerung: --list-columns, --keep-cols, --drop-cols, --reorder-cols

Beispiel:
    python xdf_to_csv_with_markers_ibi_utc_py39.py input.xdf --outdir ./csv_out ^
        --marker-name ExperimentMarkers --ppg-name PPG_RED ^
        --sep ";" --decimal "," --encoding utf-8 --bom --clean-headers ^
        --excel --sheet-name merged

Abhängigkeiten:
    pip install pyxdf pandas numpy openpyxl
"""
from __future__ import annotations

import argparse
import sys
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# pyxdf ist erforderlich
try:
    import pyxdf
except Exception as e:
    sys.stderr.write("Fehler: pyxdf ist nicht installiert. Installiere mit: pip install pyxdf\n")
    raise

# ----------------- Hilfsfunktionen -----------------

def iso_to_epoch(iso_str: str) -> float:
    """ISO-8601 (z.B. 2025-09-15T14:02:03Z) -> Unix Epoch (Sekunden, float)"""
    # Akzeptiere 'Z' oder Offset
    try:
        if iso_str.endswith("Z"):
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(iso_str)
        return dt.timestamp()
    except Exception:
        raise ValueError(f"Ungültiges ISO-Datum: {iso_str!r}")

def epoch_to_iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(float(epoch_s), tz=timezone.utc).isoformat().replace("+00:00", "Z")

def stream_basename(s: dict) -> str:
    nm = (s.get("info", {}).get("name", [""])[0] or "").strip()
    tp = (s.get("info", {}).get("type", [""])[0] or "").strip()
    return f"{nm or 'Stream'}__{tp or 'Type'}"

def channel_labels(s: dict) -> List[str]:
    """Erzeuge Kanal-Labels aus s['info'].['desc'][0]['channels'][0]... falls vorhanden"""
    try:
        desc = s["info"]["desc"][0]
        chs = desc.get("channels", [])[0].get("channel", [])
        labs = [ch.get("label", [""])[0] for ch in chs]
        uniq = []
        seen = {}
        for lab in labs:
            lab = str(lab).strip() or "ch"
            if lab in seen:
                seen[lab] += 1
                uniq.append(f"{lab}_{seen[lab]}")
            else:
                seen[lab] = 1
                uniq.append(lab)
        return uniq
    except Exception:
        n = int(s["info"]["channel_count"][0])
        return [f"ch_{i+1}" for i in range(n)]

def pick_stream_by_name_type(streams: List[dict], name_sub: Optional[str]=None, type_sub: Optional[str]=None,
                             type_fallback: Optional[str]=None) -> Optional[dict]:
    """Wähle Stream anhand Substrings in NAME/TYPE. Falls nichts gefunden, optional Fallback-Typ (z.B. 'Markers')."""
    cand = []
    for s in streams:
        nm = (s["info"]["name"][0] if s["info"]["name"] else "") or ""
        tp = (s["info"]["type"][0] if s["info"]["type"] else "") or ""
        if name_sub and name_sub.lower() in nm.lower():
            cand.append(s)
        elif type_sub and type_sub.lower() in tp.lower():
            cand.append(s)
        elif (not name_sub and not type_sub) and type_fallback and tp.lower() == type_fallback.lower():
            cand.append(s)
    cand.sort(key=lambda s: len(s.get("time_stamps", [])), reverse=True)
    return cand[0] if cand else None

def extract_markers(marker_stream: dict) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(marker_stream.get("time_stamps", []), float)
    labels = []
    for row in marker_stream.get("time_series", []):
        if isinstance(row, (list, tuple)) and len(row) >= 1:
            labels.append(str(row[0]))
        else:
            labels.append(str(row))
    labels = np.asarray(labels, dtype=object)
    if ts.size:
        idx = np.argsort(ts)
        ts = ts[idx]
        labels = labels[idx]
        good = (labels != "") & pd.notna(labels)
        ts, labels = ts[good], labels[good]
    return ts, labels

def segments_from_markers(ts: np.ndarray, labels: np.ndarray) -> List[Tuple[float, float, str]]:
    """Erzeuge (t_start, t_end, label) Segmente aus Markerzeiten/Labels (vorwärts-offen, letztes Segment bis +inf)."""
    segs: List[Tuple[float, float, str]] = []
    if ts.size == 0:
        return segs
    for i in range(len(ts)):
        t0 = float(ts[i])
        t1 = float(ts[i+1]) if i+1 < len(ts) else float("inf")
        segs.append((t0, t1, str(labels[i])))
    return segs


def add_labels_to_df(df: pd.DataFrame, segs: List[Tuple[float, float, str]]) -> pd.DataFrame:
    # Ensure df is sorted by timestamp
    df = df.sort_values("timestamp")
    if not segs:
        # Create or sanitize an existing label column
        if "label" not in df.columns:
            df["label"] = ""
        else:
            df["label"] = df["label"].fillna("")
        return df

    # Build segment dataframe
    seg_df = pd.DataFrame({"timestamp": [s[0] for s in segs], "label": [s[2] for s in segs]}).sort_values("timestamp")

    # If df already had a 'label' column, temporarily rename it to avoid suffix confusion
    had_label = "label" in df.columns
    if had_label:
        df = df.rename(columns={"label": "_label_existing"})

    merged_df = pd.merge_asof(df, seg_df, on="timestamp", direction="backward", suffixes=("", "_m"))

    # Decide which column contains the merged label
    label_source = None
    for cand in ["label", "label_m", "label_y", "label_x"]:
        if cand in merged_df.columns:
            label_source = cand
            break

    if label_source is None:
        # fallback: create an empty label column
        merged_df["label"] = ""
    elif label_source != "label":
        merged_df["label"] = merged_df[label_source]

    # Final cleanup
    merged_df["label"] = merged_df["label"].fillna("")
    for col in ["label_m", "label_x", "label_y", "_label_existing"]:
        if col in merged_df.columns:
            merged_df = merged_df.drop(columns=[col])

    return merged_df

def compute_utc_offset(streams: List[dict], xdf_path: Path, utc_start: Optional[str]) -> float:
    """OFFSET = UTC_Start - LSL_Start; LSL_Start = min(time_stamps) aller Streams."""
    # LSL-Start
    lsl_starts = []
    for s in streams:
        ts = np.asarray(s.get("time_stamps", []), float)
        if ts.size:
            lsl_starts.append(float(ts.min()))
    if not lsl_starts:
        raise RuntimeError("Keine time_stamps in XDF gefunden.")
    lsl_start = float(min(lsl_starts))

    # UTC-Start
    if utc_start:
        utc_start_epoch = iso_to_epoch(utc_start)
    else:
        # Heuristik: Dateimtime als UTC
        mtime = xdf_path.stat().st_mtime
        utc_start_epoch = float(mtime)
        sys.stderr.write("Warnung: --utc-start nicht gesetzt, benutze Datei-mtime als Heuristik.\n")
    return float(utc_start_epoch - lsl_start)

def sanitize_headers(cols: List[str]) -> List[str]:
    safe = []
    for c in cols:
        cc = str(c)
        # Entferne potenzielle CSV-Trennzeichen (Komma/Semikolon/Tab) aus Spaltennamen
        cc = cc.replace("\t", "_")
        cc = cc.replace(",", "_")
        cc = cc.replace(";", "_")
        safe.append(cc)
    return safe

# --------- Einfache Peak-Detektion für PPG (rudimentär aber robust) ---------

def smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")

def detect_peaks(time_s: np.ndarray, signal: np.ndarray,
                 min_ibi_s: float = 0.3,
                 win: int = 11,
                 thr_k: float = 0.5) -> List[float]:
    """Einfache Detektion: geglättetes Signal, lokale Maxima, adaptiver Schwellwert."""
    t = np.asarray(time_s, float)
    x = smooth(np.asarray(signal, float), w=win)
    peaks: List[float] = []
    last_t = None
    n = len(x)
    for i in range(1, n-1):
        if not (x[i] >= x[i-1] and x[i] >= x[i+1]):
            continue
        local = x[max(0, i-50):min(n, i+51)]
        thr = local.mean() + thr_k * local.std(ddof=0)
        if x[i] < thr:
            continue
        ti = float(t[i])
        if last_t is not None and (ti - last_t) < min_ibi_s:
            continue
        peaks.append(ti)
        last_t = ti
    return peaks

def ibi_hr_from_peaks(peaks_t: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(peaks_t) < 2:
        return np.array([]), np.array([]), np.array([])
    t = np.asarray(peaks_t, float)
    ibi = np.diff(t)
    hr = 60.0 / np.maximum(ibi, 1e-6)
    t_mid = t[1:]
    return t_mid, ibi, hr

# ----------------- Hauptlogik -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xdf", type=str, help="Pfad zur XDF-Datei")
    ap.add_argument("--outdir", type=str, default="./csv_out", help="Zielordner")

    # Marker-/PPG-Erkennung
    ap.add_argument("--marker-name", type=str, default=None, help="Substring zum Finden des Marker-Streams nach NAME")
    ap.add_argument("--marker-type", type=str, default=None, help="Substring zum Finden des Marker-Streams nach TYPE (z.B. 'Markers')")
    ap.add_argument("--ppg-name", type=str, default=None, help="Substring zum Finden des PPG/BVP-Streams nach NAME")
    ap.add_argument("--ppg-type", type=str, default=None, help="Substring zum Finden des PPG/BVP-Streams nach TYPE (z.B. 'PPGRed', 'BVP')")

    # Merge & Ausgabe
    ap.add_argument("--tolerance", type=float, default=0.02, help="Merge-Toleranz (s) für merged_all_streams.csv")
    ap.add_argument("--split-channels", action="store_true", help="Pro Kanal einzelne CSVs ausgeben")

    # UTC
    ap.add_argument("--utc-start", type=str, default=None, help="ISO-Startzeit der Aufnahme, sonst Datei-mtime Heuristik")

    # CSV/XLSX-Ausgabe-Optionen
    ap.add_argument("--sep", type=str, default=",", help="CSV Separator (z.B. ';' für deutsches Excel)")
    ap.add_argument("--decimal", type=str, default=".", help="Dezimaltrennzeichen (z.B. ',' für deutsches Excel)")
    ap.add_argument("--encoding", type=str, default="utf-8", help="Datei-Encoding (z.B. 'utf-8', 'cp1252')")
    ap.add_argument("--bom", action="store_true", help="UTF-8 mit BOM schreiben (für Excel-Autodetektion)")
    ap.add_argument("--lineterm", type=str, default="\n", help="Zeilenende (Standard '\\n')")
    ap.add_argument("--clean-headers", action="store_true", help="Komma/Semikolon/Tab aus Spaltennamen entfernen")
    ap.add_argument("--excel", action="store_true", help="Zusätzlich eine XLSX-Datei für merged_all_streams schreiben")
    ap.add_argument("--sheet-name", type=str, default="merged", help="XLSX Tabellenblattname")

    # Spaltensteuerung für merged
    ap.add_argument("--list-columns", action="store_true", help="Nur Spaltenliste ausgeben (merged) und beenden")
    ap.add_argument("--keep-cols", type=str, default=None, help="Kommagetrennte Liste von Spalten, die behalten werden (regex erlaubt)")
    ap.add_argument("--drop-cols", type=str, default=None, help="Kommagetrennte Liste von Spalten, die entfernt werden (regex erlaubt)")
    ap.add_argument("--reorder-cols", type=str, default=None, help="Kommagetrennte Liste, um Spaltenreihenfolge festzulegen (genaue Namen)")

    args = ap.parse_args()

    xdf_path = Path(args.xdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Laden
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    if not streams:
        raise RuntimeError("Keine Streams in der XDF gefunden.")

    # UTC Offset
    offset = compute_utc_offset(streams, xdf_path, args.utc_start)

    # Marker-Stream finden + Marker-CSV + Segmente
    marker_stream = pick_stream_by_name_type(streams, args.marker_name, args.marker_type, type_fallback="Markers")
    marker_stream_id = id(marker_stream) if marker_stream is not None else None
    marker_segs: List[Tuple[float, float, str]] = []
    marker_col_name = "marker_event"
    if marker_stream is not None:
        mts, mlab = extract_markers(marker_stream)
        # Marker-CSV (nur wenn non-empty)
        try:
            mcols = ["timestamp", "utc_epoch", "utc_iso", "marker"]
            mdf = pd.DataFrame({
                "timestamp": mts,
                "utc_epoch": mts + offset,
                "utc_iso":   [epoch_to_iso(t + offset) for t in mts],
                "marker":    mlab
            })
            if args.clean_headers:
                mdf.columns = sanitize_headers(list(mdf.columns))
            enc = "utf-8-sig" if args.bom and args.encoding.lower().startswith("utf-8") else args.encoding
            mdf.to_csv(outdir / "markers.csv", index=False,
                       sep=args.sep, encoding=enc, lineterminator=args.lineterm, decimal=args.decimal)
        except Exception as e:
            sys.stderr.write(f"Warnung: Marker-CSV konnte nicht geschrieben werden: {e}\n")
        marker_segs = segments_from_markers(mts, mlab)
    else:
        sys.stderr.write("Hinweis: Kein Marker-Stream gefunden.\n")

    # Pro Stream CSVs
    per_paths = []
    for i, s in enumerate(streams):
        # Skip marker stream here (handled separately)
        if marker_stream is not None and id(s) == marker_stream_id:
            continue
        ts = np.asarray(s.get("time_stamps", []), float)
        X_raw = s.get("time_series", [])
        # Try numeric cast; if it fails (e.g., strings), skip exporting this stream here
        try:
            X = np.asarray(X_raw, float)
        except Exception:
            sys.stderr.write(f"Überspringe nicht-numerischen Stream: {stream_basename(s)} (kein Float-Array)\n")
            continue

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        labs_ch = channel_labels(s)[: X.shape[1]]
        df = pd.DataFrame(X, columns=labs_ch)
        df.insert(0, "timestamp", ts)
        df.insert(1, "utc_epoch", df["timestamp"] + offset)
        df.insert(2, "utc_iso", [epoch_to_iso(tu) for tu in df["utc_epoch"]])
        df = add_labels_to_df(df, marker_segs)

        # Header-Cleanup
        if args.clean_headers:
            df.columns = sanitize_headers(list(df.columns))

        # Schreiben
        base = stream_basename(s)
        enc = "utf-8-sig" if args.bom and args.encoding.lower().startswith("utf-8") else args.encoding

        if args.split_channels:
            for j, ch_name in enumerate(df.columns[3:-1], start=0):  # nur Kanäle
                ch = df[["timestamp", "utc_epoch", "utc_iso", df.columns[3+j], "label"]]
                out = outdir / f"{i:02d}_{base}__{sanitize_headers([df.columns[3+j]])[0] if args.clean_headers else df.columns[3+j]}.csv"
                ch.to_csv(out, index=False, sep=args.sep, encoding=enc, lineterminator=args.lineterm, decimal=args.decimal)
                per_paths.append(out)
        else:
            out = outdir / f"{i:02d}_{base}.csv"
            df.to_csv(out, index=False, sep=args.sep, encoding=enc, lineterminator=args.lineterm, decimal=args.decimal)
            per_paths.append(out)

    # merged_all_streams.csv
    lengths = [len(s.get("time_stamps", [])) for s in streams]
    base_idx = int(np.argmax(lengths)) if lengths else 0
    base_s = streams[base_idx]
    base_ts = np.asarray(base_s.get("time_stamps", []), float)
    base_X = np.asarray(base_s.get("time_series", []), float)
    if base_X.ndim == 1:
        base_X = base_X.reshape(-1, 1)
    base_cols = [f"{stream_basename(base_s)}__{c}" for c in channel_labels(base_s)[: base_X.shape[1]]]

    merged = pd.DataFrame(base_X, columns=base_cols)
    merged.insert(0, "timestamp", base_ts)
    merged.insert(1, "utc_epoch", merged["timestamp"] + offset)
    merged.insert(2, "utc_iso", [epoch_to_iso(tu) for tu in merged["utc_epoch"]])
    merged["label"] = ""

    # andere Streams einmischen (asof)
    tol = args.tolerance
    for k, s in enumerate(streams):
        # Skip marker stream in merge (marker_event wird separat hinzugefügt)
        if marker_stream is not None and id(s) == marker_stream_id:
            continue
        if k == base_idx:
            continue
        ts = np.asarray(s.get("time_stamps", []), float)
        X_raw = s.get("time_series", [])
        # Try numeric cast; if it fails (e.g., strings), skip exporting this stream here
        try:
            X = np.asarray(X_raw, float)
        except Exception:
            sys.stderr.write(f"Überspringe nicht-numerischen Stream: {stream_basename(s)} (kein Float-Array)\n")
            continue
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = [f"{stream_basename(s)}__{c}" for c in channel_labels(s)[: X.shape[1]]]
        tmp = pd.DataFrame(X, columns=cols)
        tmp["timestamp"] = ts
        tmp = tmp.sort_values("timestamp")
        merged = pd.merge_asof(merged.sort_values("timestamp"),
                               tmp.sort_values("timestamp"),
                               on="timestamp", direction="nearest", tolerance=tol)

    # Marker-Events als Spalte
    if marker_segs:
        mts = np.array([s[0] for s in marker_segs], float)
        mlab = np.array([s[2] for s in marker_segs], object)
        mdf = pd.DataFrame({"timestamp": mts, marker_col_name: mlab}).sort_values("timestamp")
        merged = pd.merge_asof(merged.sort_values("timestamp"),
                               mdf,
                               on="timestamp", direction="backward")
    else:
        merged[marker_col_name] = ""

    # Label (Segment) in merged
    merged = add_labels_to_df(merged, marker_segs)

    # IBI/HR aus PPG/BVP
    ppg_stream = pick_stream_by_name_type(streams, args.ppg_name, args.ppg_type)
    if ppg_stream is not None:
        ts = np.asarray(ppg_stream.get("time_stamps", []), float)
        X = np.asarray(ppg_stream.get("time_series", []), float)
        if X.ndim == 1:
            sig = X
        else:
            # bevorzugt ersten Kanal
            sig = X[:, 0]

        peaks_t = detect_peaks(ts, sig, min_ibi_s=0.3, win=11, thr_k=0.5)
        t_mid, ibi, hr = ibi_hr_from_peaks(peaks_t)

        ppg_base = stream_basename(ppg_stream)
        ibi_df = pd.DataFrame({
            "timestamp": t_mid,
            "utc_epoch": t_mid + offset,
            "utc_iso":   [epoch_to_iso(t + offset) for t in t_mid],
            "ibi_s":     ibi,
            "hr_bpm":    hr,
        })
        if args.clean_headers:
            ibi_df.columns = sanitize_headers(list(ibi_df.columns))

        enc = "utf-8-sig" if args.bom and args.encoding.lower().startswith("utf-8") else args.encoding
        ibi_out = outdir / f"{ppg_base}__IBI_HR.csv"
        ibi_df.to_csv(ibi_out, index=False, sep=args.sep, encoding=enc, lineterminator=args.lineterm, decimal=args.decimal)
        print(f"IBI/HR geschrieben: {ibi_out}")

        if t_mid.size:
            add_df = pd.DataFrame({
                "timestamp": t_mid,
                f"{ppg_base}__ibi_s": ibi,
                f"{ppg_base}__hr_bpm": hr
            }).sort_values("timestamp")
            merged = pd.merge_asof(merged, add_df, on="timestamp", direction="nearest", tolerance=tol)
    else:
        sys.stderr.write("Hinweis: Kein PPG/BVP-Stream für IBI gefunden (nutze --ppg-name/--ppg-type).\n")

    # Spaltensteuerung (optional) für merged
    if args.list_columns:
        print("Spalten in merged_all_streams:")
        for c in merged.columns:
            print(c)
        return

    # drop/keep via regex
    if args.keep_cols:
        pats = [p.strip() for p in args.keep_cols.split(",") if p.strip()]
        keep_cols = []
        for c in merged.columns:
            for p in pats:
                if pd.Series([c]).str.contains(p, regex=True).iat[0]:
                    keep_cols.append(c)
                    break
        if keep_cols:
            merged = merged.loc[:, list(dict.fromkeys(keep_cols))]  # uniq & order
    if args.drop_cols:
        pats = [p.strip() for p in args.drop_cols.split(",") if p.strip()]
        drop_cols = set()
        for c in merged.columns:
            for p in pats:
                if pd.Series([c]).str.contains(p, regex=True).iat[0]:
                    drop_cols.add(c)
                    break
        merged = merged.drop(columns=list(drop_cols), errors="ignore")

    if args.reorder_cols:
        order = [c.strip() for c in args.reorder_cols.split(",") if c.strip()]
        # Bringe bekannte zuerst, Rest hinten dran
        known = [c for c in order if c in merged.columns]
        rest = [c for c in merged.columns if c not in known]
        merged = merged.loc[:, known + rest]

    # Header-Cleanup
    if args.clean_headers:
        merged.columns = sanitize_headers(list(merged.columns))

    # Schreiben merged
    enc = "utf-8-sig" if args.bom and args.encoding.lower().startswith("utf-8") else args.encoding
    merged_out = outdir / "merged_all_streams.csv"
    merged.to_csv(merged_out, index=False, sep=args.sep, encoding=enc, lineterminator=args.lineterm, decimal=args.decimal)

    # XLSX optional
    if args.excel:
        try:
            with pd.ExcelWriter(outdir / "merged_all_streams.xlsx", engine="openpyxl") as xw:
                merged.to_excel(xw, index=False, sheet_name=args.sheet_name)
        except Exception as e:
            sys.stderr.write(f"Warnung: XLSX konnte nicht geschrieben werden: {e}\n")

    # Ausgabe
    print("\nFertig.")
    print("Einzel-CSV(s):")
    for p in per_paths:
        print(" -", p)
    print("Merged CSV :", merged_out)
    print(f"UTC-Offset verwendet: {offset:.3f} s",
          "(aus Datei-mtime geschätzt)" if args.utc_start is None else "(aus --utc-start)")

if __name__ == "__main__":
    main()
