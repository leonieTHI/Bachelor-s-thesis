#!/usr/bin/env python3
"""
EmotiBit pipeline: compute HR and HRV (RMSSD) from raw green PPG, merge with EDA, Temperature,
UTC timestamps and labels, save a tidy CSV (Excel-friendly if desired), and produce fixed-scale plots with label shading.

Uses HeartPy (python-heart-rate-analysis-toolkit) and Matplotlib.

Example (sampling rate will be **estimated automatically** from timestamps)
----------------------------------------------------------------------------
python emotibit_hr_hrv_pipeline.py \
  --input /path/to/merged_all_streams.csv \
  --output /path/to/processed_timeseries.csv \
  --ppg-col PPG_GRN__PPGGreen__ch_1 \
  --eda-col EDA__EDA__ch_1 \
  --temp-col TEMP1__Temperature__ch_1 \
  --time-col utc_iso \
  --label-col label \
  --hrv-window-seconds 60 \
  --segment-overlap-seconds 30 \
  --plot-dir ./plots \
  --ppg-ylim 9000 11000 \
  --eda-ylim 0 0.5 \
  --temp-ylim 35 38 \
  --out-sep ';' --out-decimal ',' --out-encoding utf-8-sig --out-lineterm "\r\n"

Notes
-----
* If you don't know the sampling rate, simply **omit --fs** and it will be inferred from the timestamp column.
* HR and RMSSD are computed segment-wise and then assigned to every sample that falls into that segment (so your
  per-sample rows will include the segment's HR and RMSSD values).
* If HeartPy fails on a segment (e.g., too noisy), the HR/RMSSD will be NaN for that segment.


Requirements
------------
python -m pip install heartpy pandas matplotlib numpy
# For optional Excel export: python -m pip install openpyxl
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import heartpy as hp
except Exception as e:
    raise RuntimeError("HeartPy is required: pip install heartpy") from e


def read_csv_robust(path: str, delimiter: Optional[str] = None) -> pd.DataFrame:
    
    encodings = ['utf-8-sig', 'utf-8']
    if delimiter is not None and delimiter.lower() == 'tab':
        delimiter = '\t'
    last_err = None
    for enc in encodings:
        try:
            if delimiter:
                return pd.read_csv(path, sep=delimiter, engine='python', encoding=enc, skipinitialspace=True)
            # try auto-detect
            return pd.read_csv(path, sep=None, engine='python', encoding=enc, skipinitialspace=True)
        except Exception as e:
            last_err = e
            # fall through to brute-force delimiters
            for sep in [',',';','\t','|']:
                try:
                    return pd.read_csv(path, sep=sep, engine='python', encoding=enc, skipinitialspace=True)
                except Exception as e2:
                    last_err = e2
            continue
    raise last_err


def parse_args():
    p = argparse.ArgumentParser(description='EmotiBit HR/HRV (RMSSD) pipeline using HeartPy + Matplotlib')
    p.add_argument('--input', required=True, help='Input CSV file (merged streams)')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--ppg-col', default='PPG_Green', help='Column name for raw green PPG')
    p.add_argument('--eda-col', default='EDA', help='Column name for EDA')
    p.add_argument('--temp-col', default='TempC', help='Column name for temperature (°C)')
    p.add_argument('--time-col', default='utc_iso', help='Timestamp column (ISO8601 or epoch seconds)')
    p.add_argument('--label-col', default='label', help='Label column name (optional)')

    p.add_argument('--fs', type=float, default=None, help='Sampling rate in Hz (if omitted, inferred from time)')
    p.add_argument('--hrv-window-seconds', type=float, default=60.0, help='Segment/window length for HRV/HR (s)')
    p.add_argument('--segment-overlap-seconds', type=float, default=0.0, help='Segment overlap (s)')

    p.add_argument('--plot-dir', default=None, help='Directory to save plots (if omitted, plots are shown interactively)')

    # CSV parsing
    p.add_argument('--delimiter', default=None, help='CSV delimiter (auto-detect if omitted). Examples: "," ";" "\\t"')

    # Fixed scale limits for plots(min max)
    p.add_argument('--ppg-ylim', nargs=2, type=float, default=None, metavar=('MIN','MAX'))
    p.add_argument('--eda-ylim', nargs=2, type=float, default=None, metavar=('MIN','MAX'))
    p.add_argument('--temp-ylim', nargs=2, type=float, default=None, metavar=('MIN','MAX'))
    p.add_argument('--hr-ylim', nargs=2, type=float, default=(40, 180), metavar=('MIN','MAX'))
    p.add_argument('--rmssd-ylim', nargs=2, type=float, default=(0, 150), metavar=('MIN','MAX'))

    
    p.add_argument('--out-sep', default=';', help='Output CSV separator (e.g., ";" for German Excel)')
    p.add_argument('--out-decimal', default=',', help='Output decimal mark ("," for German Excel, "." otherwise)')
    p.add_argument('--out-encoding', default='utf-8-sig', help='Output text encoding (utf-8-sig recommended for Excel)')
    p.add_argument('--out-lineterm', default='\r\n', help='Line terminator for CSV (Windows-friendly default)')
    p.add_argument('--excel', action='store_true', help='Also write an .xlsx alongside the CSV')
    p.add_argument('--excel-sheet', default='timeseries', help='Worksheet name if --excel is set')

    # Optional: Header cleanup
    p.add_argument('--clean-headers', action='store_true', help='Remove commas/semicolons/tabs from column names')

    return p.parse_args()


def to_datetime_series(s: pd.Series) -> pd.Series:
    """Coerce a variety of timestamp formats to timezone-aware pandas datetimes (UTC)."""
    dt = pd.to_datetime(s, errors='coerce', utc=True)
    if dt.isna().any():
        # If many NaT, try epoch seconds then ms
        if dt.isna().mean() > 0.5:
            try:
                dt = pd.to_datetime(s.astype(float), unit='s', errors='coerce', utc=True)
            except Exception:
                pass
            if dt.isna().mean() > 0.5:
                try:
                    dt = pd.to_datetime(s.astype(float), unit='ms', errors='coerce', utc=True)
                except Exception:
                    pass
        dt = dt.ffill().bfill()
    return dt


def infer_fs_from_time(t: pd.Series) -> float:
    diffs = np.diff(t.view(np.int64)) / 1e9
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError('Cannot infer sampling rate from timestamps (insufficient or invalid data).')
    median_dt = np.median(diffs)
    fs = 1.0 / median_dt
    return float(fs)


def infer_fs_diagnostics(t: pd.Series) -> dict:
    diffs = np.diff(t.view(np.int64)) / 1e9
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.size == 0:
        return {"error": "no positive diffs"}
    median_dt = float(np.median(diffs))
    mean_dt = float(np.mean(diffs))
    std_dt = float(np.std(diffs))
    hist, bin_edges = np.histogram(diffs, bins=min(100, max(10, int(np.sqrt(diffs.size)))))
    mode_dt = float((bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2.0)
    return {
        'median_dt_s': median_dt,
        'mean_dt_s': mean_dt,
        'std_dt_s': std_dt,
        'fs_median_hz': 1.0/median_dt,
        'fs_mean_hz': 1.0/mean_dt,
        'fs_mode_hz': 1.0/mode_dt,
        'pct_large_gaps_gt2x_median': float(100.0 * np.mean(diffs > (2*median_dt)))
    }


def robust_ylim(y: pd.Series) -> Tuple[float, float]:
    y_clean = pd.to_numeric(y, errors='coerce')
    y_clean = y_clean[np.isfinite(y_clean)]
    if y_clean.empty:
        return (0.0, 1.0)
    q1, q99 = np.percentile(y_clean, [1, 99])
    pad = 0.05 * (q99 - q1 if q99 > q1 else max(abs(q1), 1.0))
    return (q1 - pad, q99 + pad)


def coerce_numeric_locale(s: pd.Series) -> pd.Series:
    """Convert strings with decimal comma to float. Handles thousands '.', decimal ','."""
    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.notna().mean() >= 0.8:
        return s_num
    s_str = s.astype(str)
    has_comma = s_str.str.contains(',', regex=False, na=False).mean() > 0.2
    if has_comma:
        s_fix = s_str.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        s_num2 = pd.to_numeric(s_fix, errors='coerce')
        if s_num2.notna().mean() > s_num.notna().mean():
            return s_num2
    return s_num


# === HR/RMSSD computation ===

def compute_segmentwise_measures(ppg: np.ndarray, fs: float, window_s: float, overlap_s: float):
    """
    Manually iterate segments, pre-filter, and run HeartPy per segment.
    Returns DataFrame with segment start/end (samples), HR_bpm, RMSSD_ms.
    """
    segment_width_samples = int(round(window_s * fs))
    if segment_width_samples <= 0:
        raise ValueError('hrv-window-seconds is too small for the sampling rate.')

    n_samples = len(ppg)
    overlap_frac = max(0.0, min(0.99, float(overlap_s / window_s))) if window_s > 0 else 0.0
    step_size_samples = int(round(segment_width_samples * (1.0 - overlap_frac)))
    if step_size_samples <= 0:
        step_size_samples = segment_width_samples

    data = np.asarray(ppg, dtype=float)

    if not np.any(np.isfinite(data)):
        print('[WARN] PPG data has no finite values. Skipping HR/HRV.')
        return pd.DataFrame()

    # interpolate NaNs across the whole series first
    data_series = pd.Series(data)
    data_interpolated = data_series.interpolate(method='linear', limit_direction='both').values

    finite_data = data[np.isfinite(data)]
    replacement_value = np.mean(finite_data) if finite_data.size > 0 else 0.0
    data_clean = np.nan_to_num(data_interpolated, nan=replacement_value)

    # light bandpass around plausible HR (0.8–3.0 Hz ~ 48–180 bpm)
    try:
        filtered_data = hp.filter_signal(data_clean, cutoff=[0.8, 3.0], sample_rate=fs, filtertype='bandpass', order=3)
    except Exception as e:
        print(f'[ERROR] HeartPy filter failed: {e}')
        return pd.DataFrame()

    measures_by_segment = []

    for seg_idx, start_sample in enumerate(range(0, n_samples, step_size_samples)):
        end_sample = min(n_samples, start_sample + segment_width_samples)
        if end_sample - start_sample < segment_width_samples:
            continue
        segment_data = filtered_data[start_sample:end_sample]
        #try:
        wd, m = hp.process(segment_data, sample_rate=fs)
        #except Exception:
            #wd, m = {}, {}
        bpm = m.get('bpm', np.nan)
        rmssd = m.get('rmssd', np.nan)
        measures_by_segment.append({
            'segment_index': seg_idx,
            'start_sample': int(start_sample),
            'end_sample': int(end_sample),
            'HR_bpm': float(bpm) if bpm is not None else np.nan,
            'RMSSD_ms': float(rmssd) if rmssd is not None else np.nan
        })

    if not measures_by_segment and n_samples > 0:
        print('[WARN] HeartPy produced no valid segments. Check signal quality or --fs.')

    return pd.DataFrame(measures_by_segment)


def assign_segment_values_to_samples(seg_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    hr = np.full(n_samples, np.nan)
    rmssd = np.full(n_samples, np.nan)
    for _, row in seg_df.iterrows():
        s = max(0, int(row['start_sample']))
        e = min(n_samples, int(row['end_sample']))
        hr[s:e] = row['HR_bpm']
        rmssd[s:e] = row['RMSSD_ms']
    return pd.DataFrame({'HR_bpm': hr, 'RMSSD_ms': rmssd})


def add_label_background(ax, time_s: pd.Series, labels: pd.Series, alpha: float = 0.18):
    if labels is None or labels.isna().all():
        return
    colors = plt.cm.tab10.colors
    lab_series = labels.ffill()
    change = (lab_series != lab_series.shift()).cumsum()
    groups = lab_series.groupby(change)
    color_map = {}
    color_idx = 0
    for _, grp in groups:
        lab = grp.iloc[0]
        if pd.isna(lab):
            continue
        if lab not in color_map:
            color_map[lab] = colors[color_idx % len(colors)]
            color_idx += 1
        start_t = time_s.loc[grp.index[0]]
        end_t = time_s.loc[grp.index[-1]]
        ax.axvspan(start_t, end_t, color=color_map[lab], alpha=alpha, linewidth=0)
    handles = [plt.Line2D([0],[0], color=c, lw=6) for _, c in color_map.items()]
    ax.legend(handles, [str(k) for k in color_map.keys()], title='Labels', fontsize='x-small', loc='upper right', framealpha=0.8)


def plot_with_labels(time_s: pd.Series, y: pd.Series, title: str, ylim: Optional[Tuple[float, float]], labels: Optional[pd.Series], outpath: Optional[str]):
    plt.figure(figsize=(12, 3))
    plt.plot(time_s, y, linewidth=0.8)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(title)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(robust_ylim(y))
    add_label_background(plt.gca(), time_s, labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
        plt.close()
    else:
        plt.show()


def main():
    args = parse_args()

    # Load data (robust to delimiters like ';' or tabs)
    df = read_csv_robust(args.input, args.delimiter)
    if args.time_col not in df.columns:
        raise KeyError(f"time column '{args.time_col}' not found. Available: {list(df.columns)}")
    if args.ppg_col not in df.columns:
        raise KeyError(f"ppg column '{args.ppg_col}' not found. Available: {list(df.columns)}")
    for opt_col in [args.eda_col, args.temp_col, args.label_col]:
        if opt_col not in df.columns:
            print(f"[WARN] optional column '{opt_col}' not found in input CSV.")

    # Prepare time
    ts = to_datetime_series(df[args.time_col])
    df['_utc_dt'] = ts
    df['time_s'] = (ts - ts.iloc[0]).dt.total_seconds()

    # Determine sampling rate (estimate if not provided)
    if args.fs is None:
        fs = infer_fs_from_time(ts)
        diag = infer_fs_diagnostics(ts)
        print(f"[FS] Estimated sampling rate: {fs:.3f} Hz | details: {diag}")
    else:
        fs = float(args.fs)
        print(f"[FS] Using provided sampling rate: {fs:.3f} Hz")

    # Coerce numeric columns (handle decimal comma)
    for col in [args.ppg_col, args.eda_col, args.temp_col]:
        if col in df.columns:
            df[col] = coerce_numeric_locale(df[col])

    # HR & RMSSD
    seg_df = compute_segmentwise_measures(df[args.ppg_col].values, fs, args.hrv_window_seconds, args.segment_overlap_seconds)
    per_sample_hrv = assign_segment_values_to_samples(seg_df, len(df)) if not seg_df.empty else pd.DataFrame({'HR_bpm': np.full(len(df), np.nan), 'RMSSD_ms': np.full(len(df), np.nan)})

    # Output columns
    out_cols = {
        'utc_iso': df[args.time_col],
        'PPG_Raw': df[args.ppg_col],
    }
    if args.eda_col in df.columns:
        out_cols['EDA'] = df[args.eda_col]
    if args.temp_col in df.columns:
        out_cols['Temperature'] = df[args.temp_col]
    out_cols['HR_bpm'] = per_sample_hrv['HR_bpm']
    out_cols['RMSSD_ms'] = per_sample_hrv['RMSSD_ms']
    if args.label_col in df.columns:
        out_cols['label'] = df[args.label_col]

    out_df = pd.DataFrame(out_cols)

    if args.clean_headers:
        out_df.columns = [str(c).replace('\t','_').replace(';','_').replace(',','_') for c in out_df.columns]

    # ==== WRITE CSV (Excel-friendly if desired) ====
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_csv_kwargs = dict(index=False, sep=args.out_sep, decimal=args.out_decimal, encoding=args.out_encoding, lineterminator=args.out_lineterm)
    out_df.to_csv(args.output, **out_csv_kwargs)
    print(f"Saved merged timeseries to: {args.output}")

    # Optional: also write XLSX
    if args.excel:
        try:
            xlsx_path = os.path.splitext(args.output)[0] + '.xlsx'
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                out_df.to_excel(writer, sheet_name=args.excel_sheet, index=False)
            print(f"Saved Excel to: {xlsx_path}")
        except Exception as e:
            print(f"[WARN] Could not write Excel file: {e}")

    # Plotting
    plot_dir = args.plot_dir
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    labels_series = df[args.label_col] if args.label_col in df.columns else pd.Series([np.nan]*len(df))

    # PPG
    plot_with_labels(df['time_s'], df[args.ppg_col], 'PPG Raw', tuple(args.ppg_ylim) if args.ppg_ylim else None,
                     labels_series, os.path.join(plot_dir, 'ppg_raw.png') if plot_dir else None)

    # EDA
    if args.eda_col in df.columns:
        plot_with_labels(df['time_s'], df[args.eda_col], 'EDA', tuple(args.eda_ylim) if args.eda_ylim else None,
                         labels_series, os.path.join(plot_dir, 'eda.png') if plot_dir else None)

    # Temperature
    if args.temp_col in df.columns:
        plot_with_labels(df['time_s'], df[args.temp_col], 'Temperature (°C)', tuple(args.temp_ylim) if args.temp_ylim else None,
                         labels_series, os.path.join(plot_dir, 'temperature.png') if plot_dir else None)

    # HR
    plot_with_labels(df['time_s'], out_df['HR_bpm'], 'HR (bpm)', tuple(args.hr_ylim) if args.hr_ylim else None,
                     labels_series, os.path.join(plot_dir, 'hr_bpm.png') if plot_dir else None)

    # RMSSD
    plot_with_labels(df['time_s'], out_df['RMSSD_ms'], 'RMSSD (ms)', tuple(args.rmssd_ylim) if args.rmssd_ylim else None,
                     labels_series, os.path.join(plot_dir, 'rmssd_ms.png') if plot_dir else None)

    print('Done.')


if __name__ == '__main__':
    main()
