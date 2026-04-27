import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
from scipy.signal import butter, filtfilt, find_peaks

from predict_realtime import predict_ecg

FS = 360
CSV_DEFAULT_FS = 100
WINDOW = 360


def bandpass(signal, fs=FS, low=0.5, high=40.0, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def parse_sample(line):
    # Supports both "1234" and "timestamp,1234" style serial output.
    raw = line.strip()
    if not raw:
        return None
    parts = raw.split(",")
    token = parts[-1].strip()
    return float(token)


def parse_csv_sample(line: str, csv_lead: str) -> float | None:
    raw = line.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # Skip header line such as: timestamp,value
    if parts and any(ch.isalpha() for ch in "".join(parts)):
        return None

    # 1) Single-signal formats:
    #    - value
    #    - timestamp,value
    if len(parts) in (1, 2):
        try:
            return float(parts[-1])
        except ValueError:
            return None

    # 2) 3-lead format:
    #    - RA,LA,LL
    #    - timestamp,RA,LA,LL
    if len(parts) in (3, 4):
        try:
            if len(parts) == 3:
                ra, la, ll = map(float, parts)
            else:
                _, ra, la, ll = map(float, parts)
        except ValueError:
            return None

        lead = csv_lead.upper()
        if lead == "I":
            return la - ra
        if lead == "II":
            return ll - ra
        if lead == "III":
            return ll - la
        raise ValueError("csv-lead must be one of: I, II, III")

    return None


def parse_csv_row(line: str, csv_lead: str):
    raw = line.strip()
    if not raw:
        return None, None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts and any(ch.isalpha() for ch in "".join(parts)):
        return None, None

    # value-only row
    if len(parts) == 1:
        try:
            return None, float(parts[0])
        except ValueError:
            return None, None

    # timestamp,value row
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None, None

    # 3/4-column lead rows (with optional timestamp prefix)
    ts = None
    try:
        if len(parts) == 3:
            ra, la, ll = map(float, parts)
        elif len(parts) == 4:
            ts = float(parts[0])
            ra, la, ll = map(float, parts[1:])
        else:
            return None, None
    except ValueError:
        return None, None

    lead = csv_lead.upper()
    if lead == "I":
        return ts, la - ra
    if lead == "II":
        return ts, ll - ra
    if lead == "III":
        return ts, ll - la
    raise ValueError("csv-lead must be one of: I, II, III")


def estimate_fs_from_timestamps(timestamps, default_fs: float) -> float:
    t = np.asarray([x for x in timestamps if x is not None], dtype=np.float64)
    if t.size < 3:
        return float(default_fs)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 3:
        return float(default_fs)
    med = float(np.median(dt))
    if med <= 0:
        return float(default_fs)
    fs = 1.0 / med
    # Guard unrealistic values from noisy timestamps.
    if not np.isfinite(fs) or fs < 20 or fs > 2000:
        return float(default_fs)
    return float(fs)


def resample_to_window(x: np.ndarray, out_len: int) -> np.ndarray:
    if x.size == out_len:
        return x.astype(np.float32, copy=False)
    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num=out_len, dtype=np.float64)
    y = np.interp(dst, src, x.astype(np.float64))
    return y.astype(np.float32, copy=False)


def run_csv_beat_mode(
    csv_path: Path,
    csv_lead: str,
    threshold: float,
    window: int,
    fs_default: float,
    no_filter: bool,
    max_preds: int,
    beat_window_sec: float,
):
    timestamps = []
    values = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts, val = parse_csv_row(line, csv_lead)
            if val is None or not np.isfinite(val):
                continue
            timestamps.append(ts)
            values.append(float(val))

    if len(values) < max(200, window):
        raise RuntimeError("CSV has too few valid samples for beat-based inference.")

    signal = np.asarray(values, dtype=np.float32)
    fs = estimate_fs_from_timestamps(timestamps, fs_default)
    print(f"CSV beat mode: samples={len(signal)} estimated_fs={fs:.2f} Hz")

    if not no_filter:
        try:
            signal = bandpass(signal, fs=fs).astype(np.float32, copy=False)
        except ValueError:
            pass

    # Detect likely R-peaks on continuous signal.
    min_dist = max(1, int(0.30 * fs))
    prom = max(1e-3, 0.6 * float(np.std(signal)))
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=prom)
    if peaks.size == 0:
        raise RuntimeError("No beat peaks detected in CSV. Check signal quality/FS.")

    half = max(2, int((beat_window_sec * fs) / 2.0))
    infer_count = 0
    for p in peaks:
        a, b = p - half, p + half
        if a < 0 or b > len(signal):
            continue
        beat = signal[a:b]
        if beat.size < 10:
            continue
        segment = resample_to_window(beat, window)
        prob = float(predict_ecg(segment))
        pred = "Abnormal" if prob >= threshold else "Normal"
        infer_count += 1
        t = datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] beat#{infer_count:04d} prob={prob:.3f} -> {pred}")
        if max_preds > 0 and infer_count >= max_preds:
            break
    print(f"CSV beat mode complete. total_predictions={infer_count}")


def main():
    parser = argparse.ArgumentParser(description="Live ECG inference from ESP32 serial stream.")
    parser.add_argument("--port", help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--csv-file",
        help="Path to CSV with rows as RA,LA,LL. If set, serial port is not used.",
    )
    parser.add_argument(
        "--csv-lead",
        default="II",
        help="Lead to derive from RA/LA/LL for model input: I, II, or III (default: II).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Abnormal alert threshold")
    parser.add_argument("--window", type=int, default=WINDOW, help="Window size in samples")
    parser.add_argument("--hop", type=int, default=180, help="Hop size between predictions")
    parser.add_argument("--fs", type=float, default=FS, help="Sampling rate for filtering if timestamps are absent")
    parser.add_argument("--no-filter", action="store_true", help="Disable bandpass filter")
    parser.add_argument(
        "--csv-mode",
        choices=["beat", "rolling"],
        default="beat",
        help="CSV inference mode: beat-centered windows (recommended) or rolling windows",
    )
    parser.add_argument(
        "--beat-window-sec",
        type=float,
        default=1.0,
        help="Beat window length (seconds) before resampling to model window in csv beat mode",
    )
    parser.add_argument(
        "--max-preds",
        type=int,
        default=0,
        help="Stop after this many predictions (0 = run full stream/file).",
    )
    args = parser.parse_args()

    if args.hop < 1 or args.hop > args.window:
        raise ValueError("hop must be in range [1, window]")
    if not args.csv_file and not args.port:
        raise ValueError("Provide --port for serial input or --csv-file for file input.")

    buf = deque(maxlen=args.window)
    sample_count = 0
    infer_count = 0

    if args.csv_file:
        path = Path(args.csv_file)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        print(
            f"Reading CSV {path} | derived lead={args.csv_lead.upper()} | mode={args.csv_mode} ..."
        )
        if args.csv_mode == "beat":
            fs_default = args.fs
            if fs_default == FS:
                # For AD8232 CSV capture (timestamp,value), 100 Hz is the expected default.
                fs_default = CSV_DEFAULT_FS
                print(
                    f"CSV mode: overriding default fs to {CSV_DEFAULT_FS} Hz "
                    f"(use --fs to change)."
                )
            run_csv_beat_mode(
                csv_path=path,
                csv_lead=args.csv_lead,
                threshold=args.threshold,
                window=args.window,
                fs_default=fs_default,
                no_filter=args.no_filter,
                max_preds=args.max_preds,
                beat_window_sec=args.beat_window_sec,
            )
            return
        line_iter = path.open("r", encoding="utf-8", errors="ignore")
        mode_name = "file"
    else:
        print(f"Opening serial {args.port} @ {args.baud} ...")
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print("Streaming... Press Ctrl+C to stop.")
        line_iter = None
        mode_name = "serial"

    try:
        while True:
            if mode_name == "file":
                line = next(line_iter, "")
                if line == "":
                    break
                sample = parse_csv_sample(line, args.csv_lead)
            else:
                line = ser.readline().decode(errors="ignore")
                sample = parse_sample(line)
            try:
                if sample is not None:
                    sample = float(sample)
            except (ValueError, TypeError):
                continue
            if sample is None or not np.isfinite(sample):
                continue

            buf.append(sample)
            sample_count += 1

            if len(buf) < args.window:
                continue
            if (sample_count - args.window) % args.hop != 0:
                continue

            segment = np.array(buf, dtype=np.float32)
            if not args.no_filter:
                try:
                    segment = bandpass(segment, fs=args.fs)
                except ValueError:
                    # In case filtering becomes unstable for edge windows.
                    continue

            prob = float(predict_ecg(segment))
            pred = "Abnormal" if prob >= args.threshold else "Normal"
            infer_count += 1
            t = datetime.now().strftime("%H:%M:%S")
            print(f"[{t}] #{infer_count:04d} prob={prob:.3f} -> {pred}")
            if args.max_preds > 0 and infer_count >= args.max_preds:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if mode_name == "file":
            line_iter.close()
            print("CSV reading complete.")
        else:
            ser.close()
            print("Serial closed.")


if __name__ == "__main__":
    main()
