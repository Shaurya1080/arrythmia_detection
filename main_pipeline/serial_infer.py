import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
from scipy.signal import butter, filtfilt

from predict_realtime import predict_ecg

FS = 360
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
    parser.add_argument("--no-filter", action="store_true", help="Disable bandpass filter")
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
        print(f"Reading CSV {path} | derived lead={args.csv_lead.upper()} ...")
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
                    segment = bandpass(segment)
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
