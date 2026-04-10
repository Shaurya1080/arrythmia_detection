import argparse
from collections import deque
from datetime import datetime

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


def main():
    parser = argparse.ArgumentParser(description="Live ECG inference from ESP32 serial stream.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Abnormal alert threshold")
    parser.add_argument("--window", type=int, default=WINDOW, help="Window size in samples")
    parser.add_argument("--hop", type=int, default=180, help="Hop size between predictions")
    parser.add_argument("--no-filter", action="store_true", help="Disable bandpass filter")
    args = parser.parse_args()

    if args.hop < 1 or args.hop > args.window:
        raise ValueError("hop must be in range [1, window]")

    print(f"Opening serial {args.port} @ {args.baud} ...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    buf = deque(maxlen=args.window)
    sample_count = 0
    infer_count = 0

    print("Streaming... Press Ctrl+C to stop.")
    try:
        while True:
            line = ser.readline().decode(errors="ignore")
            try:
                sample = parse_sample(line)
            except ValueError:
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
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        print("Serial closed.")


if __name__ == "__main__":
    main()
