# Suppress TensorFlow verbose logs (oneDNN, CPU features, etc.)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=errors only
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

from predict_realtime import predict_ecg
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.getenv("DATASET_PATH", "mitdb")
NORMAL = ['N']
ABNORMAL = ['V', 'A', 'L', 'R', 'F', 'E']
WINDOW_SIZE = 180
FS = 360
THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.5"))
MAX_BEATS = int(os.getenv("MAX_BEATS", "0"))  # 0 means full record

def bandpass(signal, fs=FS, low=0.5, high=40, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def resolve_dataset_path(path_value):
    candidates = []
    if os.path.isabs(path_value):
        candidates.append(path_value)
    else:
        candidates.append(path_value)
        candidates.append(os.path.join(BASE_DIR, path_value))
        candidates.append(os.path.join(BASE_DIR, "..", path_value))

    for p in candidates:
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"Dataset folder not found. Checked: {', '.join(os.path.abspath(c) for c in candidates)}"
    )

DATASET_PATH = resolve_dataset_path(DATASET_PATH)

# Get all records from MIT-BIH database
records = [r.split('.')[0] for r in os.listdir(DATASET_PATH) if '.dat' in r]
records = list(set(records))
if not records:
    raise RuntimeError(f"No records found in dataset path: {DATASET_PATH}")

# Pick a random record
record = records[np.random.randint(0, len(records))]

# Load raw ECG signal and annotations from MIT-BIH
signal, fields = wfdb.rdsamp(os.path.join(DATASET_PATH, record))
annotation = wfdb.rdann(os.path.join(DATASET_PATH, record), 'atr')

ecg = signal[:, 0]  # Channel 0 (MLII)
ecg = bandpass(ecg)  # Match training preprocessing

# Collect valid beat segments with labels
beats = []
labels = []
for i, symbol in enumerate(annotation.symbol):
    pos = annotation.sample[i]
    if pos - WINDOW_SIZE < 0 or pos + WINDOW_SIZE > len(ecg):
        continue
    if symbol in NORMAL:
        beats.append(ecg[pos - WINDOW_SIZE:pos + WINDOW_SIZE])
        labels.append(0)
    elif symbol in ABNORMAL:
        beats.append(ecg[pos - WINDOW_SIZE:pos + WINDOW_SIZE])
        labels.append(1)

if not beats:
    raise RuntimeError(f"No valid beats found in record {record}.")

total_beats = len(beats)
n = total_beats if MAX_BEATS <= 0 else min(MAX_BEATS, total_beats)
preds = []
probs = []
for i in range(n):
    p = float(predict_ecg(np.array(beats[i])))
    probs.append(p)
    preds.append(1 if p >= THRESHOLD else 0)
    msg = "ALERT: Abnormal ECG Detected" if p >= THRESHOLD else "Normal ECG"
    print(f"{i+1:03d} | {msg} | prob={p:.3f} | actual={'Abnormal' if labels[i] == 1 else 'Normal'}")

preds = np.array(preds, dtype=np.int32)
truth = np.array(labels[:n], dtype=np.int32)
acc = float(np.mean(preds == truth))
pred_abnormal = int(np.sum(preds == 1))
actual_abnormal = int(np.sum(truth == 1))
print("\n=== Alert Summary ===")
print(f"Record: {record}")
print(f"Threshold: {THRESHOLD:.2f}")
print(f"Total valid beats in record: {total_beats}")
print(f"Processed beats: {n}")
print(f"Predicted abnormal beats: {pred_abnormal}")
print(f"Actual abnormal beats: {actual_abnormal}")
print(f"Accuracy over processed beats: {acc:.3f}")
