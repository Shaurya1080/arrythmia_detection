import os
import random

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_dataset_path(raw: str) -> str:
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(os.path.join(_SCRIPT_DIR, raw))
        candidates.append(os.path.join(_SCRIPT_DIR, "..", raw))
    for p in candidates:
        ap = os.path.abspath(p)
        if os.path.isdir(ap):
            return ap
    raise FileNotFoundError(
        "Dataset folder not found. Tried: " + ", ".join(os.path.abspath(c) for c in candidates)
    )


DATASET_PATH = resolve_dataset_path(os.environ.get("DATASET_PATH", "mitdb"))
WINDOW_SIZE = 180
FS = 360  # MIT-BIH sampling rate
SEED = 42

NORMAL = {"N"}
ABNORMAL = {"V", "A", "L", "R", "F", "E"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def bandpass(signal, fs=FS, low=0.5, high=40.0, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def safe_normalize(x, mean, std):
    if not np.isfinite(std) or std == 0:
        raise ValueError("Global std is invalid; cannot normalize.")
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def split_records_patient_wise(records):
    records = sorted(records)
    rng = np.random.default_rng(SEED)
    rng.shuffle(records)

    n = len(records)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    if n_train + n_val >= n:
        n_val = 1
        n_train = max(1, n - 2)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_records = set(records[:n_train])
    val_records = set(records[n_train : n_train + n_val])
    test_records = set(records[n_train + n_val :])
    return train_records, val_records, test_records


set_seed(SEED)

records = [r.split(".")[0] for r in os.listdir(DATASET_PATH) if r.endswith(".dat")]
records = sorted(list(set(records)))
if not records:
    raise RuntimeError("No MIT-BIH records found (.dat files missing).")

train_records, val_records, test_records = split_records_patient_wise(records)
print(
    f"Records split -> train: {len(train_records)} | val: {len(val_records)} | test: {len(test_records)}"
)

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

for record in records:
    signal, _ = wfdb.rdsamp(os.path.join(DATASET_PATH, record))
    annotation = wfdb.rdann(os.path.join(DATASET_PATH, record), "atr")
    ecg = bandpass(signal[:, 0])

    for i, symbol in enumerate(annotation.symbol):
        pos = annotation.sample[i]
        if pos - WINDOW_SIZE < 0 or pos + WINDOW_SIZE > len(ecg):
            continue
        if symbol not in NORMAL and symbol not in ABNORMAL:
            continue

        beat = ecg[pos - WINDOW_SIZE : pos + WINDOW_SIZE]
        label = 0 if symbol in NORMAL else 1

        if record in train_records:
            X_train.append(beat)
            y_train.append(label)
        elif record in val_records:
            X_val.append(beat)
            y_val.append(label)
        else:
            X_test.append(beat)
            y_test.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.int32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise RuntimeError("Empty split detected. Check dataset and label mapping.")

global_mean = float(np.mean(X_train))
global_std = float(np.std(X_train))

X_train = safe_normalize(X_train, global_mean, global_std).reshape(-1, 360, 1)
X_val = safe_normalize(X_val, global_mean, global_std).reshape(-1, 360, 1)
X_test = safe_normalize(X_test, global_mean, global_std).reshape(-1, 360, 1)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
np.savez("norm_stats.npz", mean=global_mean, std=global_std)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print(
    "Train class counts -> Normal:",
    int(np.sum(y_train == 0)),
    "| Abnormal:",
    int(np.sum(y_train == 1)),
)
print("Saved: X/y train-val-test and norm_stats.npz")
