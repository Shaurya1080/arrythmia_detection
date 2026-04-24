# ECG Arrhythmia Main Pipeline

This folder contains the final ECG pipeline with:
- Patient-wise data split
- CNN training and evaluation
- Classical ML baseline models (SVM, RandomForest, LogisticRegression)
- Real-time style alert simulation

## Files

- `prepare_data.py`: preprocess ECG, build beat windows, patient-wise train/val/test split, save normalization stats
- `train_cnn.py`: train 1D CNN with class weights, dropout, early stopping, LR scheduling
- `test_cnn.py`: evaluate CNN using precision/recall/F1/ROC-AUC/confusion matrix/classification report
- `train_classical_ml.py`: feature extraction + optional PCA + SVM/RF/LogReg comparison
- `compare_models.py`: generate final CNN vs ML comparison table for report
- `predict_realtime.py`: robust inference helper with saved global normalization
- `alert_system.py`: batch beat alert simulation with configurable threshold
- `live_plot_serial.py`: quick live single-channel serial waveform plotter
- `live_beats.py`: cleaned real-time ECG plot + BPM estimate from serial input
- `saving_file.py`: capture AD8232 serial signal and save as `ecg_data.csv` (`timestamp,value`)
- `sample_quick_test.py`: fast smoke test using tiny synthetic data
- `run_full_pipeline_output.py`: full pipeline with all logs + `final_comparison.md` saved under `output/<timestamp>/` (and mirrored to `output/latest/`)

## Full run with saved outputs

Runs `prepare_data` → `train_cnn` → `test_cnn` → `train_classical_ml` → `compare_models`, and writes:

- `output/<timestamp>/final_comparison.md`
- `output/<timestamp>/cnn_test_metrics.txt` (full CNN evaluation printout)
- `output/<timestamp>/accuracy_summary.txt` (one-line accuracy)
- `output/<timestamp>/RUN_SUMMARY.txt`
- `output/<timestamp>/log_*.txt` for each step

```bash
cd main_pipeline
set CNN_EPOCHS=10
set USE_CLASS_WEIGHTS=1
python run_full_pipeline_output.py
```

Optional: `set DATASET_PATH=mitdb` (default). The script resolves `mitdb` next to the project or under `../mitdb`.

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
python prepare_data.py
python train_cnn.py
python test_cnn.py
python train_classical_ml.py
python compare_models.py
python alert_system.py
```

## Quick sample test (fast)

Use this to validate the core pipeline quickly (without full dataset/training):

```bash
python sample_quick_test.py
```

It performs a smoke test in a temporary folder:
- creates tiny synthetic data
- builds a tiny CNN model artifact
- runs `test_cnn.py`
- runs `compare_models.py`

This is a fast sanity check, not a clinical/performance evaluation.

## Arduino Uno + AD8232 live capture/plot

1) Upload Arduino sketch that prints one AD8232 sample per line over serial.
   Expected format: `value` (integer), one sample per line.

2) Save incoming live stream to CSV:

```bash
python saving_file.py
```

This writes `ecg_data.csv` with columns:
- `timestamp,value`

3) View cleaned live ECG + BPM from serial:

```bash
python live_beats.py --port COM6 --baud 115200 --fs 200 --window 500
```

Useful flags:
- `--port` serial port (e.g. `COM6`)
- `--baud` baud rate (default in script: `115200`)
- `--fs` sampling frequency (set to your actual Arduino sampling rate)
- `--window` number of samples shown in plot
- `--batch` serial samples read per refresh

If COM/baud is different, edit constants in `saving_file.py` or pass CLI flags to `live_beats.py`.

## Config knobs (env vars)

- `ALERT_THRESHOLD` (default: `0.5`)
- `MAX_BEATS` (default: `0`, means process full record)
- `DATASET_PATH` (optional; default resolves `mitdb` near script/project root)

Example:

```bash
set ALERT_THRESHOLD=0.4
set MAX_BEATS=50
set DATASET_PATH=..\mitdb
python alert_system.py
```

## Final Comparison Section

Run:

```bash
python compare_models.py
```

It creates `final_comparison.md` with a table:

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| CNN | ... | ... | ... | ... | Best performance |
| SVM | ... | ... | ... | ... | Strong baseline |
| RandomForest | ... | ... | ... | ... | Stable |

Key insight to report:
- CNN gives best performance due to automatic feature learning.
- Classical ML is more interpretable and often faster.
- This is a practical trade-off between **performance vs interpretability**.
