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

## Config knobs (env vars)

- `ALERT_THRESHOLD` (default: `0.5`)
- `MAX_BEATS` (default: `0`, means process full record)

Example:

```bash
set ALERT_THRESHOLD=0.4
set MAX_BEATS=50
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
