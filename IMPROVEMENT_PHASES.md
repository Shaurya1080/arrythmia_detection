# Med Instru — 3-Phase Improvement Plan

---

## Phase 1 — Critical Fixes ✅ DONE

| # | Issue | Implementation |
|---|-------|----------------|
| 1 | **Data leakage in test_cnn.py** | `prepare_data.py` saves X_train, y_train, X_test, y_test. `test_cnn.py` loads only X_test, y_test. |
| 2 | **No bandpass filter in prepare_data.py** | 0.5–40 Hz bandpass filter added in `prepare_data.py` and `alert_system.py`. |
| 3 | **Class imbalance** | `class_weight='balanced'` in `train_cnn.py` via `compute_class_weight`. |

**Run order:** `prepare_data.py` → `train_cnn.py` → `test_cnn.py`

---

## Phase 2 — Model & Training Improvements

| # | Improvement | Action |
|---|-------------|--------|
| 4 | Early stopping | Add `EarlyStopping` and `ModelCheckpoint` in `train_cnn.py`. |
| 5 | Better metrics | Add precision, recall, F1, confusion matrix. |
| 6 | More epochs | Use 30–50 epochs with early stopping. |
| 7 | Evaluation script | Create `evaluate_model.py` with accuracy, precision, recall, F1, AUC-ROC, confusion matrix. |

---

## Phase 3 — New Features & Project Hygiene

| # | Task | Action |
|---|------|--------|
| 8 | Visualize predictions | Plot sample beats with predicted vs actual labels. |
| 9 | Multi-class classification | Classify N, V, A, L, R separately (optional). |
| 10 | Patient-level split | Split by record ID (optional). |
| 11 | Data augmentation | Add time shift, scaling, noise (optional). |
| 12 | requirements.txt | Add wfdb, numpy, scipy, scikit-learn, tensorflow, matplotlib. |
| 13 | README | Document setup, download, train, test. |
| 14 | Unify pipeline | Mark or remove `simple_version/` as legacy. |
