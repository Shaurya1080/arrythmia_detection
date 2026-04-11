import os
import random

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
USE_SMOTE = False
APPLY_PCA = [False, True]
PCA_COMPONENTS = 5
# RBF SVC does not scale to huge n; subsample for SVM only (RF/LogReg use full data).
SVM_MAX_SAMPLES = int(os.environ.get("SVM_MAX_SAMPLES", "10000"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def extract_features(x_360):
    x = np.asarray(x_360).reshape(-1)
    if x.shape[0] != 360:
        raise ValueError(f"Expected 360 samples, got {x.shape[0]}")
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = float(np.mean(x))
    std = float(np.std(x))
    peak_amp = float(np.max(np.abs(x)))
    energy = float(np.sum(x * x) / len(x))

    # Simple RR proxy: distance between two strongest peaks in first/second half.
    first_peak = int(np.argmax(x[:180]))
    second_peak = int(np.argmax(x[180:])) + 180
    rr_proxy = float(abs(second_peak - first_peak))

    return np.array([mean, std, peak_amp, energy, rr_proxy], dtype=np.float32)


def build_feature_matrix(X):
    return np.vstack([extract_features(beat) for beat in X])


def maybe_smote(X, y):
    if not USE_SMOTE:
        return X, y
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("imblearn not installed, skipping SMOTE.")
        return X, y
    sm = SMOTE(random_state=SEED)
    return sm.fit_resample(X, y)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba
        y_prob = y_pred.astype(float)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Precision:", float(precision))
    print("Recall:", float(recall))
    print("F1:", float(f1))
    print("ROC-AUC:", float(auc))
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"], zero_division=0))
    print("False Negatives:", int(cm[1, 0]))
    return model


def main():
    set_seed(SEED)
    os.makedirs("models", exist_ok=True)

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    Xtr = build_feature_matrix(X_train)
    Xte = build_feature_matrix(X_test)
    Xtr, ytr = maybe_smote(Xtr, y_train)

    print("Classical features shape:", Xtr.shape)
    print("Running models with and without PCA...")

    for use_pca in APPLY_PCA:
        tag = "with_pca" if use_pca else "no_pca"
        steps = [("scaler", StandardScaler())]
        if use_pca:
            steps.append(("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)))

        models = {
            f"SVM_{tag}": Pipeline(
                steps + [("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED))]
            ),
            f"RandomForest_{tag}": Pipeline(
                steps + [("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced"))]
            ),
            f"LogReg_{tag}": Pipeline(
                steps + [("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]
            ),
        }

        for name, model in models.items():
            X_fit, y_fit = Xtr, ytr
            if name.startswith("SVM_") and len(Xtr) > SVM_MAX_SAMPLES:
                X_fit, _, y_fit, _ = train_test_split(
                    Xtr,
                    ytr,
                    train_size=SVM_MAX_SAMPLES,
                    stratify=ytr,
                    random_state=SEED,
                )
                print(
                    f"{name}: SVM training on stratified subset n={len(X_fit)} "
                    f"(SVM_MAX_SAMPLES={SVM_MAX_SAMPLES})"
                )
            fitted = evaluate_model(name, model, X_fit, y_fit, Xte, y_test)
            joblib.dump(fitted, f"models/{name}.joblib")

    print("\nSaved classical models in models/*.joblib")


if __name__ == "__main__":
    main()
