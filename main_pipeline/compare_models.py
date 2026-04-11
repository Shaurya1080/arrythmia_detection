import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train_classical_ml import build_feature_matrix


def binary_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def format_pct(x):
    return f"{100.0 * float(x):.2f}%"


def main():
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    rows = []

    # CNN
    cnn = tf.keras.models.load_model("models/ecg_cnn.keras")
    y_prob = cnn.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    m = binary_metrics(y_test, y_pred)
    rows.append(
        {
            "model": "CNN",
            **m,
            "notes": "Best performance, automatic feature learning",
        }
    )

    # Classical ML models
    X_feat = build_feature_matrix(X_test)
    for fname in sorted(os.listdir("models")):
        if not fname.endswith(".joblib"):
            continue
        path = os.path.join("models", fname)
        model = joblib.load(path)
        pred = model.predict(X_feat)
        m = binary_metrics(y_test, pred)
        name = fname.replace(".joblib", "")
        notes = "Strong baseline"
        if "RandomForest" in name:
            notes = "Stable baseline"
        if "LogReg" in name:
            notes = "Interpretable and fast"
        rows.append({"model": name, **m, "notes": notes})

    # Print markdown table for report
    header = "| Model | Accuracy | Precision | Recall | F1 | Notes |"
    sep = "|---|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {format_pct(r['accuracy'])} | {format_pct(r['precision'])} | "
            f"{format_pct(r['recall'])} | {format_pct(r['f1'])} | {r['notes']} |"
        )

    md = "\n".join(lines)
    print("\n=== Final Comparison Table ===")
    print(md)

    out_md = os.environ.get("COMPARISON_MD", "final_comparison.md")
    _parent = os.path.dirname(os.path.abspath(out_md))
    if _parent:
        os.makedirs(_parent, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Final Model Comparison\n\n")
        f.write(md)
        f.write(
            "\n\n## Key Insight\n"
            "- CNN: best performance due to automatic feature learning.\n"
            "- Classical ML: more interpretable and often faster.\n"
            "- Practical trade-off: **performance vs interpretability**.\n"
        )

    print(f"\nSaved: {out_md}")


if __name__ == "__main__":
    main()
