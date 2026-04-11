import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


def build_tiny_cnn_model(model_path: Path):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(360, 1)),
            tf.keras.layers.Conv1D(8, 5, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x = np.random.randn(32, 360, 1).astype(np.float32)
    y = np.random.randint(0, 2, size=(32,)).astype(np.float32)
    model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    model.save(model_path)


def run_cmd(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    script_dir = Path(__file__).resolve().parent
    compare_script = script_dir / "compare_models.py"
    test_script = script_dir / "test_cnn.py"
    train_classical_module = script_dir / "train_classical_ml.py"

    if not compare_script.exists() or not test_script.exists() or not train_classical_module.exists():
        raise FileNotFoundError("Expected main pipeline scripts are missing.")

    print("Running quick smoke test in temporary folder...")
    with tempfile.TemporaryDirectory(prefix="ecg_quick_test_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "models").mkdir(exist_ok=True)

        # Tiny synthetic test set
        x_test = np.random.randn(40, 360, 1).astype(np.float32)
        y_test = np.random.randint(0, 2, size=(40,), dtype=np.int32)
        np.save(tmp_path / "X_test.npy", x_test)
        np.save(tmp_path / "y_test.npy", y_test)

        # Normalization stats required by predict_realtime.py / serial flow
        np.savez(tmp_path / "norm_stats.npz", mean=float(np.mean(x_test)), std=float(np.std(x_test) + 1e-6))

        # Build and save a tiny CNN model quickly
        build_tiny_cnn_model(tmp_path / "models" / "ecg_cnn.keras")

        # Create at least one classical model so compare_models can include ML row(s)
        sys.path.insert(0, str(script_dir))
        from train_classical_ml import build_feature_matrix

        x_feat = build_feature_matrix(x_test)
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        clf.fit(x_feat, y_test)
        joblib.dump(clf, tmp_path / "models" / "LogReg_quick.joblib")

        # 1) Run test_cnn.py on tiny artifacts
        code, out, err = run_cmd([sys.executable, str(test_script)], cwd=tmp_path)
        if code != 0:
            print("test_cnn.py failed")
            print(out)
            print(err)
            raise SystemExit(1)
        print("test_cnn.py OK")

        # 2) Run compare_models.py and check final table output
        code, out, err = run_cmd([sys.executable, str(compare_script)], cwd=tmp_path)
        if code != 0:
            print("compare_models.py failed")
            print(out)
            print(err)
            raise SystemExit(1)
        print("compare_models.py OK")

        report_path = tmp_path / "final_comparison.md"
        if not report_path.exists():
            raise FileNotFoundError("final_comparison.md was not generated.")

        print("\nQuick smoke test passed.")
        print(f"Temporary outputs created at: {tmp_path}")
        print("This test validates model loading, evaluation, and comparison flow quickly.")


if __name__ == "__main__":
    main()
