"""
Run the full ECG pipeline and save logs + final_comparison.md under output/<timestamp>/.

Also mirrors key files to output/latest/ for convenience.

Usage (from main_pipeline):
  python run_full_pipeline_output.py

Optional env:
  DATASET_PATH   - default: mitdb (resolved vs script dir and ../mitdb)
  CNN_EPOCHS     - default: 10
  USE_CLASS_WEIGHTS - default: 1
  SVM_MAX_SAMPLES - default: 10000 (train_classical_ml: stratified subset for RBF SVM only)
  PYTHON         - interpreter path (default: sys.executable)
"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main() -> int:
    os.chdir(BASE)
    py = os.environ.get("PYTHON", sys.executable)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE / "output" / stamp
    latest = BASE / "output" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    latest.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("DATASET_PATH", "mitdb")
    env.setdefault("CNN_EPOCHS", "10")
    env.setdefault("USE_CLASS_WEIGHTS", "1")
    env.setdefault("SVM_MAX_SAMPLES", "10000")
    env["COMPARISON_MD"] = str(out_dir / "final_comparison.md")

    summary_lines: list[str] = [
        f"Run started: {stamp}",
        f"Output directory: {out_dir}",
        f"DATASET_PATH={env['DATASET_PATH']}",
        f"CNN_EPOCHS={env['CNN_EPOCHS']}",
        f"USE_CLASS_WEIGHTS={env['USE_CLASS_WEIGHTS']}",
        f"SVM_MAX_SAMPLES={env['SVM_MAX_SAMPLES']}",
        "",
    ]

    steps = [
        ("prepare_data.py", out_dir / "log_prepare_data.txt"),
        ("train_cnn.py", out_dir / "log_train_cnn.txt"),
        ("test_cnn.py", out_dir / "cnn_test_metrics.txt"),
        ("train_classical_ml.py", out_dir / "log_train_classical_ml.txt"),
        ("compare_models.py", out_dir / "log_compare_models.txt"),
    ]

    for script, log_path in steps:
        cmd = [py, str(BASE / script)]
        summary_lines.append(f"--- {script} ---")
        proc = subprocess.run(
            cmd,
            cwd=str(BASE),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log_text = (proc.stdout or "") + (proc.stderr or "")
        log_path.write_text(log_text, encoding="utf-8")
        summary_lines.append(f"exit_code={proc.returncode} log={log_path.name}")
        if proc.returncode != 0:
            summary_lines.append("FAILED — see log file above.")
            (out_dir / "RUN_SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
            _mirror_to_latest(out_dir, latest)
            print(f"Pipeline failed at {script}. See {log_path}")
            return proc.returncode

    # Short accuracy file for reports
    metrics_path = out_dir / "cnn_test_metrics.txt"
    acc_line = _extract_accuracy(metrics_path.read_text(encoding="utf-8"))
    acc_file = out_dir / "accuracy_summary.txt"
    acc_file.write_text(
        acc_line + "\n\nFull metrics in cnn_test_metrics.txt\n",
        encoding="utf-8",
    )
    summary_lines.extend(["", "--- Summary ---", acc_line, "", "Done."])

    (out_dir / "RUN_SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    _mirror_to_latest(out_dir, latest)

    print(f"Saved run under: {out_dir}")
    print(f"Mirrored to: {latest}")
    print(acc_line)
    return 0


def _extract_accuracy(text: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith("accuracy:"):
            return line.strip()
    return "Accuracy: (see cnn_test_metrics.txt)"


def _mirror_to_latest(src: Path, latest: Path) -> None:
    for p in latest.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    for item in src.iterdir():
        dest = latest / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest)


if __name__ == "__main__":
    raise SystemExit(main())
