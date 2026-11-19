#!/usr/bin/env python3
"""Training baseline detektor plat nomor ALPRO (Milestone 5).

Menjalankan Ultralytics YOLO training dengan parameter baseline yang disepakati,
menyimpan artefak ke models/trained/detection/<experiment_id>/ dan ringkasan ke
outputs/reports/detection/<experiment_id>/.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ALPRO YOLO plate detector baseline.")
    p.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path dataset YAML kerja")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Model/bobot awal (pretrained)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cache", action="store_true", default=False, help="Enable Ultralytics cache")
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--project", type=str, default="models/trained/detection", help="Direktori project output training")
    p.add_argument("--experiment-id", type=str, default=None, help="Nama run/experiment id")
    p.add_argument("--exist-ok", action="store_true", default=False, help="Izinkan menimpa/lanjut di direktori run yang ada")
    p.add_argument("--reports-dir", type=str, default="outputs/reports/detection", help="Direktori laporan eksperimen")
    return p.parse_args(argv)


def make_experiment_id(provided: Optional[str]) -> str:
    if provided:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in provided)
    return f"yolov8n_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass


def collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import ultralytics

        info["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as e:
        info["ultralytics"] = f"unavailable: {e}"
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as e:
        info["torch"] = f"unavailable: {e}"
    return info


def read_results_csv(run_dir: Path) -> Dict[str, Any]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    try:
        import csv

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]
        if not rows:
            return {"path": str(csv_path), "rows": 0}
        # normalize keys (ultralytics often pads names)
        def norm_row(row: Dict[str, str]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in row.items():
                nk = (k or "").strip()
                try:
                    out[nk] = float(v) if v not in (None, "") else None
                except Exception:
                    out[nk] = v
            return out

        norm_rows = [norm_row(r) for r in rows]
        last = norm_rows[-1]
        best_epoch = None
        # common metric key in ultralytics results.csv
        metric_key = None
        for cand in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50(B)", "metrics/mAP50"):
            if cand in last:
                metric_key = cand
                break
        if metric_key:
            vals = [(i, r.get(metric_key)) for i, r in enumerate(norm_rows)]
            vals = [(i, v) for i, v in vals if isinstance(v, (int, float))]
            if vals:
                bi, bv = max(vals, key=lambda x: x[1])
                best_epoch = {"index": bi, "epoch": norm_rows[bi].get("epoch", bi + 1), metric_key: bv}
        return {
            "path": str(csv_path),
            "rows": len(norm_rows),
            "metric_key_used_for_best": metric_key,
            "best_epoch": best_epoch,
            "last_row": last,
        }
    except Exception as e:
        return {"path": str(csv_path), "error": str(e)}


def copy_if_exists(src: Path, dst_dir: Path) -> Optional[str]:
    if not src.exists():
        return None
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    experiment_id = make_experiment_id(args.experiment_id)
    run_project = Path(args.project)
    report_root = Path(args.reports_dir)
    run_dir = run_project / experiment_id
    report_dir = report_root / experiment_id
    ensure_dir(run_project)
    ensure_dir(report_dir)

    print(f"[train] experiment_id={experiment_id}")
    print(f"[train] data={args.data} model={args.model} device={args.device}")
    print(f"[train] epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} seed={args.seed} patience={args.patience} workers={args.workers}")

    set_global_seeds(args.seed, args.deterministic)
    env_info = collect_env_info()

    config_snapshot = {
        "experiment_id": experiment_id,
        "timestamp": env_info["timestamp"],
        "data": args.data,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "seed": args.seed,
        "patience": args.patience,
        "workers": args.workers,
        "cache": args.cache,
        "deterministic": args.deterministic,
        "project": str(run_project),
        "name": experiment_id,
        "exist_ok": args.exist_ok,
        "decisions": {
            "6.1": "A_cpu_only",
            "6.2": "yolov8n",
            "6.3": "adopted_recommended_baseline_params",
            "6.4": "A_use_configs_dataset_yaml_as_is",
            "6.5": "A_report_val_and_test_once_no_tuning_on_test",
            "6.6": "A_organized_project_artifacts",
            "6.7": "A_visualizations_val_test_15_each",
            "6.8": "A_reusable_plate_detector_module",
            "6.9": "A_update_detector_config_and_experiment_report",
        },
        "data_notes": {
            "config": "configs/dataset.yaml",
            "split_counts": {"train": 1085, "val": 107, "test": 53},
            "cross_split_duplicates": 0,
            "intra_split_duplicate_groups": 374,
            "intra_split_duplicate_files": 1085,
            "empty_labels_provisional": 4,
        },
        "environment": env_info,
    }
    (report_dir / "run_config.json").write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        from ultralytics import YOLO
    except Exception as e:
        err = {"error": f"Gagal import ultralytics: {e}", "experiment_id": experiment_id}
        (report_dir / "training_summary.json").write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[train] ERROR: {e}")
        return 2

    try:
        model = YOLO(args.model)
        train_results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            seed=args.seed,
            patience=args.patience,
            workers=args.workers,
            cache=args.cache,
            deterministic=args.deterministic,
            project=str(run_project),
            name=experiment_id,
            exist_ok=args.exist_ok,
            plots=True,
            save=True,
            val=True,
            verbose=True,
        )
    except Exception as e:
        err = {"error": f"Training gagal: {e}", "experiment_id": experiment_id, "run_dir": str(run_dir)}
        (report_dir / "training_summary.json").write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[train] ERROR: {e}")
        return 1

    # ultralytics may adjust save_dir; prefer actual
    actual_run_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", run_dir))
    if not actual_run_dir.exists():
        actual_run_dir = run_dir

    results_summary = read_results_csv(actual_run_dir)
    weights_dir = actual_run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    copied = {}
    for fname in ["args.yaml", "results.csv", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png", "labels.jpg", "labels_correlogram.jpg"]:
        p = actual_run_dir / fname
        c = copy_if_exists(p, report_dir / "training_artifacts")
        if c:
            copied[fname] = c

    summary = {
        "experiment_id": experiment_id,
        "status": "completed",
        "run_dir": str(actual_run_dir),
        "report_dir": str(report_dir),
        "weights": {
            "best": str(best_pt) if best_pt.exists() else None,
            "last": str(last_pt) if last_pt.exists() else None,
        },
        "results": results_summary,
        "copied_artifacts": copied,
        "train_return_type": str(type(train_results)),
        "environment": env_info,
    }
    (report_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # basic markdown note (final consolidated report produced/updated by evaluate script)
    best_w = summary["weights"]["best"] or "(tidak ditemukan)"
    last_w = summary["weights"]["last"] or "(tidak ditemukan)"
    lines = [
        f"# Training Summary — {experiment_id}",
        "",
        f"- Status: completed",
        f"- Waktu: {env_info['timestamp']}",
        f"- Data: `{args.data}`",
        f"- Model awal: `{args.model}`",
        f"- Device: `{args.device}` | epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} seed={args.seed} patience={args.patience} workers={args.workers}",
        f"- Run dir: `{actual_run_dir}`",
        f"- Best weights: `{best_w}`",
        f"- Last weights: `{last_w}`",
        "",
        "## Catatan",
        "- Baseline mengikuti keputusan 6.1A (CPU-only), 6.2 (yolov8n), 6.3-6.9 (adopsi rekomendasi).",
        "- Data kerja: `configs/dataset.yaml` (split bawaan, seluruh train dalam grup duplikat intra-split, 4 label kosong provisional).",
        "- Evaluasi eksplisit val/test dan laporan konsolidasi: jalankan `scripts/evaluate_detector.py`.",
    ]
    if results_summary.get("best_epoch"):
        lines.append(f"- Best epoch (berdasarkan {results_summary.get('metric_key_used_for_best')}): {results_summary['best_epoch']}")
    (report_dir / "training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[train] COMPLETED experiment_id={experiment_id}")
    print(f"[train] run_dir={actual_run_dir}")
    print(f"[train] best={best_w}")
    print(f"[train] last={last_w}")
    print(f"[train] report_dir={report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
