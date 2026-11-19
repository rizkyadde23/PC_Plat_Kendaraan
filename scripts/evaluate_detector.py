#!/usr/bin/env python3
"""Evaluasi baseline detektor plat nomor ALPRO (Milestone 5).

Menjalankan evaluasi eksplisit pada split val dan test (sekali), menyimpan metrik,
visualisasi prediksi deterministik, dan laporan eksperimen konsolidasi.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ALPRO YOLO plate detector baseline.")
    p.add_argument("--weights", type=str, default=None, help="Path ke bobot model (.pt), mis. best.pt")
    p.add_argument("--experiment-id", type=str, default=None, help="Experiment id untuk mencari best.pt dan menamai output")
    p.add_argument("--data", type=str, default="configs/dataset.yaml")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=10)
    p.add_argument("--viz-per-split", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trained-root", type=str, default="models/trained/detection")
    p.add_argument("--reports-dir", type=str, default="outputs/reports/detection")
    p.add_argument("--viz-dir", type=str, default="outputs/visualizations/detection")
    p.add_argument("--pred-dir", type=str, default="outputs/predictions/detection")
    return p.parse_args(argv)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    except Exception as e:
        info["torch"] = f"unavailable: {e}"
    try:
        import cv2 as _cv2

        info["opencv"] = _cv2.__version__
    except Exception:
        pass
    return info


def derive_experiment_id(weights: Optional[str], experiment_id: Optional[str], trained_root: Path) -> str:
    if experiment_id:
        return experiment_id
    if weights:
        wp = Path(weights)
        # .../<experiment_id>/weights/best.pt or .../<experiment_id>/best.pt
        if wp.parent.name == "weights":
            return wp.parent.parent.name
        return wp.parent.name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_weights(weights: Optional[str], experiment_id: str, trained_root: Path) -> Path:
    if weights:
        return Path(weights)
    candidate = trained_root / experiment_id / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Bobot tidak ditemukan. Berikan --weights atau pastikan ada: {candidate}"
    )


def load_data_yaml(data_path: Path) -> Dict[str, Any]:
    try:
        import yaml

        with data_path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        raise RuntimeError(f"Gagal membaca dataset yaml {data_path}: {e}")


def resolve_split_images(data_yaml_path: Path, data_cfg: Dict[str, Any], split: str) -> Path:
    # split: 'val' or 'test' -> keys in our yaml: val/test
    key = split
    if key not in data_cfg:
        # try common aliases
        if split == "val" and "valid" in data_cfg:
            key = "valid"
    val = data_cfg.get(key)
    if not val:
        raise RuntimeError(f"Entry split '{split}' tidak ditemukan di {data_yaml_path}")
    base = data_cfg.get("path")
    if base:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = (Path.cwd() / base_path).resolve()
    else:
        base_path = data_yaml_path.parent.resolve()
    img_dir = Path(str(val))
    if not img_dir.is_absolute():
        img_dir = (base_path / img_dir).resolve()
    return img_dir


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def select_deterministic(paths: Sequence[Path], n: int, seed: int) -> List[Path]:
    if n <= 0 or not paths:
        return []
    items = list(paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    picked = items[: min(n, len(items))]
    # stable output order for readability
    return sorted(picked, key=lambda p: p.name)


def extract_metrics(det_metrics: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Ultralytics DetMetrics: .box has mp, mr, map50, map, etc; .results_dict has flat keys
    try:
        rd = getattr(det_metrics, "results_dict", None)
        if isinstance(rd, dict):
            out["results_dict"] = {str(k): (float(v) if isinstance(v, (int, float)) else v) for k, v in rd.items()}
    except Exception:
        pass
    box = getattr(det_metrics, "box", None)
    if box is not None:
        # common attributes
        mapping = {
            "precision": ["mp"],
            "recall": ["mr"],
            "mAP50": ["map50"],
            "mAP50-95": ["map"],
            "mAP75": ["map75"],
        }
        for out_key, attrs in mapping.items():
            for a in attrs:
                if hasattr(box, a):
                    try:
                        val = getattr(box, a)
                        out[out_key] = float(val)
                        break
                    except Exception:
                        pass
        # also capture arrays/extra if useful
        for a in ["p", "r", "ap50", "ap", "maps", "nc"]:
            if a in out:
                continue
            if hasattr(box, a):
                try:
                    val = getattr(box, a)
                    if isinstance(val, (int, float)):
                        out[a] = float(val)
                    elif hasattr(val, "tolist"):
                        out[a] = val.tolist()
                except Exception:
                    pass
    # fallback: try top-level speed etc.
    try:
        speed = getattr(det_metrics, "speed", None)
        if isinstance(speed, dict):
            out["speed"] = {str(k): float(v) for k, v in speed.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return out


def save_metrics_csv(metrics_by_split: Dict[str, Dict[str, Any]], path: Path) -> None:
    rows = []
    for split, m in metrics_by_split.items():
        rows.append({
            "split": split,
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "mAP50": m.get("mAP50"),
            "mAP50-95": m.get("mAP50-95"),
        })
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "precision", "recall", "mAP50", "mAP50-95"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def visualize_split(
    model: Any,
    image_paths: Sequence[Path],
    out_dir: Path,
    split: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    max_det: int,
) -> List[Dict[str, Any]]:
    ensure_dir(out_dir)
    records: List[Dict[str, Any]] = []
    if not image_paths:
        return records
    # run prediction in one batch for efficiency
    try:
        results = model.predict(
            source=[str(p) for p in image_paths],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            verbose=False,
        )
    except Exception as e:
        return [{"split": split, "error": str(e)}]

    names = getattr(model, "names", {}) or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    for res, src_path in zip(results, image_paths):
        dets = []
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                cid = int(clss[i])
                dets.append({
                    "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                    "confidence": float(confs[i]),
                    "class_id": cid,
                    "class_name": names.get(cid, str(cid)),
                })
        dets.sort(key=lambda d: d["confidence"], reverse=True)

        # annotated image
        try:
            plotted = res.plot()  # BGR numpy
            out_path = out_dir / f"{src_path.stem}_pred{src_path.suffix}"
            # ensure extension is writable
            if out_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                out_path = out_path.with_suffix(".jpg")
            cv2.imwrite(str(out_path), plotted)
            vis_path = str(out_path)
        except Exception:
            vis_path = None

        records.append({
            "split": split,
            "source": str(src_path),
            "visualization": vis_path,
            "num_detections": len(dets),
            "detections": dets,
        })
    return records


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    trained_root = Path(args.trained_root)
    reports_root = Path(args.reports_dir)
    viz_root = Path(args.viz_dir)
    pred_root = Path(args.pred_dir)

    try:
        experiment_id = derive_experiment_id(args.weights, args.experiment_id, trained_root)
        weights_path = resolve_weights(args.weights, experiment_id, trained_root)
    except Exception as e:
        print(f"[eval] ERROR: {e}")
        return 2

    report_dir = reports_root / experiment_id
    viz_exp_dir = viz_root / experiment_id
    pred_exp_dir = pred_root / experiment_id
    ensure_dir(report_dir)
    ensure_dir(viz_exp_dir)
    ensure_dir(pred_exp_dir)

    print(f"[eval] experiment_id={experiment_id}")
    print(f"[eval] weights={weights_path}")
    print(f"[eval] data={args.data} device={args.device} imgsz={args.imgsz} conf={args.conf} iou={args.iou}")

    if not weights_path.exists():
        print(f"[eval] ERROR: weights tidak ditemukan: {weights_path}")
        return 2

    env_info = collect_env_info()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[eval] ERROR: gagal import ultralytics: {e}")
        return 2

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        print(f"[eval] ERROR: gagal memuat model: {e}")
        return 1

    # explicit evaluation on val and test (once)
    metrics_by_split: Dict[str, Dict[str, Any]] = {}
    eval_errors: Dict[str, str] = {}
    for split in ["val", "test"]:
        try:
            print(f"[eval] running val() on split={split}")
            m = model.val(
                data=args.data,
                split=split,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                plots=True,
                save_json=False,
                project=str(pred_exp_dir),
                name=f"val_{split}",
                exist_ok=True,
                verbose=True,
            )
            metrics_by_split[split] = extract_metrics(m)
        except Exception as e:
            eval_errors[split] = str(e)
            metrics_by_split[split] = {"error": str(e)}
            print(f"[eval] ERROR split={split}: {e}")

    save_metrics_csv(metrics_by_split, report_dir / "evaluation_metrics.csv")
    (report_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics_by_split, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # deterministic visualizations for val and test
    viz_records: Dict[str, Any] = {}
    try:
        data_cfg = load_data_yaml(Path(args.data))
        data_yaml_path = Path(args.data)
        for split in ["val", "test"]:
            try:
                img_dir = resolve_split_images(data_yaml_path, data_cfg, split)
                all_imgs = list_images(img_dir)
                picked = select_deterministic(all_imgs, args.viz_per_split, args.seed)
                out_dir = viz_exp_dir / split
                print(f"[eval] visualizing split={split} selected={len(picked)}/{len(all_imgs)} dir={img_dir}")
                recs = visualize_split(
                    model=model,
                    image_paths=picked,
                    out_dir=out_dir,
                    split=split,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    max_det=args.max_det,
                )
                viz_records[split] = {
                    "image_dir": str(img_dir),
                    "total_images": len(all_imgs),
                    "selected": len(picked),
                    "seed": args.seed,
                    "output_dir": str(out_dir),
                    "records": recs,
                }
            except Exception as e:
                viz_records[split] = {"error": str(e)}
                print(f"[eval] ERROR visualization split={split}: {e}")
    except Exception as e:
        viz_records["error"] = str(e)
        print(f"[eval] ERROR preparing visualizations: {e}")

    (report_dir / "visualization_index.json").write_text(
        json.dumps(viz_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # consolidated report
    training_summary = read_json_if_exists(report_dir / "training_summary.json")
    run_config = read_json_if_exists(report_dir / "run_config.json")

    eval_summary = {
        "experiment_id": experiment_id,
        "timestamp": env_info["timestamp"],
        "weights": str(weights_path),
        "data": args.data,
        "params": {
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "viz_per_split": args.viz_per_split,
            "seed": args.seed,
        },
        "metrics_by_split": metrics_by_split,
        "evaluation_errors": eval_errors,
        "artifacts": {
            "report_dir": str(report_dir),
            "visualizations_dir": str(viz_exp_dir),
            "predictions_dir": str(pred_exp_dir),
        },
        "methodology": {
            "explicit_val_and_test_once": True,
            "tuning_on_test": False,
            "notes": "Test split dievaluasi sekali untuk pelaporan baseline; tidak digunakan untuk tuning hyperparameter.",
        },
        "environment": env_info,
    }
    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(eval_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # markdown report
    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    lines: List[str] = []
    lines.append(f"# Laporan Eksperimen Deteksi — {experiment_id}")
    lines.append("")
    lines.append(f"- Waktu evaluasi: {env_info['timestamp']}")
    lines.append(f"- Bobot: `{weights_path}`")
    lines.append(f"- Data: `{args.data}`")
    lines.append(f"- Device: `{args.device}` | imgsz={args.imgsz} conf={args.conf} iou={args.iou} max_det={args.max_det}")
    lines.append("")
    lines.append("## Keputusan & Konfigurasi Baseline")
    lines.append("- 6.1A CPU-only, 6.2 `yolov8n` (`yolov8n.pt`), 6.3-6.9 adopsi rekomendasi praktis.")
    lines.append("- Training params (rencana/baseline): epochs=50, batch=8, imgsz=640, seed=42, patience=20, workers=2, optimizer/augmentasi default Ultralytics, cache=false, deterministic=true.")
    lines.append("- Data kerja: `configs/dataset.yaml` (split bawaan 1085/107/53, 0 duplikat lintas-split, 374 grup duplikat intra-split/1085 file train, 4 label kosong provisional).")
    if run_config:
        tr = run_config
        lines.append(f"- Run config tercatat: epochs={tr.get('epochs')} batch={tr.get('batch')} seed={tr.get('seed')} device={tr.get('device')} model={tr.get('model')}")
    lines.append("")
    lines.append("## Hasil Training (ringkasan)")
    if training_summary:
        tw = (training_summary.get("weights") or {})
        lines.append(f"- Run dir: `{training_summary.get('run_dir', 'n/a')}`")
        lines.append(f"- Best weights: `{tw.get('best', 'n/a')}`")
        lines.append(f"- Last weights: `{tw.get('last', 'n/a')}`")
        res = training_summary.get("results") or {}
        if res.get("best_epoch"):
            lines.append(f"- Best epoch: {res.get('best_epoch')} (key: {res.get('metric_key_used_for_best')})")
        if res.get("last_row"):
            lr = res.get("last_row")
            # show a few common columns if present
            interesting = [k for k in ["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "val/box_loss"] if k in lr]
            if interesting:
                lines.append("- Last row (subset): " + ", ".join([f"{k}={fmt(lr.get(k))}" for k in interesting]))
    else:
        lines.append("- Ringkasan training (`training_summary.json`) tidak ditemukan di direktori laporan ini.")
    lines.append("")
    lines.append("## Metrik Evaluasi Eksplisit (val & test, sekali)")
    lines.append("")
    lines.append("| Split | Precision | Recall | mAP@50 | mAP@50-95 | Catatan |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for split in ["val", "test"]:
        m = metrics_by_split.get(split, {})
        note = "error: " + m.get("error", "") if "error" in m else "ok"
        lines.append(f"| {split} | {fmt(m.get('precision'))} | {fmt(m.get('recall'))} | {fmt(m.get('mAP50'))} | {fmt(m.get('mAP50-95'))} | {note} |")
    lines.append("")
    lines.append("> Metodologi: evaluasi eksplisit pada `val` dan **satu kali** pada `test` untuk pelaporan baseline; `test` **tidak** digunakan untuk tuning hyperparameter.")
    lines.append("")
    lines.append("## Visualisasi Prediksi")
    for split in ["val", "test"]:
        vr = viz_records.get(split, {}) if isinstance(viz_records, dict) else {}
        if isinstance(vr, dict) and "error" not in vr:
            lines.append(f"- **{split}**: dipilih {vr.get('selected', 'n/a')} dari {vr.get('total_images', 'n/a')} gambar (seed={vr.get('seed', args.seed)}), output: `{vr.get('output_dir', '')}`")
        else:
            err = vr.get("error") if isinstance(vr, dict) else vr
            lines.append(f"- **{split}**: gagal/tidak tersedia ({err})")
    lines.append("")
    lines.append("## Artefak")
    lines.append(f"- Laporan: `{report_dir}`")
    lines.append(f"- Visualisasi: `{viz_exp_dir}`")
    lines.append(f"- Prediksi/evaluasi Ultralytics: `{pred_exp_dir}`")
    if training_summary:
        lines.append(f"- Training run: `{training_summary.get('run_dir', 'n/a')}`")
    lines.append("")
    lines.append("## Analisis Singkat & Keterbatasan")
    lines.append("- Hasil di atas adalah performa baseline aktual pada konfigurasi dan data kerja saat ini (CPU-only, `yolov8n`).")
    lines.append("- Seluruh data train (1085) berada dalam grup duplikat intra-split (audit/persiapan M4), sehingga metrik dapat dipengaruhi over-representasi; 4 label kosong bersifat provisional dan memerlukan verifikasi visual.")
    lines.append("- Tidak ada duplikat lintas-split yang teridentifikasi pada audit M3 (risiko leakage hash-identik rendah), namun interpretasi tetap harus mempertimbangkan karakteristik dataset.")
    lines.append("- Variasi hasil antar environment mungkin terjadi meskipun seed/deterministic diupayakan.")
    lines.append("")
    lines.append("## Reproduksi (ringkas)")
    lines.append(f"1. Training: `python scripts/train_detector.py --data {args.data} --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --device cpu --seed 42 --patience 20 --workers 2`")
    lines.append(f"2. Evaluasi: `python scripts/evaluate_detector.py --experiment-id {experiment_id} --data {args.data} --device {args.device}`")
    lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[eval] COMPLETED experiment_id={experiment_id}")
    print(f"[eval] report={report_dir / 'report.md'}")
    if eval_errors:
        print(f"[eval] warnings/errors: {eval_errors}")
        return 1 if metrics_by_split else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
