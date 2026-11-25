#!/usr/bin/env python3
"""Eksperimen preprocessing Milestone 6 (ALPRO).

Alur: load image (val split) -> deteksi (best.pt M5) -> crop top-confidence + padding
-> jalankan matriks varian preprocessing -> simpan artefak & laporan.

Fokus M6: perbandingan visual/artefak preprocessing, tanpa klaim metrik OCR.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpro.detection.plate_detector import PlateDetector  # noqa: E402
from alpro.preprocessing import ImageLoader, PlateCropper, ImagePreprocessor  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Milestone 6 preprocessing experiment.")
    p.add_argument("--config", type=str, default="configs/preprocessing.yaml")
    p.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    p.add_argument("--weights", type=str, default=None, help="Override detector weights")
    p.add_argument("--split", type=str, default=None, help="Override split (default: config experiment.split)")
    p.add_argument("--experiment-id", type=str, default=None)
    p.add_argument("--curated-count", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--max-images", type=int, default=None, help="Batasi jumlah gambar diproses (debug)")
    return p.parse_args(argv)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_experiment_id(provided: Optional[str]) -> str:
    if provided:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in provided)
        return safe or f"m6_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"m6_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_split_dir(dataset_cfg: Dict[str, Any], dataset_yaml_path: Path, split: str) -> Path:
    key = split
    if key not in dataset_cfg and split == "valid":
        key = "val"
    if key not in dataset_cfg and split == "val" and "valid" in dataset_cfg:
        key = "valid"
    val = dataset_cfg.get(key)
    if not val:
        raise RuntimeError(f"Split '{split}' tidak ditemukan di {dataset_yaml_path}")
    base = dataset_cfg.get("path")
    if base:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = (PROJECT_ROOT / base_path).resolve()
    else:
        base_path = dataset_yaml_path.parent.resolve()
    img_dir = Path(str(val))
    if not img_dir.is_absolute():
        img_dir = (base_path / img_dir).resolve()
    return img_dir


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def select_curated(paths: Sequence[Path], n: int, seed: int) -> List[Path]:
    if n <= 0 or not paths:
        return []
    items = list(paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    picked = items[: min(n, len(items))]
    return sorted(picked, key=lambda p: p.name)


def save_image(path: Path, image) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Gagal menyimpan gambar: {path}")


def get_variants(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = cfg.get("variants")
    if isinstance(variants, list) and variants:
        out = []
        for v in variants:
            if isinstance(v, dict) and v.get("name"):
                steps = v.get("steps", [])
                if not isinstance(steps, list):
                    steps = []
                out.append({"name": str(v["name"]), "steps": steps})
        if out:
            return out
    # fallback
    return [{"name": "original_crop", "steps": []}, {"name": "grayscale", "steps": [{"name": "grayscale", "enabled": True}]}]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    dataset_cfg_path = (PROJECT_ROOT / args.dataset_config).resolve() if not Path(args.dataset_config).is_absolute() else Path(args.dataset_config)

    cfg = load_yaml(cfg_path)
    dataset_cfg = load_yaml(dataset_cfg_path)
    exp_cfg = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}

    split = str(args.split or exp_cfg.get("split", "val"))
    curated_count = int(args.curated_count if args.curated_count is not None else exp_cfg.get("curated_count", 20))
    seed = int(args.seed if args.seed is not None else exp_cfg.get("curated_seed", 42))
    weights = str(args.weights or exp_cfg.get("detector_weights", "yolov8n.pt"))
    device = str(args.device or exp_cfg.get("detector_device", "cpu"))
    conf = float(args.conf if args.conf is not None else exp_cfg.get("detector_conf", 0.25))
    iou = float(args.iou if args.iou is not None else exp_cfg.get("detector_iou", 0.45))
    imgsz = int(args.imgsz if args.imgsz is not None else exp_cfg.get("detector_imgsz", 640))
    padding_ratio = float(defaults.get("padding_ratio", 0.05))
    crop_strategy = str(defaults.get("crop_strategy", "top_confidence"))
    min_crop_size = int(defaults.get("min_crop_size", 8))
    save_intermediate = bool(defaults.get("save_intermediate", True))

    out_viz_root = Path(str(exp_cfg.get("output_visualizations", "outputs/visualizations/preprocessing")))
    out_rep_root = Path(str(exp_cfg.get("output_reports", "outputs/reports/preprocessing")))
    if not out_viz_root.is_absolute():
        out_viz_root = (PROJECT_ROOT / out_viz_root).resolve()
    if not out_rep_root.is_absolute():
        out_rep_root = (PROJECT_ROOT / out_rep_root).resolve()

    experiment_id = make_experiment_id(args.experiment_id)
    viz_dir = out_viz_root / experiment_id
    rep_dir = out_rep_root / experiment_id
    ensure_dir(viz_dir)
    ensure_dir(rep_dir)

    split_dir = resolve_split_dir(dataset_cfg, dataset_cfg_path, split)
    image_paths = list_images(split_dir)
    if args.max_images is not None:
        image_paths = image_paths[: max(0, int(args.max_images))]
    curated_paths = select_curated(image_paths, curated_count, seed)
    curated_set = {str(p.resolve()) for p in curated_paths}

    variants = get_variants(cfg)

    print(f"[m6] experiment_id={experiment_id}")
    print(f"[m6] split={split} images={len(image_paths)} curated={len(curated_paths)} variants={len(variants)}")
    print(f"[m6] weights={weights} device={device} conf={conf} iou={iou} imgsz={imgsz} padding={padding_ratio}")

    detector = PlateDetector(weights=weights, device=device, conf=conf, iou=iou, imgsz=imgsz)

    index_rows: List[Dict[str, Any]] = []
    variant_counts: Dict[str, int] = {v["name"]: 0 for v in variants}
    status_counts: Dict[str, int] = {}
    no_detection_images: List[str] = []
    crop_failed_images: List[str] = []

    curated_manifest: List[Dict[str, Any]] = []

    for i, img_path in enumerate(image_paths, start=1):
        rel_name = img_path.name
        stem = img_path.stem
        is_curated = str(img_path.resolve()) in curated_set
        record_base: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "split": split,
            "image_path": str(img_path),
            "image_name": rel_name,
            "is_curated": is_curated,
        }

        try:
            loaded = ImageLoader.load(img_path)
            image = loaded["image"]
        except Exception as e:
            status_counts["load_error"] = status_counts.get("load_error", 0) + 1
            row = {**record_base, "status": "load_error", "error": str(e)}
            index_rows.append(row)
            if is_curated:
                curated_manifest.append(row)
            continue

        try:
            det_out = detector.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
            detections = det_out.get("detections", [])
        except Exception as e:
            status_counts["detect_error"] = status_counts.get("detect_error", 0) + 1
            row = {**record_base, "status": "detect_error", "error": str(e)}
            index_rows.append(row)
            if is_curated:
                curated_manifest.append(row)
            continue

        crop_out = PlateCropper.crop_detections(
            image,
            detections,
            padding_ratio=padding_ratio,
            strategy=crop_strategy,
            min_size=min_crop_size,
        )

        top_det = detections[0] if detections else None
        base_status = crop_out.get("status", "unknown")
        status_counts[base_status] = status_counts.get(base_status, 0) + 1
        if base_status == "no_detection":
            no_detection_images.append(rel_name)
        if base_status == "crop_failed":
            crop_failed_images.append(rel_name)

        row = {
            **record_base,
            "status": base_status,
            "num_detections": len(detections),
            "selected_count": crop_out.get("selected_count", 0),
            "ok_count": crop_out.get("ok_count", 0),
            "top_confidence": float(top_det.get("confidence", 0.0)) if top_det else None,
            "top_bbox_xyxy": json.dumps(top_det.get("bbox_xyxy")) if top_det else None,
        }

        # save full processing only for curated images (to keep artifact size manageable)
        if is_curated:
            sample_dir = viz_dir / "curated" / stem
            ensure_dir(sample_dir)
            # original annotated
            try:
                ann = PlateDetector.annotate(image, detections)
                save_image(sample_dir / "original_annotated.jpg", ann)
            except Exception:
                pass

            crop_records = []
            for c in crop_out.get("crops", []):
                c_status = c.get("status")
                c_entry = {
                    "index": c.get("index"),
                    "status": c_status,
                    "bbox_original_xyxy": c.get("bbox_original_xyxy"),
                    "bbox_padded_xyxy": c.get("bbox_padded_xyxy"),
                    "detection_confidence": (c.get("detection") or {}).get("confidence"),
                }
                if c_status == "ok" and c.get("crop") is not None:
                    crop_img = c["crop"]
                    crop_path = sample_dir / f"crop_{c.get('index', 0)}.jpg"
                    save_image(crop_path, crop_img)
                    c_entry["crop_path"] = str(crop_path.relative_to(PROJECT_ROOT))

                    variant_outputs = {}
                    for v in variants:
                        vname = v["name"]
                        vsteps = v.get("steps", [])
                        pre = ImagePreprocessor({"pipeline": {"steps": vsteps}})
                        res = pre.process(crop_img, return_intermediate=save_intermediate, output_mode="bgr")
                        out_img = res["image"]
                        v_path = sample_dir / f"variant_{vname}.jpg"
                        save_image(v_path, out_img)
                        variant_counts[vname] = variant_counts.get(vname, 0) + 1
                        variant_outputs[vname] = str(v_path.relative_to(PROJECT_ROOT))

                        # save intermediate steps per variant (optional but useful)
                        if save_intermediate:
                            inter_dir = sample_dir / "intermediate" / vname
                            for sname, simg in (res.get("intermediate") or {}).items():
                                save_image(inter_dir / f"{sname}.jpg", simg)
                    c_entry["variants"] = variant_outputs
                crop_records.append(c_entry)

            row["curated_dir"] = str(sample_dir.relative_to(PROJECT_ROOT))
            row["crop_records"] = crop_records
            curated_manifest.append(row)
        else:
            # non-curated: still count one "would-be" processing target for ok crops? no, keep curated artifacts only
            pass

        index_rows.append({k: v for k, v in row.items() if k != "crop_records"})

        if i % 20 == 0 or i == len(image_paths):
            print(f"[m6] processed {i}/{len(image_paths)}")

    # save index CSV (all images)
    index_csv = rep_dir / "index.csv"
    csv_fields = [
        "experiment_id", "split", "image_name", "image_path", "is_curated", "status",
        "num_detections", "selected_count", "ok_count", "top_confidence", "top_bbox_xyxy",
        "curated_dir", "error",
    ]
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in index_rows:
            w.writerow({k: r.get(k, "") for k in csv_fields})

    # curated manifest json
    curated_json = rep_dir / "curated_manifest.json"
    curated_json.write_text(json.dumps(curated_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(cfg_path),
        "dataset_config_path": str(dataset_cfg_path),
        "split": split,
        "split_dir": str(split_dir),
        "total_images": len(image_paths),
        "curated_count_requested": curated_count,
        "curated_count_actual": len(curated_paths),
        "curated_seed": seed,
        "detector": {
            "weights": weights,
            "device": device,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
        },
        "crop": {
            "strategy": crop_strategy,
            "padding_ratio": padding_ratio,
            "min_crop_size": min_crop_size,
        },
        "variants": [v["name"] for v in variants],
        "variant_output_counts_curated_crops": variant_counts,
        "status_counts": status_counts,
        "no_detection_count": len(no_detection_images),
        "no_detection_images": no_detection_images,
        "crop_failed_count": len(crop_failed_images),
        "crop_failed_images": crop_failed_images,
        "artifacts": {
            "visualizations_dir": str(viz_dir),
            "reports_dir": str(rep_dir),
            "index_csv": str(index_csv),
            "curated_manifest_json": str(curated_json),
        },
        "notes": [
            "M6 fokus pada deteksi->crop->preprocessing dan perbandingan visual/artefak.",
            "Tidak ada klaim metrik OCR kuantitatif (GT teks belum tersedia/diisi).",
            "data/raw tidak diubah; artefak disimpan di outputs/.",
        ],
    }
    (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown report
    lines = []
    lines.append(f"# Laporan Eksperimen Preprocessing — {experiment_id}")
    lines.append("")
    lines.append(f"- Waktu: {summary['timestamp']}")
    lines.append(f"- Split: `{split}` ({len(image_paths)} gambar), curated: {len(curated_paths)} (seed={seed})")
    lines.append(f"- Detector: `{weights}` (device={device}, conf={conf}, iou={iou}, imgsz={imgsz})")
    lines.append(f"- Crop: strategy={crop_strategy}, padding_ratio={padding_ratio}, min_crop_size={min_crop_size}")
    lines.append("")
    lines.append("## Varian Preprocessing (6.3A)")
    for v in variants:
        lines.append(f"- `{v['name']}`: {json.dumps(v.get('steps', []), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Ringkasan Status")
    for k, v in sorted(status_counts.items()):
        lines.append(f"- {k}: {v}")
    lines.append(f"- no_detection: {len(no_detection_images)}")
    lines.append(f"- crop_failed: {len(crop_failed_images)}")
    lines.append("")
    lines.append("## Artefak")
    lines.append(f"- Visualisasi curated: `{viz_dir / 'curated'}`")
    lines.append(f"- Index semua gambar: `{index_csv}`")
    lines.append(f"- Curated manifest: `{curated_json}`")
    lines.append(f"- Summary JSON: `{rep_dir / 'summary.json'}`")
    lines.append("")
    lines.append("## Catatan Metodologis")
    lines.append("- Fokus M6: deteksi → crop → preprocessing dan perbandingan visual/artefak.")
    lines.append("- Tidak ada klaim metrik OCR kuantitatif (CER/WER/EMA) karena GT teks belum tersedia.")
    lines.append("- `data/raw/` tidak diubah. Test set tidak digunakan untuk pemilihan parameter pada eksperimen ini.")
    lines.append("")
    if no_detection_images:
        lines.append("## Gambar Tanpa Deteksi (split ini)")
        for n in no_detection_images:
            lines.append(f"- {n}")
        lines.append("")
    (rep_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[m6] DONE. report={rep_dir / 'report.md'}")
    print(f"[m6] curated_viz={viz_dir / 'curated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
