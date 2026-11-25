#!/usr/bin/env python3
"""Eksperimen/evaluasi OCR Milestone 7 (ALPRO).

Alur: deteksi (M5) -> crop (M6) -> preprocessing varian (M6) -> OCR (EasyOCR) ->
normalisasi -> (opsional) metrik CER/WER/EMA jika GT terverifikasi tersedia.

Default mengikuti rekomendasi M7:
- Curated val subset: semua 10 varian (kualitatif terkontrol)
- Indeks val: 1-2 varian (runtime CPU wajar)
- Kuantitatif hanya pada GT verified=true, unreadable!=true, plate_text_gt non-empty.
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

try:
    from jiwer import cer as jiwer_cer, wer as jiwer_wer
except Exception:  # pragma: no cover
    jiwer_cer = None
    jiwer_wer = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpro.detection.plate_detector import PlateDetector  # noqa: E402
from alpro.preprocessing import ImageLoader, PlateCropper, ImagePreprocessor  # noqa: E402
from alpro.recognition import OCREngine, TextNormalizer  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Milestone 7 OCR experiment/evaluation.")
    p.add_argument("--config", type=str, default="configs/ocr.yaml")
    p.add_argument("--preprocess-config", type=str, default="configs/preprocessing.yaml")
    p.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    p.add_argument("--experiment-id", type=str, default=None)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--curated-count", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-images", type=int, default=None, help="Batasi total gambar (debug).")
    p.add_argument("--index-variants", type=str, default=None, help="Comma-separated variant names for index run.")
    p.add_argument("--skip-index", action="store_true", help="Lewati run indeks (hanya curated).")
    p.add_argument("--skip-curated", action="store_true", help="Lewati run curated (hanya indeks).")
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
        return safe or f"m7_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"m7_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_split_dir(dataset_cfg: Dict[str, Any], split: str) -> Path:
    key = split
    if key not in dataset_cfg and split == "valid":
        key = "val"
    if key not in dataset_cfg and split == "val" and "valid" in dataset_cfg:
        key = "valid"
    val = dataset_cfg.get(key)
    if not val:
        raise RuntimeError(f"Split '{split}' tidak ditemukan di dataset config")
    base = dataset_cfg.get("path")
    if base:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = (PROJECT_ROOT / base_path).resolve()
    else:
        base_path = PROJECT_ROOT
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


def get_variants(preprocess_cfg: Dict[str, Any], names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    raw = preprocess_cfg.get("variants")
    variants: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for v in raw:
            if isinstance(v, dict) and v.get("name"):
                steps = v.get("steps", [])
                variants.append({"name": str(v["name"]), "steps": steps if isinstance(steps, list) else []})
    if not variants:
        variants = [
            {"name": "original_crop", "steps": []},
            {"name": "grayscale", "steps": [{"name": "grayscale", "enabled": True}]},
        ]
    if names is None:
        return variants
    name_set = {n.strip() for n in names if n and n.strip()}
    selected = [v for v in variants if v["name"] in name_set]
    # preserve requested order for unknown? keep known only
    missing = [n for n in name_set if n not in {v["name"] for v in selected}]
    for m in missing:
        selected.append({"name": m, "steps": []})  # fallback
    # order by requested names when provided
    order = [n.strip() for n in names if n and n.strip()]
    selected.sort(key=lambda v: order.index(v["name"]) if v["name"] in order else 9999)
    return selected


def normalize_gt_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y"}


def build_gt_map(gt_csv_paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """Map image_path (normalized relative/absolute forms) to GT row. Later files override earlier."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in gt_csv_paths:
        for r in normalize_gt_rows(p):
            raw_path = str(r.get("image_path", "") or "").strip()
            if not raw_path:
                continue
            keys = set()
            keys.add(raw_path.replace("\\", "/"))
            try:
                abs_p = (PROJECT_ROOT / raw_path).resolve()
                keys.add(str(abs_p).replace("\\", "/"))
                keys.add(abs_p.name)
            except Exception:
                keys.add(Path(raw_path).name)
            for k in keys:
                out[k] = r
    return out


def lookup_gt(gt_map: Dict[str, Dict[str, Any]], image_path: Path) -> Optional[Dict[str, Any]]:
    keys = [
        str(image_path.resolve()).replace("\\", "/"),
        str(image_path).replace("\\", "/"),
        image_path.name,
    ]
    try:
        rel = image_path.resolve().relative_to(PROJECT_ROOT.resolve())
        keys.append(str(rel).replace("\\", "/"))
    except Exception:
        pass
    for k in keys:
        if k in gt_map:
            return gt_map[k]
    return None


def compute_metrics(pred: str, gt: str) -> Dict[str, Any]:
    if jiwer_cer is None or jiwer_wer is None:
        return {"cer": None, "wer": None, "exact_match": int(pred == gt), "metric_error": "jiwer unavailable"}
    try:
        cer_v = float(jiwer_cer(gt, pred))
    except Exception:
        cer_v = None
    try:
        wer_v = float(jiwer_wer(gt, pred))
    except Exception:
        wer_v = None
    return {
        "cer": cer_v,
        "wer": wer_v,
        "exact_match": int(pred == gt),
    }


def save_image(path: Path, image) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Gagal menyimpan gambar: {path}")


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def run_ocr_on_crop(
    crop_img,
    variants: Sequence[Dict[str, Any]],
    ocr_engine: OCREngine,
    normalizer: TextNormalizer,
    sample_dir: Optional[Path],
    save_visuals: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for v in variants:
        vname = v["name"]
        pre = ImagePreprocessor({"pipeline": {"steps": v.get("steps", [])}})
        pre_res = pre.process(crop_img, return_intermediate=False, output_mode="bgr")
        proc_img = pre_res["image"]
        ocr_res = ocr_engine.read(proc_img)
        norm_res = normalizer.normalize(ocr_res.get("text_raw", ""))
        row = {
            "variant": vname,
            "ocr_status": ocr_res.get("status"),
            "text_raw": ocr_res.get("text_raw", ""),
            "text_normalized": norm_res.text_normalized,
            "ocr_confidence": float(ocr_res.get("confidence", 0.0) or 0.0),
            "ocr_candidates": json.dumps(ocr_res.get("filtered_candidates", ocr_res.get("candidates", [])), ensure_ascii=False),
            "ocr_error": ocr_res.get("error"),
        }
        if save_visuals and sample_dir is not None:
            v_path = sample_dir / f"variant_{vname}.jpg"
            save_image(v_path, proc_img)
            row["variant_image_path"] = safe_rel(v_path)
        rows.append(row)
    return rows


def aggregate_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_variant.setdefault(str(r.get("variant")), []).append(r)

    summary = {}
    for v, items in by_variant.items():
        n = len(items)
        ok = [r for r in items if r.get("ocr_status") == "ok"]
        non_empty_norm = [r for r in items if str(r.get("text_normalized", "")).strip()]
        confs = [float(r.get("ocr_confidence", 0.0) or 0.0) for r in ok]
        eval_rows = [r for r in items if r.get("gt_available") and r.get("cer") is not None]
        cer_vals = [float(r["cer"]) for r in eval_rows if r.get("cer") is not None]
        wer_vals = [float(r["wer"]) for r in eval_rows if r.get("wer") is not None]
        em_vals = [int(r["exact_match"]) for r in eval_rows if r.get("exact_match") is not None]
        summary[v] = {
            "count": n,
            "ocr_ok": len(ok),
            "ocr_empty": sum(1 for r in items if r.get("ocr_status") == "empty_ocr"),
            "ocr_error": sum(1 for r in items if r.get("ocr_status") == "ocr_error"),
            "det_or_crop_fail": sum(1 for r in items if r.get("pipeline_status") != "ok"),
            "non_empty_normalized": len(non_empty_norm),
            "avg_confidence_ok": (sum(confs) / len(confs)) if confs else 0.0,
            "gt_evaluated_count": len(eval_rows),
            "avg_cer": (sum(cer_vals) / len(cer_vals)) if cer_vals else None,
            "avg_wer": (sum(wer_vals) / len(wer_vals)) if wer_vals else None,
            "exact_match_rate": (sum(em_vals) / len(em_vals)) if em_vals else None,
        }
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    pre_cfg_path = (PROJECT_ROOT / args.preprocess_config).resolve() if not Path(args.preprocess_config).is_absolute() else Path(args.preprocess_config)
    ds_cfg_path = (PROJECT_ROOT / args.dataset_config).resolve() if not Path(args.dataset_config).is_absolute() else Path(args.dataset_config)

    cfg = load_yaml(cfg_path)
    pre_cfg = load_yaml(pre_cfg_path)
    ds_cfg = load_yaml(ds_cfg_path)
    exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    engine_cfg = cfg.get("engine", {}) if isinstance(cfg.get("engine"), dict) else {}
    norm_cfg = cfg.get("normalization", {}) if isinstance(cfg.get("normalization"), dict) else {}

    experiment_id = make_experiment_id(args.experiment_id)
    split = str(args.split or exp.get("split", "val"))
    weights = str(args.weights or exp.get("detector_weights", "yolov8n.pt"))
    device = str(args.device or exp.get("detector_device", "cpu"))
    conf = float(exp.get("detector_conf", 0.25))
    iou = float(exp.get("detector_iou", 0.45))
    imgsz = int(exp.get("detector_imgsz", 640))
    curated_count = int(args.curated_count if args.curated_count is not None else exp.get("curated_count", 20))
    seed = int(args.seed if args.seed is not None else exp.get("curated_seed", 42))
    padding_ratio = float(exp.get("padding_ratio", 0.05))
    crop_strategy = str(exp.get("crop_strategy", "top_confidence"))
    min_crop_size = int(exp.get("min_crop_size", 8))

    out_reports = Path(str(exp.get("output_reports", "outputs/reports/ocr")))
    out_viz = Path(str(exp.get("output_visualizations", "outputs/visualizations/ocr")))
    if not out_reports.is_absolute():
        out_reports = (PROJECT_ROOT / out_reports).resolve()
    if not out_viz.is_absolute():
        out_viz = (PROJECT_ROOT / out_viz).resolve()
    report_dir = out_reports / experiment_id
    viz_dir = out_viz / experiment_id
    ensure_dir(report_dir)
    ensure_dir(viz_dir)

    split_dir = resolve_split_dir(ds_cfg, split)
    image_paths = list_images(split_dir)
    if args.max_images is not None:
        image_paths = image_paths[: max(0, int(args.max_images))]

    curated_paths = select_curated(image_paths, curated_count, seed)
    curated_set = {str(p.resolve()) for p in curated_paths}

    all_variants = get_variants(pre_cfg, None)
    curated_variants_cfg = exp.get("curated_variants", "all")
    if curated_variants_cfg == "all":
        curated_variants = list(all_variants)
    elif isinstance(curated_variants_cfg, list):
        curated_variants = get_variants(pre_cfg, [str(x) for x in curated_variants_cfg])
    else:
        curated_variants = list(all_variants)

    index_variants_arg = args.index_variants
    if index_variants_arg:
        index_variant_names = [s.strip() for s in index_variants_arg.split(",") if s.strip()]
    else:
        idx_cfg = exp.get("index_variants", ["original_crop", "gray_clahe"])
        index_variant_names = [str(x) for x in idx_cfg] if isinstance(idx_cfg, list) else ["original_crop", "gray_clahe"]
    index_variants = get_variants(pre_cfg, index_variant_names)

    # GT maps
    gt_paths = []
    for k in ["gt_annotations_csv", "gt_candidates_csv"]:
        v = eval_cfg.get(k)
        if v:
            p = Path(str(v))
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            gt_paths.append(p)
    gt_map = build_gt_map(gt_paths)

    print(f"[m7] experiment_id={experiment_id}")
    print(f"[m7] split={split} images={len(image_paths)} curated={len(curated_paths)}")
    print(f"[m7] curated_variants={len(curated_variants)} index_variants={len(index_variants)}")
    print(f"[m7] weights={weights} device={device} conf={conf} iou={iou} imgsz={imgsz}")

    detector = PlateDetector(weights=weights, device=device, conf=conf, iou=iou, imgsz=imgsz)
    ocr_engine = OCREngine(
        lang=engine_cfg.get("lang", ["en"]),
        gpu=bool(engine_cfg.get("gpu", False)),
        detail=int(engine_cfg.get("detail", 1)),
        paragraph=bool(engine_cfg.get("paragraph", False)),
        decoder=engine_cfg.get("decoder", None),
        beamWidth=int(engine_cfg.get("beamWidth", 5)),
        batch_size=int(engine_cfg.get("batch_size", 1)),
        min_confidence=float(engine_cfg.get("min_confidence", 0.0)),
    )
    normalizer = TextNormalizer(norm_cfg)

    all_rows: List[Dict[str, Any]] = []
    run_errors: List[Dict[str, Any]] = []

    def process_set(paths: Sequence[Path], variants: Sequence[Dict[str, Any]], mode: str, save_visuals: bool) -> None:
        for i, img_path in enumerate(paths, start=1):
            is_curated = str(img_path.resolve()) in curated_set
            gt_row = lookup_gt(gt_map, img_path)
            gt_text = str(gt_row.get("plate_text_gt", "") or "").strip() if gt_row else ""
            gt_verified = to_bool(gt_row.get("verified", False)) if gt_row else False
            gt_unreadable = to_bool(gt_row.get("unreadable", False)) if gt_row else False
            gt_available = bool(gt_row) and gt_verified and (not gt_unreadable) and bool(gt_text)

            base = {
                "experiment_id": experiment_id,
                "mode": mode,
                "split": split,
                "image_name": img_path.name,
                "image_path": safe_rel(img_path),
                "is_curated": is_curated,
                "plate_text_gt": gt_text,
                "gt_verified": gt_verified,
                "gt_unreadable": gt_unreadable,
                "gt_available": gt_available,
            }

            try:
                loaded = ImageLoader.load(img_path)
                image = loaded["image"]
            except Exception as e:
                for v in variants:
                    all_rows.append({
                        **base, "variant": v["name"], "pipeline_status": "load_error",
                        "det_status": "load_error", "crop_status": "load_error", "ocr_status": "skipped",
                        "text_raw": "", "text_normalized": "", "ocr_confidence": 0.0,
                        "cer": None, "wer": None, "exact_match": None, "error": str(e),
                    })
                run_errors.append({"image": img_path.name, "mode": mode, "error": f"load: {e}"})
                continue

            try:
                det_out = detector.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
                detections = det_out.get("detections", [])
            except Exception as e:
                for v in variants:
                    all_rows.append({
                        **base, "variant": v["name"], "pipeline_status": "detect_error",
                        "det_status": "detect_error", "crop_status": "skipped", "ocr_status": "skipped",
                        "text_raw": "", "text_normalized": "", "ocr_confidence": 0.0,
                        "cer": None, "wer": None, "exact_match": None, "error": str(e),
                    })
                run_errors.append({"image": img_path.name, "mode": mode, "error": f"detect: {e}"})
                continue

            crop_out = PlateCropper.crop_detections(
                image, detections, padding_ratio=padding_ratio,
                strategy=crop_strategy, min_size=min_crop_size,
            )
            det_status = "ok" if detections else "no_detection"
            crop_status = str(crop_out.get("status", "unknown"))

            sample_dir = None
            if save_visuals:
                sample_dir = viz_dir / mode / img_path.stem
                ensure_dir(sample_dir)
                try:
                    ann = PlateDetector.annotate(image, detections)
                    save_image(sample_dir / "original_annotated.jpg", ann)
                except Exception:
                    pass

            # no detection / crop fail: record per variant without OCR
            if crop_status != "ok":
                for v in variants:
                    all_rows.append({
                        **base, "variant": v["name"], "pipeline_status": crop_status,
                        "det_status": det_status, "crop_status": crop_status, "ocr_status": "skipped",
                        "num_detections": len(detections),
                        "text_raw": "", "text_normalized": "", "ocr_confidence": 0.0,
                        "cer": None, "wer": None, "exact_match": None,
                        "error": None,
                    })
                if i % 20 == 0 or i == len(paths):
                    print(f"[m7:{mode}] processed {i}/{len(paths)}")
                continue

            # take first ok crop (top-confidence strategy produces first as top)
            ok_crops = [c for c in crop_out.get("crops", []) if c.get("status") == "ok" and c.get("crop") is not None]
            crop = ok_crops[0]
            crop_img = crop["crop"]
            if save_visuals and sample_dir is not None:
                save_image(sample_dir / "crop_0.jpg", crop_img)

            ocr_rows = run_ocr_on_crop(crop_img, variants, ocr_engine, normalizer, sample_dir, save_visuals)
            for r in ocr_rows:
                row = {
                    **base,
                    "pipeline_status": "ok",
                    "det_status": det_status,
                    "crop_status": "ok",
                    "num_detections": len(detections),
                    "selected_confidence": float((crop.get("detection") or {}).get("confidence", 0.0) or 0.0),
                    "bbox_padded_xyxy": json.dumps(crop.get("bbox_padded_xyxy")),
                    **r,
                }
                if gt_available:
                    m = compute_metrics(str(row.get("text_normalized", "")), gt_text.upper())
                    row.update(m)
                else:
                    row.update({"cer": None, "wer": None, "exact_match": None})
                all_rows.append(row)

            if i % 10 == 0 or i == len(paths):
                print(f"[m7:{mode}] processed {i}/{len(paths)}")

    if not args.skip_curated:
        process_set(curated_paths, curated_variants, mode="curated", save_visuals=bool(exp.get("save_curated_visuals", True)))
    if not args.skip_index:
        process_set(image_paths, index_variants, mode="index", save_visuals=bool(exp.get("save_index_visuals", False)))

    # Outputs
    fields = [
        "experiment_id", "mode", "split", "image_name", "image_path", "is_curated", "variant",
        "pipeline_status", "det_status", "crop_status", "ocr_status", "num_detections", "selected_confidence",
        "bbox_padded_xyxy", "text_raw", "text_normalized", "ocr_confidence", "ocr_candidates", "ocr_error",
        "plate_text_gt", "gt_verified", "gt_unreadable", "gt_available", "cer", "wer", "exact_match",
        "variant_image_path", "error",
    ]
    write_csv(report_dir / "predictions.csv", all_rows, fields)
    (report_dir / "predictions.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": safe_rel(cfg_path),
        "split": split,
        "total_images": len(image_paths),
        "curated_images": len(curated_paths),
        "curated_variants": [v["name"] for v in curated_variants],
        "index_variants": [v["name"] for v in index_variants],
        "detector": {"weights": weights, "device": device, "conf": conf, "iou": iou, "imgsz": imgsz},
        "ocr": engine_cfg,
        "normalization": norm_cfg,
        "gt_sources": [safe_rel(p) for p in gt_paths],
        "gt_verified_available_count": sum(1 for r in all_rows if r.get("gt_available")),
        "rows": len(all_rows),
        "run_errors": run_errors,
        "by_mode_variant": {
            "curated": aggregate_summary([r for r in all_rows if r.get("mode") == "curated"]),
            "index": aggregate_summary([r for r in all_rows if r.get("mode") == "index"]),
        },
        "notes": [
            "M7 fokus OCR+normalisasi; evaluasi kuantitatif hanya pada GT verified=true & readable & non-empty.",
            "Jika gt_verified_available_count==0, metrik CER/WER/EMA belum dapat diklaim.",
            "data/raw tidak diubah; test tidak dipakai tuning.",
        ],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report markdown
    lines = []
    lines.append(f"# Laporan Eksperimen OCR — {experiment_id}")
    lines.append("")
    lines.append(f"- Waktu: {summary['timestamp']}")
    lines.append(f"- Split: `{split}` ({len(image_paths)} gambar), curated: {len(curated_paths)} (seed={seed})")
    lines.append(f"- Detector: `{weights}` (device={device}, conf={conf}, iou={iou}, imgsz={imgsz})")
    lines.append(f"- OCR: EasyOCR lang={engine_cfg.get('lang')} gpu={engine_cfg.get('gpu')} detail={engine_cfg.get('detail')} paragraph={engine_cfg.get('paragraph')}")
    lines.append(f"- Normalisasi: uppercase/strip/collapse/allowed_pattern, char_substitution enabled={bool((norm_cfg.get('char_substitution') or {}).get('enabled', False))}")
    lines.append("")
    lines.append("## Cakupan")
    lines.append(f"- Curated variants ({len(curated_variants)}): {', '.join(v['name'] for v in curated_variants)}")
    lines.append(f"- Index variants ({len(index_variants)}): {', '.join(v['name'] for v in index_variants)}")
    lines.append(f"- GT verified+readable rows available: {summary['gt_verified_available_count']}")
    lines.append("")
    for mode in ["curated", "index"]:
        lines.append(f"## Ringkasan {mode.title()}")
        mv = summary["by_mode_variant"].get(mode, {})
        if not mv:
            lines.append("- (tidak ada data)")
            lines.append("")
            continue
        lines.append("| Variant | N | OCR OK | Empty | Err | AvgConf OK | NonEmptyNorm | GT Eval | Avg CER | Avg WER | EM |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for vname, s in mv.items():
            def fmt(x):
                return "" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
            lines.append(f"| {vname} | {s['count']} | {s['ocr_ok']} | {s['ocr_empty']} | {s['ocr_error']} | {fmt(s['avg_confidence_ok'])} | {s['non_empty_normalized']} | {s['gt_evaluated_count']} | {fmt(s['avg_cer'])} | {fmt(s['avg_wer'])} | {fmt(s['exact_match_rate'])} |")
        lines.append("")
    lines.append("## Catatan Metodologis")
    lines.append("- Prediksi mentah (`text_raw`) disimpan terpisah dari `text_normalized`.")
    lines.append("- Normalisasi default non-spekulatif; koreksi O/0/I/1/B/8 default OFF.")
    lines.append("- CER/WER/Exact Match hanya dihitung pada GT `verified=true`, `unreadable!=true`, `plate_text_gt` non-empty.")
    if summary["gt_verified_available_count"] == 0:
        lines.append("- **Belum ada GT terverifikasi**: metrik kuantitatif OCR belum dapat diklaim; laporan ini berisi bukti pipeline/kualitatif.")
    lines.append("- `data/raw/` tidak diubah; test set tidak dipakai tuning.")
    lines.append("")
    lines.append("## Artefak")
    lines.append(f"- Predictions CSV: `{safe_rel(report_dir / 'predictions.csv')}`")
    lines.append(f"- Summary JSON: `{safe_rel(report_dir / 'summary.json')}`")
    lines.append(f"- Visualizations: `{safe_rel(viz_dir)}`")
    if run_errors:
        lines.append("")
        lines.append("## Run Errors")
        for e in run_errors[:50]:
            lines.append(f"- {e}")
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[m7] DONE. report={report_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
