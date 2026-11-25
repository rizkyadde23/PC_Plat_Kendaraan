#!/usr/bin/env python3
"""Milestone 9 - konsolidasi evaluasi & smoke end-to-end ALPRO.

Tidak mengubah data/raw. Tidak mengklaim metrik OCR kuantitatif jika GT terverifikasi tidak tersedia.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpro.core import PlateRecognitionPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M9 final evaluation consolidation + smoke tests")
    p.add_argument("--output-dir", default="outputs/reports/final/m9_evaluation")
    p.add_argument("--samples-per-case", type=int, default=3)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_artifacts() -> Dict[str, Any]:
    artifacts = {
        "detection_report": "outputs/reports/detection/yolov8n_baseline_m5_20261118_cpu/report.md",
        "detection_metrics": "outputs/reports/detection/yolov8n_baseline_m5_20261118_cpu/evaluation_metrics.json",
        "preprocessing_report": "outputs/reports/preprocessing/m6_preprocessing_val_20261125/report.md",
        "ocr_report": "outputs/reports/ocr/m7_ocr_val_20261125/report.md",
        "ocr_summary": "outputs/reports/ocr/m7_ocr_val_20261125/summary.json",
        "pipeline_config": "configs/pipeline.yaml",
        "app": "app.py",
    }
    status = {}
    for k, v in artifacts.items():
        p = PROJECT_ROOT / v
        status[k] = {"path": v, "exists": p.exists()}
    return {"paths": artifacts, "status": status}


def run_pipeline_smoke(samples_per_case: int) -> Dict[str, Any]:
    pipeline = PlateRecognitionPipeline()
    cases: List[Dict[str, Any]] = []

    def record_case(name: str, source: Any, **kwargs) -> None:
        t0 = time.perf_counter() * 1000
        try:
            res = pipeline.process_image(source, **kwargs)
            elapsed = time.perf_counter() * 1000 - t0
            d = res.to_dict(include_images=False)
            cases.append({
                "case": name,
                "status": d.get("status"),
                "ok": bool(d.get("status") == "ok"),
                "elapsed_ms": round(elapsed, 2),
                "pipeline_timings_ms": d.get("timings_ms", {}),
                "detection_count": len(d.get("detections", [])),
                "ocr_status": (d.get("ocr") or {}).get("status"),
                "text_normalized": (d.get("ocr") or {}).get("text_normalized"),
                "warnings": d.get("warnings", []),
                "errors": d.get("errors", []),
            })
        except Exception as e:
            elapsed = time.perf_counter() * 1000 - t0
            cases.append({"case": name, "status": "exception", "ok": False, "elapsed_ms": round(elapsed, 2), "error": str(e)})

    # valid samples from val split
    val_dir = PROJECT_ROOT / "data/raw/indonesian_license_plate/valid/images"
    val_images = sorted(val_dir.glob("*.jpg"))[:samples_per_case]
    for p in val_images:
        record_case(f"valid_val:{p.name}", str(p))

    # known no-detection cases from M6/M7
    for name in [
        "IMG-20180108-WA0024_jpg.rf.f9fd66333357558710a1c39b2e7e639f.jpg",
        "IMG-20180108-WA0104_jpg.rf.9463b0d9f63615c551f3ddb492cfd879.jpg",
    ]:
        p = val_dir / name
        if p.exists():
            record_case(f"known_no_detection:{name}", str(p))

    # invalid path
    record_case("invalid_path", str(PROJECT_ROOT / "data/raw/indonesian_license_plate/valid/images/DOES_NOT_EXIST.jpg"))
    # invalid empty image
    record_case("invalid_empty_ndarray", np.array([], dtype=np.uint8))

    summary = {
        "total": len(cases),
        "by_status": {},
        "avg_elapsed_ms": round(sum(c.get("elapsed_ms", 0.0) for c in cases) / max(1, len(cases)), 2),
    }
    for c in cases:
        s = str(c.get("status"))
        summary["by_status"][s] = summary["by_status"].get(s, 0) + 1
    return {"summary": summary, "cases": cases}


def main() -> int:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    ensure_dir(out_dir)

    artifacts = collect_artifacts()
    det_metrics = load_json(PROJECT_ROOT / artifacts["paths"]["detection_metrics"]) or {}
    ocr_summary = load_json(PROJECT_ROOT / artifacts["paths"]["ocr_summary"]) or {}

    smoke = run_pipeline_smoke(args.samples_per_case)

    summary = {
        "milestone": "M9",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": artifacts,
        "detection_metrics": det_metrics,
        "ocr_summary_snapshot": {
            "experiment_id": ocr_summary.get("experiment_id"),
            "gt_verified_available_count": ocr_summary.get("gt_verified_available_count"),
            "rows": ocr_summary.get("rows"),
            "notes": ocr_summary.get("notes", []),
            "by_mode_variant": ocr_summary.get("by_mode_variant", {}),
        },
        "pipeline_smoke": smoke,
        "claims_policy": {
            "detection_quantitative": True,
            "ocr_quantitative_claimed": bool((ocr_summary.get("gt_verified_available_count") or 0) > 0),
            "ocr_quantitative_note": "CER/WER/EMA hanya boleh diklaim jika GT terverifikasi tersedia.",
        },
    }
    (out_dir / "m9_evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report
    det = det_metrics.get("metrics", det_metrics)
    val = det.get("val", {}) if isinstance(det, dict) else {}
    test = det.get("test", {}) if isinstance(det, dict) else {}
    lines = []
    lines.append("# Laporan Final Evaluasi — M9 ALPRO")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append("- Cakupan: konsolidasi M1–M8, smoke pipeline end-to-end, status evaluasi OCR.")
    lines.append("")
    lines.append("## Status Artefak Kunci")
    for k, v in artifacts["status"].items():
        lines.append(f"- {k}: {'OK' if v['exists'] else 'MISSING'} (`{v['path']}`)")
    lines.append("")
    lines.append("## Deteksi (M5, kuantitatif)")
    lines.append("| Split | Precision | Recall | mAP@50 | mAP@50-95 |")
    lines.append("|---|---:|---:|---:|---:|")
    def fmt(x):
        try: return f"{float(x):.4f}"
        except Exception: return ""
    lines.append(f"| val | {fmt(val.get('precision'))} | {fmt(val.get('recall'))} | {fmt(val.get('mAP50'))} | {fmt(val.get('mAP50-95'))} |")
    lines.append(f"| test | {fmt(test.get('precision'))} | {fmt(test.get('recall'))} | {fmt(test.get('mAP50'))} | {fmt(test.get('mAP50-95'))} |")
    lines.append("")
    lines.append("## OCR (M7, status)")
    lines.append(f"- GT verified+readable available: **{ocr_summary.get('gt_verified_available_count', 0)}**")
    lines.append(f"- Rows: {ocr_summary.get('rows', 'n/a')}")
    if (ocr_summary.get("gt_verified_available_count") or 0) == 0:
        lines.append("- **CER/WER/Exact Match belum diklaim** karena GT OCR terverifikasi belum tersedia.")
    else:
        lines.append("- GT terverifikasi tersedia; metrik kuantitatif dapat dihitung dari `scripts/evaluate_ocr.py`.")
    lines.append("")
    lines.append("## Smoke Test Pipeline End-to-End (M8)")
    lines.append(f"- Total cases: {smoke['summary']['total']}")
    lines.append(f"- Avg elapsed ms: {smoke['summary']['avg_elapsed_ms']}")
    lines.append(f"- By status: `{json.dumps(smoke['summary']['by_status'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("| Case | Status | Elapsed ms | Detections | OCR status | Text normalized |")
    lines.append("|---|---|---:|---:|---|---|")
    for c in smoke["cases"]:
        lines.append(f"| {c['case']} | {c.get('status')} | {c.get('elapsed_ms','')} | {c.get('detection_count','')} | {c.get('ocr_status','')} | {c.get('text_normalized','')} |")
    lines.append("")
    lines.append("## Keterbatasan & Analisis")
    lines.append("- `data/raw/` tidak diubah; split bawaan dipertahankan.")
    lines.append("- 2 kasus `no_detection` val diketahui (M6/M7) dan tetap muncul pada smoke test.")
    lines.append("- Duplikat intra-split train dan 4 empty-label provisional (M3/M4) dapat memengaruhi interpretasi generalisasi.")
    lines.append("- OCR kuantitatif memerlukan GT `plate_text_gt` terverifikasi; sampai saat itu hasil OCR adalah prediksi.")
    lines.append("- EasyOCR confidence bukan pengganti akurasi terhadap ground truth.")
    lines.append("- Runtime CPU-only; performa/waktu dapat berbeda pada GPU.")
    lines.append("")
    lines.append("## Cara Reproduksi Singkat")
    lines.append("```powershell")
    lines.append(".\\.venv\\Scripts\\python.exe -m pytest tests/unit tests/integration -q")
    lines.append(".\\.venv\\Scripts\\python.exe scripts/run_final_evaluation.py")
    lines.append("streamlit run app.py")
    lines.append("```")
    (out_dir / "m9_final_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"ok": True, "output_dir": str(out_dir), "summary": smoke["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
