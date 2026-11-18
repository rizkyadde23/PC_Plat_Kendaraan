#!/usr/bin/env python3
"""Persiapan dataset ALPRO (Milestone 4).

Membuat artefak turunan untuk deteksi YOLO dan kerangka anotasi OCR
berdasarkan temuan audit, tanpa memodifikasi data/raw.

Keputusan default yang diimplementasikan:
- 6.1A (provisional): 4 label kosong dipertahankan sebagai kandidat background/no-object
  (perlu verifikasi visual), dicatat di manifest.
- 6.2A: semua sampel dipertahankan; duplikat intra-split didokumentasikan.
- 6.3C: config YOLO kerja di configs/dataset.yaml + manifest di data/splits/.
- 6.4A: manifest per split (JSON/CSV) di data/splits/.
- 6.5A(ii),C(i): kerangka OCR + 50 kandidat anotasi (seed 42, proporsional, non-empty) di data/annotations/ocr/.
- 6.7A,6.8C: laporan run di outputs/reports/dataset_preparation/<run>/ dan pembaruan data/README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_EMPTY_LABEL_STEMS = {
    "train": {
        "helm_045_jpg.rf.3eafc00e8cc8ac1e40858a7df61f2429",
        "helm_045_jpg.rf.c31448415f7ac963a7423202c75e9feb",
    },
    "valid": {"99_E-2984-TW-04-18_jpg.rf.b1a1a2d9ea7528f302fe97c14a4f1091"},
    "test": {"helm_047_jpg.rf.fe6ed6f023c55690a6cd60da208be8b6"},
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ALPRO dataset artifacts (Milestone 4).")
    p.add_argument("--data-root", type=Path, default=Path("data/raw/indonesian_license_plate"))
    p.add_argument("--configs-dir", type=Path, default=Path("configs"))
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--ocr-dir", type=Path, default=Path("data/annotations/ocr"))
    p.add_argument("--report-dir", type=Path, default=Path("outputs/reports/dataset_preparation"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidate-total", type=int, default=50)
    p.add_argument("--audit-report-json", type=Path, default=Path("outputs/reports/dataset_audit/20261118_210535/report.json"))
    p.add_argument("--update-data-readme", action="store_true", default=True)
    p.add_argument("--no-update-data-readme", dest="update_data_readme", action="store_false")
    return p.parse_args(argv)


def log(msg: str) -> None:
    print(msg, flush=True)


def make_run_id(provided: Optional[str]) -> str:
    if provided:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in provided)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_posix(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def list_images(images_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not images_dir.exists():
        return out
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = p
    return out


def count_label_objects(label_path: Path) -> Tuple[int, bool, bool]:
    """Return (object_count, is_empty, is_missing_or_error)."""
    if not label_path.exists():
        return 0, True, True
    try:
        text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return 0, True, True
    if text == "":
        return 0, True, False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines), False, False


def compute_duplicate_info(data_root: Path, images_by_split: Dict[str, Dict[str, Path]]) -> Dict[str, Any]:
    hash_to_items: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    file_hash: Dict[Tuple[str, str], str] = {}
    for split, images in images_by_split.items():
        for stem, path in images.items():
            h = md5_file(path)
            file_hash[(split, stem)] = h
            hash_to_items[h].append((split, stem))
    groups: Dict[str, Dict[str, Any]] = {}
    for h, items in hash_to_items.items():
        if len(items) < 2:
            continue
        splits = sorted({s for s, _ in items})
        gtype = "cross_split" if len(splits) > 1 else "intra_split"
        groups[h] = {"hash": h, "type": gtype, "count": len(items), "splits": splits}
    # per file lookup
    per_file: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, h in file_hash.items():
        if h in groups:
            g = groups[h]
            per_file[key] = {"hash": h, "duplicate_type": g["type"], "duplicate_group_size": g["count"]}
        else:
            per_file[key] = {"hash": h, "duplicate_type": "none", "duplicate_group_size": 1}
    summary = {
        "duplicate_groups_total": len(groups),
        "duplicate_groups_intra_split": sum(1 for g in groups.values() if g["type"] == "intra_split"),
        "duplicate_groups_cross_split": sum(1 for g in groups.values() if g["type"] == "cross_split"),
        "duplicate_files_intra_split": sum(g["count"] for g in groups.values() if g["type"] == "intra_split"),
        "duplicate_files_cross_split": sum(g["count"] for g in groups.values() if g["type"] == "cross_split"),
    }
    return {"summary": summary, "per_file": per_file, "groups": groups}


def build_manifests(
    data_root: Path,
    project_root: Path,
    dup_info: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    manifests: Dict[str, List[Dict[str, Any]]] = {}
    for split in SPLITS:
        images = list_images(data_root / split / "images")
        rows: List[Dict[str, Any]] = []
        for stem in sorted(images.keys()):
            img_path = images[stem]
            lbl_path = data_root / split / "labels" / f"{stem}.txt"
            obj_count, is_empty, lbl_err = count_label_objects(lbl_path)
            # 6.1A provisional: keep known empty-label files as included background candidates
            known_empty = stem in KNOWN_EMPTY_LABEL_STEMS.get(split, set())
            empty_label = is_empty or known_empty
            d = dup_info["per_file"].get((split, stem), {"hash": "", "duplicate_type": "none", "duplicate_group_size": 1})
            included = True
            reasons: List[str] = []
            if empty_label:
                reasons.append("empty_label_provisional_background_keep_6.1A")
            if d["duplicate_type"] == "intra_split":
                reasons.append("duplicate_intra_split_kept_6.2A")
            elif d["duplicate_type"] == "cross_split":
                reasons.append("duplicate_cross_split_flagged")
            if not reasons:
                reasons.append("ok")
            rows.append({
                "split": split,
                "stem": stem,
                "image_path": rel_posix(img_path, project_root),
                "label_path": rel_posix(lbl_path, project_root),
                "included": included,
                "object_count": obj_count,
                "empty_label": bool(empty_label),
                "known_empty_label_case": bool(known_empty),
                "label_missing_or_error": bool(lbl_err),
                "duplicate_hash_md5": d.get("hash", ""),
                "duplicate_type": d.get("duplicate_type", "none"),
                "duplicate_group_size": d.get("duplicate_group_size", 1),
                "decision_reason": ";".join(reasons),
                "visual_verification_required": bool(empty_label),
            })
        manifests[split] = rows
    return manifests


def write_manifests(manifests: Dict[str, List[Dict[str, Any]]], splits_dir: Path) -> List[Path]:
    ensure_dir(splits_dir)
    written: List[Path] = []
    fields = [
        "split", "stem", "image_path", "label_path", "included", "object_count",
        "empty_label", "known_empty_label_case", "label_missing_or_error",
        "duplicate_hash_md5", "duplicate_type", "duplicate_group_size",
        "decision_reason", "visual_verification_required",
    ]
    for split, rows in manifests.items():
        jp = splits_dir / f"{split}_manifest.json"
        cp = splits_dir / f"{split}_manifest.csv"
        jp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        with cp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        written.extend([jp, cp])
    # combined summary
    summary = {s: {"total": len(r), "included": sum(1 for x in r if x["included"]),
                   "empty_label": sum(1 for x in r if x["empty_label"]),
                   "duplicate_intra_split": sum(1 for x in r if x["duplicate_type"] == "intra_split")}
               for s, r in manifests.items()}
    sp = splits_dir / "preparation_manifest_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(sp)
    return written


def write_dataset_yaml(configs_dir: Path) -> Path:
    ensure_dir(configs_dir)
    path = configs_dir / "dataset.yaml"
    content = (
        "# ALPRO - Konfigurasi dataset YOLO kerja (turunan, Milestone 4)\n"
        "# Sumber data: data/raw/indonesian_license_plate (TIDAK diubah)\n"
        "# Catatan: data.yaml bawaan raw memiliki path relatif ekspor yang tidak resolve\n"
        "# pada audit; config ini menunjuk ke struktur aktual dataset dari root proyek.\n\n"
        "path: data/raw/indonesian_license_plate\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['License-Plate']\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def proportional_allocation(total: int, splits: Sequence[str], counts: Dict[str, int]) -> Dict[str, int]:
    avail_total = sum(counts[s] for s in splits)
    if avail_total <= 0 or total <= 0:
        return {s: 0 for s in splits}
    total = min(total, avail_total)
    raw = {s: (counts[s] / avail_total) * total for s in splits}
    base = {s: int(math.floor(raw[s])) for s in splits}
    rem = total - sum(base.values())
    # distribute remainder by largest fractional parts, then by larger counts for stability
    fracs = sorted(splits, key=lambda s: (raw[s] - base[s], counts[s]), reverse=True)
    i = 0
    while rem > 0 and i < len(fracs):
        s = fracs[i]
        if base[s] < counts[s]:
            base[s] += 1
            rem -= 1
        i += 1
    # if still remainder due to caps (unlikely), fill any with capacity
    if rem > 0:
        for s in splits:
            while rem > 0 and base[s] < counts[s]:
                base[s] += 1
                rem -= 1
    return base


def build_ocr_framework(
    manifests: Dict[str, List[Dict[str, Any]]],
    ocr_dir: Path,
    seed: int,
    candidate_total: int,
    project_root: Path,
) -> Dict[str, Any]:
    ensure_dir(ocr_dir)
    written: List[str] = []

    # README/pedoman
    readme = ocr_dir / "README.md"
    readme_text = """# Kerangka Anotasi Ground Truth OCR — ALPRO (Milestone 4)

Direktori ini menyiapkan **kerangka** anotasi teks plat nomor (ground truth OCR). 
**Tidak ada teks ground truth yang dibuat otomatis/asumtif** pada Milestone 4.

## Tujuan
- Menyediakan skema/format dan daftar kandidat untuk anotasi manual transkripsi plat.
- Evaluasi OCR kuantitatif (CER/WER/Exact Match) hanya valid pada sampel dengan `plate_text_gt` yang terisi dan **terverifikasi**.

## Berkas
- `ocr_annotations_template.csv`: template skema anotasi.
- `ocr_annotation_candidates.csv`: daftar kandidat anotasi (50 sampel, deterministik, seed=42, proporsional per split) dari gambar non-empty label sesuai manifest M4. Kolom teks **kosong** dan harus diisi manual.
- `ocr_annotations.csv`: (opsional, dibuat saat Anda mulai mengisi) salin dari kandidat/template dan isi hasil anotasi.

## Skema CSV (6.5 C.i)
Kolom: `image_path,split,plate_text_gt,unreadable,notes,verified`

- `image_path`: path relatif terhadap root proyek ke gambar sumber.
- `split`: `train` / `valid` / `test` (split bawaan).
- `plate_text_gt`: transkripsi teks plat **hasil anotasi manual**. Kosongkan jika belum dianotasi.
- `unreadable`: `true/false` (atau `1/0`) — `true` jika teks plat tidak terbaca/ambigu secara visual.
- `notes`: catatan anotator (opsional, mis. kondisi buram/tertutup/ganda).
- `verified`: `true/false` (atau `1/0`) — `true` jika sudah diperiksa/QA.

## Konvensi Anotasi (disarankan)
- Gunakan **huruf kapital** sesuai tampilan plat (A–Z, 0–9).
- Pertahankan **spasi** sebagaimana terbaca pada plat (jika ragu, catat di `notes`).
- Jangan menebak karakter yang tidak terbaca; tandai `unreadable=true` dan jelaskan di `notes`.
- Jangan menyalin hasil OCR model sebagai ground truth.
- Simpan hasil anotasi di direktori ini (`data/annotations/ocr/`), bukan di `data/raw/`.

## Catatan
- Subset kandidat dipilih dari gambar dengan label deteksi non-empty (sesuai manifest M4). Label kosong & duplikat telah didokumentasikan terpisah.
- Crop plat otomatis belum tersedia (direncanakan Milestone 6); anotasi dapat dilakukan berbasis gambar sumber (dan/atau crop manual yang disimpan di lokasi kerja turunan bila diperlukan).
"""
    readme.write_text(readme_text, encoding="utf-8")
    written.append(readme.name)

    header = ["image_path", "split", "plate_text_gt", "unreadable", "notes", "verified"]

    template = ocr_dir / "ocr_annotations_template.csv"
    with template.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
    written.append(template.name)

    # eligible candidates: included and not empty label
    eligible: Dict[str, List[Dict[str, Any]]] = {}
    for split, rows in manifests.items():
        eligible[split] = [r for r in rows if r.get("included") and not r.get("empty_label")]

    counts = {s: len(eligible.get(s, [])) for s in SPLITS}
    alloc = proportional_allocation(candidate_total, list(SPLITS), counts)
    rng = random.Random(seed)
    selected_rows: List[Dict[str, Any]] = []
    selection_meta: Dict[str, Any] = {"seed": seed, "requested_total": candidate_total, "eligible_counts": counts, "allocation": alloc, "selected_counts": {}}

    for split in SPLITS:
        pool = list(eligible.get(split, []))
        pool.sort(key=lambda r: r["stem"])  # deterministic base order
        rng.shuffle(pool)
        n = min(alloc.get(split, 0), len(pool))
        chosen = pool[:n]
        # keep stable order in output by path for readability
        chosen.sort(key=lambda r: r["image_path"])
        selection_meta["selected_counts"][split] = len(chosen)
        for r in chosen:
            selected_rows.append({
                "image_path": r["image_path"],
                "split": split,
                "plate_text_gt": "",
                "unreadable": "false",
                "notes": "",
                "verified": "false",
            })

    cand_path = ocr_dir / "ocr_annotation_candidates.csv"
    with cand_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in selected_rows:
            w.writerow(r)
    written.append(cand_path.name)

    meta_path = ocr_dir / "ocr_candidates_meta.json"
    meta_path.write_text(json.dumps(selection_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(meta_path.name)

    return {"written": written, "selection": selection_meta, "candidates_path": str(cand_path)}


def generate_report(
    run_id: str,
    manifests: Dict[str, List[Dict[str, Any]]],
    dup_info: Dict[str, Any],
    dataset_yaml_path: Path,
    ocr_info: Dict[str, Any],
    report_run_dir: Path,
) -> Dict[str, Any]:
    ensure_dir(report_run_dir)
    split_stats = {}
    for split, rows in manifests.items():
        split_stats[split] = {
            "total": len(rows),
            "included": sum(1 for r in rows if r["included"]),
            "excluded": sum(1 for r in rows if not r["included"]),
            "empty_label": sum(1 for r in rows if r["empty_label"]),
            "duplicate_intra_split": sum(1 for r in rows if r["duplicate_type"] == "intra_split"),
        }
    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decisions": {
            "6.1": "A_provisional_keep_empty_labels_as_background_candidates_require_visual_verification",
            "6.2": "A_keep_all_document_intra_split_duplicates",
            "6.3": "C_configs_dataset_yaml_plus_data_splits_manifests_no_image_duplication",
            "6.4": "A_manifests_in_data_splits_json_csv",
            "6.5": "Aii_framework_plus_50_candidates_Ci_csv_schema",
            "6.6": "A_scripts_prepare_dataset_py",
            "6.7": "A_outputs_reports_dataset_preparation_run",
            "6.8": "C_update_data_readme_and_detailed_report",
        },
        "split_stats": split_stats,
        "duplicates_summary": dup_info["summary"],
        "dataset_config": str(dataset_yaml_path),
        "ocr": ocr_info.get("selection", {}),
        "readiness": {
            "detection": "READY_WITH_PREPARED_CONFIG_AND_MANIFESTS (split bawaan dipertahankan, 0 cross-split duplicate per audit M3; empty label provisional & duplikat intra-split terdokumentasi)",
            "ocr_evaluation": "PENDING_MANUAL_GROUND_TRUTH_ANNOTATION (kerangka & kandidat disiapkan, plate_text_gt belum diisi)",
        },
    }
    (report_run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append(f"# Laporan Persiapan Dataset ALPRO — Milestone 4\n")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Waktu: {report['timestamp']}")
    lines.append(f"- Config YOLO kerja: `{dataset_yaml_path.as_posix()}`")
    lines.append("")
    lines.append("## Keputusan yang Diterapkan")
    for k, v in report["decisions"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Statistik Manifest per Split")
    for s, st in split_stats.items():
        lines.append(f"- **{s}**: total={st['total']}, included={st['included']}, excluded={st['excluded']}, empty_label={st['empty_label']}, duplicate_intra_split={st['duplicate_intra_split']}")
    lines.append("")
    d = dup_info["summary"]
    lines.append("## Duplikasi (audit/konsistensi)")
    lines.append(f"- Grup total: {d['duplicate_groups_total']} | intra-split: {d['duplicate_groups_intra_split']} | lintas-split: {d['duplicate_groups_cross_split']}")
    lines.append(f"- File dalam grup intra-split: {d['duplicate_files_intra_split']} | lintas-split: {d['duplicate_files_cross_split']}")
    lines.append("")
    sel = ocr_info.get("selection", {})
    lines.append("## Kerangka OCR")
    lines.append(f"- Kandidat anotasi: {sum((sel.get('selected_counts') or {}).values())} (alokasi: {sel.get('allocation')}, eligible: {sel.get('eligible_counts')}, seed={sel.get('seed')})")
    lines.append("- Skema CSV: `image_path,split,plate_text_gt,unreadable,notes,verified` (teks GT manual, belum terisi)")
    lines.append("")
    lines.append("## Status Kesiapan")
    lines.append(f"- Deteksi: {report['readiness']['detection']}")
    lines.append(f"- Evaluasi OCR: {report['readiness']['ocr_evaluation']}")
    lines.append("")
    lines.append("## Catatan")
    lines.append("- 4 label kosong dipertahankan secara provisional (6.1A) dan **memerlukan verifikasi visual** sebelum training.")
    lines.append("- Tidak ada perubahan pada `data/raw/indonesian_license_plate/`.")
    (report_run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def update_data_readme(
    readme_path: Path,
    run_id: str,
    manifests: Dict[str, List[Dict[str, Any]]],
    dup_info: Dict[str, Any],
    ocr_info: Dict[str, Any],
) -> None:
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")
    marker = "\n## 8. Kesiapan & Rekomendasi Tindak Lanjut"
    idx = text.find(marker)
    if idx == -1:
        return
    base = text[:idx].rstrip()
    sel = ocr_info.get("selection", {})
    cand_counts = sel.get("selected_counts", {})
    cand_total = sum(cand_counts.values()) if cand_counts else 0
    new_section = f"""

## 8. Persiapan Dataset (Milestone 4)

- **Run persiapan:** `{run_id}` (`scripts/prepare_dataset.py`)
- **Laporan:** `outputs/reports/dataset_preparation/{run_id}/` (`report.md`, `report.json`)
- **Config YOLO kerja (turunan):** `configs/dataset.yaml` (menunjuk ke `data/raw/indonesian_license_plate/{{train,valid,test}}/images`; `data/raw` tidak diubah)
- **Manifest turunan:** `data/splits/` (`*_manifest.json/csv`, `preparation_manifest_summary.json`)

### Keputusan persiapan yang diterapkan
- **6.1A (provisional):** 4 label kosong **dipertahankan** sebagai kandidat background/no-object dalam manifest, **wajib verifikasi visual** sebelum training.
- **6.2A:** semua sampel **dipertahankan**; duplikat **intra-split** didokumentasikan (374 grup, 1085 file; 0 lintas-split per audit M3).
- **6.3C/6.4A:** config kerja + manifest turunan tanpa duplikasi file gambar.
- **6.5A(ii),C(i):** kerangka OCR dan **{cand_total} kandidat anotasi** (seed {sel.get('seed', 42)}, proporsional {cand_counts}) disiapkan di `data/annotations/ocr/` **tanpa** `plate_text_gt` otomatis.

### Status kesiapan pasca-persiapan
- **Deteksi (YOLO):** siap untuk baseline menggunakan `configs/dataset.yaml` dan manifest turunan (split bawaan dipertahankan, tidak ada indikasi leakage lintas-split). Perhatikan verifikasi visual label kosong dan over-representasi duplikat intra-split.
- **Evaluasi OCR:** **tertunda** hingga anotasi manual ground truth (`plate_text_gt`, terverifikasi) tersedia di `data/annotations/ocr/`.

## 9. Rekomendasi Tindak Lanjut
- Lakukan **verifikasi visual** 4 label kosong (rujuk visualisasi audit) dan perbarui manifest turunan jika keputusan per-file berubah (tanpa ubah `raw/`).
- Pertimbangkan strategi penanganan **duplikat intra-split** pada eksperimen training bila diperlukan (melalui data/manifest turunan).
- Pastikan konfigurasi training Ultralytics menggunakan path/config kerja yang resolve benar dari environment eksekusi.
- Lakukan **anotasi manual** subset OCR (mulai dari `data/annotations/ocr/ocr_annotation_candidates.csv`) dengan QA (`verified`) untuk evaluasi CER/WER/EMA.
- Pertahankan split bawaan kecuali ada alasan teknis jelas dan terdokumentasi.

## 10. Atribusi

Dataset: *Indonesian_License_Plate* oleh kontributor Roboflow Universe (`dataset-pyufc`), dilisensikan **CC BY 4.0**.  
Sumber: https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate
"""
    readme_path.write_text(base + new_section + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path.cwd()
    data_root = args.data_root
    run_id = make_run_id(args.run_id)
    report_run_dir = args.report_dir / run_id

    log(f"[prepare] run_id={run_id}")
    if not data_root.exists():
        log(f"[prepare] ERROR: data_root tidak ditemukan: {data_root}")
        return 2

    # gather images per split
    images_by_split = {s: list_images(data_root / s / "images") for s in SPLITS}
    log("[prepare] menghitung duplikat & membangun manifest")
    dup_info = compute_duplicate_info(data_root, images_by_split)
    manifests = build_manifests(data_root, project_root, dup_info)

    log("[prepare] menulis manifest (data/splits)")
    manifest_files = write_manifests(manifests, args.splits_dir)

    log("[prepare] menulis configs/dataset.yaml")
    dataset_yaml_path = write_dataset_yaml(args.configs_dir)

    log("[prepare] menyiapkan kerangka OCR & kandidat")
    ocr_info = build_ocr_framework(manifests, args.ocr_dir, args.seed, args.candidate_total, project_root)

    log("[prepare] menulis laporan persiapan")
    report = generate_report(run_id, manifests, dup_info, dataset_yaml_path, ocr_info, report_run_dir)

    if args.update_data_readme:
        log("[prepare] memperbarui data/README.md")
        update_data_readme(project_root / "data" / "README.md", run_id, manifests, dup_info, ocr_info)

    # console summary
    log("")
    log(f"[prepare] SELESAI run_id={run_id}")
    for s in SPLITS:
        st = report["split_stats"][s]
        log(f"[prepare] {s}: total={st['total']} included={st['included']} excluded={st['excluded']} empty={st['empty_label']} dup_intra={st['duplicate_intra_split']}")
    log(f"[prepare] duplicates: intra_groups={dup_info['summary']['duplicate_groups_intra_split']} cross_groups={dup_info['summary']['duplicate_groups_cross_split']}")
    log(f"[prepare] dataset_yaml={dataset_yaml_path}")
    log(f"[prepare] manifests_dir={args.splits_dir}")
    log(f"[prepare] ocr_dir={args.ocr_dir} candidates={sum((ocr_info['selection'].get('selected_counts') or {}).values())}")
    log(f"[prepare] report_dir={report_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
