#!/usr/bin/env python3
"""Audit dataset Roboflow (YOLO) untuk proyek ALPRO.

Script ini melakukan pemeriksaan read-only pada dataset di data/raw,
menghasilkan laporan/artefak di direktori output terpisah, dan opsional
membuat visualisasi sampel anotasi. Tidak memodifikasi data/raw.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"
SPLIT_NAMES = ("train", "valid", "test")


@dataclass
class Issue:
    severity: str  # ERROR, WARNING, INFO
    category: str
    message: str
    split: Optional[str] = None
    path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "split": self.split or "",
            "path": self.path or "",
            "message": self.message,
            "details": json.dumps(self.details, ensure_ascii=False) if self.details else "",
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset ALPRO (read-only pada data/raw).")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/indonesian_license_plate"),
        help="Root dataset Roboflow (default: data/raw/indonesian_license_plate)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/dataset_audit"),
        help="Direktori dasar laporan audit (default: outputs/reports/dataset_audit)",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=Path("outputs/visualizations/dataset_audit"),
        help="Direktori dasar visualisasi audit (default: outputs/visualizations/dataset_audit)",
    )
    parser.add_argument("--run-id", type=str, default=None, help="ID run; default: timestamp")
    parser.add_argument("--hash", choices=["md5", "sha256"], default="md5", help="Algoritma hash duplikasi")
    parser.add_argument("--sample-per-split", type=int, default=10, help="Jumlah sampel visualisasi per split")
    parser.add_argument("--seed", type=int, default=42, help="Seed sampling visualisasi")
    parser.add_argument("--no-visualize", action="store_true", help="Nonaktifkan pembuatan visualisasi sampel")
    parser.add_argument(
        "--skip-full-decode",
        action="store_true",
        help="Lewati full decode gambar (tidak disarankan; default: full decode aktif)",
    )
    return parser.parse_args(argv)


def log(msg: str) -> None:
    print(msg, flush=True)


def make_run_id(provided: Optional[str]) -> str:
    if provided:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in provided)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel_to(base: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        try:
            return str(path.relative_to(base))
        except Exception:
            return str(path)


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data_yaml(data_root: Path) -> Tuple[Optional[Dict[str, Any]], List[Issue], Dict[str, Any]]:
    issues: List[Issue] = []
    info: Dict[str, Any] = {"path": str(data_root / "data.yaml"), "exists": False}
    yaml_path = data_root / "data.yaml"
    if not yaml_path.exists():
        issues.append(Issue("ERROR", "structure", "data.yaml tidak ditemukan", path=str(yaml_path)))
        return None, issues, info
    info["exists"] = True
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        issues.append(Issue("ERROR", "yaml", f"Gagal membaca/parse data.yaml: {e}", path=str(yaml_path)))
        return None, issues, info

    info["content"] = data
    nc = data.get("nc")
    names = data.get("names")
    if nc != 1:
        issues.append(Issue("WARNING", "yaml", f"nc di data.yaml = {nc!r} (ekspektasi 1)", path=str(yaml_path), details={"nc": nc}))
    if names not in (["License-Plate"], {0: "License-Plate"}, {"License-Plate": 0}):
        # tolerate common forms but report if odd
        if names is None or (isinstance(names, list) and len(names) == 1 and names[0] == "License-Plate"):
            pass
        else:
            issues.append(Issue("WARNING", "yaml", f"names di data.yaml tidak umum: {names!r}", path=str(yaml_path), details={"names": names}))

    # path resolution checks
    resolved = {}
    for key in ("train", "val", "test"):
        raw_val = data.get(key)
        entry: Dict[str, Any] = {"raw": raw_val}
        if isinstance(raw_val, str):
            p_yaml = (yaml_path.parent / raw_val).resolve()
            p_root = (data_root / raw_val).resolve()
            p_cwd = (Path.cwd() / raw_val).resolve() if not Path(raw_val).is_absolute() else Path(raw_val).resolve()
            entry.update({
                "resolved_from_yaml_dir": str(p_yaml),
                "exists_from_yaml_dir": p_yaml.exists(),
                "resolved_from_data_root": str(p_root),
                "exists_from_data_root": p_root.exists(),
                "resolved_from_cwd": str(p_cwd),
                "exists_from_cwd": p_cwd.exists(),
            })
            if not p_yaml.exists():
                issues.append(Issue("WARNING", "yaml_path", f"Path '{key}' pada data.yaml tidak resolve dari lokasi yaml", path=str(yaml_path), details={"key": key, **entry}))
        else:
            issues.append(Issue("WARNING", "yaml", f"Entry path '{key}' tidak ditemukan/tidak valid", path=str(yaml_path), details={"key": key, "value": raw_val}))
        resolved[key] = entry
    info["path_resolution"] = resolved
    return data, issues, info


def list_split_files(data_root: Path, split: str) -> Tuple[Dict[str, Path], Dict[str, Path], List[Issue], Dict[str, Any]]:
    issues: List[Issue] = []
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    info: Dict[str, Any] = {
        "split": split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "images_dir_exists": images_dir.exists(),
        "labels_dir_exists": labels_dir.exists(),
    }
    if not images_dir.exists():
        issues.append(Issue("ERROR", "structure", f"Direktori images tidak ditemukan untuk split '{split}'", split=split, path=str(images_dir)))
    if not labels_dir.exists():
        issues.append(Issue("ERROR", "structure", f"Direktori labels tidak ditemukan untuk split '{split}'", split=split, path=str(labels_dir)))

    images: Dict[str, Path] = {}
    labels: Dict[str, Path] = {}
    image_exts_counter: Counter[str] = Counter()
    label_exts_counter: Counter[str] = Counter()
    nonstandard_files: List[str] = []

    if images_dir.exists():
        for p in images_dir.iterdir():
            if p.is_file():
                ext = p.suffix.lower()
                if ext in IMAGE_EXTS:
                    images[p.stem] = p
                    image_exts_counter[ext] += 1
                else:
                    nonstandard_files.append(str(p))
    if labels_dir.exists():
        for p in labels_dir.iterdir():
            if p.is_file():
                ext = p.suffix.lower()
                if ext == LABEL_EXT:
                    labels[p.stem] = p
                    label_exts_counter[ext] += 1
                else:
                    nonstandard_files.append(str(p))

    if nonstandard_files:
        issues.append(Issue("WARNING", "structure", f"Ditemukan {len(nonstandard_files)} file non-standar di split '{split}'", split=split, details={"examples": nonstandard_files[:10], "count": len(nonstandard_files)}))

    info["image_exts"] = dict(image_exts_counter)
    info["label_exts"] = dict(label_exts_counter)
    info["num_images"] = len(images)
    info["num_labels"] = len(labels)
    info["nonstandard_count"] = len(nonstandard_files)
    return images, labels, issues, info


def parse_and_validate_labels(labels: Dict[str, Path], split: str) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    per_file: Dict[str, Dict[str, Any]] = {}
    objects_per_image: List[int] = []
    bbox_widths: List[float] = []
    bbox_heights: List[float] = []
    bbox_areas: List[float] = []
    empty_files: List[str] = []
    invalid_files = 0

    for stem, path in labels.items():
        entry: Dict[str, Any] = {"path": str(path), "num_objects": 0, "empty": False, "valid": True, "errors": []}
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except Exception:
            # try permissive
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                entry["errors"].append("decode_warning")
            except Exception as e:
                entry["valid"] = False
                entry["errors"].append(f"read_error:{e}")
                invalid_files += 1
                issues.append(Issue("ERROR", "label_io", f"Gagal membaca label: {e}", split=split, path=str(path)))
                per_file[stem] = entry
                continue

        if text.strip() == "":
            entry["empty"] = True
            entry["num_objects"] = 0
            empty_files.append(str(path))
            objects_per_image.append(0)
            issues.append(Issue("WARNING", "empty_label", "Label kosong (0 objek) - perlu verifikasi visual", split=split, path=str(path)))
            per_file[stem] = entry
            continue

        lines = [ln for ln in text.splitlines() if ln.strip() != ""]
        file_valid = True
        obj_count = 0
        for i, ln in enumerate(lines, start=1):
            parts = ln.strip().split()
            if len(parts) != 5:
                file_valid = False
                entry["errors"].append(f"line_{i}_field_count:{len(parts)}")
                issues.append(Issue("ERROR", "label_format", f"Jumlah kolom bukan 5 (line {i})", split=split, path=str(path), details={"line": i, "content": ln, "count": len(parts)}))
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except Exception:
                file_valid = False
                entry["errors"].append(f"line_{i}_parse_error")
                issues.append(Issue("ERROR", "label_format", f"Nilai label tidak dapat diparse (line {i})", split=split, path=str(path), details={"line": i, "content": ln}))
                continue
            if cls != 0:
                file_valid = False
                entry["errors"].append(f"line_{i}_class:{cls}")
                issues.append(Issue("ERROR", "label_class", f"class_id != 0 (line {i}, cls={cls})", split=split, path=str(path), details={"line": i, "cls": cls}))
            # bbox validity (normalized)
            vals = (x, y, w, h)
            if any((v <= 0 or v > 1.0) for v in vals):
                file_valid = False
                entry["errors"].append(f"line_{i}_out_of_range")
                issues.append(Issue("ERROR", "label_bbox", f"Nilai bbox di luar rentang (0,1] (line {i})", split=split, path=str(path), details={"line": i, "xywh": [x, y, w, h]}))
            # optional logical bounds (center +/- size/2 inside [0,1]) - warning if slightly off due to float
            x1, x2 = x - w / 2.0, x + w / 2.0
            y1, y2 = y - h / 2.0, y + h / 2.0
            if x1 < -1e-6 or y1 < -1e-6 or x2 > 1.0 + 1e-6 or y2 > 1.0 + 1e-6:
                # mark as warning or error? can break training in strict cases; treat as ERROR for safety
                file_valid = False
                entry["errors"].append(f"line_{i}_bbox_outside_image")
                issues.append(Issue("ERROR", "label_bbox", f"Bbox keluar batas gambar ternormalisasi (line {i})", split=split, path=str(path), details={"line": i, "xyxy_norm": [x1, y1, x2, y2]}))
            obj_count += 1
            bbox_widths.append(w)
            bbox_heights.append(h)
            bbox_areas.append(w * h)

        entry["num_objects"] = obj_count
        entry["valid"] = file_valid
        objects_per_image.append(obj_count)
        if not file_valid:
            invalid_files += 1
        per_file[stem] = entry

    summary = {
        "num_label_files": len(labels),
        "empty_label_files": len(empty_files),
        "invalid_label_files": invalid_files,
        "total_objects": sum(objects_per_image),
        "objects_per_image": {
            "min": min(objects_per_image) if objects_per_image else None,
            "max": max(objects_per_image) if objects_per_image else None,
            "mean": (sum(objects_per_image) / len(objects_per_image)) if objects_per_image else None,
            "distribution": dict(Counter(objects_per_image)),
        },
        "bbox_stats": {
            "count": len(bbox_widths),
            "w_min": min(bbox_widths) if bbox_widths else None,
            "w_max": max(bbox_widths) if bbox_widths else None,
            "h_min": min(bbox_heights) if bbox_heights else None,
            "h_max": max(bbox_heights) if bbox_heights else None,
            "area_min": min(bbox_areas) if bbox_areas else None,
            "area_max": max(bbox_areas) if bbox_areas else None,
        },
        "empty_file_paths": empty_files,
    }
    return {"summary": summary, "files": per_file}, issues


def decode_images(images: Dict[str, Path], split: str, full_decode: bool) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    props: Dict[str, Dict[str, Any]] = {}
    sizes_counter: Counter[str] = Counter()
    channels_counter: Counter[int] = Counter()
    file_sizes: List[int] = []
    unreadable: List[str] = []
    unexpected_size: List[str] = []

    for stem, path in images.items():
        rec: Dict[str, Any] = {"path": str(path), "file_size_bytes": None, "readable": None}
        try:
            rec["file_size_bytes"] = path.stat().st_size
            file_sizes.append(rec["file_size_bytes"])
        except Exception as e:
            rec["file_size_bytes"] = None
            issues.append(Issue("WARNING", "image_stat", f"Gagal membaca ukuran file: {e}", split=split, path=str(path)))

        if not full_decode:
            rec["readable"] = None
            props[stem] = rec
            continue

        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            img = None
            rec["read_error"] = str(e)
        if img is None:
            rec.update({"readable": False, "width": None, "height": None, "channels": None})
            unreadable.append(str(path))
            issues.append(Issue("ERROR", "image_corrupt", "Gambar tidak dapat didecode/dibaca", split=split, path=str(path)))
            props[stem] = rec
            continue

        h, w = img.shape[:2]
        c = 1 if img.ndim == 2 else img.shape[2]
        rec.update({"readable": True, "width": w, "height": h, "channels": c})
        sizes_counter[f"{w}x{h}"] += 1
        channels_counter[c] += 1
        if (w, h) != (640, 640):
            unexpected_size.append(str(path))
            issues.append(Issue("WARNING", "image_property", f"Dimensi tidak 640x640 ({w}x{h})", split=split, path=str(path), details={"width": w, "height": h}))
        # grayscale export expectation - flag if not 1-channel and not obviously grayscale-in-3ch? record only
        if c not in (1, 3, 4):
            issues.append(Issue("WARNING", "image_property", f"Jumlah channel tidak umum ({c})", split=split, path=str(path), details={"channels": c}))
        props[stem] = rec

    summary = {
        "num_images": len(images),
        "full_decode": full_decode,
        "unreadable_count": len(unreadable),
        "unexpected_size_count": len(unexpected_size),
        "sizes": dict(sizes_counter),
        "channels": {str(k): v for k, v in channels_counter.items()},
        "file_size_bytes": {
            "min": min(file_sizes) if file_sizes else None,
            "max": max(file_sizes) if file_sizes else None,
            "mean": (sum(file_sizes) / len(file_sizes)) if file_sizes else None,
        },
        "unreadable_paths": unreadable,
    }
    return {"summary": summary, "files": props}, issues


def analyze_duplicates(
    per_split_images: Dict[str, Dict[str, Path]], algo: str
) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    hash_to_items: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    file_hashes: Dict[str, Dict[str, str]] = {}

    for split, images in per_split_images.items():
        for stem, path in images.items():
            try:
                h = file_hash(path, algo)
            except Exception as e:
                issues.append(Issue("ERROR", "hash", f"Gagal menghitung hash: {e}", split=split, path=str(path)))
                continue
            item = {"split": split, "stem": stem, "path": str(path), "hash": h}
            hash_to_items[h].append(item)
            file_hashes.setdefault(split, {})[stem] = h

    groups: List[Dict[str, Any]] = []
    intra_groups = 0
    cross_groups = 0
    cross_split_files = 0
    intra_split_files = 0
    for h, items in hash_to_items.items():
        if len(items) < 2:
            continue
        splits = sorted({it["split"] for it in items})
        gtype = "cross_split" if len(splits) > 1 else "intra_split"
        group = {
            "hash": h,
            "algo": algo,
            "count": len(items),
            "type": gtype,
            "splits": splits,
            "members": items,
        }
        groups.append(group)
        if gtype == "cross_split":
            cross_groups += 1
            cross_split_files += len(items)
            issues.append(Issue(
                "ERROR", "duplicate_cross_split",
                f"Duplikat lintas-split terdeteksi (hash={h[:12]}..., count={len(items)}, splits={','.join(splits)})",
                details={"hash": h, "count": len(items), "splits": splits, "paths": [it["path"] for it in items]},
            ))
        else:
            intra_groups += 1
            intra_split_files += len(items)
            issues.append(Issue(
                "WARNING", "duplicate_intra_split",
                f"Duplikat intra-split terdeteksi di {splits[0]} (hash={h[:12]}..., count={len(items)})",
                split=splits[0],
                details={"hash": h, "count": len(items), "split": splits[0], "paths": [it["path"] for it in items]},
            ))

    # stable order: cross first, then larger groups
    groups.sort(key=lambda g: (0 if g["type"] == "cross_split" else 1, -g["count"], g["hash"]))

    summary = {
        "algo": algo,
        "total_files_hashed": sum(len(v) for v in file_hashes.values()),
        "duplicate_groups_total": len(groups),
        "duplicate_groups_cross_split": cross_groups,
        "duplicate_groups_intra_split": intra_groups,
        "duplicate_files_cross_split": cross_split_files,
        "duplicate_files_intra_split": intra_split_files,
    }
    return {"summary": summary, "groups": groups, "file_hashes": file_hashes}, issues


def pair_checks(
    split: str,
    images: Dict[str, Path],
    labels: Dict[str, Path],
) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    img_keys = set(images.keys())
    lbl_keys = set(labels.keys())
    img_wo_lbl = sorted(img_keys - lbl_keys)
    lbl_wo_img = sorted(lbl_keys - img_keys)
    for stem in img_wo_lbl:
        issues.append(Issue("ERROR", "pairing", "Image tanpa label pasangan", split=split, path=str(images[stem]), details={"stem": stem}))
    for stem in lbl_wo_img:
        issues.append(Issue("ERROR", "pairing", "Label tanpa image pasangan", split=split, path=str(labels[stem]), details={"stem": stem}))
    summary = {
        "images_without_label": len(img_wo_lbl),
        "labels_without_image": len(lbl_wo_img),
        "images_without_label_examples": img_wo_lbl[:10],
        "labels_without_image_examples": lbl_wo_img[:10],
    }
    return summary, issues


def choose_visualization_samples(
    per_split_images: Dict[str, Dict[str, Path]],
    label_info: Dict[str, Any],
    dup_info: Dict[str, Any],
    sample_per_split: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    selected: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in per_split_images}

    def add(split: str, stem: str, reason: str) -> None:
        if split not in selected or stem not in per_split_images.get(split, {}):
            return
        rec = selected[split].setdefault(stem, {"split": split, "stem": stem, "reasons": []})
        if reason not in rec["reasons"]:
            rec["reasons"].append(reason)

    # forced: empty labels
    for split, info in label_info.items():
        for p in info.get("summary", {}).get("empty_file_paths", []):
            stem = Path(p).stem
            add(split, stem, "empty_label")

    # forced: representatives from duplicate groups (cross-split priority)
    for g in dup_info.get("groups", []):
        reason = "dup_cross_split" if g.get("type") == "cross_split" else "dup_intra_split"
        # include up to 2 members per group to show evidence, prefer covering splits
        members = g.get("members", [])
        picked = members[:2]
        # try to include different splits for cross-split
        if g.get("type") == "cross_split":
            seen_s = set()
            picked2 = []
            for m in members:
                if m["split"] not in seen_s:
                    picked2.append(m)
                    seen_s.add(m["split"])
                if len(picked2) >= 2:
                    break
            picked = picked2 or picked
        for m in picked:
            add(m["split"], m["stem"], reason)

    # forced: multi-object examples if present
    for split, info in label_info.items():
        files = info.get("files", {})
        multi = [stem for stem, rec in files.items() if (rec.get("num_objects") or 0) > 1]
        for stem in multi[:2]:
            add(split, stem, "multi_object")

    # base random per split
    for split, images in per_split_images.items():
        stems = sorted(images.keys())
        rng.shuffle(stems)
        for stem in stems:
            if len(selected[split]) >= sample_per_split and stem not in selected[split]:
                continue
            if stem not in selected[split]:
                # fill until reaching sample_per_split total (including forced)
                if len([1 for v in selected[split].values() if "random_base" in v["reasons"] or not v["reasons"]]) >= sample_per_split:
                    pass
            # simpler: ensure at least sample_per_split total by adding randoms
        # deterministic fill
        current = selected[split]
        if len(current) < sample_per_split:
            remaining = [s for s in stems if s not in current]
            rng.shuffle(remaining)
            need = sample_per_split - len(current)
            for stem in remaining[:need]:
                add(split, stem, "random_base")
        # if still less (e.g. tiny split), fine

    # flatten
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, recs in selected.items():
        vals = list(recs.values())
        vals.sort(key=lambda r: r["stem"])
        out[split] = vals
    return out


def draw_label_visualization(
    image_path: Path,
    label_path: Optional[Path],
    out_path: Path,
    split: str,
) -> Dict[str, Any]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    result: Dict[str, Any] = {"image": str(image_path), "output": str(out_path), "ok": False}
    if img is None:
        result["error"] = "cannot_read_image"
        return result
    h, w = img.shape[:2]
    lines: List[Tuple[int, float, float, float, float]] = []
    empty_label = False
    if label_path and label_path.exists():
        try:
            text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            text = ""
        if text == "":
            empty_label = True
        else:
            for ln in text.splitlines():
                parts = ln.strip().split()
                if len(parts) == 5:
                    try:
                        cls = int(float(parts[0]))
                        x, y, bw, bh = map(float, parts[1:])
                        lines.append((cls, x, y, bw, bh))
                    except Exception:
                        pass
            if not lines and text != "":
                # had content but unparsable; still show note
                empty_label = False
    else:
        empty_label = True

    for cls, x, y, bw, bh in lines:
        cx, cy, ww, hh = x * w, y * h, bw * w, bh * h
        x1 = int(round(cx - ww / 2))
        y1 = int(round(cy - hh / 2))
        x2 = int(round(cx + ww / 2))
        y2 = int(round(cy + hh / 2))
        x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"cls:{cls}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    header = f"{split} | {image_path.name} | objs:{len(lines)}"
    if empty_label and not lines:
        header += " | EMPTY LABEL / 0 objects"
        cv2.putText(img, "EMPTY LABEL / 0 objects", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    # top bar background for readability
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, header[:120], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    ensure_dir(out_path.parent)
    ok = cv2.imwrite(str(out_path), img)
    result.update({"ok": bool(ok), "num_objects_drawn": len(lines), "empty_label": empty_label and not lines})
    if not ok:
        result["error"] = "imwrite_failed"
    return result


def determine_status(issues: List[Issue]) -> Tuple[str, Dict[str, int], List[str]]:
    counts = Counter(i.severity for i in issues)
    reasons: List[str] = []
    errors = [i for i in issues if i.severity == "ERROR"]
    warnings = [i for i in issues if i.severity == "WARNING"]
    if errors:
        # summarize key reasons
        cats = Counter(i.category for i in errors)
        reasons.append(f"Terdapat {len(errors)} ERROR ({', '.join(f'{k}:{v}' for k,v in cats.most_common())})")
        status = "NOT_READY"
    elif warnings:
        cats = Counter(i.category for i in warnings)
        reasons.append(f"Tidak ada ERROR, namun terdapat {len(warnings)} WARNING ({', '.join(f'{k}:{v}' for k,v in cats.most_common())})")
        status = "READY_WITH_WARNINGS"
    else:
        reasons.append("Tidak ditemukan ERROR/WARNING material")
        status = "READY"
    return status, dict(counts), reasons


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def build_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    status = report["status"]
    lines: List[str] = []
    lines.append(f"# Laporan Audit Dataset ALPRO")
    lines.append("")
    lines.append(f"- **Run ID:** `{report['run_id']}`")
    lines.append(f"- **Waktu (lokal):** {report['timestamp']}")
    lines.append(f"- **Data root:** `{report['data_root']}`")
    lines.append(f"- **Status:** **{status['value']}**")
    for r in status.get("reasons", []):
        lines.append(f"- **Alasan:** {r}")
    lines.append("")
    lines.append("## Ringkasan")
    totals = s.get("totals", {})
    lines.append(f"- Total gambar: **{totals.get('images', 0)}** | Total label: **{totals.get('labels', 0)}**")
    lines.append(f"- Split: train={totals.get('train_images',0)}/{totals.get('train_labels',0)}, valid={totals.get('valid_images',0)}/{totals.get('valid_labels',0)}, test={totals.get('test_images',0)}/{totals.get('test_labels',0)} (images/labels)")
    dups = s.get("duplicates", {})
    lines.append(f"- Duplikat grup: total={dups.get('duplicate_groups_total',0)}, lintas-split={dups.get('duplicate_groups_cross_split',0)} (ERROR), intra-split={dups.get('duplicate_groups_intra_split',0)} (WARNING)")
    sev = status.get("counts", {})
    lines.append(f"- Isu: ERROR={sev.get('ERROR',0)}, WARNING={sev.get('WARNING',0)}, INFO={sev.get('INFO',0)}")
    lines.append("")
    lines.append("## Temuan Utama")
    if dups.get("duplicate_groups_cross_split", 0) > 0:
        lines.append(f"- **ERROR:** Terdeteksi duplikat lintas-split ({dups.get('duplicate_groups_cross_split')} grup, {dups.get('duplicate_files_cross_split')} file). Berisiko data leakage; penanganan harus pada data turunan (bukan `data/raw`).")
    # empty labels
    empty_total = 0
    for sp in s.get("splits", {}).values():
        empty_total += sp.get("labels", {}).get("empty_label_files", 0)
    if empty_total:
        lines.append(f"- **WARNING:** {empty_total} label kosong terdeteksi (perlu verifikasi visual; dapat berarti background/no-object atau anotasi hilang).")
    # unreadable
    unread = 0
    for sp in s.get("splits", {}).values():
        unread += sp.get("images", {}).get("unreadable_count", 0)
    if unread:
        lines.append(f"- **ERROR:** {unread} gambar tidak dapat didecode/unreadable.")
    inv = 0
    for sp in s.get("splits", {}).values():
        inv += sp.get("labels", {}).get("invalid_label_files", 0)
    if inv:
        lines.append(f"- **ERROR:** {inv} file label invalid.")
    lines.append("")
    lines.append("## Per Split")
    for split, sp in s.get("splits", {}).items():
        lines.append(f"### {split}")
        lines.append(f"- Images: {sp.get('num_images',0)} | Labels: {sp.get('num_labels',0)} | Ekstensi img: {sp.get('image_exts',{})} | Ekstensi lbl: {sp.get('label_exts',{})}")
        pair = sp.get("pairing", {})
        lines.append(f"- Pairing: image tanpa label={pair.get('images_without_label',0)}, label tanpa image={pair.get('labels_without_image',0)}")
        ls = sp.get("labels", {})
        lines.append(f"- Label: empty={ls.get('empty_label_files',0)}, invalid={ls.get('invalid_label_files',0)}, total_objects={ls.get('total_objects',0)}, distribusi objek/gambar={ls.get('objects_per_image',{}).get('distribution',{})}")
        im = sp.get("images", {})
        lines.append(f"- Gambar: unreadable={im.get('unreadable_count',0)}, sizes={im.get('sizes',{})}, channels={im.get('channels',{})}")
        lines.append("")
    lines.append("## data.yaml & Kompatibilitas")
    y = s.get("yaml", {})
    lines.append(f"- Ada: {y.get('exists')} | nc/names: {y.get('content',{}).get('nc')} / {y.get('content',{}).get('names')}")
    lines.append("- Resolusi path (ringkas): lihat `report.json` (path_resolution). Catatan: path relatif bawaan Roboflow umum terhadap lokasi yaml; pastikan resolusi saat training.")
    lines.append("")
    lines.append("## Ground Truth OCR")
    lines.append("- Tidak ditemukan transkripsi teks plat di dataset (label YOLO hanya bounding box). Evaluasi OCR kuantitatif memerlukan anotasi teks terpisah di `data/annotations/ocr/`.")
    lines.append("")
    lines.append("## Rekomendasi")
    status_val = status["value"]
    if status_val == "NOT_READY":
        lines.append("- **Status NOT_READY:** selesaikan/mitigasi isu ERROR sebelum melanjutkan ke persiapan/training. Duplikat lintas-split dan/atau data corrupt/invalid harus ditangani pada data turunan/split kerja (mis. `data/interim/`, `data/processed/`, `data/splits/`), **tanpa** mengubah `data/raw/`.")
    elif status_val == "READY_WITH_WARNINGS":
        lines.append("- **Status READY_WITH_WARNINGS:** lanjut ke Milestone 4 dengan kehati-hatian. Verifikasi visual label kosong dan pertimbangkan dampak duplikat intra-split; tetapkan strategi penanganan pada data turunan.")
    else:
        lines.append("- **Status READY:** dataset dapat dilanjutkan ke tahap persiapan (tetap verifikasi visual sampling disarankan).")
    lines.append("- Pertahankan split bawaan kecuali ada alasan teknis jelas; siapkan rencana anotasi ground truth OCR (subset manual) untuk evaluasi teks.")
    lines.append("")
    lines.append("## Artefak")
    lines.append("- `report.json`, `report.md`, `issues.csv`, `duplicates.csv` (dan `duplicates.json`) di folder run laporan.")
    lines.append(f"- Visualisasi sampel (jika aktif): `{report.get('visualization_dir','')}`")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_root: Path = args.data_root
    run_id = make_run_id(args.run_id)
    report_dir = args.output_dir / run_id
    vis_dir = args.visualization_dir / run_id
    ensure_dir(report_dir)

    log(f"[audit] run_id={run_id}")
    log(f"[audit] data_root={data_root}")

    all_issues: List[Issue] = []
    report: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "report_dir": str(report_dir),
        "visualization_dir": str(vis_dir) if not args.no_visualize else None,
        "config": {
            "hash": args.hash,
            "full_decode": not args.skip_full_decode,
            "visualize": not args.no_visualize,
            "sample_per_split": args.sample_per_split,
            "seed": args.seed,
        },
    }

    # structure/yaml
    log("[audit] tahap 1/6: struktur & data.yaml")
    if not data_root.exists():
        all_issues.append(Issue("ERROR", "structure", "Direktori dataset tidak ditemukan", path=str(data_root)))
        status, counts, reasons = determine_status(all_issues)
        report.update({"status": {"value": status, "counts": counts, "reasons": reasons}, "summary": {}, "issues": [i.to_row() for i in all_issues]})
        (report_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[audit] SELESAI status={status} (dataset tidak ditemukan)")
        return 2

    yaml_data, yaml_issues, yaml_info = load_data_yaml(data_root)
    all_issues.extend(yaml_issues)
    # source readmes metadata note
    for name in ("README.dataset.txt", "README.roboflow.txt"):
        p = data_root / name
        if not p.exists():
            all_issues.append(Issue("INFO", "metadata", f"File sumber {name} tidak ditemukan", path=str(p)))

    # splits inventory
    log("[audit] tahap 2/6: inventaris & pairing")
    per_split_images: Dict[str, Dict[str, Path]] = {}
    per_split_labels: Dict[str, Dict[str, Path]] = {}
    split_infos: Dict[str, Any] = {}
    for split in SPLIT_NAMES:
        images, labels, inv_issues, inv_info = list_split_files(data_root, split)
        all_issues.extend(inv_issues)
        pair_summary, pair_issues = pair_checks(split, images, labels)
        all_issues.extend(pair_issues)
        per_split_images[split] = images
        per_split_labels[split] = labels
        split_infos[split] = {**inv_info, "pairing": pair_summary}

    # labels validation
    log("[audit] tahap 3/6: validasi label")
    label_info: Dict[str, Any] = {}
    for split, labels in per_split_labels.items():
        info, issues = parse_and_validate_labels(labels, split)
        all_issues.extend(issues)
        label_info[split] = info
        split_infos[split]["labels"] = info["summary"]

    # image decode
    log("[audit] tahap 4/6: pemeriksaan gambar (decode & properti)")
    image_info: Dict[str, Any] = {}
    for split, images in per_split_images.items():
        info, issues = decode_images(images, split, full_decode=not args.skip_full_decode)
        all_issues.extend(issues)
        image_info[split] = info
        split_infos[split]["images"] = info["summary"]

    # duplicates
    log("[audit] tahap 5/6: analisis duplikat hash")
    dup_info, dup_issues = analyze_duplicates(per_split_images, args.hash)
    all_issues.extend(dup_issues)

    # OCR ground truth note
    all_issues.append(Issue("INFO", "ocr_ground_truth", "Tidak ditemukan ground truth teks OCR di dataset (hanya anotasi bbox YOLO). Evaluasi OCR kuantitatif memerlukan anotasi teks terpisah."))

    # visualizations
    vis_results: Dict[str, Any] = {"enabled": not args.no_visualize, "output_dir": str(vis_dir), "samples": {}}
    if not args.no_visualize:
        log("[audit] tahap 6/6: visualisasi sampel")
        ensure_dir(vis_dir)
        samples = choose_visualization_samples(per_split_images, label_info, dup_info, args.sample_per_split, args.seed)
        for split, items in samples.items():
            vis_results["samples"][split] = []
            for it in items:
                stem = it["stem"]
                img_path = per_split_images[split].get(stem)
                lbl_path = per_split_labels[split].get(stem)
                if not img_path:
                    continue
                reason_tag = "_".join(it.get("reasons", ["sample"]))[:40]
                out_name = f"{stem}__{reason_tag}.jpg"
                out_path = vis_dir / split / out_name
                res = draw_label_visualization(img_path, lbl_path, out_path, split)
                res.update({"stem": stem, "reasons": it.get("reasons", [])})
                vis_results["samples"][split].append(res)
    else:
        log("[audit] tahap 6/6: visualisasi dilewati (--no-visualize)")

    # summary & status
    totals = {
        "images": sum(len(v) for v in per_split_images.values()),
        "labels": sum(len(v) for v in per_split_labels.values()),
    }
    for split in SPLIT_NAMES:
        totals[f"{split}_images"] = len(per_split_images.get(split, {}))
        totals[f"{split}_labels"] = len(per_split_labels.get(split, {}))

    status_value, status_counts, status_reasons = determine_status(all_issues)
    report.update({
        "status": {"value": status_value, "counts": status_counts, "reasons": status_reasons},
        "summary": {
            "totals": totals,
            "yaml": yaml_info,
            "splits": split_infos,
            "duplicates": dup_info["summary"],
            "visualization": {k: v for k, v in vis_results.items() if k != "samples"},
        },
        "duplicates": {"summary": dup_info["summary"], "groups": dup_info["groups"]},
        "visualization": vis_results,
        "issues": [i.to_row() for i in all_issues],
    })

    # write outputs
    log("[audit] menulis laporan")
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "report.md").write_text(build_markdown(report), encoding="utf-8")
    write_csv(report_dir / "issues.csv", [i.to_row() for i in all_issues], ["severity", "category", "split", "path", "message", "details"])
    # duplicates csv/json
    dup_rows = []
    for g in dup_info["groups"]:
        for m in g.get("members", []):
            dup_rows.append({
                "hash": g["hash"], "algo": g["algo"], "type": g["type"], "group_count": g["count"],
                "group_splits": ",".join(g.get("splits", [])), "member_split": m.get("split", ""),
                "member_stem": m.get("stem", ""), "member_path": m.get("path", ""),
            })
    write_csv(report_dir / "duplicates.csv", dup_rows, ["hash", "algo", "type", "group_count", "group_splits", "member_split", "member_stem", "member_path"])
    (report_dir / "duplicates.json").write_text(json.dumps(dup_info, indent=2, ensure_ascii=False), encoding="utf-8")

    # console summary
    log("")
    log(f"[audit] STATUS: {status_value}")
    for r in status_reasons:
        log(f"[audit] - {r}")
    log(f"[audit] totals images={totals['images']} labels={totals['labels']} | duplicates cross={dup_info['summary']['duplicate_groups_cross_split']} intra={dup_info['summary']['duplicate_groups_intra_split']}")
    log(f"[audit] report_dir={report_dir}")
    if not args.no_visualize:
        log(f"[audit] visualization_dir={vis_dir}")
    return 0 if status_value != "NOT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
