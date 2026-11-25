"""Config loading utilities for ALPRO."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def resolve_path(path: Union[str, Path, None], root: Optional[Path] = None) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    base = root or PROJECT_ROOT
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_pipeline_config(path: Union[str, Path] = "configs/pipeline.yaml") -> Dict[str, Any]:
    cfg = load_yaml(path)
    # resolve common paths for convenience
    det = cfg.get("detector", {}) if isinstance(cfg.get("detector"), dict) else {}
    if det.get("weights"):
        det["weights"] = str(resolve_path(det["weights"]))
        cfg["detector"] = det
    pre = cfg.get("preprocessing", {}) if isinstance(cfg.get("preprocessing"), dict) else {}
    if pre.get("config_path"):
        pre["config_path"] = str(resolve_path(pre["config_path"]))
        cfg["preprocessing"] = pre
    ocr = cfg.get("ocr", {}) if isinstance(cfg.get("ocr"), dict) else {}
    if ocr.get("config_path"):
        ocr["config_path"] = str(resolve_path(ocr["config_path"]))
        cfg["ocr"] = ocr
    out = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}
    if out.get("dir"):
        out["dir"] = str(resolve_path(out["dir"]))
        cfg["output"] = out
    return cfg
