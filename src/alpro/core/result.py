"""Pipeline result contracts for ALPRO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


def _utc_now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_json_safe(obj: Any, strip_ndarray: bool = False) -> Any:
    """Convert objects into JSON-serializable structures.

    When strip_ndarray=True, numpy arrays are removed/omitted because they are
    heavy image payloads not intended for JSON reports.
    """
    if isinstance(obj, np.ndarray):
        return None if not strip_ndarray else "__ndarray_stripped__"
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if strip_ndarray and isinstance(v, np.ndarray):
                continue
            out[str(k)] = _to_json_safe(v, strip_ndarray=strip_ndarray)
        return out
    if isinstance(obj, (list, tuple)):
        items = []
        for v in obj:
            if strip_ndarray and isinstance(v, np.ndarray):
                continue
            items.append(_to_json_safe(v, strip_ndarray=strip_ndarray))
        return items
    if isinstance(obj, PathLike):
        return str(obj)
    return obj


PathLike = (str, bytes)  # simple guard; pathlib.Path handled by __fspath__/str below


def _normalize_json_safe(obj: Any, strip_ndarray: bool = False) -> Any:
    """Wrapper that also handles pathlib.Path and common non-JSON scalars."""
    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        if strip_ndarray:
            return None
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if strip_ndarray and isinstance(v, np.ndarray):
                continue
            nv = _normalize_json_safe(v, strip_ndarray=strip_ndarray)
            if strip_ndarray and nv is None and isinstance(v, np.ndarray):
                continue
            out[str(k)] = nv
        return out
    if isinstance(obj, (list, tuple)):
        items = []
        for v in obj:
            if strip_ndarray and isinstance(v, np.ndarray):
                continue
            items.append(_normalize_json_safe(v, strip_ndarray=strip_ndarray))
        return items
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback for datetime/Path-like/other simple objects
    try:
        return str(obj)
    except Exception:
        return repr(obj)


@dataclass
class PipelineResult:
    status: str
    source: Optional[str] = None
    image_shape: Optional[List[int]] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)
    selected_detection: Optional[Dict[str, Any]] = None
    crop: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, Any]] = None
    ocr: Optional[Dict[str, Any]] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_images:
            # Strip heavy image arrays from common locations
            crop = d.get("crop")
            if isinstance(crop, dict):
                crop = {k: v for k, v in crop.items() if k != "image"}
                d["crop"] = crop

            pre = d.get("preprocessing")
            if isinstance(pre, dict):
                pre = {
                    k: v
                    for k, v in pre.items()
                    if k not in {"image", "intermediate", "intermediate_native"}
                }
                d["preprocessing"] = pre

            meta = d.get("meta")
            if isinstance(meta, dict):
                meta = {k: v for k, v in meta.items() if k != "annotated_image"}
                d["meta"] = meta

            d = _normalize_json_safe(d, strip_ndarray=True)
        else:
            d = _normalize_json_safe(d, strip_ndarray=False)
        return d

    @property
    def ok(self) -> bool:
        return self.status == "ok"
