"""Pipeline result contracts for ALPRO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
            if isinstance(crop, dict) and "image" in crop:
                crop = dict(crop)
                crop.pop("image", None)
                d["crop"] = crop
            pre = d.get("preprocessing")
            if isinstance(pre, dict):
                pre = dict(pre)
                pre.pop("image", None)
                pre.pop("intermediate", None)
                pre.pop("intermediate_native", None)
                d["preprocessing"] = pre
        return d

    @property
    def ok(self) -> bool:
        return self.status == "ok"
