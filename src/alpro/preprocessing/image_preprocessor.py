"""Configurable image preprocessing pipeline for plate crops.

Mendukung tahap modular: grayscale, resize, denoise, contrast, threshold, morphology.
Setiap tahap dapat diaktifkan/nonaktifkan melalui config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class StepResult:
    name: str
    enabled: bool
    params: Dict[str, Any]
    image: np.ndarray


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Format gambar tidak didukung untuk grayscale: {image.shape}")


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Format gambar tidak didukung untuk BGR: {image.shape}")


class ImagePreprocessor:
    """Pipeline preprocessing citra berbasis daftar step terkonfigurasi."""

    DEFAULT_STEPS = [
        {"name": "grayscale", "enabled": True},
        {"name": "resize", "enabled": False, "width": 320, "keep_aspect": True},
        {"name": "denoise", "enabled": False, "method": "gaussian", "ksize": 3},
        {"name": "contrast", "enabled": False, "method": "clahe", "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        {"name": "threshold", "enabled": False, "method": "otsu"},
        {"name": "morphology", "enabled": False, "operation": "close", "ksize": [3, 3]},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        pipeline = cfg.get("pipeline", cfg) if isinstance(cfg, dict) else {}
        steps = pipeline.get("steps") if isinstance(pipeline, dict) else None
        self.steps: List[Dict[str, Any]] = self._normalize_steps(steps if isinstance(steps, list) else self.DEFAULT_STEPS)

    @staticmethod
    def _normalize_steps(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "")).strip().lower()
            if not name:
                continue
            item = dict(s)
            item["name"] = name
            item["enabled"] = bool(s.get("enabled", False))
            out.append(item)
        return out

    def process(
        self,
        image: np.ndarray,
        return_intermediate: bool = True,
        output_mode: str = "bgr",
    ) -> Dict[str, Any]:
        """Proses image melalui step aktif.

        Args:
            image: input image (BGR/gray).
            return_intermediate: simpan hasil tiap step aktif.
            output_mode: 'bgr' (default, mudah untuk simpan/visualisasi) atau 'native'.

        Returns:
            {
              "image": np.ndarray,
              "steps": [{"name","enabled","params","shape"}],
              "intermediate": {"step_name": np.ndarray} (opsional)
            }
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image kosong/tidak valid")

        current = image.copy()
        applied: List[Dict[str, Any]] = []
        intermediate: Dict[str, np.ndarray] = {}

        for step in self.steps:
            name = step["name"]
            enabled = bool(step.get("enabled", False))
            params = {k: v for k, v in step.items() if k not in {"name", "enabled"}}
            if not enabled:
                applied.append({"name": name, "enabled": False, "params": params, "applied": False})
                continue
            current = self._apply_step(current, name, params)
            applied.append({
                "name": name,
                "enabled": True,
                "params": params,
                "applied": True,
                "shape": [int(current.shape[0]), int(current.shape[1]), int(current.shape[2]) if current.ndim == 3 else 1],
            })
            if return_intermediate:
                intermediate[name] = current.copy()

        out_img = current if output_mode == "native" else _to_bgr(current)
        result = {"image": out_img, "steps": applied}
        if return_intermediate:
            # also provide BGR intermediates for easy saving/visual display
            result["intermediate"] = {k: _to_bgr(v) for k, v in intermediate.items()}
            result["intermediate_native"] = intermediate
        return result

    def _apply_step(self, image: np.ndarray, name: str, params: Dict[str, Any]) -> np.ndarray:
        if name == "grayscale":
            return _to_gray(image)
        if name == "resize":
            return self._step_resize(image, params)
        if name == "denoise":
            return self._step_denoise(image, params)
        if name == "contrast":
            return self._step_contrast(image, params)
        if name == "threshold":
            return self._step_threshold(image, params)
        if name == "morphology":
            return self._step_morphology(image, params)
        raise ValueError(f"Step preprocessing tidak dikenal: {name}")

    @staticmethod
    def _step_resize(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        h, w = image.shape[:2]
        width = params.get("width")
        height = params.get("height")
        keep_aspect = bool(params.get("keep_aspect", True))
        interpolation = params.get("interpolation", "linear")
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interp = interp_map.get(str(interpolation).lower(), cv2.INTER_LINEAR)

        if width is None and height is None:
            scale = float(params.get("scale", 1.0))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
        elif keep_aspect:
            if width is not None and height is None:
                new_w = max(1, int(width))
                scale = new_w / max(1, w)
                new_h = max(1, int(round(h * scale)))
            elif height is not None and width is None:
                new_h = max(1, int(height))
                scale = new_h / max(1, h)
                new_w = max(1, int(round(w * scale)))
            else:
                # fit within box preserving aspect
                new_w = max(1, int(width))
                new_h = max(1, int(height))
                scale = min(new_w / max(1, w), new_h / max(1, h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
        else:
            new_w = max(1, int(width if width is not None else w))
            new_h = max(1, int(height if height is not None else h))
        return cv2.resize(image, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _step_denoise(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        method = str(params.get("method", "gaussian")).lower()
        ksize = int(params.get("ksize", 3))
        ksize = max(1, ksize)
        if ksize % 2 == 0:
            ksize += 1
        if method == "gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        if method == "median":
            return cv2.medianBlur(image, ksize)
        if method == "bilateral":
            d = int(params.get("d", 5))
            sigma_color = float(params.get("sigma_color", 75))
            sigma_space = float(params.get("sigma_space", 75))
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        raise ValueError(f"Metode denoise tidak dikenal: {method}")

    @staticmethod
    def _step_contrast(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        method = str(params.get("method", "clahe")).lower()
        if method != "clahe":
            # simple fallback: histogram equalization
            if image.ndim == 3:
                gray = _to_gray(image)
                return cv2.equalizeHist(gray)
            return cv2.equalizeHist(image)

        clip_limit = float(params.get("clip_limit", 2.0))
        tile = params.get("tile_grid_size", [8, 8])
        if isinstance(tile, (list, tuple)) and len(tile) == 2:
            tile_grid = (int(tile[0]), int(tile[1]))
        else:
            tile_grid = (8, 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        if image.ndim == 3:
            # apply CLAHE on L channel to preserve some color info
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return clahe.apply(image)

    @staticmethod
    def _step_threshold(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        method = str(params.get("method", "otsu")).lower()
        gray = _to_gray(image)
        if method == "otsu":
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th
        if method == "adaptive":
            block_size = int(params.get("block_size", 11))
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1
            c = int(params.get("c", 2))
            adaptive_method_name = str(params.get("adaptive_method", "gaussian")).lower()
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adaptive_method_name == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
            th = cv2.adaptiveThreshold(gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c)
            return th
        # binary fallback
        thresh_val = float(params.get("thresh", 127))
        _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        return th

    @staticmethod
    def _step_morphology(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        operation = str(params.get("operation", "close")).lower()
        ksize = params.get("ksize", [3, 3])
        if isinstance(ksize, (list, tuple)) and len(ksize) == 2:
            k = (max(1, int(ksize[0])), max(1, int(ksize[1])))
        else:
            k = (3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k)
        src = image
        # morphology is most meaningful on binary/gray; apply on gray then keep single channel
        if src.ndim == 3:
            src = _to_gray(src)
        if operation == "open":
            return cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
        if operation == "close":
            return cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
        if operation == "dilate":
            return cv2.dilate(src, kernel, iterations=int(params.get("iterations", 1)))
        if operation == "erode":
            return cv2.erode(src, kernel, iterations=int(params.get("iterations", 1)))
        raise ValueError(f"Operasi morphology tidak dikenal: {operation}")
