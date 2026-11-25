"""Plate cropping utilities.

Memotong area plat dari gambar berdasarkan bounding box deteksi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class PlateCropper:
    """Crop area plat berdasarkan bbox deteksi."""

    @staticmethod
    def _clamp_bbox(
        bbox_xyxy: Sequence[float],
        width: int,
        height: int,
    ) -> Optional[List[int]]:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        # normalize invalid order
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1i = max(0, int(np.floor(x1)))
        y1i = max(0, int(np.floor(y1)))
        x2i = min(width, int(np.ceil(x2)))
        y2i = min(height, int(np.ceil(y2)))
        if x2i <= x1i or y2i <= y1i:
            return None
        return [x1i, y1i, x2i, y2i]

    @staticmethod
    def apply_padding(
        bbox_xyxy: Sequence[int],
        width: int,
        height: int,
        padding_ratio: float = 0.05,
    ) -> List[int]:
        """Tambahkan padding proporsional terhadap ukuran bbox dan clamp ke batas gambar."""
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        pad_x = int(round(bw * float(padding_ratio)))
        pad_y = int(round(bh * float(padding_ratio)))
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(width, x2 + pad_x)
        ny2 = min(height, y2 + pad_y)
        if nx2 <= nx1:
            nx2 = min(width, nx1 + 1)
        if ny2 <= ny1:
            ny2 = min(height, ny1 + 1)
        return [nx1, ny1, nx2, ny2]

    @staticmethod
    def crop(
        image: np.ndarray,
        bbox_xyxy: Sequence[float],
        padding_ratio: float = 0.05,
        min_size: int = 1,
    ) -> Dict[str, Any]:
        """Crop satu bbox dari image.

        Returns dict dengan status, crop image (jika sukses), dan metadata bbox.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            return {"status": "invalid_image", "crop": None, "error": "image kosong/tidak valid"}

        h, w = int(image.shape[0]), int(image.shape[1])
        clamped = PlateCropper._clamp_bbox(bbox_xyxy, w, h)
        if clamped is None:
            return {"status": "invalid_bbox", "crop": None, "error": "bbox di luar batas/tidak valid"}

        padded = PlateCropper.apply_padding(clamped, w, h, padding_ratio=padding_ratio)
        x1, y1, x2, y2 = padded
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return {
                "status": "too_small",
                "crop": None,
                "bbox_original_xyxy": [float(v) for v in bbox_xyxy],
                "bbox_clamped_xyxy": clamped,
                "bbox_padded_xyxy": padded,
                "error": "crop terlalu kecil",
            }

        crop_img = image[y1:y2, x1:x2].copy()
        if crop_img.size == 0:
            return {
                "status": "empty_crop",
                "crop": None,
                "bbox_original_xyxy": [float(v) for v in bbox_xyxy],
                "bbox_clamped_xyxy": clamped,
                "bbox_padded_xyxy": padded,
                "error": "hasil crop kosong",
            }

        return {
            "status": "ok",
            "crop": crop_img,
            "bbox_original_xyxy": [float(v) for v in bbox_xyxy],
            "bbox_clamped_xyxy": clamped,
            "bbox_padded_xyxy": padded,
            "padding_ratio": float(padding_ratio),
            "crop_shape": [int(crop_img.shape[0]), int(crop_img.shape[1]), int(crop_img.shape[2]) if crop_img.ndim == 3 else 1],
        }

    @staticmethod
    def select_top_detection(
        detections: Sequence[Dict[str, Any]],
        strategy: str = "top_confidence",
    ) -> Optional[Dict[str, Any]]:
        """Pilih satu deteksi berdasarkan strategi."""
        if not detections:
            return None
        dets = list(detections)
        if strategy == "top_confidence":
            dets.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
            return dets[0]
        # fallback: first as-is
        return dets[0]

    @staticmethod
    def crop_detections(
        image: np.ndarray,
        detections: Sequence[Dict[str, Any]],
        padding_ratio: float = 0.05,
        strategy: str = "top_confidence",
        min_size: int = 1,
    ) -> Dict[str, Any]:
        """Crop deteksi dari gambar sesuai strategi.

        Strategi:
        - top_confidence: ambil deteksi confidence tertinggi.
        - all: crop semua deteksi (diurutkan confidence desc).
        """
        dets = sorted(list(detections), key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        if not dets:
            return {
                "status": "no_detection",
                "selected_count": 0,
                "total_detections": 0,
                "crops": [],
            }

        if strategy == "all":
            selected = dets
        else:
            top = PlateCropper.select_top_detection(dets, strategy="top_confidence")
            selected = [top] if top else []

        crops: List[Dict[str, Any]] = []
        for i, det in enumerate(selected):
            bbox = det.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                crops.append({
                    "index": i,
                    "status": "invalid_bbox",
                    "crop": None,
                    "detection": det,
                    "error": "bbox_xyxy tidak valid",
                })
                continue
            c = PlateCropper.crop(image, bbox, padding_ratio=padding_ratio, min_size=min_size)
            c["index"] = i
            c["detection"] = det
            crops.append(c)

        ok_count = sum(1 for c in crops if c.get("status") == "ok")
        status = "ok" if ok_count > 0 else "crop_failed"
        return {
            "status": status,
            "strategy": strategy,
            "selected_count": len(selected),
            "ok_count": ok_count,
            "total_detections": len(dets),
            "crops": crops,
        }
