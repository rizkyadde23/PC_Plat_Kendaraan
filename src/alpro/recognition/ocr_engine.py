"""EasyOCR-based OCR engine wrapper for ALPRO.

Menyimpan kandidat mentah OCR agar dapat ditelusuri.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import easyocr
except Exception as e:  # pragma: no cover
    easyocr = None  # type: ignore
    _EASYOCR_IMPORT_ERROR = e
else:
    _EASYOCR_IMPORT_ERROR = None


ImageInput = np.ndarray


def _ensure_easyocr() -> None:
    if easyocr is None:
        raise ImportError(
            "EasyOCR tidak tersedia. Pastikan dependensi terpasang."
        ) from _EASYOCR_IMPORT_ERROR


class OCREngine:
    """Wrapper OCR engine dengan status terstruktur."""

    def __init__(
        self,
        lang: Optional[Sequence[str]] = None,
        gpu: bool = False,
        detail: int = 1,
        paragraph: bool = False,
        decoder: Optional[str] = None,
        beamWidth: int = 5,
        batch_size: int = 1,
        min_confidence: float = 0.0,
    ) -> None:
        _ensure_easyocr()
        self.lang = list(lang or ["en"])
        self.gpu = bool(gpu)
        self.detail = int(detail)
        self.paragraph = bool(paragraph)
        self.decoder = decoder
        self.beamWidth = int(beamWidth)
        self.batch_size = int(batch_size)
        self.min_confidence = float(min_confidence)
        self._reader = easyocr.Reader(self.lang, gpu=self.gpu)

    def read(self, image: ImageInput) -> Dict[str, Any]:
        """Jalankan OCR pada image crop.

        Returns dict terstruktur:
        - status: ok | empty_ocr | invalid_image | ocr_error
        - text_raw: gabungan kandidat (diurutkan seperti output EasyOCR)
        - confidence: rata-rata confidence kandidat (0 jika kosong)
        - candidates: daftar kandidat mentah
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "status": "invalid_image",
                "text_raw": "",
                "confidence": 0.0,
                "candidates": [],
                "error": "image kosong/tidak valid",
            }

        try:
            # EasyOCR supports grayscale or BGR/BGRA; pass as-is.
            read_kwargs: Dict[str, Any] = {
                "detail": self.detail,
                "paragraph": self.paragraph,
                "batch_size": self.batch_size,
            }
            if self.decoder is not None:
                read_kwargs["decoder"] = self.decoder
                read_kwargs["beamWidth"] = self.beamWidth
            raw_results = self._reader.readtext(image, **read_kwargs)
        except Exception as e:
            return {
                "status": "ocr_error",
                "text_raw": "",
                "confidence": 0.0,
                "candidates": [],
                "error": str(e),
            }

        candidates = self._normalize_candidates(raw_results)
        filtered = [c for c in candidates if float(c.get("confidence", 0.0)) >= self.min_confidence]

        if not filtered:
            return {
                "status": "empty_ocr",
                "text_raw": "",
                "confidence": 0.0,
                "candidates": candidates,
                "filtered_candidates": filtered,
            }

        # Join candidates in EasyOCR order with space; keep raw candidate texts available.
        text_raw = " ".join(str(c.get("text", "")).strip() for c in filtered if str(c.get("text", "")).strip()).strip()
        confs = [float(c.get("confidence", 0.0)) for c in filtered]
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return {
            "status": "ok" if text_raw else "empty_ocr",
            "text_raw": text_raw,
            "confidence": avg_conf,
            "candidates": candidates,
            "filtered_candidates": filtered,
        }

    @staticmethod
    def _normalize_candidates(raw_results: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(raw_results, list):
            return out
        for item in raw_results:
            # detail=1: [bbox, text, confidence]
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                bbox, text, conf = item[0], item[1], item[2]
                bbox_list = None
                try:
                    bbox_list = [[float(x), float(y)] for x, y in bbox]
                except Exception:
                    bbox_list = bbox
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = 0.0
                out.append({
                    "bbox": bbox_list,
                    "text": "" if text is None else str(text),
                    "confidence": conf_f,
                })
            # detail=0 or paragraph-like simple text
            elif isinstance(item, str):
                out.append({"bbox": None, "text": item, "confidence": 0.0})
            elif isinstance(item, dict):
                out.append({
                    "bbox": item.get("bbox"),
                    "text": "" if item.get("text") is None else str(item.get("text")),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                })
        return out
