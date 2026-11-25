"""End-to-end ALPRO pipeline (Milestone 8).

Orkestrasi: load/validate -> detect -> crop -> preprocess -> OCR -> normalize.
Tidak bergantung pada UI. Komponen berat di-load lazy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from alpro.config.settings import deep_merge, load_pipeline_config, load_yaml, resolve_path
from alpro.core.result import PipelineResult
from alpro.detection.plate_detector import PlateDetector
from alpro.preprocessing import ImageLoader, PlateCropper, ImagePreprocessor
from alpro.recognition import OCREngine, TextNormalizer


ImageInput = Union[str, Path, np.ndarray]


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class PlateRecognitionPipeline:
    """Pipeline inferensi end-to-end untuk ALPRO."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Union[str, Path] = "configs/pipeline.yaml"):
        base = load_pipeline_config(config_path)
        self.config = deep_merge(base, config or {})
        self._detector: Optional[PlateDetector] = None
        self._ocr: Optional[OCREngine] = None
        self._preprocess_cfg: Optional[Dict[str, Any]] = None

    @property
    def preprocess_config(self) -> Dict[str, Any]:
        if self._preprocess_cfg is None:
            pre_cfg_path = self.config.get("preprocessing", {}).get("config_path", "configs/preprocessing.yaml")
            self._preprocess_cfg = load_yaml(pre_cfg_path)
        return self._preprocess_cfg

    def get_preprocessing_variants(self) -> List[str]:
        variants = self.preprocess_config.get("variants", [])
        names = []
        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, dict) and v.get("name"):
                    names.append(str(v["name"]))
        return names or ["original_crop"]

    def get_variant_steps(self, variant_name: Optional[str]) -> List[Dict[str, Any]]:
        name = variant_name or self.config.get("preprocessing", {}).get("default_variant", "original_crop")
        variants = self.preprocess_config.get("variants", [])
        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, dict) and str(v.get("name")) == str(name):
                    steps = v.get("steps", [])
                    return steps if isinstance(steps, list) else []
        # fallback original
        return []

    def _get_detector(self) -> PlateDetector:
        if self._detector is None:
            dcfg = self.config.get("detector", {})
            self._detector = PlateDetector(
                weights=dcfg.get("weights", "yolov8n.pt"),
                device=str(dcfg.get("device", self.config.get("pipeline", {}).get("device", "cpu"))),
                conf=float(dcfg.get("conf", 0.25)),
                iou=float(dcfg.get("iou", 0.45)),
                imgsz=int(dcfg.get("imgsz", 640)),
                max_det=int(dcfg.get("max_det", 10)),
            )
        return self._detector

    def _get_ocr(self) -> OCREngine:
        if self._ocr is None:
            ocr_cfg_path = self.config.get("ocr", {}).get("config_path", "configs/ocr.yaml")
            ocr_cfg = load_yaml(ocr_cfg_path)
            eng = ocr_cfg.get("engine", {}) if isinstance(ocr_cfg.get("engine"), dict) else {}
            self._ocr = OCREngine(
                lang=eng.get("lang", ["en"]),
                gpu=bool(eng.get("gpu", False)),
                detail=int(eng.get("detail", 1)),
                paragraph=bool(eng.get("paragraph", False)),
                decoder=eng.get("decoder"),
                beamWidth=int(eng.get("beamWidth", 5)),
                batch_size=int(eng.get("batch_size", 1)),
                min_confidence=float(eng.get("min_confidence", 0.0)),
            )
            self._normalizer_cfg = ocr_cfg.get("normalization", {}) if isinstance(ocr_cfg.get("normalization"), dict) else {}
        return self._ocr

    def _get_normalizer(self) -> TextNormalizer:
        # ensure OCR config loaded for normalizer config
        if not hasattr(self, "_normalizer_cfg"):
            ocr_cfg_path = self.config.get("ocr", {}).get("config_path", "configs/ocr.yaml")
            ocr_cfg = load_yaml(ocr_cfg_path)
            self._normalizer_cfg = ocr_cfg.get("normalization", {}) if isinstance(ocr_cfg.get("normalization"), dict) else {}
        return TextNormalizer(self._normalizer_cfg)

    def process_image(
        self,
        source: ImageInput,
        *,
        variant: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        padding_ratio: Optional[float] = None,
        crop_strategy: Optional[str] = None,
        return_images: bool = True,
    ) -> PipelineResult:
        t0 = _now_ms()
        res = PipelineResult(status="started", source=str(source) if isinstance(source, (str, Path)) else None)
        timings: Dict[str, float] = {}

        # 1) load
        t = _now_ms()
        try:
            loaded = ImageLoader.load(source)
            image = loaded["image"]
            res.image_shape = loaded.get("shape")
            if loaded.get("path"):
                res.source = loaded.get("path")
        except Exception as e:
            res.status = "load_error"
            res.errors.append(f"load: {e}")
            res.timings_ms["total"] = _now_ms() - t0
            return res
        timings["load_ms"] = _now_ms() - t

        # 2) detect
        t = _now_ms()
        try:
            detector = self._get_detector()
            dcfg = self.config.get("detector", {})
            det_out = detector.predict(
                image,
                conf=float(dcfg.get("conf", 0.25) if conf is None else conf),
                iou=float(dcfg.get("iou", 0.45) if iou is None else iou),
                imgsz=int(dcfg.get("imgsz", 640) if imgsz is None else imgsz),
                device=str(dcfg.get("device", self.config.get("pipeline", {}).get("device", "cpu"))),
                verbose=False,
            )[0]
            detections = det_out.get("detections", [])
            res.detections = detections
            res.meta["annotated_available"] = True
            if return_images:
                res.meta["annotated_image"] = detector.annotate(image, detections)
        except Exception as e:
            res.status = "detect_error"
            res.errors.append(f"detect: {e}")
            res.timings_ms = timings
            res.timings_ms["total"] = _now_ms() - t0
            return res
        timings["detect_ms"] = _now_ms() - t

        # 3) crop
        t = _now_ms()
        ccfg = self.config.get("crop", {})
        crop_out = PlateCropper.crop_detections(
            image,
            detections,
            padding_ratio=float(ccfg.get("padding_ratio", 0.05) if padding_ratio is None else padding_ratio),
            strategy=str(ccfg.get("strategy", "top_confidence") if crop_strategy is None else crop_strategy),
            min_size=int(ccfg.get("min_size", 8)),
        )
        res.meta["crop_summary"] = {k: v for k, v in crop_out.items() if k != "crops"}
        crops = crop_out.get("crops", [])
        ok_crops = [c for c in crops if c.get("status") == "ok" and c.get("crop") is not None]
        if not ok_crops:
            res.status = str(crop_out.get("status", "no_detection"))
            if not detections:
                res.warnings.append("Tidak ada deteksi plat.")
            else:
                res.warnings.append("Deteksi ada tetapi crop gagal.")
            res.timings_ms = timings
            res.timings_ms["total"] = _now_ms() - t0
            return res

        chosen = ok_crops[0]
        res.selected_detection = chosen.get("detection")
        crop_payload = {k: v for k, v in chosen.items() if k != "crop"}
        if return_images:
            crop_payload["image"] = chosen["crop"]
        res.crop = crop_payload
        timings["crop_ms"] = _now_ms() - t

        # 4) preprocess
        t = _now_ms()
        try:
            variant_name = variant or self.config.get("preprocessing", {}).get("default_variant", "original_crop")
            steps = self.get_variant_steps(variant_name)
            pre = ImagePreprocessor({"pipeline": {"steps": steps}})
            pre_out = pre.process(chosen["crop"], return_intermediate=bool(return_images), output_mode="bgr")
            pre_payload = {
                "variant": variant_name,
                "steps": pre_out.get("steps", []),
            }
            if return_images:
                pre_payload["image"] = pre_out.get("image")
                pre_payload["intermediate"] = pre_out.get("intermediate", {})
            res.preprocessing = pre_payload
            pre_image = pre_out.get("image")
        except Exception as e:
            res.status = "preprocess_error"
            res.errors.append(f"preprocess: {e}")
            res.timings_ms = timings
            res.timings_ms["total"] = _now_ms() - t0
            return res
        timings["preprocess_ms"] = _now_ms() - t

        # 5) OCR + normalize
        t = _now_ms()
        try:
            ocr = self._get_ocr()
            normalizer = self._get_normalizer()
            ocr_out = ocr.read(pre_image)
            norm_out = normalizer.normalize(ocr_out.get("text_raw", ""))
            res.ocr = {
                "status": ocr_out.get("status"),
                "text_raw": ocr_out.get("text_raw", ""),
                "text_normalized": norm_out.text_normalized,
                "confidence": float(ocr_out.get("confidence", 0.0) or 0.0),
                "candidates": ocr_out.get("filtered_candidates", ocr_out.get("candidates", [])),
                "normalization_rules": norm_out.applied_rules,
                "error": ocr_out.get("error"),
            }
        except Exception as e:
            res.status = "ocr_error"
            res.errors.append(f"ocr: {e}")
            res.timings_ms = timings
            res.timings_ms["total"] = _now_ms() - t0
            return res
        timings["ocr_ms"] = _now_ms() - t

        res.status = "ok" if res.ocr.get("status") == "ok" else str(res.ocr.get("status", "ok"))
        if res.status == "empty_ocr":
            res.warnings.append("OCR tidak menghasilkan teks.")
        res.timings_ms = timings
        res.timings_ms["total"] = _now_ms() - t0
        res.meta["variant"] = variant_name
        return res

    def save_result(self, result: PipelineResult, out_dir: Union[str, Path]) -> Dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}
        # save json without heavy arrays
        json_path = out / "result.json"
        json_path.write_text(json.dumps(result.to_dict(include_images=False), ensure_ascii=False, indent=2), encoding="utf-8")
        paths["result_json"] = str(json_path)

        def _save_img(name: str, img: Any) -> None:
            if isinstance(img, np.ndarray) and img.size > 0:
                p = out / name
                cv2.imwrite(str(p), img)
                paths[name] = str(p)

        ann = result.meta.get("annotated_image") if isinstance(result.meta, dict) else None
        _save_img("annotated.jpg", ann)
        if isinstance(result.crop, dict):
            _save_img("crop.jpg", result.crop.get("image"))
        if isinstance(result.preprocessing, dict):
            _save_img("preprocessed.jpg", result.preprocessing.get("image"))
            inter = result.preprocessing.get("intermediate")
            if isinstance(inter, dict):
                inter_dir = out / "intermediate"
                inter_dir.mkdir(parents=True, exist_ok=True)
                for k, v in inter.items():
                    if isinstance(v, np.ndarray):
                        p = inter_dir / f"{k}.jpg"
                        cv2.imwrite(str(p), v)
                        paths[f"intermediate/{k}.jpg"] = str(p)
        return paths
