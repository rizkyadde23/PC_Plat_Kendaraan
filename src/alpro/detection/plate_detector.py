"""Wrapper deteksi plat nomor berbasis Ultralytics YOLO.

Modul ini menyediakan antarmuka inferensi reusable yang terpisah dari UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover - dependency/runtime guard
    YOLO = None  # type: ignore
    _YOLO_IMPORT_ERROR = e
else:
    _YOLO_IMPORT_ERROR = None


ImageSource = Union[str, Path, np.ndarray]


def _ensure_model_available() -> None:
    if YOLO is None:
        raise ImportError(
            "Ultralytics YOLO tidak tersedia. Pastikan dependensi terpasang."
        ) from _YOLO_IMPORT_ERROR


class PlateDetector:
    """Detektor plat nomor untuk inferensi.

    Args:
        weights: Path ke bobot model (.pt) atau nama model Ultralytics.
        device: Perangkat inferensi (mis. 'cpu', '0').
        conf: Confidence threshold default.
        iou: IoU threshold default (NMS).
        imgsz: Ukuran input inferensi.
        max_det: Batas maksimum deteksi per gambar.
    """

    def __init__(
        self,
        weights: Union[str, Path],
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 10,
    ) -> None:
        _ensure_model_available()
        self.weights = str(weights)
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.model = YOLO(self.weights)

    @property
    def names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", {}) or {}
        # ultralytics may provide list or dict
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return {int(k): str(v) for k, v in names.items()}

    def predict(
        self,
        source: Union[ImageSource, Sequence[ImageSource]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        max_det: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Jalankan inferensi dan kembalikan hasil terstruktur per gambar.

        Returns:
            List item per gambar:
            {
              "source": <path|ndarray>,
              "image_shape": [h, w, c] (jika diketahui),
              "detections": [
                 {
                   "bbox_xyxy": [x1,y1,x2,y2],
                   "confidence": float,
                   "class_id": int,
                   "class_name": str
                 }, ...
              ]
            }
        """
        conf = self.conf if conf is None else float(conf)
        iou = self.iou if iou is None else float(iou)
        imgsz = self.imgsz if imgsz is None else int(imgsz)
        device = self.device if device is None else str(device)
        max_det = self.max_det if max_det is None else int(max_det)

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            verbose=verbose,
        )

        outputs: List[Dict[str, Any]] = []
        names = self.names
        for res in results:
            dets: List[Dict[str, Any]] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    cid = int(clss[i])
                    dets.append(
                        {
                            "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                            "confidence": float(confs[i]),
                            "class_id": cid,
                            "class_name": names.get(cid, str(cid)),
                        }
                    )
            # sort by confidence desc for consistency
            dets.sort(key=lambda d: d["confidence"], reverse=True)

            orig_shape = getattr(res, "orig_shape", None)
            shape = None
            if orig_shape is not None:
                try:
                    h, w = int(orig_shape[0]), int(orig_shape[1])
                    shape = [h, w, 3]
                except Exception:
                    shape = None

            src = getattr(res, "path", None)
            if not src:
                # for ndarray input, ultralytics may not set path
                src = source if not isinstance(source, (list, tuple)) else None

            outputs.append(
                {
                    "source": str(src) if isinstance(src, (str, Path)) else src,
                    "image_shape": shape,
                    "detections": dets,
                }
            )
        return outputs

    @staticmethod
    def annotate(
        image_bgr: np.ndarray,
        detections: Sequence[Dict[str, Any]],
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Gambar bbox dan label confidence pada salinan image BGR."""
        if image_bgr is None:
            raise ValueError("image_bgr tidak boleh None")
        out = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
            conf = float(det.get("confidence", 0.0))
            name = str(det.get("class_name", "obj"))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(y1, th + baseline + 2)
            cv2.rectangle(out, (x1, y_text - th - baseline - 2), (x1 + tw + 2, y_text), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 1, y_text - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return out
