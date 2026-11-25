from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpro.core.pipeline import PlateRecognitionPipeline


class FakeDetector:
    def predict(self, image, **kwargs):
        h, w = image.shape[:2]
        return [{"source": None, "image_shape": [h, w, 3], "detections": [
            {"bbox_xyxy": [10, 10, 100, 50], "confidence": 0.9, "class_id": 0, "class_name": "License-Plate"}
        ]}]

    @staticmethod
    def annotate(image, detections):
        return image.copy()


class FakeOCR:
    def read(self, image):
        return {
            "status": "ok",
            "text_raw": "b 1234 xyz",
            "confidence": 0.88,
            "candidates": [{"text": "b 1234 xyz", "confidence": 0.88, "bbox": None}],
            "filtered_candidates": [{"text": "b 1234 xyz", "confidence": 0.88, "bbox": None}],
        }


def _base_config():
    return {
        "detector": {"weights": "dummy.pt", "device": "cpu", "conf": 0.25, "iou": 0.45, "imgsz": 640, "max_det": 10},
        "crop": {"strategy": "top_confidence", "padding_ratio": 0.05, "min_size": 1},
        "preprocessing": {"config_path": "configs/preprocessing.yaml", "default_variant": "original_crop"},
        "ocr": {"config_path": "configs/ocr.yaml"},
        "output": {"save_by_default": False, "dir": "outputs/predictions/pipeline"},
        "pipeline": {"device": "cpu"},
    }


def test_pipeline_end_to_end_mocked(monkeypatch):
    # avoid loading real models
    import alpro.core.pipeline as pmod

    monkeypatch.setattr(pmod, "PlateDetector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(pmod, "OCREngine", lambda *args, **kwargs: FakeOCR())

    pipe = PlateRecognitionPipeline(config=_base_config(), config_path="configs/pipeline.yaml")
    img = np.full((120, 240, 3), 255, dtype=np.uint8)
    result = pipe.process_image(img, variant="original_crop", return_images=True)

    assert result.status == "ok"
    assert result.ocr["text_raw"] == "b 1234 xyz"
    assert result.ocr["text_normalized"] == "B 1234 XYZ"
    assert result.selected_detection is not None
    assert isinstance(result.crop.get("image"), np.ndarray)
    assert isinstance(result.preprocessing.get("image"), np.ndarray)


def test_pipeline_no_detection_mocked(monkeypatch):
    import alpro.core.pipeline as pmod

    class NoDet:
        def predict(self, image, **kwargs):
            h, w = image.shape[:2]
            return [{"source": None, "image_shape": [h, w, 3], "detections": []}]

        @staticmethod
        def annotate(image, detections):
            return image.copy()

    monkeypatch.setattr(pmod, "PlateDetector", lambda *args, **kwargs: NoDet())
    monkeypatch.setattr(pmod, "OCREngine", lambda *args, **kwargs: FakeOCR())

    pipe = PlateRecognitionPipeline(config=_base_config(), config_path="configs/pipeline.yaml")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipe.process_image(img)
    assert result.status == "no_detection"
    assert result.ocr is None
