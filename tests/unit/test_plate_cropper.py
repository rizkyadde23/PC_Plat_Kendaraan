from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpro.preprocessing.plate_cropper import PlateCropper


def test_crop_clamps_and_pads():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    bbox = [-10, 10, 50, 60]
    out = PlateCropper.crop(img, bbox, padding_ratio=0.1)
    assert out["status"] == "ok"
    assert out["crop"].shape[0] > 0 and out["crop"].shape[1] > 0
    x1, y1, x2, y2 = out["bbox_padded_xyxy"]
    assert x1 >= 0 and y1 >= 0 and x2 <= 200 and y2 <= 100


def test_crop_detections_no_detection():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    out = PlateCropper.crop_detections(img, [])
    assert out["status"] == "no_detection"
    assert out["selected_count"] == 0
