from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpro.core.result import PipelineResult


def test_pipeline_result_to_dict_strips_images():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    r = PipelineResult(status="ok", crop={"image": img, "bbox_padded_xyxy": [0, 0, 1, 1]})
    d = r.to_dict(include_images=False)
    assert "image" not in d["crop"]
    assert d["crop"]["bbox_padded_xyxy"] == [0, 0, 1, 1]


def test_pipeline_result_ok_flag():
    assert PipelineResult(status="ok").ok is True
    assert PipelineResult(status="no_detection").ok is False
