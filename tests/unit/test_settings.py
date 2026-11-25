from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpro.config.settings import deep_merge, resolve_path


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    over = {"b": {"d": 99, "e": 4}}
    out = deep_merge(base, over)
    assert out == {"a": 1, "b": {"c": 2, "d": 99, "e": 4}}


def test_resolve_path_relative():
    p = resolve_path("configs/pipeline.yaml")
    assert p is not None
    assert p.name == "pipeline.yaml"
