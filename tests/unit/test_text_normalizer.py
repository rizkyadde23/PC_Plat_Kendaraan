from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpro.recognition.text_normalizer import TextNormalizer


def test_normalizer_basic():
    n = TextNormalizer()
    r = n.normalize("  b 1234 xy ")
    assert r.text_raw == "  b 1234 xy "
    assert r.text_normalized == "B 1234 XY"


def test_normalizer_removes_non_allowed():
    n = TextNormalizer()
    r = n.normalize("B@1234#XY!")
    assert r.text_normalized == "B1234XY"


def test_normalizer_substitution_default_off():
    n = TextNormalizer()
    r = n.normalize("B 1O8I")
    assert r.text_normalized == "B 1O8I"
