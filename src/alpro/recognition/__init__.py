"""Recognition/OCR modules for ALPRO (Milestone 7)."""

from .ocr_engine import OCREngine
from .text_normalizer import NormalizationResult, TextNormalizer

__all__ = ["OCREngine", "TextNormalizer", "NormalizationResult"]
