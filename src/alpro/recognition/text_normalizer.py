"""Text normalization for OCR results.

Normalisasi ringan dan dapat diaudit. Prediksi mentah selalu disimpan terpisah.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NormalizationResult:
    text_raw: str
    text_normalized: str
    applied_rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_raw": self.text_raw,
            "text_normalized": self.text_normalized,
            "applied_rules": self.applied_rules,
        }


class TextNormalizer:
    """Normalisasi teks OCR secara ringan dan non-spekulatif."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "uppercase": True,
        "strip": True,
        "collapse_whitespace": True,
        "allowed_pattern": "[A-Z0-9 -]",
        "replace_non_allowed_with": "",
        "char_substitution": {"enabled": False, "map": {}},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(self.DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        # nested merge for char_substitution
        cs_default = dict(self.DEFAULT_CONFIG.get("char_substitution", {}))
        cs_user = config.get("char_substitution", {}) if config else {}
        if isinstance(cs_user, dict):
            cs_default.update(cs_user)
        cfg["char_substitution"] = cs_default
        self.config = cfg

    def normalize(self, text_raw: Any) -> NormalizationResult:
        raw = "" if text_raw is None else str(text_raw)
        text = raw
        applied: Dict[str, Any] = {}

        if self.config.get("uppercase", True):
            text = text.upper()
            applied["uppercase"] = True

        # Char substitution is explicit and default OFF (non-speculative by default)
        cs = self.config.get("char_substitution", {}) or {}
        if isinstance(cs, dict) and cs.get("enabled", False):
            mapping = cs.get("map", {}) or {}
            if isinstance(mapping, dict) and mapping:
                before = text
                for src, dst in mapping.items():
                    text = text.replace(str(src).upper(), str(dst).upper())
                applied["char_substitution"] = {"enabled": True, "map": mapping, "changed": before != text}

        allowed_pattern = str(self.config.get("allowed_pattern", "[A-Z0-9 -]"))
        replace_with = str(self.config.get("replace_non_allowed_with", ""))
        try:
            # Build "not allowed" class from allowed pattern body.
            pattern = re.compile(f"[^{allowed_pattern.strip('[]')}]")
            before = text
            text = pattern.sub(replace_with, text)
            applied["allowed_pattern"] = allowed_pattern
            applied["removed_non_allowed"] = before != text
        except re.error as e:
            applied["allowed_pattern_error"] = str(e)

        if self.config.get("collapse_whitespace", True):
            before = text
            text = re.sub(r"\s+", " ", text)
            applied["collapse_whitespace"] = before != text

        if self.config.get("strip", True):
            before = text
            text = text.strip()
            applied["strip"] = before != text

        return NormalizationResult(text_raw=raw, text_normalized=text, applied_rules=applied)
