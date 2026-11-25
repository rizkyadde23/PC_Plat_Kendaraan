"""Streamlit UI helpers for ALPRO."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np
import streamlit as st


STATUS_LABELS = {
    "ok": "Berhasil",
    "no_detection": "Plat tidak terdeteksi",
    "load_error": "Gagal membaca gambar",
    "detect_error": "Gagal deteksi",
    "crop_failed": "Gagal potong plat",
    "preprocess_error": "Gagal preprocessing",
    "ocr_error": "Gagal OCR",
    "empty_ocr": "OCR kosong",
}


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        return image
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(image, np.ndarray) and image.ndim == 2:
        return image
    return image


def status_label(status: Optional[str]) -> str:
    key = str(status or "-")
    return STATUS_LABELS.get(key, key.replace("_", " ").title())


def show_image(title: str, image: Any, caption: str = "") -> bool:
    """Render an image section. Returns False when there is nothing to show."""
    if image is None:
        return False
    st.markdown(f'<div class="alpro-card-title">{title}</div>', unsafe_allow_html=True)
    if isinstance(image, np.ndarray):
        st.image(bgr_to_rgb(image), caption=caption, use_container_width=True)
    else:
        st.write(image)
    return True


def show_result_summary(result: Dict[str, Any]) -> None:
    status = str(result.get("status", "-"))
    ocr = result.get("ocr") or {}
    ocr_status = str(ocr.get("status", "-"))
    conf = float(ocr.get("confidence", 0.0) or 0.0)
    text_raw = str(ocr.get("text_raw", "")).strip()
    text_norm = str(ocr.get("text_normalized", "")).strip()

    st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
    st.markdown("### Hasil Pembacaan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status Proses", status_label(status))
    c2.metric("Status OCR", status_label(ocr_status))
    c3.metric("Tingkat Keyakinan OCR", f"{conf:.2%}")

    st.markdown("**Teks plat (hasil normalisasi)**")
    st.code(text_norm if text_norm else "-", language=None)
    st.markdown("**Teks mentah OCR**")
    st.code(text_raw if text_raw else "-", language=None)

    if status == "no_detection":
        st.warning("Sistem belum menemukan area plat pada gambar ini. Coba gambar yang lebih jelas atau turunkan sedikit 'Tingkat keyakinan deteksi'.")
    elif ocr_status == "empty_ocr":
        st.warning("Plat terdeteksi, tetapi OCR belum dapat membaca teks dengan jelas. Coba ubah mode pemrosesan gambar atau gunakan foto yang lebih tajam/terang.")

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(str(w))
    if result.get("errors"):
        for e in result["errors"]:
            st.error(str(e))
    st.markdown("</div>", unsafe_allow_html=True)
