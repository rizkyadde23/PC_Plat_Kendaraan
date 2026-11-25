"""Streamlit UI helpers for ALPRO."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
import streamlit as st


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        return image
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(image, np.ndarray) and image.ndim == 2:
        return image
    return image


def show_image(title: str, image: Any, caption: str = "") -> None:
    if image is None:
        st.info(f"{title}: tidak tersedia")
        return
    st.markdown(f"**{title}**")
    if isinstance(image, np.ndarray):
        st.image(bgr_to_rgb(image), caption=caption, use_container_width=True)
    else:
        st.write(image)


def show_result_summary(result: Dict[str, Any]) -> None:
    status = result.get("status")
    ocr = result.get("ocr") or {}
    st.subheader("Hasil OCR")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status pipeline", str(status))
    c2.metric("OCR status", str(ocr.get("status", "-")))
    c3.metric("OCR confidence", f"{float(ocr.get('confidence', 0.0) or 0.0):.3f}")
    st.text_input("Teks mentah (raw)", value=str(ocr.get("text_raw", "")), disabled=True)
    st.text_input("Teks normalisasi", value=str(ocr.get("text_normalized", "")), disabled=True)
    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)
    if result.get("errors"):
        for e in result["errors"]:
            st.error(e)
