"""Reusable UI components for ALPRO Streamlit app."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st


STATUS_LABELS = {
    "ok": "Berhasil",
    "no_detection": "Plat Tidak Terdeteksi",
    "load_error": "Gagal Membaca Gambar",
    "detect_error": "Gagal Deteksi",
    "crop_failed": "Gagal Crop Plat",
    "preprocess_error": "Gagal Preprocessing",
    "ocr_error": "Gagal OCR",
    "empty_ocr": "OCR Kosong",
}


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        return image
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def status_label(status: Optional[str]) -> str:
    key = str(status or "-")
    return STATUS_LABELS.get(key, key.replace("_", " ").title())


def status_badge(status: Optional[str]) -> str:
    key = str(status or "")
    if key == "ok":
        cls, text = "alpro-badge-success", "License Plate Detected"
    elif key in {"no_detection", "empty_ocr"}:
        cls, text = "alpro-badge-warn", status_label(key)
    elif key in {"load_error", "detect_error", "crop_failed", "preprocess_error", "ocr_error"}:
        cls, text = "alpro-badge-error", status_label(key)
    else:
        cls, text = "alpro-badge-info", status_label(key)
    return f'<span class="alpro-badge-status {cls}">{text}</span>'


def render_sidebar_brand() -> None:
    st.markdown(
        """
<div style="padding:0.25rem 0 0.75rem 0;">
  <div style="font-size:1.35rem;font-weight:800;color:#0F172A;letter-spacing:-0.02em;">ALPRO</div>
  <div style="font-size:0.82rem;color:#64748B;line-height:1.45;margin-top:0.15rem;">Automatic License Plate Recognition Online</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_result_panel(result: Dict[str, Any]) -> None:
    status = str(result.get("status", "-"))
    ocr = result.get("ocr") or {}
    det = result.get("selected_detection") or {}
    detections = result.get("detections") or []
    text_norm = str(ocr.get("text_normalized", "")).strip()
    text_raw = str(ocr.get("text_raw", "")).strip()
    ocr_conf = float(ocr.get("confidence", 0.0) or 0.0)
    det_conf = float(det.get("confidence", 0.0) or 0.0)

    display_text = text_norm or text_raw or "Tidak terbaca"
    badge = status_badge(status)
    ocr_status = status_label(str(ocr.get("status", status)))

    st.markdown(
        f"""
<div class="alpro-result-panel">
  <div class="alpro-result-label">Recognition Result</div>
  <div class="alpro-plate-text">{display_text}</div>
  <div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">{badge}</div>
  <div class="alpro-result-meta">
    <div class="alpro-meta-item strong"><div class="k">OCR Confidence</div><div class="v">{ocr_conf:.2%}</div></div>
    <div class="alpro-meta-item strong"><div class="k">Detection Confidence</div><div class="v">{det_conf:.2%}</div></div>
    <div class="alpro-meta-item"><div class="k">Detected Plates</div><div class="v">{len(detections)}</div></div>
    <div class="alpro-meta-item"><div class="k">OCR Status</div><div class="v">{ocr_status}</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if text_norm or text_raw:
        st.caption("Teks normalisasi digunakan sebagai hasil utama; teks mentah OCR ditampilkan di detail.")

    if status == "no_detection":
        st.warning("Plat nomor tidak terdeteksi pada gambar ini. Coba gunakan gambar yang lebih jelas, atau sesuaikan confidence threshold di Configuration.")
    elif status == "empty_ocr":
        st.warning("Plat terdeteksi, tetapi OCR belum menghasilkan teks yang jelas. Coba ubah mode preprocessing atau gunakan gambar dengan pencahayaan lebih baik.")
    elif status not in {"ok"}:
        msg = (result.get("errors") or ["Terjadi kendala saat memproses gambar."])[0]
        st.error(f"Pemrosesan tidak berhasil: {msg}")

    for w in result.get("warnings", []):
        if str(w).strip() and str(w) not in {"OCR tidak menghasilkan teks."}:
            st.info(str(w))


def render_image_card(title: str, image: Any, caption: str = "") -> bool:
    if image is None or not isinstance(image, np.ndarray):
        return False
    st.markdown(f"**{title}**")
    st.image(bgr_to_rgb(image), caption=caption, use_container_width=True)
    return True


def render_preprocessing_steps(intermediate: Dict[str, Any]) -> None:
    items = [(k, v) for k, v in (intermediate or {}).items() if isinstance(v, np.ndarray)]
    if not items:
        st.caption("Tidak ada langkah preprocessing antara untuk mode ini.")
        return
    cols = st.columns(2)
    for i, (name, img) in enumerate(items):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            st.image(bgr_to_rgb(img), use_container_width=True)
