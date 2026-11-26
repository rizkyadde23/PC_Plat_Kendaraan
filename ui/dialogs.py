"""Configuration dialog for ALPRO."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import streamlit as st

from ui.state import get_config, get_default_config, set_config


def _ensure_draft() -> Dict[str, Any]:
    if "alpro_config_draft" not in st.session_state:
        st.session_state["alpro_config_draft"] = deepcopy(get_config())
    return st.session_state["alpro_config_draft"]


@st.dialog("Configuration", width="large")
def configuration_dialog(variant_options: List[str]) -> None:
    draft = _ensure_draft()
    det = draft.setdefault("detector", {})
    crop = draft.setdefault("crop", {})
    pre = draft.setdefault("preprocessing", {})
    ocr = draft.setdefault("ocr", {})
    ui_cfg = draft.setdefault("ui", {})

    st.caption("Atur parameter pipeline ALPRO. Perubahan disimpan hanya saat Anda menekan Save Configuration.")

    st.markdown("#### Detection Configuration")
    c1, c2 = st.columns(2)
    with c1:
        det["conf"] = st.slider(
            "Confidence threshold",
            0.05, 1.0, float(det.get("conf", 0.25)), 0.05,
            help="Ambang keyakinan minimum deteksi plat. Lebih tinggi = lebih selektif.",
        )
        det["imgsz"] = st.selectbox(
            "Inference image size",
            options=[320, 480, 640, 800, 960],
            index=[320, 480, 640, 800, 960].index(int(det.get("imgsz", 640))) if int(det.get("imgsz", 640)) in [320, 480, 640, 800, 960] else 2,
            help="Ukuran input inferensi YOLO. 640 adalah default baseline.",
        )
    with c2:
        det["iou"] = st.slider(
            "IoU threshold",
            0.05, 1.0, float(det.get("iou", 0.45)), 0.05,
            help="Ambang IoU untuk NMS saat menggabungkan deteksi tumpang tindih.",
        )
        det["max_det"] = st.number_input(
            "Max detections",
            min_value=1, max_value=20, value=int(det.get("max_det", 10)), step=1,
            help="Jumlah maksimum deteksi plat per gambar.",
        )

    st.divider()
    st.markdown("#### Preprocessing Configuration")
    default_variant = str(pre.get("default_variant", "original_crop"))
    if variant_options and default_variant not in variant_options:
        default_variant = variant_options[0]
    pre["default_variant"] = st.selectbox(
        "Default preprocessing variant",
        options=variant_options or ["original_crop"],
        index=(variant_options or ["original_crop"]).index(default_variant) if default_variant in (variant_options or ["original_crop"]) else 0,
        help="Mode preprocessing default sebelum OCR. original_crop direkomendasikan sebagai baseline aman.",
    )
    crop["padding_ratio"] = st.slider(
        "Crop padding ratio",
        0.0, 0.30, float(crop.get("padding_ratio", 0.05)), 0.01,
        help="Margin tambahan di sekitar bbox plat untuk menghindari karakter terpotong.",
    )

    st.divider()
    st.markdown("#### OCR Configuration")
    st.text_input(
        "OCR config path",
        value=str(ocr.get("config_path", "configs/ocr.yaml")),
        disabled=True,
        help="Path konfigurasi OCR saat ini (read-only di dialog ini).",
    )
    st.caption("Pengaturan engine OCR lanjutan dikelola melalui file konfigurasi OCR.")

    st.divider()
    st.markdown("#### Application Preferences")
    ui_cfg["show_intermediate"] = st.checkbox(
        "Tampilkan langkah preprocessing antara secara default",
        value=bool(ui_cfg.get("show_intermediate", False)),
    )
    ui_cfg["allow_save"] = st.checkbox(
        "Aktifkan opsi simpan hasil",
        value=bool(ui_cfg.get("allow_save", True)),
    )

    st.divider()
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("Save Configuration", type="primary", use_container_width=True):
            set_config(deepcopy(draft))
            st.session_state.pop("alpro_config_draft", None)
            st.success("Konfigurasi disimpan.")
            st.rerun()
    with b2:
        if st.button("Reset to Default", use_container_width=True):
            st.session_state["alpro_config_draft"] = get_default_config()
            st.rerun()
    with b3:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("alpro_config_draft", None)
            st.rerun()
