"""ALPRO - Plate Recognition Page."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpro.core.pipeline import PlateRecognitionPipeline  # noqa: E402
from alpro.core.result import PipelineResult  # noqa: E402
from ui.components import render_preprocessing_steps, render_result_panel  # noqa: E402
from ui.dialogs import configuration_dialog  # noqa: E402
from ui.state import get_config, init_config_state  # noqa: E402
from ui.styles import inject_css  # noqa: E402


st.set_page_config(
    page_title="ALPRO — Plate Recognition",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()
init_config_state()


@st.cache_resource(show_spinner=False)
def get_pipeline(config_signature: str, config_json: str) -> PlateRecognitionPipeline:
    # config_signature/config_json ensure cache key changes when config changes
    _ = config_signature
    cfg = json.loads(config_json)
    return PlateRecognitionPipeline(config=cfg, config_path="configs/pipeline.yaml")


def config_signature(cfg: Dict[str, Any]) -> str:
    det = cfg.get("detector", {})
    crop = cfg.get("crop", {})
    pre = cfg.get("preprocessing", {})
    ocr = cfg.get("ocr", {})
    return "|".join([
        str(det.get("weights", "")),
        str(det.get("device", "cpu")),
        str(det.get("conf", "")),
        str(det.get("iou", "")),
        str(det.get("imgsz", "")),
        str(det.get("max_det", "")),
        str(crop.get("padding_ratio", "")),
        str(pre.get("default_variant", "")),
        str(ocr.get("config_path", "")),
    ])


def decode_upload(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def get_variant_options(cfg: Dict[str, Any]) -> list[str]:
    try:
        p = PlateRecognitionPipeline(config=cfg, config_path="configs/pipeline.yaml")
        return p.get_preprocessing_variants()
    except Exception:
        return ["original_crop"]


cfg = get_config()
variant_options = get_variant_options(cfg)

# Top navigation / actions
top_left, top_right = st.columns([1, 0.35], vertical_alignment="center")
with top_left:
    if st.button("← Kembali ke Home", key="back_home"):
        st.switch_page("app.py")
with top_right:
    st.markdown('<div class="alpro-config-btn">', unsafe_allow_html=True)
    if st.button("⚙️ Configuration", type="primary", use_container_width=True, key="open_config"):
        configuration_dialog(variant_options)
    st.markdown("</div>", unsafe_allow_html=True)

# Header
st.markdown('<div class="alpro-kicker">ALPRO Feature</div>', unsafe_allow_html=True)
st.title("Plate Recognition")
st.caption("Unggah gambar kendaraan untuk mendeteksi plat nomor dan membaca teksnya melalui pipeline YOLO, OpenCV, dan EasyOCR.")

# Upload + action
upload_col, info_col = st.columns([1.4, 1], gap="large")
with upload_col:
    st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
    st.markdown("#### Upload Gambar Kendaraan")
    st.markdown('<p class="alpro-muted-text">Format yang didukung: JPG, JPEG, PNG, BMP, WEBP. Gunakan gambar yang jelas dan plat terlihat cukup besar.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    run = st.button("Mulai Pembacaan", type="primary", use_container_width=True, disabled=uploaded is None)
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="alpro-card-soft">', unsafe_allow_html=True)
    st.markdown("#### Tips Hasil Lebih Baik")
    st.markdown(
        '<div class="alpro-muted-text">• Pastikan plat tidak blur dan pencahayaan cukup.<br>• Hindari sudut terlalu miring.<br>• Jika plat tidak terdeteksi, turunkan confidence threshold di Configuration.<br>• Jika OCR sulit terbaca, coba preprocessing variant lain.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

image = decode_upload(uploaded) if uploaded is not None else None
if uploaded is not None and image is None:
    st.error("File gambar tidak dapat dibaca. Pastikan file tidak rusak dan formatnya didukung.")

# Process
result: Optional[PipelineResult] = None
if uploaded is not None and image is not None and run:
    with st.spinner("Memproses gambar... mendeteksi plat, preprocessing, dan membaca OCR."):
        try:
            sig = config_signature(cfg)
            pipeline = get_pipeline(sig, json.dumps(cfg, sort_keys=True))
            det = cfg.get("detector", {})
            crop = cfg.get("crop", {})
            pre = cfg.get("preprocessing", {})
            result = pipeline.process_image(
                image,
                variant=str(pre.get("default_variant", "original_crop")),
                conf=float(det.get("conf", 0.25)),
                iou=float(det.get("iou", 0.45)),
                imgsz=int(det.get("imgsz", 640)),
                padding_ratio=float(crop.get("padding_ratio", 0.05)),
                return_images=True,
            )
            result.meta["input_filename"] = uploaded.name
        except Exception as e:
            st.error("Terjadi error saat memproses gambar. Periksa konfigurasi atau model, lalu coba lagi.")
            st.caption(f"Detail teknis: {e}")

# Results layout
if result is not None:
    result_dict = result.to_dict(include_images=False)
    render_result_panel(result_dict)

    main_left, main_right = st.columns([1.35, 1], gap="large")

    with main_left:
        st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
        st.markdown("#### Detection Result")
        ann = result.meta.get("annotated_image") if isinstance(result.meta, dict) else None
        if isinstance(ann, np.ndarray):
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption="Gambar dengan bounding box deteksi plat.", use_container_width=True)
            dets = result_dict.get("detections", [])
            m1, m2 = st.columns(2)
            m1.metric("Jumlah Plat Terdeteksi", len(dets))
            if dets:
                m2.metric("Confidence Deteksi Tertinggi", f"{max(float(d.get('confidence',0) or 0) for d in dets):.2%}")
            else:
                m2.metric("Confidence Deteksi Tertinggi", "-")
        else:
            st.info("Tidak ada visual deteksi yang tersedia.")
        st.markdown("</div>", unsafe_allow_html=True)

    with main_right:
        st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
        st.markdown("#### Cropped Plate")
        crop_img = result.crop.get("image") if isinstance(result.crop, dict) else None
        if isinstance(crop_img, np.ndarray):
            st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), caption="Area plat hasil crop.", use_container_width=True)
        else:
            st.info("Crop plat tidak tersedia. Ini dapat terjadi jika plat tidak terdeteksi.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
        st.markdown("#### Preprocessing Result")
        pre_img = result.preprocessing.get("image") if isinstance(result.preprocessing, dict) else None
        inter = result.preprocessing.get("intermediate", {}) if isinstance(result.preprocessing, dict) else {}
        variant = str(result_dict.get("meta", {}).get("variant", cfg.get("preprocessing", {}).get("default_variant", "original_crop")))
        if isinstance(pre_img, np.ndarray):
            st.image(cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB), caption=f"Output preprocessing ({variant}).", use_container_width=True)
        else:
            st.info("Hasil preprocessing tidak tersedia.")
        if inter:
            with st.expander("Lihat langkah preprocessing"):
                render_preprocessing_steps(inter)
        st.markdown("</div>", unsafe_allow_html=True)

    # Additional info / details
    with st.expander("Additional Information", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        timings = result_dict.get("timings_ms", {})
        d1.metric("Total Time", f"{float(timings.get('total', 0)):.0f} ms")
        d2.metric("Detect", f"{float(timings.get('detect_ms', 0)):.0f} ms")
        d3.metric("OCR", f"{float(timings.get('ocr_ms', 0)):.0f} ms")
        d4.metric("Status", str(result_dict.get("status", "-")))
        st.markdown("**Detections**")
        st.json(result_dict.get("detections", []))
        st.markdown("**OCR Candidates**")
        st.json((result_dict.get("ocr") or {}).get("candidates", []))

    # Save option
    if bool(cfg.get("ui", {}).get("allow_save", True)):
        with st.expander("Simpan Hasil", expanded=False):
            st.caption("Simpan gambar dan JSON hasil ke folder outputs/predictions/pipeline/<timestamp>/")
            if st.button("Save Result", use_container_width=False):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = PROJECT_ROOT / "outputs" / "predictions" / "pipeline" / ts
                try:
                    sig = config_signature(cfg)
                    pipeline = get_pipeline(sig, json.dumps(cfg, sort_keys=True))
                    paths = pipeline.save_result(result, out_dir)
                    st.success(f"Hasil disimpan ke: {out_dir}")
                    st.json(paths)
                except Exception as e:
                    st.error(f"Gagal menyimpan hasil: {e}")

elif uploaded is not None and image is not None:
    st.markdown('<div class="alpro-card-soft">', unsafe_allow_html=True)
    st.markdown("#### Gambar Siap Diproses")
    st.markdown('<p class="alpro-muted-text">Klik <strong>Mulai Pembacaan</strong> untuk menjalankan deteksi plat, preprocessing, dan OCR.</p>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=uploaded.name, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="alpro-card-soft">', unsafe_allow_html=True)
    st.markdown("#### Belum Ada Gambar")
    st.markdown('<p class="alpro-muted-text">Unggah gambar kendaraan untuk memulai. Hasil pembacaan plat nomor akan tampil di sini sebagai panel utama.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption("ALPRO Plate Recognition • Confidence OCR berasal dari model OCR, bukan ground truth akurasi.")
