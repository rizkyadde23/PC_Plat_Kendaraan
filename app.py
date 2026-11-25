"""ALPRO Streamlit app - Milestone 8.

UI memanggil PlateRecognitionPipeline; logika inti tetap di modul src/alpro.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpro.core.pipeline import PlateRecognitionPipeline  # noqa: E402
from alpro.config.settings import load_pipeline_config  # noqa: E402
from ui.components import show_image, show_result_summary  # noqa: E402


st.set_page_config(page_title="ALPRO - Plate Recognition", layout="wide")
st.title("ALPRO — Automatic License Plate Recognition Online")
st.caption("Milestone 8: Integrasi pipeline + UI. Prediksi OCR ditampilkan apa adanya; evaluasi kuantitatif membutuhkan GT terverifikasi.")


@st.cache_resource(show_spinner=False)
def get_pipeline(config_path: str, weights: str, device: str) -> PlateRecognitionPipeline:
    cfg = load_pipeline_config(config_path)
    cfg.setdefault("detector", {})
    cfg["detector"]["weights"] = weights
    cfg["detector"]["device"] = device
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["device"] = device
    return PlateRecognitionPipeline(config=cfg, config_path=config_path)


def decode_upload(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


with st.sidebar:
    st.header("Konfigurasi")
    config_path = st.text_input("Pipeline config", value="configs/pipeline.yaml")
    base_cfg = load_pipeline_config(config_path) if Path(config_path).exists() or (PROJECT_ROOT / config_path).exists() else {}
    det_cfg = base_cfg.get("detector", {}) if isinstance(base_cfg, dict) else {}
    pre_cfg_path = (base_cfg.get("preprocessing", {}) or {}).get("config_path", "configs/preprocessing.yaml") if isinstance(base_cfg, dict) else "configs/preprocessing.yaml"
    weights = st.text_input("Detector weights", value=str(det_cfg.get("weights", "runs/detect/models/trained/detection/yolov8n_baseline_m5_20261118_cpu/weights/best.pt")))
    device = st.selectbox("Device", options=["cpu"], index=0)
    conf = st.slider("Detection conf", 0.05, 1.0, float(det_cfg.get("conf", 0.25)), 0.05)
    iou = st.slider("Detection IoU", 0.05, 1.0, float(det_cfg.get("iou", 0.45)), 0.05)
    padding = st.slider("Crop padding ratio", 0.0, 0.3, float((base_cfg.get("crop", {}) or {}).get("padding_ratio", 0.05)), 0.01)

    # variant options
    try:
        tmp_pipeline = PlateRecognitionPipeline(config_path=config_path)
        variant_options = tmp_pipeline.get_preprocessing_variants()
    except Exception:
        variant_options = ["original_crop"]
    default_variant = (base_cfg.get("preprocessing", {}) or {}).get("default_variant", "original_crop")
    if default_variant not in variant_options:
        default_variant = variant_options[0]
    variant = st.selectbox("Preprocessing variant", options=variant_options, index=variant_options.index(default_variant))
    show_intermediate = st.checkbox("Tampilkan intermediate preprocessing", value=True)
    save_result = st.checkbox("Simpan hasil ke outputs/", value=False)


uploaded = st.file_uploader("Upload gambar kendaraan/plat", type=["jpg", "jpeg", "png", "bmp", "webp"])
run = st.button("Proses Gambar", type="primary", disabled=uploaded is None)

if uploaded is not None:
    image = decode_upload(uploaded)
    if image is None:
        st.error("Gagal membaca file gambar. Pastikan format gambar valid.")
    else:
        st.subheader("Input")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=uploaded.name, use_container_width=True)

        if run:
            with st.spinner("Memuat pipeline dan memproses gambar..."):
                pipeline = get_pipeline(config_path, weights, device)
                result = pipeline.process_image(
                    image,
                    variant=variant,
                    conf=conf,
                    iou=iou,
                    padding_ratio=padding,
                    return_images=True,
                )
                result.meta["input_filename"] = uploaded.name

            result_dict = result.to_dict(include_images=False)
            show_result_summary(result_dict)

            col1, col2 = st.columns(2)
            with col1:
                ann = result.meta.get("annotated_image") if isinstance(result.meta, dict) else None
                show_image("Deteksi (annotated)", ann)
                crop_img = result.crop.get("image") if isinstance(result.crop, dict) else None
                show_image("Crop plat", crop_img)
            with col2:
                pre_img = result.preprocessing.get("image") if isinstance(result.preprocessing, dict) else None
                show_image(f"Preprocessing ({variant})", pre_img)
                if show_intermediate and isinstance(result.preprocessing, dict):
                    inter = result.preprocessing.get("intermediate") or {}
                    if inter:
                        st.markdown("**Intermediate steps**")
                        for k, v in inter.items():
                            st.image(cv2.cvtColor(v, cv2.COLOR_BGR2RGB) if v.ndim == 3 else v, caption=k, use_container_width=True)

            with st.expander("Detail deteksi"):
                st.json(result_dict.get("detections", []))
            with st.expander("Detail OCR candidates"):
                ocr = result_dict.get("ocr") or {}
                st.json(ocr.get("candidates", []))
            with st.expander("Pipeline JSON (tanpa image array)"):
                st.json(result_dict)

            if save_result:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = PROJECT_ROOT / "outputs" / "predictions" / "pipeline" / ts
                paths = pipeline.save_result(result, out_dir)
                st.success(f"Hasil disimpan ke: {out_dir}")
                st.json(paths)

st.divider()
st.markdown("**Catatan:** M8 berfokus pada integrasi dan demonstrasi pipeline. Tanpa GT OCR terverifikasi, hasil OCR adalah prediksi model dan belum dievaluasi kuantitatif (CER/WER/EMA).")
