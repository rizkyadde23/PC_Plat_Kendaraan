"""ALPRO Streamlit app - Milestone 8/9 UI refresh.

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
from ui.styles import inject_css  # noqa: E402


st.set_page_config(page_title="ALPRO - Plate Recognition", page_icon=":car:", layout="wide")
inject_css()

st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.20rem;"><span class="alpro-badge">ALPRO • Plate Recognition Demo</span></div>', unsafe_allow_html=True)
st.title("ALPRO — Pengenalan Plat Nomor Kendaraan")
st.caption("Unggah gambar kendaraan, lalu sistem akan mendeteksi plat, memotong area plat, memproses gambar, dan membaca teksnya. Hasil OCR adalah prediksi model, bukan klaim akurasi ground truth.")


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


# Load default config for initial values
default_config_path = "configs/pipeline.yaml"
try:
    initial_cfg = load_pipeline_config(default_config_path)
except Exception:
    initial_cfg = {}
initial_det = initial_cfg.get("detector", {}) if isinstance(initial_cfg, dict) else {}
initial_crop = initial_cfg.get("crop", {}) if isinstance(initial_cfg, dict) else {}
initial_pre = initial_cfg.get("preprocessing", {}) if isinstance(initial_cfg, dict) else {}


with st.sidebar:
    st.markdown("## Pengaturan")
    st.markdown('<div class="alpro-help">Atur cara sistem membaca plat. Default sudah bisa dipakai; ubah jika hasil belum sesuai.</div>', unsafe_allow_html=True)

    st.markdown('<div class="alpro-card-title">Pengaturan Mudah</div>', unsafe_allow_html=True)

    conf = st.slider(
        "Tingkat keyakinan deteksi",
        min_value=0.05,
        max_value=1.0,
        value=float(initial_det.get("conf", 0.25)),
        step=0.05,
        help="Seberapa yakin sistem sebelum menandai area sebagai plat. Lebih tinggi = lebih ketat; lebih rendah = lebih sensitif.",
    )
    st.markdown('<div class="alpro-help">Jika plat tidak terdeteksi, coba turunkan sedikit. Jika banyak deteksi salah, coba naikkan.</div>', unsafe_allow_html=True)

    iou = st.slider(
        "Toleransi tumpang tindih deteksi",
        min_value=0.05,
        max_value=1.0,
        value=float(initial_det.get("iou", 0.45)),
        step=0.05,
        help="Mengatur bagaimana sistem menggabungkan kotak deteksi yang saling tumpang tindih. Biasanya biarkan default.",
    )
    st.markdown('<div class="alpro-help">Umumnya tidak perlu diubah. Ubah hanya jika kotak deteksi terlihat ganda atau kurang pas.</div>', unsafe_allow_html=True)

    padding = st.slider(
        "Margin potongan plat",
        min_value=0.0,
        max_value=0.30,
        value=float(initial_crop.get("padding_ratio", 0.05)),
        step=0.01,
        help="Menambah sedikit area di sekitar plat saat dipotong agar karakter di tepi tidak terpotong.",
    )
    st.markdown('<div class="alpro-help">Nilai kecil (0.03–0.08) biasanya cukup. Terlalu besar bisa membawa latar belakang berlebih.</div>', unsafe_allow_html=True)

    # Variants
    try:
        tmp_pipeline = PlateRecognitionPipeline(config_path=default_config_path)
        variant_options = tmp_pipeline.get_preprocessing_variants()
    except Exception:
        variant_options = ["original_crop"]
    default_variant = str(initial_pre.get("default_variant", "original_crop"))
    if default_variant not in variant_options:
        default_variant = variant_options[0]

    variant = st.selectbox(
        "Mode pemrosesan gambar",
        options=variant_options,
        index=variant_options.index(default_variant),
        help="Cara menyiapkan gambar plat sebelum dibaca OCR. 'original_crop' paling natural; opsi lain mencoba meningkatkan kontras/binerisasi.",
    )
    st.markdown('<div class="alpro-help">Mulai dari <b>original_crop</b>. Coba <b>gray_clahe</b> atau mode lain jika teks sulit terbaca.</div>', unsafe_allow_html=True)

    show_intermediate = st.checkbox(
        "Tampilkan langkah pemrosesan antara",
        value=False,
        help="Menampilkan hasil tiap tahap preprocessing. Berguna untuk memahami proses, tapi membuat tampilan lebih panjang.",
    )
    save_result = st.checkbox(
        "Simpan hasil ke folder outputs",
        value=False,
        help="Menyimpan gambar hasil dan data prediksi ke outputs/predictions/pipeline/<waktu>/.",
    )

    with st.expander("Pengaturan Lanjutan (Teknis)"):
        st.caption("Biasanya tidak perlu diubah oleh pengguna umum.")
        config_path = st.text_input("Pipeline config", value=default_config_path, help="File konfigurasi pipeline YAML.")
        weights = st.text_input(
            "Detector weights",
            value=str(initial_det.get("weights", "runs/detect/models/trained/detection/yolov8n_baseline_m5_20261118_cpu/weights/best.pt")),
            help="Path bobot model deteksi YOLO (.pt).",
        )
        device = st.selectbox("Device", options=["cpu"], index=0, help="Perangkat inferensi. Saat ini disetel CPU.")


# Main content
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
    st.markdown("### 1. Upload Gambar")
    st.markdown('<div class="alpro-help">Format: JPG, JPEG, PNG, BMP, WEBP. Gunakan foto yang jelas agar plat mudah terbaca.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih gambar kendaraan/plat", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    run = st.button("Proses Gambar", type="primary", disabled=uploaded is None, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        image = decode_upload(uploaded)
        if image is None:
            st.error("Gagal membaca file gambar. Pastikan format gambar valid.")
        else:
            st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
            st.markdown("### Gambar Input")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=uploaded.name, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        image = None
        st.markdown('<div class="alpro-card-muted">', unsafe_allow_html=True)
        st.markdown("**Tips foto yang baik**")
        st.markdown("- Plat terlihat jelas dan tidak terlalu blur\n- Pencahayaan cukup\n- Plat tidak terlalu kecil dalam gambar\n- Sudut pengambilan tidak terlalu miring")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    if uploaded is not None and image is not None and run:
        with st.spinner("Memproses gambar... Sistem sedang mendeteksi dan membaca plat."):
            try:
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
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                st.stop()

        result_dict = result.to_dict(include_images=False)
        show_result_summary(result_dict)

        ann = result.meta.get("annotated_image") if isinstance(result.meta, dict) else None
        crop_img = result.crop.get("image") if isinstance(result.crop, dict) else None
        pre_img = result.preprocessing.get("image") if isinstance(result.preprocessing, dict) else None
        inter = {}
        if show_intermediate and isinstance(result.preprocessing, dict):
            inter = result.preprocessing.get("intermediate") or {}

        has_visual = any(isinstance(img, np.ndarray) for img in [ann, crop_img, pre_img]) or bool(inter)
        if has_visual:
            st.markdown('<div class="alpro-card">', unsafe_allow_html=True)
            st.markdown("### 2. Tahapan Visual")
            v1, v2 = st.columns(2)
            with v1:
                show_image("Deteksi Plat", ann, caption="Kotak hijau menunjukkan area plat yang terdeteksi.")
                show_image("Hasil Potongan Plat", crop_img, caption="Area plat yang dipotong untuk dibaca OCR.")
            with v2:
                show_image(f"Hasil Pemrosesan ({variant})", pre_img, caption="Gambar setelah disiapkan untuk OCR.")
                if inter:
                    st.markdown("**Langkah pemrosesan antara**")
                    for k, v in inter.items():
                        if isinstance(v, np.ndarray):
                            st.image(cv2.cvtColor(v, cv2.COLOR_BGR2RGB) if v.ndim == 3 else v, caption=k, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Detail Teknis (Opsional)"):
            st.caption("Bagian ini untuk kebutuhan debugging/evaluasi teknis.")
            st.markdown("**Deteksi**")
            st.json(result_dict.get("detections", []))
            st.markdown("**Kandidat OCR**")
            ocr = result_dict.get("ocr") or {}
            st.json(ocr.get("candidates", []))
            st.markdown("**Waktu proses (ms)**")
            st.json(result_dict.get("timings_ms", {}))
            st.markdown("**Pipeline JSON (tanpa image array)**")
            st.json(result_dict)

        if save_result:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = PROJECT_ROOT / "outputs" / "predictions" / "pipeline" / ts
            paths = pipeline.save_result(result, out_dir)
            st.success(f"Hasil disimpan ke: {out_dir}")
            with st.expander("File tersimpan"):
                st.json(paths)

    elif uploaded is not None and image is not None:
        st.markdown('<div class="alpro-card-muted">', unsafe_allow_html=True)
        st.markdown("### 2. Siap Diproses")
        st.markdown("Klik **Proses Gambar** untuk menjalankan deteksi plat dan pembacaan OCR.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="alpro-card-muted">', unsafe_allow_html=True)
        st.markdown("### 2. Hasil Akan Muncul di Sini")
        st.markdown("Setelah gambar diunggah dan diproses, sistem akan menampilkan status, teks plat, dan tahapan visual.")
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption("ALPRO Demo • Tema putih minimalis • OCR adalah prediksi model; evaluasi kuantitatif memerlukan ground truth terverifikasi.")
