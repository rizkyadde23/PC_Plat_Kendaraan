"""ALPRO - Home / Landing Page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ui.state import init_config_state  # noqa: E402
from ui.styles import inject_css  # noqa: E402


st.set_page_config(
    page_title="ALPRO — Home",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()
init_config_state()


# Hero
left, right = st.columns([1.1, 1], gap="large")
with left:
    st.markdown(
        """
<div class="alpro-hero">
  <div class="alpro-hero-badge">AI Computer Vision Application</div>
  <h1>Smart Indonesian <span class="accent">License Plate</span> Recognition</h1>
  <p>Deteksi, proses, dan baca plat nomor kendaraan Indonesia secara otomatis menggunakan Computer Vision dan Optical Character Recognition.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="alpro-big-btn" style="max-width:320px;">', unsafe_allow_html=True)
    if st.button("Mulai Pembacaan", type="primary", use_container_width=True, icon=":material/arrow_forward:"):
        st.switch_page("pages/1_Plate_Recognition.py")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="alpro-muted-text" style="margin-top:0.65rem;">Upload gambar kendaraan dan dapatkan hasil OCR secara otomatis.</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
<div class="alpro-hero-visual">
  <div class="alpro-visual-frame">
    <div class="alpro-visual-car">
      <div style="text-align:center;">
        <div style="font-size:0.85rem;color:#64748B;">Vehicle Image</div>
        <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.25rem;">YOLO detection area</div>
      </div>
      <div class="alpro-bbox"><span class="alpro-bbox-label">License Plate</span></div>
    </div>
    <div class="alpro-visual-result">
      <div>
        <div style="font-size:0.72rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Illustration Preview</div>
        <div class="plate">PLATE • OCR</div>
      </div>
      <div class="status">Illustration</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# About
st.markdown('<div class="alpro-section-title">Tentang Aplikasi</div>', unsafe_allow_html=True)
st.markdown("## Apa itu ALPRO?")
a1, a2 = st.columns([1.2, 1], gap="large")
with a1:
    st.markdown(
        """
<div class="alpro-card">
  <p style="margin:0;color:#334155;line-height:1.8;">
    <strong>ALPRO (Automatic License Plate Recognition Online)</strong> adalah aplikasi pengenalan plat nomor kendaraan Indonesia
    yang menggabungkan <strong>Object Detection</strong> dan <strong>Optical Character Recognition (OCR)</strong>.
    Sistem mendeteksi lokasi plat nomor pada gambar kendaraan, memproses area plat, lalu membaca karakter yang terdapat di dalamnya.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
with a2:
    st.markdown(
        """
<div class="alpro-card-soft">
  <div style="font-weight:700;color:#0F172A;margin-bottom:0.5rem;">Manfaat utama</div>
  <div class="alpro-muted-text">• Membantu membaca plat nomor dari gambar kendaraan.<br>
  • Menampilkan tahapan proses secara transparan.<br>
  • Memberikan hasil OCR beserta confidence dan status pemrosesan.</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# How it works
st.markdown('<div class="alpro-section-title">Alur Kerja</div>', unsafe_allow_html=True)
st.markdown("## How It Works")
st.caption("Pipeline ALPRO memproses gambar dari upload hingga hasil pembacaan teks plat.")
steps = [
    ("1", "Upload Image", "Pengguna mengunggah gambar kendaraan. Sistem memvalidasi format dan membaca gambar.", ""),
    ("2", "Plate Detection", "Model YOLO mendeteksi area plat nomor dan menghasilkan bounding box.", ""),
    ("3", "Image Preprocessing", "OpenCV memproses crop plat: grayscale, denoise, thresholding, atau mode lain sesuai konfigurasi.", "teal"),
    ("4", "Text Recognition", "EasyOCR membaca karakter pada gambar plat yang telah diproses.", ""),
    ("5", "Recognition Result", "Sistem menormalkan teks dan menampilkan hasil, confidence, serta status.", "teal"),
]
cols = st.columns(5)
for col, (num, title, desc, icon_cls) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
<div class="alpro-grid-card">
  <div class="alpro-icon {icon_cls}">{num}</div>
  <h4>{title}</h4>
  <p>{desc}</p>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# Technology
st.markdown('<div class="alpro-section-title">Teknologi</div>', unsafe_allow_html=True)
st.markdown("## Technology Stack")
tech = [
    ("Python", "Bahasa utama untuk pipeline, integrasi model, dan aplikasi."),
    ("YOLO", "Mendeteksi lokasi plat nomor pada gambar kendaraan."),
    ("OpenCV", "Preprocessing citra untuk meningkatkan kualitas input OCR."),
    ("EasyOCR", "Membaca karakter teks dari area plat nomor."),
    ("Streamlit", "Antarmuka web lokal yang interaktif dan cepat dikembangkan."),
]
tcols = st.columns(5)
for col, (name, desc) in zip(tcols, tech):
    with col:
        st.markdown(
            f"""
<div class="alpro-tech">
  <div class="name">{name}</div>
  <div class="desc">{desc}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

# CTA
st.markdown(
    """
<div class="alpro-cta">
  <div class="alpro-section-title" style="justify-content:center;">Mulai Sekarang</div>
  <h2>Ready to Recognize a License Plate?</h2>
  <p>Upload gambar kendaraan dan biarkan ALPRO memproses plat nomor secara otomatis.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="alpro-big-btn alpro-center-cta">', unsafe_allow_html=True)
if st.button("Start Recognition", type="primary", use_container_width=True, icon=":material/document_scanner:", key="cta_bottom"):
    st.switch_page("pages/1_Plate_Recognition.py")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.caption("ALPRO • White Modern AI Interface • Prediksi OCR bukan klaim akurasi ground truth.")
