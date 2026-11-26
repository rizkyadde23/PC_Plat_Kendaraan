# ALPRO — Automatic License Plate Recognition Online

ALPRO adalah aplikasi **pengenalan plat nomor kendaraan Indonesia** berbasis Computer Vision dan OCR. Aplikasi ini menggabungkan **YOLO** untuk deteksi area plat, **OpenCV** untuk preprocessing citra, dan **EasyOCR** untuk pembacaan karakter, lalu menampilkan hasilnya melalui antarmuka **Streamlit**.

> **Status proyek:** Finalisasi teknis M1–M9 selesai. Deteksi memiliki metrik kuantitatif (val/test). **OCR kuantitatif (CER/WER/Exact Match) belum diklaim** karena ground truth teks terverifikasi belum tersedia (`gt_verified_available_count=0`).

## Fitur Utama

- **Landing page informatif** — menjelaskan apa itu ALPRO, alur kerja, dan teknologi yang digunakan.
- **Plate Recognition end-to-end** — upload gambar → deteksi plat → crop → preprocessing → OCR → normalisasi teks.
- **Recognition Result panel** — hasil OCR menjadi fokus utama, dilengkapi status, OCR confidence, detection confidence, dan jumlah plat terdeteksi.
- **Visualisasi tahapan** — bounding box deteksi, crop plat, hasil preprocessing, dan langkah intermediate opsional.
- **Configuration dialog** — pengaturan deteksi/preprocessing/UI dapat diubah dari dialog tanpa mengubah kode.
- **Pipeline reusable** — logika inti tersedia di `PlateRecognitionPipeline` dan dapat dipakai tanpa UI.
- **Penanganan error ramah** — status `no_detection`, `empty_ocr`, `load_error`, dll. ditampilkan tanpa crash.

## Pipeline

```text
Gambar kendaraan
      │
      ▼
Validasi & pembacaan gambar
      │
      ▼
Deteksi plat nomor (YOLOv8)
      │
      ▼
Cropping area plat
      │
      ▼
Preprocessing citra (OpenCV)
      │
      ▼
OCR (EasyOCR)
      │
      ▼
Normalisasi teks
      │
      ▼
Hasil & visualisasi (Streamlit / Python API)
```

## Tech Stack

| Teknologi | Peran |
|---|---|
| Python | Bahasa utama aplikasi |
| Streamlit | Antarmuka web lokal |
| Ultralytics YOLOv8 | Deteksi area plat nomor |
| OpenCV | Pembacaan gambar, crop, preprocessing |
| EasyOCR | Pembacaan karakter plat |
| NumPy / Pandas | Pengolahan data dan evaluasi |
| PyYAML | Konfigurasi pipeline |
| pytest / jiwer | Pengujian dan metrik evaluasi |

## Struktur Repository

```text
.
├── app.py                          # Landing page Streamlit (entry point)
├── pages/
│   └── 1_Plate_Recognition.py      # Halaman Plate Recognition
├── ui/
│   ├── components.py               # Komponen UI reusable
│   ├── dialogs.py                  # Configuration dialog
│   ├── state.py                    # Session state/config helpers
│   └── styles.py                   # CSS white modern theme
├── configs/
│   ├── app.yaml
│   ├── dataset.yaml
│   ├── detector.yaml
│   ├── ocr.yaml
│   ├── pipeline.yaml
│   └── preprocessing.yaml
├── src/alpro/
│   ├── config/                     # Loader/merge config
│   ├── core/                       # PipelineResult & PlateRecognitionPipeline
│   ├── detection/                  # PlateDetector (YOLO wrapper)
│   ├── preprocessing/              # Loader, cropper, preprocessor
│   ├── recognition/                # OCREngine, TextNormalizer
│   ├── evaluation/                 # Metrik/evaluator (kerangka/eksperimen)
│   └── utils/
├── scripts/                        # Audit, training, evaluasi, eksperimen
├── tests/                          # Unit & integration tests
├── data/
│   ├── raw/indonesian_license_plate/   # Dataset mentah (JANGAN diubah)
│   ├── annotations/ocr/                # Template/kandidat/GT OCR
│   └── splits/                         # Manifest turunan
├── outputs/                        # Predictions, visualizations, reports
├── docs/                           # Checklist smoke UI & panduan GT OCR
├── konteks.md
├── perencanaan.md
└── struktur.md
```

## Prasyarat

- **Python 3.10+** (proyek ini dikembangkan/diuji pada Python 3.13.3, Windows).
- **pip** dan **virtual environment**.
- RAM/CPU cukup untuk inferensi EasyOCR + YOLO (CPU-first). GPU opsional.
- Dataset mentah tersedia lokal di `data/raw/indonesian_license_plate/` jika ingin menjalankan script training/evaluasi dataset.

## Instalasi

### 1. Clone repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Buat dan aktifkan virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows CMD**
```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux/macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Catatan: instalasi `torch`, `ultralytics`, dan `easyocr` dapat memakan waktu. Untuk GPU, instal PyTorch sesuai CUDA sebelum/bersamaan kebutuhan Ultralytics.

## Menjalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Lalu buka URL lokal yang ditampilkan Streamlit (biasanya `http://localhost:8501`).

### Cara pakai UI

1. Buka **Home** (landing page ALPRO).
2. Klik **Mulai Pembacaan** / **Start Recognition**.
3. Di halaman **Plate Recognition**, klik **Configuration** bila ingin mengubah parameter.
4. Upload gambar kendaraan (JPG/JPEG/PNG/BMP/WEBP).
5. Klik **Mulai Pembacaan**.
6. Lihat **Recognition Result** (fokus utama), lalu detail deteksi, crop, preprocessing, dan informasi tambahan.

### Konfigurasi di UI

Dialog **Configuration** mencakup:
- **Detection**: confidence threshold, IoU, image size, max detections.
- **Preprocessing**: default variant, crop padding ratio.
- **OCR**: path config OCR (read-only di dialog).
- **Application Preferences**: tampilkan langkah intermediate, aktifkan simpan hasil.

Perubahan berlaku saat **Save Configuration** ditekan. **Reset to Default** mengembalikan nilai default; **Cancel** membatalkan perubahan draft.

## Menjalankan Pipeline via Python (tanpa UI)

```python
import sys, cv2
sys.path.insert(0, "src")

from alpro.core import PlateRecognitionPipeline

pipeline = PlateRecognitionPipeline(config_path="configs/pipeline.yaml")
image = cv2.imread("path/to/image.jpg")

result = pipeline.process_image(image, return_images=True)
print(result.status)
print(result.ocr["text_normalized"] if result.ocr else None)
```

`PipelineResult` berisi status, detections, selected detection, crop, preprocessing, OCR raw/normalized/confidence, timings, warnings/errors, dan metadata.

## Konfigurasi

File konfigurasi utama:
- `configs/pipeline.yaml` — orkestrasi end-to-end (detector, crop, preprocessing, OCR, output, UI).
- `configs/detector.yaml` — parameter deteksi/eksperimen YOLO.
- `configs/preprocessing.yaml` — varian preprocessing.
- `configs/ocr.yaml` — engine OCR dan normalisasi.

Default penting saat ini:
- Device: **CPU**.
- Bobot detektor: `runs/detect/models/trained/detection/yolov8n_baseline_m5_20261118_cpu/weights/best.pt`.
- Preprocessing default: `original_crop` (konservatif; belum diklaim terbaik secara kuantitatif OCR).

## Pengujian & Evaluasi

### Test otomatis

```bash
pytest tests/unit tests/integration -q
```

Status saat ini: **11 passed** (unit + integration mock).

### Smoke evaluasi final M9

```bash
python scripts/run_final_evaluation.py
```

Artefak:
- `outputs/reports/final/m9_evaluation/m9_final_report.md`
- `outputs/reports/final/m9_evaluation/m9_evaluation_summary.json`

### Smoke UI manual

Gunakan checklist: [`docs/m9_smoke_ui_checklist.md`](docs/m9_smoke_ui_checklist.md).

## Hasil Evaluasi (Ringkasan)

### Deteksi (M5 baseline, kuantitatif)

| Split | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| val | 0.9406 | 0.8958 | 0.9010 | 0.7146 |
| test (sekali) | 0.9833 | 0.9483 | 0.9547 | 0.7614 |

Sumber: `outputs/reports/detection/yolov8n_baseline_m5_20261118_cpu/`.

### OCR (M7, status)

- Ground truth terverifikasi tersedia: **0**.
- **CER/WER/Exact Match belum diklaim**.
- Evaluator sudah tersedia (`scripts/evaluate_ocr.py`); metrik kuantitatif baru boleh diklaim setelah GT terverifikasi diisi.

Panduan: [`docs/m9_gt_ocr_verification_guide.md`](docs/m9_gt_ocr_verification_guide.md).

## Dataset & Lisensi Data

- Dataset: **Indonesian_License_Plate** (Roboflow Universe, `dataset-pyufc/indonesian_license_plate`).
- Lisensi dataset: **CC BY 4.0**. Pertahankan atribusi saat menggunakan/distribusi turunan.
- Data lokal: `data/raw/indonesian_license_plate/` (**jangan diubah/commit**).
- Data turunan: `data/interim/`, `data/processed/`, `data/annotations/`, `data/splits/`.
- Detail: [`data/README.md`](data/README.md).

## Batasan & Catatan Penting

- “Online” pada nama ALPRO adalah identitas proyek, **bukan** klaim deployment publik/real-time.
- Input saat ini: **gambar statis** (bukan CCTV/video real-time).
- OCR adalah **prediksi model**; confidence OCR bukan akurasi ground truth.
- Metrik OCR kuantitatif hanya valid dengan GT teks terverifikasi.
- Dataset memiliki catatan audit: 4 label kosong provisional, 374 grup duplikat intra-split train, 2 kasus no-detection val diketahui.
- `data/raw/` tidak boleh diubah.

## Troubleshooting

- **Model/weight tidak ditemukan**: cek `detector.weights` di `configs/pipeline.yaml` atau lewat Configuration dialog (path harus valid).
- **Import error setelah clone**: pastikan `.venv` aktif dan `pip install -r requirements.txt` sukses.
- **EasyOCR/torch lambat di CPU**: normal untuk CPU-first; gunakan gambar kecil/jelas dan pertimbangkan GPU bila tersedia.
- **Plat tidak terdeteksi**: turunkan confidence threshold di Configuration, atau gunakan gambar yang lebih jelas.
- **OCR kosong/salah baca**: coba preprocessing variant lain (`gray_clahe`, dll.), pastikan crop plat cukup jelas.
- **Streamlit tidak terbuka**: pastikan port 8501 tidak dipakai; jalankan `streamlit run app.py --server.port 8502`.

## Dokumentasi Proyek

- [`konteks.md`](konteks.md) — konteks, tujuan, ruang lingkup, keputusan.
- [`perencanaan.md`](perencanaan.md) — roadmap dan status milestone/fase.
- [`struktur.md`](struktur.md) — struktur direktori dan tanggung jawab modul.
- [`data/README.md`](data/README.md) — dokumentasi dataset dan audit.
- [`docs/m9_smoke_ui_checklist.md`](docs/m9_smoke_ui_checklist.md) — checklist smoke UI manual.
- [`docs/m9_gt_ocr_verification_guide.md`](docs/m9_gt_ocr_verification_guide.md) — panduan verifikasi GT OCR.

## Kontribusi / Pengembangan

1. Gunakan branch terpisah untuk perubahan.
2. Jaga pemisahan UI vs logika pipeline.
3. Jangan ubah `data/raw/`.
4. Tambahkan/ubah test bila perilaku berubah.
5. Jalankan `pytest tests/unit tests/integration -q` sebelum commit.

## Atribusi

- Dataset: *Indonesian_License_Plate* oleh kontributor Roboflow Universe (`dataset-pyufc`), **CC BY 4.0**.  
  https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate
- Proyek akademik: ALPRO — Pengolahan Citra Digital, Informatika, 2026/2027.
