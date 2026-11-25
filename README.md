# ALPRO — Automatic License Plate Recognition Online

ALPRO adalah proyek aplikasi pengenalan plat nomor kendaraan Indonesia berbasis pengolahan citra digital dan machine learning. Pipeline yang direncanakan meliputi deteksi plat nomor (YOLO), cropping, preprocessing citra (OpenCV), dan OCR (EasyOCR), dengan antarmuka pengguna berbasis Streamlit.

> **Status (s.d. Milestone 9):** M1–M8 selesai dan M9 finalisasi/evaluasi menyeluruh telah dijalankan (pytest lulus, smoke pipeline end-to-end, laporan final M9). **Deteksi** memiliki metrik kuantitatif (val/test). **OCR kuantitatif (CER/WER/Exact Match) belum diklaim** karena GT teks terverifikasi belum tersedia (`gt_verified_available_count=0`).

## Ruang Lingkup Awal
- Input: gambar statis (upload).
- Fokus: deteksi area plat, pemrosesan citra, dan pengenalan karakter.
- Di luar ruang lingkup awal: video/CCTV real-time, deployment publik, sistem monitoring/identifikasi pemilik, autentikasi.
- "Online" pada nama proyek adalah identitas proyek, bukan kewajiban pemrosesan real-time atau deployment.

## Prasyarat
- **Python** (disarankan menggunakan virtual environment). Milestone 2 saat ini menggunakan `.venv` dengan **Python 3.13.3** (keputusan 1A).
- **pip** terbaru di dalam environment.
- (Opsional, nanti) **GPU NVIDIA + driver/CUDA** untuk akselerasi training/inferensi. Target awal Milestone 2: **CPU-first** (keputusan 2C).

## Struktur Repository (ringkas)
- `configs/` — konfigurasi aplikasi, detektor, preprocessing (kerangka).
- `data/raw/indonesian_license_plate/` — dataset mentah Roboflow (sudah tersedia; **jangan diubah/commit**).
- `data/interim/`, `data/processed/`, `data/annotations/`, `data/splits/` — data turunan/anotasi kerja (di luar `raw/`).
- `models/pretrained/`, `models/trained/` — bobot model.
- `src/alpro/` — paket kode inti (kerangka modul: config, core, detection, preprocessing, recognition, evaluation, utils).
- `ui/` — komponen antarmuka (kerangka).
- `scripts/` — script audit/training/evaluasi (akan ditambahkan sesuai milestone).
- `tests/unit/`, `tests/integration/` — pengujian otomatis.
- `outputs/predictions/`, `outputs/visualizations/`, `outputs/reports/` — artefak eksperimen/laporan.
- `konteks.md`, `perencanaan.md`, `struktur.md` — dokumen acuan proyek.

## Setup Environment (Windows / PowerShell)

1. **Masuk ke direktori proyek**
   ```powershell
   cd "D:\Code\Python\Pengolahan Citra\Projek Akhir"
   ```

2. **Aktifkan virtual environment**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   (Alternatif CMD: `.\.venv\Scripts\activate.bat`)

3. **Perbarui pip (disarankan)**
   ```powershell
   python -m pip install --upgrade pip
   ```

4. **Instal dependency baseline**
   ```powershell
   pip install -r requirements.txt
   ```
   - Strategi versi dependency: **hybrid (3C)** — `requirements.txt` menggunakan constraint/range wajar; versi aktual yang terpasang perlu dicatat setelah verifikasi untuk reproducibility.
   - OpenCV menggunakan **`opencv-python`** (keputusan 6A).

## Catatan CPU / Upgrade ke CUDA (2C)
- Konfigurasi awal menargetkan **CPU** (`configs/app.yaml`, `configs/detector.yaml`, `.env.example`).
- Untuk menggunakan **CUDA** di kemudian hari:
  1. Pastikan GPU NVIDIA, driver, dan toolkit yang kompatibel tersedia.
  2. Instal build **PyTorch** yang sesuai CUDA (mengikuti panduan resmi PyTorch) sebelum/bersamaan kebutuhan Ultralytics.
  3. Ubah pengaturan perangkat (mis. `runtime.device` / `inference.device` / `ALPRO_DEVICE`) ke `cuda` (atau indeks GPU spesifik) saat environment siap.
  4. Verifikasi ulang instalasi dan inferensi/training pada perangkat target.

## Verifikasi Environment (dasar)
Setelah instalasi dependency, verifikasi import paket inti:

```powershell
python -c "import cv2, numpy, pandas, yaml, streamlit, ultralytics, easyocr, matplotlib, pytest, jiwer; print('imports_ok')"
python --version
pip show ultralytics opencv-python easyocr streamlit numpy pandas matplotlib pytest jiwer PyYAML
```

> Jika ada paket yang gagal terpasang/diimpor (mis. karena kompatibilitas Python 3.13), catat pesan error dan versi paket; sesuaikan environment/dependency secara terkontrol.

## Dataset
- Dataset utama: **Indonesian_License_Plate** (Roboflow Universe) — lisensi **CC BY 4.0**.
- Lokasi lokal: `data/raw/indonesian_license_plate/` (sudah terekstrak).
- **Jangan** mengunduh/mengekstrak ulang, memindahkan, atau mengubah file dataset mentah. Simpan hasil olahan di `data/interim/` atau `data/processed/`.
- Detail sumber, metadata, jumlah aktual, temuan audit awal, dan kebijakan: lihat [`data/README.md`](data/README.md).

## Menjalankan Pipeline (Python, tanpa UI)

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; sys.path.insert(0,'src'); from alpro.core import PlateRecognitionPipeline; import cv2; p=PlateRecognitionPipeline(); img=cv2.imread('data/raw/indonesian_license_plate/test/images/<file>.jpg'); r=p.process_image(img); print(r.status, r.ocr['text_normalized'] if r.ocr else None)"
```

## Menjalankan UI (Streamlit)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

UI mendukung upload gambar, konfigurasi deteksi/crop/varian preprocessing, tampilan tahapan (annotated, crop, preprocessed, intermediate), hasil OCR raw+normalized+confidence, dan opsi simpan artefak ke `outputs/predictions/pipeline/<timestamp>/`.

## Evaluasi Final (M9)

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/unit tests/integration -q
.\.venv\Scripts\python.exe scripts/run_final_evaluation.py
```

Artefak M9:
- `outputs/reports/final/m9_evaluation/m9_final_report.md`
- `outputs/reports/final/m9_evaluation/m9_evaluation_summary.json`
- `docs/m9_smoke_ui_checklist.md` (smoke UI manual)
- `docs/m9_gt_ocr_verification_guide.md` (cara verifikasi GT OCR)

Catatan: integration test pipeline menggunakan mock detector/OCR agar cepat dan deterministik. OCR kuantitatif hanya boleh diklaim setelah GT terverifikasi diisi dan dievaluasi.

## Pengujian
- Kerangka direktori pengujian tersedia di `tests/unit/` dan `tests/integration/`.
- Test fitur akan ditambahkan seiring implementasi modul (belum ada suite fitur pada Milestone 2).
- Menjalankan pytest (ketika test tersedia):
  ```powershell
  pytest
  ```

## Dokumentasi Proyek
- [`konteks.md`](konteks.md) — konteks, tujuan, ruang lingkup, dan keputusan.
- [`perencanaan.md`](perencanaan.md) — roadmap fase/milestone.
- [`struktur.md`](struktur.md) — rancangan struktur direktori dan tanggung jawab modul.
- [`data/README.md`](data/README.md) — dokumentasi dataset dan audit.

## Catatan Pengembangan
- Jaga modularitas (UI terpisah dari logika inti), konfigurasi terpisah dari implementasi, dan reproducibility eksperimen.
- Laporkan hasil pengujian/evaluasi secara faktual; metrik deteksi dan OCR dievaluasi terpisah dan metrik OCR kuantitatif memerlukan ground truth teks.
