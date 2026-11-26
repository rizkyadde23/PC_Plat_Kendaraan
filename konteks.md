# Konteks Proyek ALPRO

## 1. Identitas Proyek

| Elemen | Keterangan |
|---|---|
| Nama proyek | **ALPRO — Automatic License Plate Recognition Online** |
| Mata kuliah | Pengolahan Citra Digital |
| Program studi | Informatika |
| Tahun akademik | 2026/2027 |
| Bahasa pemrograman | Python |
| Bentuk aplikasi | Aplikasi lokal berbasis web menggunakan Streamlit |
| Fokus pengerjaan | Pengembangan perangkat lunak aplikasi |
| Domain masalah | Transportasi dan pengolahan citra |
| Status | **Finalisasi teknis selesai (M1–M9)**; OCR kuantitatif tertunda sampai GT terverifikasi tersedia |

## 2. Latar Belakang

Plat nomor kendaraan merupakan identitas registrasi kendaraan. Membaca plat dari gambar secara manual membutuhkan waktu dan rentan kesalahan. ALPRO dikembangkan sebagai aplikasi pengenalan plat nomor kendaraan Indonesia yang menggabungkan deteksi objek, preprocessing citra, dan OCR dalam satu alur yang transparan dan dapat dievaluasi.

## 3. Permasalahan

Proyek berfokus pada bagaimana membangun aplikasi Python yang mampu:

1. Mendeteksi lokasi plat nomor pada gambar kendaraan.
2. Mengambil area plat dari gambar.
3. Menerapkan preprocessing citra agar karakter lebih mudah dibaca.
4. Membaca karakter plat menggunakan OCR.
5. Menampilkan hasil dan tahapan pemrosesan kepada pengguna.
6. Mengevaluasi kinerja deteksi dan status evaluasi OCR secara jujur.

Kendala umum: pencahayaan, resolusi, sudut, blur, noise, kontras, plat tertutup, dan kualitas anotasi.

## 4. Tujuan Proyek

- Mengembangkan aplikasi pengenalan plat nomor berbasis Python.
- Menerapkan deteksi plat menggunakan YOLO.
- Menerapkan preprocessing citra menggunakan OpenCV.
- Mengintegrasikan EasyOCR sebagai baseline OCR.
- Menyediakan UI Streamlit yang modern dan mudah digunakan.
- Menyediakan pipeline yang dapat diuji tanpa UI.
- Mendokumentasikan hasil, keterbatasan, dan kondisi gagal secara faktual.

## 5. Pengguna dan Skenario Utama

**Pengguna sasaran:** mahasiswa, dosen, dan pengguna demo yang ingin menguji pembacaan plat dari gambar.

**Skenario utama:**
1. Pengguna membuka halaman Home ALPRO.
2. Pengguna masuk ke halaman Plate Recognition.
3. Pengguna mengatur konfigurasi (opsional) melalui dialog Configuration.
4. Pengguna mengunggah gambar kendaraan.
5. Sistem mendeteksi plat, crop, preprocessing, OCR, dan normalisasi.
6. UI menampilkan hasil OCR sebagai panel utama, beserta deteksi/crop/preprocessing/detail.
7. Jika plat tidak terdeteksi/OCR kosong/error, sistem menampilkan status informatif.

## 6. Ruang Lingkup

### Termasuk
- Input gambar statis (upload).
- Deteksi area plat (YOLOv8).
- Cropping area plat.
- Preprocessing OpenCV (varian terkontrol).
- OCR EasyOCR + normalisasi ringan.
- UI Streamlit (landing page + plate recognition + configuration dialog).
- Pengujian otomatis dan smoke evaluasi.
- Dokumentasi teknis dan artefak evaluasi.

### Di luar ruang lingkup awal
- Video/CCTV real-time.
- Pelacakan kendaraan antar-frame.
- Deployment publik/produksi.
- Integrasi parkir/tol/kepolisian.
- Identifikasi pemilik kendaraan.
- Klaim akurasi universal pada semua kondisi lapangan.

> “Online” pada nama ALPRO adalah identitas proyek, bukan kewajiban real-time/deployment publik.

## 7. Arsitektur & Pipeline

```text
Gambar kendaraan
  -> Validasi & load
  -> Deteksi plat (YOLO)
  -> Crop plat
  -> Preprocessing (OpenCV)
  -> OCR (EasyOCR)
  -> Normalisasi teks
  -> Tampilan hasil (UI/API)
```

Implementasi utama:
- `src/alpro/core/pipeline.py` — `PlateRecognitionPipeline.process_image(...)`
- `src/alpro/core/result.py` — `PipelineResult`
- `app.py` — landing page
- `pages/1_Plate_Recognition.py` — halaman recognition
- `ui/dialogs.py` — configuration dialog

## 8. Dataset

Sumber utama: **Indonesian_License_Plate** (Roboflow Universe, `dataset-pyufc/indonesian_license_plate`), lisensi **CC BY 4.0**.

- Lokasi lokal: `data/raw/indonesian_license_plate/` (**jangan diubah**).
- Data turunan disimpan di `data/interim/`, `data/processed/`, `data/annotations/`, `data/splits/`.
- Audit M3: struktur/pairing valid, 0 cross-split duplicate terdeteksi, 374 grup duplikat intra-split train, 4 label kosong provisional.
- Anotasi bbox tersedia; **GT teks OCR terverifikasi belum tersedia**.

Detail: [`data/README.md`](data/README.md).

## 9. Status Evaluasi

### Deteksi (kuantitatif, M5)
- Val: P=0.9406, R=0.8958, mAP@50=0.9010, mAP@50-95=0.7146
- Test (sekali): P=0.9833, R=0.9483, mAP@50=0.9547, mAP@50-95=0.7614

### OCR (M7/M9)
- Pipeline/evaluator OCR siap.
- GT terverifikasi: **0**.
- **CER/WER/Exact Match belum diklaim**.
- Hasil OCR di UI adalah prediksi model; confidence OCR bukan akurasi ground truth.

### Software/UI
- `pytest tests/unit tests/integration -q`: **11 passed**.
- Smoke pipeline end-to-end M9: `ok=3, no_detection=2, load_error=2`.
- Smoke UI manual: checklist tersedia (`docs/m9_smoke_ui_checklist.md`).

## 10. Keputusan Teknis Final (Ringkas)

| Keputusan | Status |
|---|---|
| Dataset utama | Indonesian_License_Plate (Roboflow, CC BY 4.0), lokal read-only |
| Deteksi | YOLOv8n baseline CPU (`best.pt` M5) |
| Preprocessing default | `original_crop` (konservatif), varian lain dapat dipilih |
| OCR | EasyOCR baseline + normalisasi ringan non-spekulatif |
| UI | Streamlit white modern; landing + recognition + configuration dialog |
| Config UI | Dialog di halaman Plate Recognition; session state; save/reset/cancel |
| Evaluasi OCR kuantitatif | Ditunda sampai GT terverifikasi tersedia |
| Deployment publik/real-time | Di luar ruang lingkup |

## 11. Batasan & Risiko

- Tidak ada GT OCR terverifikasi → tidak ada klaim CER/WER/EMA.
- Kasus no-detection diketahui masih ada pada sebagian sampel val.
- Duplikat intra-split train dan label kosong provisional dapat memengaruhi interpretasi.
- Performa CPU-only; latency bervariasi tergantung perangkat dan ukuran gambar.
- Hasil OCR sensitif terhadap kualitas crop/preprocessing/foto.

## 12. Dokumentasi Terkait

- [`README.md`](README.md) — cara instalasi, run UI/API, testing, evaluasi.
- [`perencanaan.md`](perencanaan.md) — status milestone/fase dan realisasi.
- [`struktur.md`](struktur.md) — struktur repo dan tanggung jawab modul.
- [`data/README.md`](data/README.md) — detail dataset/audit/kebijakan.
- [`docs/m9_smoke_ui_checklist.md`](docs/m9_smoke_ui_checklist.md) — smoke UI manual.
- [`docs/m9_gt_ocr_verification_guide.md`](docs/m9_gt_ocr_verification_guide.md) — verifikasi GT OCR.
