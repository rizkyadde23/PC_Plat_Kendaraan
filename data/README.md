# Dokumentasi Dataset — ALPRO

Dokumen ini mencatat sumber, lisensi, metadata, hasil audit lokal, dan kebijakan pengelolaan dataset untuk proyek **ALPRO (Automatic License Plate Recognition Online)**.

## 1. Sumber Dataset

- **Nama:** Indonesian_License_Plate (Roboflow Universe)
- **Project:** `dataset-pyufc/indonesian_license_plate`
- **Version:** 1
- **Workspace:** `dataset-pyufc`
- **URL project:** https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate
- **URL version (v1):** https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate/dataset/1
- **Lisensi:** **CC BY 4.0** (sesuai metadata sumber). Atribusi sumber dan ketentuan lisensi harus dipertahankan saat penggunaan/distribusi hasil turunan.
- **Metadata ekspor (dari README dataset lokal):**
  - `README.dataset.txt`: "Indonesian_License_Plate > 2025-05-15 3:36pm"
  - `README.roboflow.txt`: "exported via roboflow.com on May 16, 2025 at 7:39 AM GMT", anotasi **YOLOv8 format**

> Catatan: halaman sumber dapat menampilkan angka yang berbeda antar bagian; **jumlah dan kondisi yang digunakan proyek mengacu pada hasil audit lokal** terhadap dataset yang tersedia di repository ini.

## 2. Lokasi & Kebijakan Data

- **Lokasi dataset mentah (lokal):** `data/raw/indonesian_license_plate/`
- **Status:** dataset sudah tersedia dalam kondisi terekstrak (tidak diunduh/ekstrak ulang oleh proses pengembangan).
- **Kebijakan:**
  - **Jangan** memindahkan, mengubah, menimpa, atau menghapus file di `data/raw/indonesian_license_plate/`.
  - Hasil konversi, pembersihan, subset, anotasi turunan, atau preprocessing disimpan di direktori terpisah: `data/interim/`, `data/processed/`, `data/annotations/`, `data/splits/`.
  - Dataset mentah dan data turunan berukuran besar **tidak dimasukkan ke Git** (lihat `.gitignore`).

## 3. Struktur Aktual

Direktori utama:
- `train/images/`, `train/labels/`
- `valid/images/`, `valid/labels/`
- `test/images/`, `test/labels/`
- `data.yaml`
- `README.dataset.txt`, `README.roboflow.txt`

Konfigurasi `data.yaml` (Roboflow):
- `train: ../train/images`, `val: ../valid/images`, `test: ../test/images` (path relatif bawaan ekspor)
- `nc: 1`
- `names: ['License-Plate']`

**Catatan kompatibilitas (audit formal):** pada pemeriksaan resolusi path, entri `train/val/test` di `data.yaml` **tidak resolve** dari lokasi `data.yaml` maupun dari root dataset dengan interpretasi path mentah sebagaimana tertulis; struktur aktual dataset berada di `data/raw/indonesian_license_plate/{train,valid,test}/images`. Saat training/inferensi Ultralytics, pastikan resolusi path yang digunakan benar (mis. jalankan dari direktori yang sesuai atau gunakan konfigurasi/path kerja turunan) **tanpa mengubah file di `data/raw/`**.

## 4. Jumlah Data Aktual (audit formal)

| Split | Gambar | Label | Ekstensi |
|---|---:|---:|---|
| train | 1085 | 1085 | `.jpg` / `.txt` |
| valid | 107 | 107 | `.jpg` / `.txt` |
| test | 53 | 53 | `.jpg` / `.txt` |
| **Total** | **1245** | **1245** | |

## 5. Format Anotasi

- **Tipe tugas:** Object Detection (plat nomor kendaraan)
- **Format label:** YOLO (`class_id x_center y_center width height`, nilai ternormalisasi)
- **Jumlah kelas:** 1 (`License-Plate`, class id `0`)
- **Distribusi objek per gambar (audit formal):**
  - train: {1: 861, 2: 75, 3: 50, 4: 53, 5: 24, 6: 12, 7: 3, 8: 3, 9: 2, 0: 2} (total 1628 objek)
  - valid: {1: 85, 2: 6, 3: 6, 4: 7, 7: 1, 9: 1, 0: 1} (total 159 objek)
  - test: {1: 48, 2: 1, 3: 2, 6: 1, 0: 1} (total 62 objek)

## 6. Preprocessing & Augmentasi (sesuai metadata ekspor Roboflow)

Diterapkan pada versi dataset yang diekspor (bukan data mentah asli):
- **Preprocessing:** auto-orientation (EXIF-orientation stripped), resize stretch ke **640×640**, grayscale (CRT phosphor)
- **Augmentasi:** pembuatan **3 versi per source image** dengan random Gaussian blur **0–2.5 px**

**Catatan audit formal (properti gambar teramati):** seluruh gambar dapat didecode, berukuran **640×640** dan terbaca sebagai **3 channel** (BGR/3ch) oleh OpenCV, meskipun metadata ekspor menyebutkan grayscale. Hal ini perlu diperhatikan saat merancang pipeline preprocessing.

## 7. Temuan Audit Formal (Milestone 3)

- **Run audit:** `20261118_210535` (script: `scripts/audit_dataset.py`)
- **Laporan:** `outputs/reports/dataset_audit/20261118_210535/` (`report.md`, `report.json`, `issues.csv`, `duplicates.csv`, `duplicates.json`)
- **Visualisasi sampel:** `outputs/visualizations/dataset_audit/20261118_210535/`
- **Status:** **`READY_WITH_WARNINGS`** (ERROR=0, WARNING=381, INFO=1)

### 7.1 Konsistensi & validitas
- **Pairing image-label:** tidak ada image tanpa label dan tidak ada label tanpa image (semua split).
- **Keterbacaan gambar:** 0 gambar corrupt/unreadable (full decode seluruh 1245 file).
- **Validasi label:** 0 file label invalid (format/kolom/class/bbox dalam rentang valid). Statistik bbox tercatat dalam laporan JSON.
- **Label kosong (4 file, WARNING — perlu verifikasi visual):**
  - `train/labels/helm_045_jpg.rf.3eafc00e8cc8ac1e40858a7df61f2429.txt`
  - `train/labels/helm_045_jpg.rf.c31448415f7ac963a7423202c75e9feb.txt`
  - `valid/labels/99_E-2984-TW-04-18_jpg.rf.b1a1a2d9ea7528f302fe97c14a4f1091.txt`
  - `test/labels/helm_047_jpg.rf.fe6ed6f023c55690a6cd60da208be8b6.txt`

### 7.2 Duplikasi (MD5, byte-identik)
- **Total grup duplikat:** 374
- **Duplikat lintas-split:** **0 grup** (tidak terdeteksi) — tidak ada indikasi leakage antar split berdasarkan hash identik.
- **Duplikat intra-split:** **374 grup** (WARNING, over-representasi), melibatkan **1085 file**.
- Detail lengkap anggota grup tersedia pada `duplicates.csv`/`duplicates.json` di folder laporan run.

### 7.3 Kompatibilitas & metadata
- **`data.yaml`:** terbaca (nc=1, names=['License-Plate']), namun terdapat **3 warning** terkait resolusi path relatif bawaan (`train/val/test`) terhadap lokasi yaml/root dataset (lihat §3).
- **Ground truth teks OCR:** **tidak tersedia**. Label hanya berupa bounding box deteksi; evaluasi OCR kuantitatif (CER/WER/Exact Match) memerlukan anotasi teks terpisah pada subset tervalidasi (disimpan di `data/annotations/ocr/`, bukan di `data/raw/`).

## 8. Persiapan Dataset (Milestone 4)

- **Run persiapan:** `20261118_212026` (`scripts/prepare_dataset.py`)
- **Laporan:** `outputs/reports/dataset_preparation/20261118_212026/` (`report.md`, `report.json`)
- **Config YOLO kerja (turunan):** `configs/dataset.yaml` (menunjuk ke `data/raw/indonesian_license_plate/{train,valid,test}/images`; `data/raw` tidak diubah)
- **Manifest turunan:** `data/splits/` (`*_manifest.json/csv`, `preparation_manifest_summary.json`)

### Keputusan persiapan yang diterapkan
- **6.1A (provisional):** 4 label kosong **dipertahankan** sebagai kandidat background/no-object dalam manifest, **wajib verifikasi visual** sebelum training.
- **6.2A:** semua sampel **dipertahankan**; duplikat **intra-split** didokumentasikan (374 grup, 1085 file; 0 lintas-split per audit M3).
- **6.3C/6.4A:** config kerja + manifest turunan tanpa duplikasi file gambar.
- **6.5A(ii),C(i):** kerangka OCR dan **50 kandidat anotasi** (seed 42, proporsional {'train': 44, 'valid': 4, 'test': 2}) disiapkan di `data/annotations/ocr/` **tanpa** `plate_text_gt` otomatis.

### Status kesiapan pasca-persiapan
- **Deteksi (YOLO):** siap untuk baseline menggunakan `configs/dataset.yaml` dan manifest turunan (split bawaan dipertahankan, tidak ada indikasi leakage lintas-split). Perhatikan verifikasi visual label kosong dan over-representasi duplikat intra-split.
- **Evaluasi OCR:** **tertunda** hingga anotasi manual ground truth (`plate_text_gt`, terverifikasi) tersedia di `data/annotations/ocr/`.

## 9. Rekomendasi Tindak Lanjut
- Lakukan **verifikasi visual** 4 label kosong (rujuk visualisasi audit) dan perbarui manifest turunan jika keputusan per-file berubah (tanpa ubah `raw/`).
- Pertimbangkan strategi penanganan **duplikat intra-split** pada eksperimen training bila diperlukan (melalui data/manifest turunan).
- Pastikan konfigurasi training Ultralytics menggunakan path/config kerja yang resolve benar dari environment eksekusi.
- Lakukan **anotasi manual** subset OCR (mulai dari `data/annotations/ocr/ocr_annotation_candidates.csv`) dengan QA (`verified`) untuk evaluasi CER/WER/EMA.
- Pertahankan split bawaan kecuali ada alasan teknis jelas dan terdokumentasi.

## 10. Atribusi

Dataset: *Indonesian_License_Plate* oleh kontributor Roboflow Universe (`dataset-pyufc`), dilisensikan **CC BY 4.0**.  
Sumber: https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate

