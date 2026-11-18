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

## 3. Struktur Aktual (hasil audit lokal)

Direktori utama:
- `train/images/`, `train/labels/`
- `valid/images/`, `valid/labels/`
- `test/images/`, `test/labels/`
- `data.yaml`
- `README.dataset.txt`, `README.roboflow.txt`

Konfigurasi `data.yaml` (Roboflow):
- `train: ../train/images`, `val: ../valid/images`, `test: ../test/images` (path relatif terhadap lokasi `data.yaml`)
- `nc: 1`
- `names: ['License-Plate']`

**Catatan kompatibilitas:** Struktur `images/labels` per split dan format label YOLO umumnya kompatibel dengan Ultralytics YOLO. Resolusi path `data.yaml` bawaan perlu diperhatikan saat training/inferensi (bergantung working directory / cara pemanggilan) tanpa mengubah file di `data/raw`.

## 4. Jumlah Data Aktual (audit lokal)

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
- **Contoh baris label:** `0 0.5 0.45625 0.978125 0.9125`

## 6. Preprocessing & Augmentasi (sesuai metadata ekspor Roboflow)

Diterapkan pada versi dataset yang diekspor (bukan data mentah asli):
- **Preprocessing:** auto-orientation (EXIF-orientation stripped), resize stretch ke **640×640**, grayscale (CRT phosphor)
- **Augmentasi:** pembuatan **3 versi per source image** dengan random Gaussian blur **0–2.5 px**

Implikasi: karakteristik data (grayscale, resolusi seragam, blur augmentasi) perlu dipertimbangkan dalam strategi preprocessing dan evaluasi model.

## 7. Temuan Audit Awal (Milestone 1, pemeriksaan read-only)

- **Konsistensi pasangan file:** tidak ditemukan image tanpa label atau label tanpa image (berdasarkan kecocokan basename per split).
- **Validasi format/nilai label dasar (scan teks):** tidak ditemukan jumlah kolom ≠ 5, `class_id != 0`, atau nilai bbox di luar rentang `(0, 1]`.
- **Label kosong (0 byte, 4 file):** perlu verifikasi visual (dalam YOLO dapat berarti no-object/background, namun perlu dikonfirmasi konteksnya):
  - `train/labels/helm_045_jpg.rf.3eafc00e8cc8ac1e40858a7df61f2429.txt`
  - `train/labels/helm_045_jpg.rf.c31448415f7ac963a7423202c75e9feb.txt`
  - `valid/labels/99_E-2984-TW-04-18_jpg.rf.b1a1a2d9ea7528f302fe97c14a4f1091.txt`
  - `test/labels/helm_047_jpg.rf.fe6ed6f023c55690a6cd60da208be8b6.txt`
- **Duplikasi gambar (hash MD5 identik):** terdeteksi **374 kelompok duplikat** pada pemeriksaan awal. Pola nama/varian konsisten dengan augmentasi 3-versi, namun **distribusi intra-split vs lintas-split dan dampaknya terhadap evaluasi/leakage belum dikonfirmasi sepenuhnya** dan perlu analisis lanjutan pada audit skrip.
- **Korupsi gambar:** belum diverifikasi menyeluruh melalui decode seluruh file; akan diperiksa pada audit dataset formal (Milestone 3).
- **Ground truth teks OCR:** **tidak tersedia** di dataset ini. Label yang ada adalah bounding box deteksi, bukan transkripsi karakter plat. Evaluasi OCR kuantitatif (CER/WER/Exact Match) memerlukan anotasi teks terpisah pada subset yang tervalidasi (disimpan di `data/annotations/ocr/`, bukan di `data/raw/`).

## 8. Kesiapan & Tindak Lanjut

- **Status awal:** dataset **ditemukan** dan format dasarnya **kompatibel** untuk deteksi YOLO (1 kelas), namun **belum dinyatakan final siap latih** sebelum audit formal meliputi verifikasi visual bbox/label kosong, analisis duplikat lintas-split, uji baca gambar, dan penegasan strategi penggunaan `data.yaml`.
- **Rencana audit formal:** `scripts/audit_dataset.py` (Milestone 3) akan menghasilkan laporan terperinci ke `outputs/reports/` tanpa memodifikasi `data/raw/`.

## 9. Atribusi

Dataset: *Indonesian_License_Plate* oleh kontributor Roboflow Universe (`dataset-pyufc`), dilisensikan **CC BY 4.0**.  
Sumber: https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate
