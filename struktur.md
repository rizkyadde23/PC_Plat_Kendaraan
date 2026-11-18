# Struktur Folder Proyek ALPRO

Dokumen ini menjelaskan susunan direktori proyek **ALPRO — Automatic License Plate Recognition Online**. Seluruh dokumentasi proyek (`konteks.md`, `perencanaan.md`, dan `struktur.md`) disimpan langsung di root repository.

## 1. Struktur direktori

```text
ALPRO/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── konteks.md
├── perencanaan.md
├── struktur.md
│
├── configs/
│   ├── app.yaml
│   ├── detector.yaml
│   └── preprocessing.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── indonesian_license_plate/
│   │       └── (hasil ekstraksi ZIP Roboflow)
│   ├── interim/
│   ├── processed/
│   ├── annotations/
│   │   ├── detection/
│   │   └── ocr/
│   └── splits/
│
├── models/
│   ├── pretrained/
│   └── trained/
│
├── src/
│   └── alpro/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   └── result.py
│       │
│       ├── detection/
│       │   ├── __init__.py
│       │   └── plate_detector.py
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── image_loader.py
│       │   ├── plate_cropper.py
│       │   └── image_preprocessor.py
│       │
│       ├── recognition/
│       │   ├── __init__.py
│       │   ├── ocr_engine.py
│       │   └── text_normalizer.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── detection_metrics.py
│       │   ├── ocr_metrics.py
│       │   └── evaluator.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── image_utils.py
│           └── logger.py
│
├── ui/
│   ├── __init__.py
│   ├── home_page.py
│   └── evaluation_page.py
│
├── scripts/
│   ├── audit_dataset.py
│   ├── visualize_annotations.py
│   ├── train_detector.py
│   ├── evaluate_detector.py
│   └── evaluate_ocr.py
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_text_normalizer.py
│   │   └── test_metrics.py
│   └── integration/
│       └── test_pipeline.py
│
└── outputs/
    ├── predictions/
    ├── visualizations/
    └── reports/
```

> Nama folder hasil ekstraksi di dalam `data/raw/indonesian_license_plate/` dapat berbeda mengikuti nama ZIP Roboflow. Sesuaikan path konfigurasi dengan struktur aktual setelah ekstraksi; jangan mengubah nama file dataset tanpa kebutuhan.

## 2. Penjelasan file root

| File               | Tanggung jawab                                                       |
| ------------------ | -------------------------------------------------------------------- |
| `app.py`           | Entry point untuk menjalankan aplikasi Streamlit                     |
| `README.md`        | Panduan ringkas instalasi, konfigurasi, dan penggunaan               |
| `requirements.txt` | Daftar dependency Python                                             |
| `.gitignore`       | Mengabaikan environment, cache, dataset besar, bobot, dan file lokal |
| `.env.example`     | Contoh variabel lingkungan tanpa menyimpan secret                    |
| `konteks.md`       | Konteks, tujuan, ruang lingkup, dan keputusan proyek                 |
| `perencanaan.md`   | Fase, milestone, aktivitas, penjelasan, dan keluaran                 |
| `struktur.md`      | Dokumentasi struktur direktori dan tanggung jawab modul              |

Ketiga file dokumentasi proyek berada di root repository agar mudah ditemukan oleh seluruh anggota tim.

## 3. Penjelasan direktori utama

| Direktori                     | Tanggung jawab                                                        |
| ----------------------------- | --------------------------------------------------------------------- |
| `configs/`                    | Menyimpan konfigurasi aplikasi, model, dan preprocessing              |
| `data/raw/`                   | Menyimpan hasil ekstraksi dataset Roboflow tanpa perubahan            |
| `data/interim/`               | Menyimpan data sementara selama proses konversi/pemeriksaan           |
| `data/processed/`             | Menyimpan data yang telah diproses dan siap untuk eksperimen tertentu |
| `data/annotations/detection/` | Menyimpan atau mencatat anotasi deteksi                               |
| `data/annotations/ocr/`       | Menyimpan ground truth teks plat untuk evaluasi OCR                   |
| `data/splits/`                | Menyimpan daftar pembagian data jika diperlukan                       |
| `models/pretrained/`          | Menyimpan bobot awal/pretrained yang digunakan                        |
| `models/trained/`             | Menyimpan bobot hasil training proyek                                 |
| `src/alpro/`                  | Kode inti aplikasi yang dipisahkan berdasarkan tanggung jawab         |
| `ui/`                         | Halaman dan komponen antarmuka                                        |
| `scripts/`                    | Script audit data, visualisasi anotasi, training, dan evaluasi        |
| `tests/`                      | Unit test dan integration test                                        |
| `outputs/`                    | Prediksi, visualisasi, dan laporan eksperimen                         |

## 4. Tanggung jawab modul `src/alpro/`

### `config/`

Memuat konfigurasi dan parameter aplikasi. Path dataset, path model, confidence threshold, dan parameter preprocessing sebaiknya diatur melalui konfigurasi, bukan ditulis berulang kali di berbagai modul.

### `core/`

Mengatur alur pemrosesan utama.

- `pipeline.py`: mengorkestrasi validasi gambar, deteksi, cropping, preprocessing, OCR, dan penyusunan hasil.
- `result.py`: mendefinisikan struktur hasil yang konsisten, seperti status, bounding box, teks, confidence, durasi, dan pesan error.

### `detection/`

Berisi implementasi deteksi plat.

- `plate_detector.py`: memuat model YOLO dan menghasilkan koordinat bounding box, confidence, serta kelas.

Modul ini tidak bertanggung jawab menampilkan komponen UI.

### `preprocessing/`

- `image_loader.py`: membaca dan memvalidasi file gambar.
- `plate_cropper.py`: memotong area plat berdasarkan bounding box.
- `image_preprocessor.py`: menjalankan grayscale, resize, denoising, contrast enhancement, thresholding, dan morphology sesuai konfigurasi eksperimen.

### `recognition/`

- `ocr_engine.py`: adapter untuk menjalankan OCR, dengan EasyOCR sebagai baseline awal.
- `text_normalizer.py`: normalisasi hasil teks. Simpan hasil OCR mentah agar prediksi asli tetap dapat diperiksa.

### `evaluation/`

Menghitung metrik deteksi dan OCR. Metrik OCR membutuhkan ground truth teks; anotasi bounding box saja tidak cukup.

### `utils/`

Fungsi pendukung seperti visualisasi gambar dan logging. Hindari menempatkan seluruh logika aplikasi di direktori ini.

## 5. Tanggung jawab direktori lain

### `ui/`

Berisi halaman Streamlit. UI bertugas menerima input pengguna dan menyajikan hasil pipeline, bukan menjalankan detail algoritma deteksi atau OCR secara langsung.

### `scripts/`

Script yang dijalankan untuk pekerjaan terpisah dari aplikasi:

- `audit_dataset.py`: memeriksa file gambar, label, split, dan konfigurasi dataset.
- `visualize_annotations.py`: menggambar bounding box untuk pemeriksaan anotasi.
- `train_detector.py`: menjalankan eksperimen training detektor.
- `evaluate_detector.py`: mengevaluasi deteksi.
- `evaluate_ocr.py`: mengevaluasi OCR dengan ground truth teks.

### `tests/`

- `unit/`: menguji fungsi kecil secara terpisah.
- `integration/`: menguji hubungan antarmodul dan pipeline.

### `outputs/`

Menyimpan artefak eksperimen. Atur penamaan berdasarkan tanggal atau ID eksperimen agar hasil tidak tertimpa dan dapat dibandingkan.

## 6. Alur dependensi

```text
app.py
  └── ui/
       └── core/pipeline.py
             ├── detection/plate_detector.py
             ├── preprocessing/image_loader.py
             ├── preprocessing/plate_cropper.py
             ├── preprocessing/image_preprocessor.py
             └── recognition/ocr_engine.py
                    └── recognition/text_normalizer.py

scripts/
  ├── audit_dataset.py
  ├── visualize_annotations.py
  ├── train_detector.py
  └── evaluate_*.py
       └── evaluation/
```

Prinsip dependensi:

- UI memanggil pipeline.
- Pipeline mengatur urutan proses.
- Modul deteksi, preprocessing, dan OCR tidak bergantung pada UI.
- Evaluasi dapat dijalankan melalui script atau halaman evaluasi.
- Modul inti harus dapat diuji tanpa menjalankan Streamlit.

## 7. Kebijakan dataset dan model

1. Dataset Roboflow disimpan di `data/raw/indonesian_license_plate/` setelah diekstrak.
2. Pertahankan file asli dan anotasi sebagaimana diunduh.
3. Jangan menyimpan ZIP dan seluruh dataset besar ke Git.
4. Catat URL sumber, versi dataset, tanggal unduh, lisensi, jumlah file aktual, serta konfigurasi preprocessing.
5. Simpan hasil modifikasi atau konversi di `interim/` atau `processed/`, bukan menimpa `raw/`.
6. Bobot model besar disimpan di direktori `models/` dan tidak perlu di-commit kecuali repository memang disiapkan untuk itu.
7. Jangan memasukkan data pribadi, API key, atau secret ke repository publik.

Contoh `.gitignore`:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.env
.streamlit/secrets.toml

data/raw/*
data/interim/*
data/processed/*
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep

models/pretrained/*
models/trained/*
!models/pretrained/.gitkeep
!models/trained/.gitkeep

outputs/predictions/*
outputs/visualizations/*
outputs/reports/*
```

Sesuaikan `.gitignore` jika tim perlu menyimpan contoh data kecil yang memang boleh dibagikan. Dokumentasi dataset tetap disimpan di `data/README.md`, bukan file dataset itu sendiri.

## 8. Urutan pembuatan struktur

1. Buat root repository `ALPRO/`.
2. Letakkan `konteks.md`, `perencanaan.md`, dan `struktur.md` langsung di root.
3. Buat environment, `requirements.txt`, `.gitignore`, dan README.
4. Ekstrak ZIP Roboflow ke `data/raw/indonesian_license_plate/`.
5. Buat script audit sebelum memulai training.
6. Buat modul `src/alpro/` sesuai fase implementasi.
7. Buat `tests/` bersamaan dengan pengembangan modul.
8. Buat `ui/` setelah pipeline inti berjalan.
9. Gunakan `outputs/` untuk artefak hasil eksperimen.

Struktur ini adalah rancangan yang dapat berkembang. Penambahan atau pengubahan file diperbolehkan selama tanggung jawab modul tetap jelas dan dokumentasi diperbarui.
