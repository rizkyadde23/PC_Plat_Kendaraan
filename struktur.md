# Struktur Folder Proyek ALPR

Dokumen ini menjelaskan struktur direktori yang direncanakan untuk aplikasi pengenalan plat nomor kendaraan berbasis Python.

## 1. Struktur direktori

```text
alpr-project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── configs/
│   ├── app.yaml
│   ├── detector.yaml
│   └── preprocessing.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── annotations/
│   │   ├── detection/
│   │   └── ocr/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── models/
│   ├── pretrained/
│   │   └── .gitkeep
│   └── trained/
│       └── .gitkeep
│
├── src/
│   └── alpr/
│       ├── __init__.py
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
│   ├── split_dataset.py
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
├── outputs/
│   ├── predictions/
│   ├── visualizations/
│   └── reports/
│
└── docs/
    ├── konteks.md
    ├── perencanaan.md
    ├── struktur.md
    ├── diagrams/
    └── references/
```

## 2. Penjelasan direktori utama

| Direktori/file | Tanggung jawab |
|---|---|
| `app.py` | Entry point untuk menjalankan aplikasi Streamlit |
| `requirements.txt` | Daftar dependency Python dan versinya |
| `README.md` | Deskripsi, instalasi, cara menjalankan, dan penggunaan aplikasi |
| `configs/` | Parameter aplikasi, model deteksi, dan preprocessing |
| `data/raw/` | Dataset asli hasil unduhan/pengumpulan; tidak diedit |
| `data/interim/` | Data sementara setelah ekstraksi atau konversi awal |
| `data/processed/` | Dataset yang telah divalidasi dan siap digunakan |
| `data/annotations/` | Label bounding box dan transkripsi teks OCR |
| `data/splits/` | Daftar data train, validation, dan test |
| `models/pretrained/` | Bobot model awal/publik yang digunakan |
| `models/trained/` | Bobot hasil pelatihan proyek |
| `src/alpr/` | Kode inti aplikasi yang dipisahkan berdasarkan tanggung jawab |
| `ui/` | Halaman dan komponen antarmuka Streamlit |
| `scripts/` | Script untuk audit data, pembagian data, training, dan evaluasi |
| `tests/` | Unit test dan integration test |
| `outputs/` | Hasil prediksi, visualisasi, dan laporan eksperimen |
| `docs/` | Dokumentasi perencanaan, konteks, struktur, diagram, dan referensi |

## 3. Tanggung jawab modul pada `src/alpr/`

### `config/`
Membaca konfigurasi aplikasi dan parameter. Hindari menulis path model, threshold, atau nilai konfigurasi berulang kali di banyak modul.

### `core/`
Mengatur alur utama aplikasi. `pipeline.py` memanggil detector, preprocessing, dan OCR sesuai urutan. `result.py` mendefinisikan bentuk hasil yang konsisten, misalnya status, bounding box, teks OCR, confidence, durasi, dan pesan error.

### `detection/`
Berisi implementasi deteksi plat. `plate_detector.py` bertanggung jawab memuat model dan mengembalikan koordinat bounding box serta confidence. Modul ini tidak mengurus tampilan UI.

### `preprocessing/`
- `image_loader.py`: validasi dan pembacaan gambar.
- `plate_cropper.py`: pemotongan area plat berdasarkan bounding box.
- `image_preprocessor.py`: operasi grayscale, resize, denoising, contrast enhancement, thresholding, dan morphology.

### `recognition/`
- `ocr_engine.py`: adapter untuk menjalankan OCR.
- `text_normalizer.py`: normalisasi hasil OCR tanpa menghilangkan prediksi mentah.

### `evaluation/`
Menghitung metrik deteksi dan OCR serta menyusun ringkasan evaluasi. Metrik harus menggunakan ground truth yang sesuai.

### `utils/`
Fungsi pendukung umum seperti visualisasi gambar dan logging. Jangan menjadikan direktori ini tempat menumpuk seluruh logika aplikasi.

## 4. Alur dependensi yang disarankan

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
  ├── train_detector.py
  └── evaluate_*.py
       └── evaluation/
```

Prinsip desain:
- UI hanya menangani interaksi dan presentasi.
- Pipeline mengorkestrasi proses, bukan mengimplementasikan seluruh algoritma.
- Setiap modul memiliki tanggung jawab spesifik.
- Modul inti dapat diuji tanpa menjalankan Streamlit.
- Konfigurasi dan path tidak di-hardcode di banyak tempat.
- Gunakan type hints, docstring, logging, dan penanganan error.
- Simpan hasil eksperimen agar proses dapat direproduksi.

## 5. Kebijakan penyimpanan data dan model

- Jangan memasukkan dataset besar ke Git.
- Jangan commit file rahasia, token, atau kredensial.
- Jangan commit bobot model besar kecuali repository memang disiapkan untuk itu.
- Catat sumber, versi, tanggal unduh, lisensi, dan checksum dataset jika memungkinkan.
- Pertahankan dataset asli di `data/raw/`; lakukan pemrosesan ke direktori lain.
- Pastikan `.gitignore` mengecualikan isi dataset, model besar, cache, environment, dan output sementara.

Contoh `.gitignore`:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.streamlit/secrets.toml
.env

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

outputs/*
!outputs/predictions/
!outputs/visualizations/
!outputs/reports/
```

Sesuaikan aturan ignore jika tim perlu menyimpan artefak kecil atau contoh data yang memang boleh dibagikan.

## 6. Urutan pembuatan folder

Buat struktur secara bertahap agar tidak terlalu banyak direktori kosong:

1. Buat root project, virtual environment, `requirements.txt`, dan `README.md`.
2. Buat `docs/` dan lengkapi konteks serta perencanaan.
3. Buat `data/` setelah sumber dataset dipilih.
4. Buat `src/alpr/` ketika mulai mengimplementasikan modul.
5. Buat `scripts/` untuk pekerjaan dataset dan eksperimen.
6. Buat `ui/` setelah pipeline inti berjalan.
7. Buat `tests/` sejak modul pertama dikembangkan.
8. Buat direktori output untuk menyimpan artefak evaluasi.

Struktur ini merupakan rancangan awal. Nama file dapat disesuaikan selama pemisahan tanggung jawab dan dokumentasinya tetap terjaga.
