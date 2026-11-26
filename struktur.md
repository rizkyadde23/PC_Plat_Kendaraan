# Struktur Folder Proyek ALPRO

Dokumen ini menjelaskan susunan direktori proyek **ALPRO — Automatic License Plate Recognition Online**. Dokumen acuan utama proyek (`konteks.md`, `perencanaan.md`, `struktur.md`) disimpan di root repository.

## 1. Struktur Direktori (Aktual, Final Teknis)

```text
ALPRO/
├── app.py                          # Entry point Streamlit: Landing/Home page
├── pages/
│   └── 1_Plate_Recognition.py      # Halaman Plate Recognition
├── ui/
│   ├── __init__.py
│   ├── components.py               # Komponen UI reusable (result panel, cards, helpers)
│   ├── dialogs.py                  # Configuration dialog (@st.dialog)
│   ├── state.py                    # Session state & config defaults
│   └── styles.py                   # CSS white modern theme
├── configs/
│   ├── app.yaml
│   ├── dataset.yaml
│   ├── detector.yaml
│   ├── ocr.yaml
│   ├── pipeline.yaml
│   └── preprocessing.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── indonesian_license_plate/   # Dataset mentah (JANGAN diubah)
│   ├── interim/
│   ├── processed/
│   ├── annotations/
│   │   ├── detection/
│   │   └── ocr/                        # Template, kandidat, panduan GT OCR
│   └── splits/                         # Manifest turunan
├── src/alpro/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py             # load_yaml, resolve_path, deep_merge, load_pipeline_config
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # PlateRecognitionPipeline (end-to-end)
│   │   └── result.py               # PipelineResult dataclass
│   ├── detection/
│   │   ├── __init__.py
│   │   └── plate_detector.py       # YOLO wrapper
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_loader.py
│   │   ├── plate_cropper.py
│   │   └── image_preprocessor.py
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── ocr_engine.py
│   │   └── text_normalizer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── detection_metrics.py
│   │   ├── ocr_metrics.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py
│       └── logger.py
├── scripts/
│   ├── audit_dataset.py
│   ├── evaluate_detector.py
│   ├── evaluate_ocr.py
│   ├── prepare_dataset.py
│   ├── run_final_evaluation.py
│   ├── run_preprocessing_experiment.py
│   └── train_detector.py
├── tests/
│   ├── unit/
│   └── integration/
├── outputs/
│   ├── predictions/
│   ├── visualizations/
│   └── reports/
├── docs/
│   ├── m9_gt_ocr_verification_guide.md
│   └── m9_smoke_ui_checklist.md
├── models/
│   ├── pretrained/
│   └── trained/
├── runs/                           # Artefak training/evaluasi Ultralytics (lokal)
├── konteks.md
├── perencanaan.md
├── struktur.md
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

> Catatan: `runs/`, `outputs/`, `models/`, dan data turunan dapat berisi artefak lokal besar. Struktur di atas menekankan file/folder yang relevan untuk pengembangan dan reproduksi.

## 2. Penjelasan File Root

| File | Tanggung jawab |
|---|---|
| `app.py` | Entry point Streamlit; menampilkan Landing/Home page ALPRO |
| `pages/1_Plate_Recognition.py` | Halaman utama fitur pembacaan plat nomor |
| `README.md` | Panduan instalasi, penggunaan UI/API, evaluasi, batasan |
| `requirements.txt` | Dependency Python (strategi versioning hybrid) |
| `.env.example` | Contoh variabel environment |
| `.gitignore` | Mengabaikan env, cache, data mentah/turunan besar, artefak lokal |
| `konteks.md` | Konteks, tujuan, ruang lingkup, keputusan |
| `perencanaan.md` | Roadmap, status milestone/fase, realisasi |
| `struktur.md` | Dokumentasi struktur direktori dan tanggung jawab modul |

## 3. Penjelasan Direktori Utama

| Direktori | Tanggung jawab |
|---|---|
| `pages/` | Halaman Streamlit multipage (Plate Recognition) |
| `ui/` | Komponen presentasi, styling, dialog config, session state |
| `configs/` | Konfigurasi app/dataset/detector/preprocessing/ocr/pipeline |
| `data/raw/` | Dataset mentah Roboflow; **read-only** |
| `data/annotations/ocr/` | Template/kandidat/GT OCR manual terverifikasi |
| `data/splits/` | Manifest turunan persiapan data |
| `src/alpro/config/` | Load/resolve/merge config |
| `src/alpro/core/` | Orkestrasi pipeline dan kontrak hasil |
| `src/alpro/detection/` | Deteksi plat YOLO |
| `src/alpro/preprocessing/` | Load/crop/preprocess gambar |
| `src/alpro/recognition/` | OCR dan normalisasi teks |
| `src/alpro/evaluation/` | Metrik deteksi/OCR dan evaluator |
| `src/alpro/utils/` | Utilitas umum |
| `scripts/` | Script CLI untuk audit, training, evaluasi, eksperimen |
| `tests/` | Unit test dan integration test |
| `outputs/` | Prediksi, visualisasi, laporan eksperimen/final |
| `docs/` | Checklist smoke UI dan panduan GT OCR |

## 4. Alur Dependensi

```text
app.py (Landing/Home)
  └── ui/* (styles/state/components)

pages/1_Plate_Recognition.py
  └── ui/* (styles/state/dialogs/components)
       └── src/alpro/core/pipeline.py
            ├── src/alpro/config/settings.py
            ├── src/alpro/detection/plate_detector.py
            ├── src/alpro/preprocessing/image_loader.py
            ├── src/alpro/preprocessing/plate_cropper.py
            ├── src/alpro/preprocessing/image_preprocessor.py
            ├── src/alpro/recognition/ocr_engine.py
            └── src/alpro/recognition/text_normalizer.py

scripts/*
  └── src/alpro/* / configs/* / outputs/*
```

Prinsip:
- UI memanggil pipeline; UI tidak mengimplementasikan algoritma deteksi/OCR inti.
- Pipeline mengorkestrasi modul dan menangani status/error.
- Modul core/detection/preprocessing/recognition dapat diuji tanpa Streamlit.

## 5. Konvensi Penting

- **`data/raw/` tidak boleh diubah**. Simpan turunan di `data/interim/`, `data/processed/`, `data/annotations/`, `data/splits/`, atau `outputs/`.
- Simpan artefak eksperimen dengan ID/timestamp agar dapat ditelusuri.
- Jangan klaim metrik OCR kuantitatif (CER/WER/Exact Match) tanpa GT teks terverifikasi.
- Confidence OCR adalah confidence model, bukan ground-truth accuracy.
- Rahasiakan `.env`/secret; jangan commit data/model besar yang tidak diperlukan.

## 6. Catatan Evolusi Struktur

Struktur awal di dokumen perencanaan berevolusi selama implementasi. Perubahan penting:
- `app.py` menjadi landing page; halaman recognition dipindah ke `pages/`.
- `ui/` diperluas dengan `styles.py`, `state.py`, `dialogs.py` untuk desain white modern dan config dialog.
- `scripts/run_final_evaluation.py` ditambahkan untuk konsolidasi M9.
- Artefak training Ultralytics sebagian berada di `runs/detect/...` (sesuai perilaku runtime), sementara ringkasan/report dikonsolidasikan ke `outputs/`.

Dokumen ini harus diperbarui jika struktur atau tanggung jawab modul berubah.
