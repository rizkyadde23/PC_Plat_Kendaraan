# Perencanaan Pengembangan ALPRO

> **Nama proyek:** ALPRO — Automatic License Plate Recognition Online  
> **Mata kuliah:** Pengolahan Citra Digital  
> **Bahasa pemrograman:** Python  
> **Fokus:** Pengembangan aplikasi perangkat lunak  
> **Status:** Milestone 1–5 selesai (analisis, setup, audit dataset, persiapan data, baseline deteksi YOLO). Milestone 6+ (crop/preprocessing, OCR, integrasi pipeline, UI, evaluasi menyeluruh, finalisasi) belum dimulai. Pembaruan: Nov 2026.

Dokumen ini menjadi roadmap pelaksanaan proyek. Pengembangan dibagi menjadi fase/milestone agar setiap tahap mempunyai tujuan, aktivitas, penjelasan, dan keluaran yang dapat diperiksa sebelum melanjutkan ke tahap berikutnya.

## Status pelaksanaan (ringkasan)

| Milestone kerja | Fase rujukan | Status | Realisasi utama |
| --- | --- | --- | --- |
| M1 — Analisis & inisiasi | Fase 1 | **Selesai** | Tinjauan `konteks.md`, `perencanaan.md`, `struktur.md`, audit awal repo/dataset, identifikasi gap & risiko |
| M2 — Setup repository & environment | Fase 1–2 | **Selesai** | Struktur/scaffold, `.venv` (Python 3.13.3), `requirements.txt` (hybrid), `.env.example`, `configs/{app,detector,preprocessing}.yaml`, `data/README.md`, `.gitignore`, `README.md`; dependensi terpasang & import inti terverifikasi (CPU-first, `app.py` ditunda ke Milestone UI/Fase 9) |
| M3 — Audit dataset | Fase 3 | **Selesai** | `scripts/audit_dataset.py`; run `outputs/reports/dataset_audit/20261118_210535/` (**READY_WITH_WARNINGS**: 1245 img/lbl, 0 error struktur/pairing, 0 cross-split, 374 grup duplikat intra-split/1085 file, 4 label kosong, isu path `data.yaml` raw, tidak ada GT OCR) |
| M4 — Persiapan dataset | Fase 4 | **Selesai** | `scripts/prepare_dataset.py`; `configs/dataset.yaml` (kerja), manifest `data/splits/*`, kerangka OCR + 50 kandidat `data/annotations/ocr/` (GT kosong/manual), laporan `outputs/reports/dataset_preparation/20261118_212026/`, `data/README.md` diperbarui; `data/raw/` tidak diubah |
| M5 — Baseline deteksi YOLO | Fase 5 | **Selesai** | `scripts/train_detector.py`, `scripts/evaluate_detector.py`, `src/alpro/detection/plate_detector.py`, `configs/detector.yaml` (yolov8n, CPU); experiment `yolov8n_baseline_m5_20261118_cpu` (early stop epoch 40, best 20). Evaluasi eksplisit: **val** P=0.941 R=0.896 mAP@50=0.901 mAP@50-95=0.715; **test (sekali)** P=0.983 R=0.948 mAP@50=0.955 mAP@50-95=0.761. Laporan `outputs/reports/detection/.../report.md`, visualisasi `outputs/visualizations/detection/.../{val,test}/` |
| M6 — Crop & preprocessing | Fase 6 | Belum dimulai | - |
| M7 — OCR & normalisasi | Fase 7 | Belum dimulai | Bergantung ketersediaan/strategi GT teks untuk evaluasi kuantitatif |
| M8 — Integrasi pipeline & UI | Fase 8–9 | Belum dimulai | `app.py`/Streamlit sesuai keputusan M2 |
| M9 — Evaluasi & finalisasi | Fase 10–11 | Belum dimulai | Pengujian menyeluruh, laporan, artikel, demo |

## Gambaran umum proyek

ALPRO merupakan aplikasi pengenalan plat nomor kendaraan dari gambar. Sistem dirancang untuk mendeteksi area plat, memotong area tersebut, menerapkan prapemrosesan citra, lalu membaca karakter menggunakan OCR. Aplikasi akan menyediakan antarmuka agar pengguna dapat mengunggah gambar dan melihat hasil setiap tahap pemrosesan.

Pipeline utama:

```text
Gambar kendaraan
      |
      v
Validasi dan pembacaan gambar
      |
      v
Deteksi area plat (YOLO)
      |
      v
Cropping area plat
      |
      v
Prapemrosesan citra (OpenCV)
      |
      v
Pengenalan karakter (OCR)
      |
      v
Normalisasi dan penyajian hasil
      |
      v
Evaluasi dan dokumentasi
```

YOLO bertugas menemukan lokasi plat, sedangkan OCR bertugas membaca teks di dalam area plat. Keduanya merupakan komponen yang berbeda dan perlu dievaluasi secara terpisah.

---

## Fase 1 — Inisiasi proyek dan penetapan kebutuhan

**Tujuan:** Menetapkan arah pengembangan, kebutuhan aplikasi, dan batasan agar implementasi tidak melebar.

### Aktivitas

1. Meninjau kembali ketentuan tugas akhir Pengolahan Citra Digital.
2. Menetapkan nama proyek ALPRO — Automatic License Plate Recognition Online.
3. Menentukan pengguna sasaran dan skenario utama penggunaan.
4. Menetapkan input berupa gambar kendaraan dan output berupa hasil deteksi serta teks plat.
5. Menetapkan batasan versi awal: aplikasi memproses gambar statis, bukan video atau CCTV real-time.
6. Menentukan kebutuhan fungsional dan nonfungsional.
7. Menentukan pembagian tanggung jawab anggota kelompok.

### Penjelasan

Tahap ini memastikan aplikasi tidak hanya menjadi demonstrasi OCR, tetapi memperlihatkan rangkaian pengolahan citra yang diminta pada tugas. Pemrosesan inti juga direncanakan terpisah dari antarmuka agar mudah diuji dan dikembangkan.

### Keluaran fase

- `konteks.md` yang menjelaskan konteks dan keputusan proyek.
- Daftar kebutuhan fungsional dan nonfungsional.
- Ruang lingkup dan batasan proyek.
- Pembagian tugas awal anggota kelompok.

### Realisasi (aktual)

- **Selesai (M1):** analisis dokumen acuan (`konteks.md`, `perencanaan.md`, `struktur.md`), peninjauan kondisi repo/dataset, serta identifikasi gap, risiko, dan rekomendasi awal.
- Dokumen konteks, ruang lingkup/batasan, dan arah pipeline tersedia di repo; pembagian tugas anggota dapat diperinci/diperbarui terpisah sesuai kebutuhan kelompok.

### Kriteria selesai

Seluruh anggota memahami tujuan, alur sistem, fitur versi awal, dan hal-hal yang tidak termasuk ruang lingkup. *(Tujuan & ruang lingkup inti telah ditetapkan; rincian operasional tim dapat dilengkapi.)*

---

## Fase 2 — Persiapan repository dan lingkungan pengembangan

**Tujuan:** Menyiapkan fondasi teknis agar pengembangan konsisten dan dapat direproduksi.

### Aktivitas

1. Membuat repository Git dengan nama proyek `ALPRO`.
2. Membuat struktur direktori sesuai `struktur.md`.
3. Membuat virtual environment Python.
4. Menentukan versi Python yang kompatibel dengan dependency.
5. Memasang library secara bertahap.
6. Mencatat versi library dan spesifikasi perangkat pengembangan.
7. Membuat `requirements.txt`, `.gitignore`, dan README penggunaan awal.

### Kandidat teknologi

| Teknologi            | Peran                                        |
| -------------------- | -------------------------------------------- |
| Python               | Bahasa utama                                 |
| OpenCV               | Pembacaan, cropping, dan prapemrosesan citra |
| Ultralytics          | Training/inferensi model YOLO                |
| EasyOCR              | OCR baseline untuk membaca karakter          |
| Streamlit            | Antarmuka aplikasi                           |
| NumPy dan Pandas     | Pengolahan array dan data hasil              |
| Matplotlib           | Visualisasi hasil eksperimen                 |
| jiwer / scikit-learn | Metrik evaluasi yang sesuai                  |
| pytest               | Pengujian otomatis                           |

Versi dan kebutuhan perangkat keras belum dikunci. Kompatibilitas PyTorch, CUDA, dan sistem operasi harus diperiksa sebelum menetapkan dependency final.

### Penjelasan

Lingkungan virtual mencegah dependency proyek bercampur dengan instalasi Python global. Pencatatan versi mempermudah anggota lain menjalankan aplikasi pada perangkat berbeda.

### Keluaran fase

- Repository ALPRO.
- Struktur folder awal.
- Virtual environment aktif.
- `requirements.txt`.
- `.gitignore`.
- README berisi instruksi setup awal.

### Realisasi (aktual, M2)

- Struktur direktori & scaffold paket dibuat sesuai rancangan (`src/alpro/**`, `ui/`, `scripts/`, `configs/`, `tests/`, `data/`, `models/`, `outputs/`, dll.).
- `.venv` aktif (Python 3.13.3, Windows) dengan strategi **CPU-first** (catatan upgrade CUDA didokumentasi); `requirements.txt` (versioning hybrid), `.env.example`, `.gitignore`, `README.md`, dan `data/README.md` disiapkan.
- Konfigurasi kerangka: `configs/app.yaml`, `configs/detector.yaml`, `configs/preprocessing.yaml`.
- Dependensi terpasang dan import inti terverifikasi (termasuk `ultralytics`, `torch/torchvision`, `opencv-python`, `easyocr`, `streamlit`, `pytest`, `jiwer`).
- **Keputusan alur:** `app.py` ditunda ke fase UI (Fase 9/Milestone UI) agar fondasi pipeline/modul stabil terlebih dahulu.

### Kriteria selesai

Proyek dapat dijalankan pada environment lokal dan dependency utama berhasil diimpor. *(Terpenuhi.)*

---

## Fase 3 — Integrasi dan audit dataset Roboflow

**Tujuan:** Memastikan dataset yang telah diunduh benar, dapat dibaca, dan sesuai untuk tugas deteksi plat.

### Sumber dataset tunggal

**Indonesian_License_Plate — Roboflow Universe**  
https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate

Dataset yang digunakan dalam proyek ini dipusatkan pada sumber tersebut. Referensi dataset lain tidak digunakan sebagai sumber data proyek dalam dokumentasi ini.

Halaman Roboflow mencantumkan lisensi CC BY 4.0. Halaman versi dataset yang tersedia menampilkan versi v1 dengan tanggal pembuatan 15 Mei 2025, 1.245 gambar, dan pembagian train 1.085, valid 107, serta test 53. Halaman overview juga menampilkan angka 534 gambar; karena terdapat perbedaan angka antarbagian halaman, jumlah final yang digunakan harus mengikuti ZIP yang benar-benar diunduh dan metadata versi di dalamnya. Catat versi, tanggal unduh, jumlah file aktual, serta lisensi di dokumentasi dataset.

### Aktivitas

1. Menempatkan ZIP hasil unduhan ke lokasi sementara, bukan langsung ke direktori kode.
2. Mengekstrak ZIP ke `data/raw/indonesian_license_plate/`.
3. Memeriksa isi arsip: gambar, label, file YAML, dan pembagian data.
4. Membaca konfigurasi dataset dan memastikan nama kelas sesuai.
5. Menghitung jumlah gambar dan label pada setiap split.
6. Memeriksa apakah setiap gambar memiliki pasangan anotasi.
7. Memvalidasi koordinat bounding box dan memvisualisasikan sampel anotasi.
8. Memeriksa gambar rusak, label kosong, duplikasi, serta variasi kualitas gambar.
9. Mencatat preprocessing dan augmentasi yang sudah diterapkan saat pembuatan versi dataset.

### Hal penting tentang versi dataset

Halaman versi Roboflow mencantumkan Auto-Orient, resize stretch ke 640×640, dan grayscale sebagai preprocessing. Halaman tersebut juga mencantumkan augmentasi keluaran tiga kali, termasuk grayscale pada sebagian gambar dan blur hingga 2,5 px. Verifikasi konfigurasi ini terhadap versi ZIP yang diunduh. Jangan menganggap data ZIP sebagai gambar mentah tanpa preprocessing.

### Penjelasan

Audit diperlukan sebelum training. Kesalahan label, pasangan file yang hilang, atau pemahaman yang keliru tentang preprocessing dapat memengaruhi hasil model dan evaluasi. Dataset deteksi umumnya berisi bounding box, bukan transkripsi karakter plat.

### Keluaran fase

- Dataset terekstrak di `data/raw/indonesian_license_plate/`.
- Catatan dataset di `data/README.md`.
- Ringkasan jumlah gambar dan label per split.
- Hasil pemeriksaan pasangan gambar-label.
- Visualisasi sampel bounding box.
- Daftar temuan audit dan masalah dataset.

### Realisasi (aktual, M3)

- Script audit: `scripts/audit_dataset.py`; run formal `outputs/reports/dataset_audit/20261118_210535/` (`report.md`, `report.json`, `issue_summary.csv`, detail CSV) dan visualisasi `outputs/visualizations/dataset_audit/20261118_210535/`.
- **Status audit:** `READY_WITH_WARNINGS` (ERROR 0, WARNING 381, INFO 1). 1245/1245 pasangan gambar-label, pairing OK, 0 unreadable/invalid, seluruh citra 640×640 (3 channel), 1 kelas `License-Plate`.
- **Temuan utama:** 0 duplikat lintas-split; 374 grup duplikat intra-split (1085 file, seluruhnya di train); 4 label kosong (candidate background/no-object, perlu verifikasi visual); isu resolusi path pada `data.yaml` bawaan raw; tidak ada ground truth teks OCR.
- `data/README.md` diperbarui dengan temuan formal; `data/raw/indonesian_license_plate/` tidak diubah.

### Kriteria selesai

Struktur dataset, kelas, anotasi, jumlah data aktual, pembagian data, preprocessing, dan lisensinya telah dipahami serta terdokumentasi. *(Terpenuhi.)*

---

## Fase 4 — Penyiapan dataset untuk eksperimen

**Tujuan:** Menyiapkan data deteksi dan data evaluasi OCR secara benar.

### Aktivitas

1. Memastikan struktur train, valid, dan test dari ZIP.
2. Memastikan path pada YAML sesuai dengan struktur direktori hasil ekstraksi.
3. Memvalidasi label YOLO: `class_id x_center y_center width height`.
4. Memvisualisasikan bounding box pada sejumlah sampel dari setiap split.
5. Mempertahankan split bawaan dataset jika valid dan terdokumentasi; jangan melakukan split ulang tanpa alasan.
6. Memastikan test set tidak digunakan untuk tuning parameter.
7. Memeriksa apakah dataset menyediakan transkripsi teks plat.
8. Jika transkripsi teks tidak tersedia, menyiapkan subset anotasi OCR secara manual dengan pemeriksaan ulang.

### Penjelasan

Anotasi bounding box hanya memberi tahu lokasi plat. Anotasi tersebut tidak memberi tahu teks yang benar di dalam plat. Karena itu, evaluasi deteksi dapat memakai label bounding box, sedangkan evaluasi OCR membutuhkan ground truth teks yang terpisah. Jangan melaporkan akurasi OCR tanpa data teks acuan.

### Keluaran fase

- Konfigurasi dataset yang siap digunakan oleh Ultralytics.
- Dataset yang telah lolos validasi anotasi.
- Ringkasan split train/valid/test.
- Dataset evaluasi OCR berisi pasangan crop dan teks acuan, jika akan melakukan evaluasi OCR kuantitatif.
- Catatan prosedur anotasi dan verifikasi teks.

### Realisasi (aktual, M4)

- Script persiapan: `scripts/prepare_dataset.py`; run `outputs/reports/dataset_preparation/20261118_212026/` (`report.md`, `report.json`).
- **Config YOLO kerja:** `configs/dataset.yaml` menunjuk ke `data/raw/indonesian_license_plate/{train,valid,test}/images` (tanpa duplikasi gambar; `data.yaml` raw tidak diubah).
- **Manifest turunan:** `data/splits/*_manifest.{json,csv}` dan `preparation_manifest_summary.json` (1245 included, 0 excluded; flag empty-label provisional & duplikat intra-split terdokumentasi; split bawaan dipertahankan).
- **Kerangka OCR:** `data/annotations/ocr/` (`README.md`, template, `ocr_annotation_candidates.csv` berisi 50 kandidat deterministik seed 42 proporsional non-empty, skema `image_path,split,plate_text_gt,unreadable,notes,verified`) — `plate_text_gt` **belum diisi** (tidak ada GT otomatis/asumtif).
- `data/README.md` diperbarui; `data/raw/` tidak diubah.

### Kriteria selesai

Data deteksi dapat dimuat oleh framework dan tersedia rencana ground truth yang jelas untuk evaluasi OCR. *(Terpenuhi untuk deteksi; GT OCR kuantitatif masih menunggu anotasi manual/terverifikasi.)*

---

## Fase 5 — Implementasi baseline deteksi plat

**Tujuan:** Membuat komponen yang dapat menemukan bounding box plat pada gambar kendaraan.

### Aktivitas

1. Memilih versi model YOLO yang kompatibel dengan format dataset dan Ultralytics.
2. Memuat konfigurasi YAML dataset.
3. Menentukan parameter awal training seperti image size, batch size, epoch, dan confidence threshold.
4. Menjalankan baseline training atau menguji bobot model yang memang dipilih untuk baseline.
5. Menyimpan bobot terbaik dan log training.
6. Menguji inferensi pada gambar validasi dan test.
7. Membuat visualisasi bounding box hasil deteksi.

### Penjelasan

Pada fase ini targetnya adalah baseline yang dapat diukur, bukan langsung mengejar performa maksimum. Parameter, bobot, versi library, dan hasil eksperimen perlu dicatat supaya hasil dapat diulang.

### Keluaran fase

- Modul `plate_detector.py`.
- Bobot model hasil eksperimen, misalnya `best.pt`.
- Log training dan konfigurasi eksperimen.
- Contoh gambar hasil deteksi.
- Metrik deteksi awal.

### Realisasi (aktual, M5)

- **Keputusan baseline:** 6.1A CPU-only, 6.2 `yolov8n` (`yolov8n.pt`), parameter 6.3 (epochs 50, batch 8, imgsz 640, seed 42, patience 20, workers 2, default optimizer/augmentasi, cache false, deterministic true), data `configs/dataset.yaml` apa adanya (6.4A), evaluasi eksplisit val + test sekali tanpa tuning test (6.5A), visualisasi 15/split seed 42 (6.7A).
- **Implementasi:** `scripts/train_detector.py`, `scripts/evaluate_detector.py`, `src/alpro/detection/plate_detector.py` (+ `__init__`), `configs/detector.yaml` diperbarui.
- **Experiment:** `yolov8n_baseline_m5_20261118_cpu` (CPU, AMD Ryzen 5 6600H) — training selesai dengan early stopping pada **epoch 40/50** (best epoch 20, mAP@50-95(B)=0.73756), durasi ~2.85 jam.
- **Artefak aktual:** training run `runs/detect/models/trained/detection/yolov8n_baseline_m5_20261118_cpu/` (`weights/best.pt`, `last.pt`, `results.csv`, plot/args); laporan & metrik `outputs/reports/detection/yolov8n_baseline_m5_20261118_cpu/` (`report.md`, `evaluation_metrics.json/csv`, ringkasan training); visualisasi `outputs/visualizations/detection/yolov8n_baseline_m5_20261118_cpu/{val,test}/` (15+15); output evaluasi Ultralytics `runs/detect/outputs/predictions/detection/.../{val_val,val_test}/`. *(Catatan: lokasi training aktual berada di bawah `runs/detect/...` sesuai resolusi path Ultralytics, berbeda dari target konseptual `models/trained/detection/`.)*
- **Metrik evaluasi eksplisit (best.pt):** **val** (107 img, 159 inst) P=0.941, R=0.896, mAP@50=0.901, mAP@50-95=0.715; **test sekali** (53 img, 62 inst) P=0.983, R=0.948, mAP@50=0.955, mAP@50-95=0.761.
- Interpretasi harus mempertimbangkan konteks data (duplikat intra-split train & 4 label kosong provisional) dan batasan CPU-only.

### Kriteria selesai

Model dapat menerima gambar dan menghasilkan koordinat bounding box beserta confidence, termasuk kondisi ketika plat tidak terdeteksi. *(Terpenuhi; baseline terukur, tervalidasi eksplisit, dan terdokumentasi.)*

---

## Fase 6 — Implementasi cropping dan prapemrosesan citra

**Tujuan:** Mengambil area plat dan menguji teknik pengolahan citra untuk membantu pembacaan karakter.

### Aktivitas

1. Membuat fungsi cropping berdasarkan koordinat deteksi.
2. Menjaga koordinat agar tidak keluar dari batas gambar.
3. Menampilkan dan menyimpan hasil crop.
4. Mengimplementasikan grayscale.
5. Menguji resize untuk crop beresolusi kecil.
6. Menguji denoising menggunakan Gaussian atau median blur.
7. Menguji peningkatan kontras seperti CLAHE.
8. Membandingkan thresholding Otsu dan adaptive threshold.
9. Menguji morphology opening/closing bila dibutuhkan.
10. Menyimpan hasil antara agar efek setiap operasi dapat dibandingkan.

### Penjelasan

Tidak semua operasi harus diterapkan sekaligus. Prapemrosesan diperlakukan sebagai eksperimen: bandingkan hasil OCR pada crop asli dan beberapa variasi preprocessing. Pilihan akhir didasarkan pada hasil, bukan asumsi bahwa citra biner selalu lebih baik.

### Keluaran fase

- Modul `plate_cropper.py`.
- Modul `image_preprocessor.py`.
- Kumpulan contoh crop dan hasil preprocessing.
- Catatan parameter serta perbandingan visual.

### Kriteria selesai

Crop plat konsisten dan setiap operasi preprocessing dapat dipanggil serta diuji secara terpisah.

---

## Fase 7 — Implementasi OCR dan normalisasi teks

**Tujuan:** Mengubah citra plat menjadi teks yang dapat ditampilkan aplikasi.

### Aktivitas

1. Mengintegrasikan EasyOCR sebagai baseline.
2. Menguji OCR pada crop asli.
3. Menguji OCR pada variasi preprocessing.
4. Mengambil teks dan confidence jika disediakan engine.
5. Menyimpan hasil mentah OCR.
6. Membuat normalisasi ringan untuk spasi dan karakter yang tidak diperlukan.
7. Menambahkan validasi pola plat sebagai pemeriksaan opsional, bukan sebagai pengganti OCR.
8. Menguji kasus OCR tidak menghasilkan teks atau menghasilkan beberapa kandidat.

### Penjelasan

OCR merupakan komponen pengenalan teks, bukan detektor lokasi plat. Simpan prediksi mentah agar perubahan normalisasi tidak menyembunyikan kesalahan. Jangan mengubah karakter secara spekulatif hanya agar hasil tampak seperti format plat yang diharapkan.

### Keluaran fase

- Modul `ocr_engine.py`.
- Modul `text_normalizer.py`.
- Contoh hasil OCR dari beberapa kondisi citra.
- Catatan konfigurasi OCR dan keterbatasannya.

### Kriteria selesai

OCR dapat dijalankan pada crop plat dan menghasilkan status serta teks prediksi yang dapat ditelusuri.

---

## Fase 8 — Integrasi pipeline ALPRO

**Tujuan:** Menghubungkan seluruh komponen menjadi satu proses end-to-end.

### Aktivitas

1. Membuat `pipeline.py` sebagai pengatur urutan proses.
2. Menghubungkan validasi input, deteksi, cropping, preprocessing, OCR, dan normalisasi.
3. Menetapkan struktur hasil yang konsisten: status, bounding box, crop, hasil preprocessing, teks, confidence, durasi, dan pesan error.
4. Menangani kondisi gambar tidak valid, plat tidak ditemukan, atau OCR gagal.
5. Memisahkan logika pipeline dari antarmuka.
6. Membuat integration test untuk alur utama.

### Penjelasan

Pipeline menjadi satu pintu pemanggilan proses ALPRO. Antarmuka tidak perlu mengetahui detail implementasi setiap algoritma; UI cukup mengirim gambar dan menampilkan objek hasil dari pipeline.

### Keluaran fase

- `core/pipeline.py`.
- `core/result.py`.
- Integration test pipeline.
- Contoh hasil end-to-end melalui pemanggilan Python.

### Kriteria selesai

Pipeline dapat dijalankan tanpa Streamlit dan tidak menghentikan seluruh proses ketika satu tahap gagal.

---

## Fase 9 — Pengembangan antarmuka aplikasi

**Tujuan:** Menyediakan antarmuka yang mudah digunakan untuk demonstrasi dan pengujian.

### Aktivitas

1. Membuat halaman utama Streamlit.
2. Menyediakan fitur upload gambar.
3. Menampilkan gambar asli.
4. Menampilkan hasil deteksi dan bounding box.
5. Menampilkan crop plat.
6. Menampilkan tahapan preprocessing.
7. Menampilkan teks OCR dan confidence jika tersedia.
8. Menampilkan pesan ketika plat tidak ditemukan atau OCR gagal.
9. Menyediakan halaman atau bagian evaluasi jika diperlukan.

### Penjelasan

UI dibuat setelah pipeline inti berfungsi. Dengan demikian, perubahan pada model atau preprocessing tidak mengharuskan seluruh antarmuka ditulis ulang.

### Keluaran fase

- `app.py`.
- Halaman upload dan hasil pemrosesan.
- Tampilan visual tahapan deteksi sampai OCR.
- Petunjuk penggunaan aplikasi.

### Kriteria selesai

Pengguna dapat mengunggah gambar dan memperoleh hasil pemrosesan melalui antarmuka tanpa menjalankan script secara manual.

---

## Fase 10 — Pengujian perangkat lunak dan evaluasi model

**Tujuan:** Mengukur kinerja sistem dengan skenario dan metrik yang sesuai.

### Aktivitas pengujian perangkat lunak

- Menguji file gambar valid dan tidak valid.
- Menguji gambar tanpa plat.
- Menguji plat yang buram, gelap, miring, atau beresolusi rendah.
- Menguji lebih dari satu kendaraan jika didukung.
- Menguji kondisi deteksi kosong dan OCR kosong.
- Menjalankan unit test dan integration test.

### Metrik evaluasi

| Komponen   | Metrik                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| Deteksi    | Precision, recall, IoU, dan mAP sesuai kebutuhan                        |
| OCR        | Character Error Rate (CER), Word Error Rate (WER), Exact Match Accuracy |
| Aplikasi   | Waktu pemrosesan dan tingkat keberhasilan proses                        |
| Kualitatif | Analisis contoh benar, salah, dan kondisi kegagalan                     |

### Penjelasan

Metrik harus disesuaikan dengan tugasnya. Metrik deteksi tidak sama dengan metrik OCR. Evaluasi OCR harus membandingkan prediksi dengan transkripsi teks acuan. Hasil test set tidak boleh dipakai berulang kali untuk memilih parameter.

### Keluaran fase

- Test case dan hasil pengujian.
- Tabel metrik deteksi dan OCR.
- Grafik hasil evaluasi.
- Dokumentasi contoh keberhasilan dan kegagalan.
- Analisis keterbatasan sistem.

### Kriteria selesai

Setiap klaim performa didukung data uji, metrik yang tepat, dan penjelasan prosedur pengujian.

---

## Fase 11 — Finalisasi dokumentasi dan penyerahan

**Tujuan:** Menyiapkan seluruh luaran akademik dan teknis sesuai ketentuan tugas.

### Aktivitas

1. Merapikan source code dan komentar penting.
2. Memastikan README menjelaskan instalasi, konfigurasi, dan penggunaan.
3. Memperbarui dokumentasi dataset, lisensi, versi, dan preprocessing.
4. Menyusun diagram arsitektur dan flowchart.
5. Mengumpulkan screenshot input, deteksi, crop, preprocessing, dan hasil OCR.
6. Menyusun laporan proyek akhir.
7. Menyusun draf artikel ilmiah.
8. Menyusun daftar pustaka minimal 10 referensi, termasuk minimal 5 artikel jurnal/prosiding lima tahun terakhir sesuai ketentuan tugas.
9. Menyiapkan slide dan skenario demonstrasi.
10. Mencatat pembagian kontribusi anggota kelompok.

### Penjelasan

Dokumentasi harus menggambarkan implementasi yang benar-benar dibuat dan hasil pengujian yang benar-benar diperoleh. Jangan menuliskan angka performa sebelum eksperimen selesai. Cantumkan keterbatasan dan kondisi gagal secara terbuka.

### Keluaran fase

- Source code final.
- README dan dokumentasi teknis.
- Laporan proyek akhir.
- Draf artikel ilmiah.
- Presentasi dan bukti demonstrasi.
- Arsip hasil evaluasi.

### Kriteria selesai

Aplikasi dapat dijalankan oleh orang lain mengikuti README dan seluruh luaran tugas tersedia.

---

## Ringkasan milestone dan keluaran

> Pemetaan di bawah mengikuti **alur kerja aktual** yang telah dijalankan (M1–M5 selesai). Fase 1–11 tetap menjadi rujukan ruang lingkup teknis; milestone kerja berikutnya menyesuaikan urutan implementasi modul inti sebelum integrasi/UI dan evaluasi akhir.

| Milestone kerja | Fase rujukan | Status | Keluaran utama (aktual/rencana) |
| --- | --- | --- | --- |
| M1 — Analisis & inisiasi | 1 | **Selesai** | Analisis repo & dokumen acuan, pemahaman dataset/scope, gap & risiko, rencana bertahap |
| M2 — Setup repository & environment | 1–2 | **Selesai** | Scaffold, `.venv` + dependensi terverifikasi (CPU-first), configs kerangka, `requirements.txt`, `.gitignore`, README/data docs (UI `app.py` ditunda) |
| M3 — Audit dataset | 3 | **Selesai** | Audit formal, laporan & visualisasi, temuan kualitas/split/duplikat/label kosong, status kesiapan data |
| M4 — Persiapan dataset | 4 | **Selesai** | `configs/dataset.yaml` kerja, manifest `data/splits/`, kerangka & kandidat anotasi OCR (`data/annotations/ocr/`), laporan persiapan; deteksi siap baseline, GT OCR belum terisi |
| M5 — Baseline deteksi YOLO | 5 | **Selesai** | `yolov8n` CPU baseline, `best.pt/last.pt`, metrik val & test (sekali), visualisasi prediksi, modul deteksi & script train/eval, laporan eksperimen |
| M6 — Plate crop & preprocessing OCR | 6 | Belum dimulai | `plate_cropper.py`, `image_preprocessor.py`, hasil crop/preprocessing & perbandingan untuk OCR |
| M7 — OCR & normalisasi teks | 7 | Belum dimulai | `ocr_engine.py`, `text_normalizer.py`, hasil OCR mentah/normalisasi; evaluasi kuantitatif bila GT teks tersedia |
| M8 — Integrasi pipeline & UI aplikasi | 8–9 | Belum dimulai | `pipeline.py`/`result.py`, integration test, Streamlit (`app.py`) & tampilan tahapan proses |
| M9 — Evaluasi menyeluruh & finalisasi | 10–11 | Belum dimulai | Pengujian end-to-end, metrik deteksi/OCR/aplikasi, analisis error, dokumentasi final, laporan/artikel/presentasi/demo |

## Catatan pengelolaan eksperimen

Setiap eksperimen sebaiknya mencatat:

- ID eksperimen dan tanggal.
- Versi dataset dan split yang digunakan.
- Versi Python dan library.
- Nama/versi model dan bobot awal.
- Parameter training dan inferensi.
- Konfigurasi preprocessing dan OCR.
- Metrik hasil serta lokasi artefak.

Jangan menetapkan target angka performa tanpa baseline. Tentukan target setelah audit dataset dan eksperimen awal.

## Checklist progres

- [x] Fase 1 — Inisiasi dan kebutuhan *(inti: analisis & konteks/scope terdokumentasi; detail operasional tim dapat dilengkapi)*
- [x] Fase 2 — Repository dan environment *(scaffold, venv, dependensi terverifikasi, config kerangka; UI ditunda)*
- [x] Fase 3 — Audit dataset Roboflow *(audit formal + laporan/visualisasi, temuan terdokumentasi)*
- [x] Fase 4 — Penyiapan dataset dan ground truth OCR *(config/manifest/kerangka & kandidat OCR siap; GT teks manual belum terisi)*
- [x] Fase 5 — Baseline deteksi *(training & evaluasi eksplisit val/test selesai, artefak & laporan tersedia)*
- [ ] Fase 6 — Cropping dan preprocessing
- [ ] Fase 7 — OCR dan normalisasi
- [ ] Fase 8 — Integrasi pipeline
- [ ] Fase 9 — Antarmuka aplikasi
- [ ] Fase 10 — Pengujian dan evaluasi
- [ ] Fase 11 — Dokumentasi dan penyerahan

## Referensi dataset

Roboflow Universe. _Indonesian_License_Plate_.  
https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate

Roboflow Universe. _Indonesian_License_Plate — Dataset Version 1_.  
https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate/dataset/1

Lisensi yang tercantum pada halaman proyek: CC BY 4.0. Periksa kembali ketentuan dan atribusi pada halaman sumber saat penggunaan atau distribusi hasil.
