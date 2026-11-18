# Perencanaan Pengembangan ALPRO

> **Nama proyek:** ALPRO — Automatic License Plate Recognition Online  
> **Mata kuliah:** Pengolahan Citra Digital  
> **Bahasa pemrograman:** Python  
> **Fokus:** Pengembangan aplikasi perangkat lunak  
> **Status:** Dataset utama telah diunduh dari Roboflow Universe dalam bentuk ZIP.

Dokumen ini menjadi roadmap pelaksanaan proyek. Pengembangan dibagi menjadi fase/milestone agar setiap tahap mempunyai tujuan, aktivitas, penjelasan, dan keluaran yang dapat diperiksa sebelum melanjutkan ke tahap berikutnya.

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

### Kriteria selesai

Seluruh anggota memahami tujuan, alur sistem, fitur versi awal, dan hal-hal yang tidak termasuk ruang lingkup.

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

### Kriteria selesai

Proyek dapat dijalankan pada environment lokal dan dependency utama berhasil diimpor.

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

### Kriteria selesai

Struktur dataset, kelas, anotasi, jumlah data aktual, pembagian data, preprocessing, dan lisensinya telah dipahami serta terdokumentasi.

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

### Kriteria selesai

Data deteksi dapat dimuat oleh framework dan tersedia rencana ground truth yang jelas untuk evaluasi OCR.

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

### Kriteria selesai

Model dapat menerima gambar dan menghasilkan koordinat bounding box beserta confidence, termasuk kondisi ketika plat tidak terdeteksi.

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

| Milestone                | Fase | Keluaran utama                                                   |
| ------------------------ | ---- | ---------------------------------------------------------------- |
| M1 — Project setup       | 1–2  | Konteks, kebutuhan, repository, environment                      |
| M2 — Dataset ready       | 3–4  | Dataset terverifikasi, konfigurasi, audit, ground truth OCR plan |
| M3 — Detection baseline  | 5    | Model deteksi, bobot, log, metrik awal                           |
| M4 — Image processing    | 6    | Cropping dan preprocessing                                       |
| M5 — Text recognition    | 7    | OCR dan normalisasi teks                                         |
| M6 — End-to-end pipeline | 8    | Pipeline ALPRO yang terintegrasi                                 |
| M7 — Application UI      | 9    | Aplikasi Streamlit                                               |
| M8 — Evaluation          | 10   | Test report, metrik, analisis error                              |
| M9 — Final delivery      | 11   | Source code, laporan, artikel, presentasi, demo                  |

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

- [ ] Fase 1 — Inisiasi dan kebutuhan
- [ ] Fase 2 — Repository dan environment
- [ ] Fase 3 — Audit dataset Roboflow
- [ ] Fase 4 — Penyiapan dataset dan ground truth OCR
- [ ] Fase 5 — Baseline deteksi
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
