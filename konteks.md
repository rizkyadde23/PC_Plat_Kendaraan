# Konteks Proyek ALPRO

## 1. Identitas proyek

| Elemen             | Keterangan                                             |
| ------------------ | ------------------------------------------------------ |
| Nama proyek        | **ALPRO — Automatic License Plate Recognition Online** |
| Mata kuliah        | Pengolahan Citra Digital                               |
| Program studi      | Informatika                                            |
| Tahun akademik     | 2026/2027                                              |
| Bahasa pemrograman | Python                                                 |
| Bentuk aplikasi    | Aplikasi lokal berbasis web menggunakan Streamlit      |
| Fokus pengerjaan   | Pengembangan perangkat lunak aplikasi                  |
| Domain masalah     | Transportasi dan pengolahan citra                      |

Nama anggota kelompok, NIM, dosen pengampu, dan informasi administratif dapat ditambahkan setelah ditetapkan.

## 2. Latar belakang

Plat nomor kendaraan merupakan rangkaian karakter alfanumerik yang digunakan sebagai identitas registrasi kendaraan. Membaca plat dari gambar secara manual membutuhkan perhatian pengguna. Pengolahan citra digital dan computer vision dapat membantu menemukan area plat dan membaca karakter yang terdapat di dalamnya.

ALPRO dikembangkan sebagai aplikasi pengenalan plat kendaraan dari gambar. Sistem menggabungkan deteksi objek, prapemrosesan citra, dan Optical Character Recognition (OCR) dalam satu alur aplikasi yang dapat diperiksa tahap demi tahap.

## 3. Permasalahan

Proyek berfokus pada pertanyaan bagaimana membangun aplikasi Python yang mampu:

1. Mendeteksi lokasi plat nomor pada gambar kendaraan.
2. Mengambil area plat dari gambar.
3. Menerapkan prapemrosesan citra agar karakter lebih mudah dikenali.
4. Membaca karakter plat menggunakan OCR.
5. Menampilkan hasil dan tahapan pemrosesan kepada pengguna.
6. Mengevaluasi kinerja deteksi dan pengenalan teks secara terukur.

Hasil sistem dapat dipengaruhi oleh pencahayaan, resolusi, sudut pengambilan gambar, blur, noise, kontras, bagian plat yang tertutup, dan kualitas anotasi dataset.

## 4. Tujuan proyek

- Mengembangkan aplikasi pengenalan plat nomor kendaraan berbasis Python.
- Menerapkan deteksi objek untuk menemukan area plat.
- Menerapkan teknik prapemrosesan citra menggunakan OpenCV.
- Mengintegrasikan OCR untuk membaca karakter plat.
- Menampilkan proses dan hasil melalui antarmuka aplikasi.
- Mengukur performa sistem dengan skenario dan metrik yang sesuai.
- Mendokumentasikan kesalahan, keterbatasan, dan peluang pengembangan.

## 5. Pengguna dan skenario penggunaan

**Pengguna sasaran:** mahasiswa, dosen, dan pengguna demonstrasi yang ingin menguji pembacaan plat dari gambar kendaraan.

**Skenario utama:**

1. Pengguna membuka aplikasi ALPRO.
2. Pengguna mengunggah gambar kendaraan.
3. Aplikasi memvalidasi dan menampilkan gambar.
4. Model mendeteksi area plat dan menampilkan bounding box.
5. Sistem memotong area plat.
6. Sistem menjalankan prapemrosesan citra.
7. OCR membaca teks pada area plat.
8. Aplikasi menampilkan teks prediksi, confidence jika tersedia, dan visualisasi proses.
9. Jika plat tidak terdeteksi atau OCR gagal, aplikasi menampilkan status yang sesuai.

## 6. Ruang lingkup

### Termasuk dalam ruang lingkup awal

- Input gambar statis melalui upload.
- Deteksi area plat kendaraan.
- Cropping area plat.
- Prapemrosesan menggunakan OpenCV.
- Pengenalan teks menggunakan OCR.
- Antarmuka aplikasi berbasis Streamlit.
- Pengujian perangkat lunak dan evaluasi model.
- Dokumentasi dataset, source code, dan hasil eksperimen.

### Di luar ruang lingkup awal

- Pemrosesan CCTV atau video real-time.
- Pelacakan kendaraan antar-frame.
- Integrasi ke sistem parkir, gerbang tol, atau kepolisian.
- Identifikasi pemilik kendaraan.
- Pengambilan keputusan hukum atau penindakan otomatis.
- Jaminan akurasi pada seluruh kondisi lapangan.

Kata “Online” pada nama ALPRO merupakan identitas nama proyek dan tidak otomatis berarti aplikasi sudah di-deploy ke layanan publik atau mendukung pemrosesan real-time. Bentuk implementasi awal adalah aplikasi lokal berbasis web.

## 7. Rancangan solusi dan teknologi

Pipeline yang direncanakan:

```text
Gambar kendaraan
      |
      v
Validasi dan pembacaan gambar
      |
      v
Deteksi plat (YOLO)
      |
      v
Cropping area plat
      |
      v
Prapemrosesan citra (OpenCV)
      |
      v
OCR (EasyOCR sebagai baseline)
      |
      v
Normalisasi teks
      |
      v
Tampilan hasil aplikasi
```

| Teknologi            | Tanggung jawab                         |
| -------------------- | -------------------------------------- |
| Python               | Bahasa pemrograman utama               |
| OpenCV               | Membaca, memotong, dan memproses citra |
| Ultralytics YOLO     | Kandidat framework deteksi plat        |
| EasyOCR              | OCR baseline                           |
| Streamlit            | Antarmuka pengguna                     |
| NumPy dan Pandas     | Pengolahan data dan hasil evaluasi     |
| Matplotlib           | Visualisasi eksperimen                 |
| jiwer / scikit-learn | Metrik evaluasi yang sesuai            |
| pytest               | Pengujian otomatis                     |

Versi model, parameter, dan dependency final ditetapkan setelah audit dataset dan eksperimen baseline.

## 8. Dataset utama

ALPRO menggunakan satu sumber dataset utama yang telah diunduh oleh pengembang dalam bentuk ZIP:

**Indonesian_License_Plate — Roboflow Universe**  
https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate

Dataset ini merupakan dataset object detection untuk mendeteksi area plat kendaraan Indonesia. Lisensi yang tercantum pada halaman proyek adalah **CC BY 4.0**. Atribusi dan ketentuan lisensi harus tetap dicantumkan ketika dataset atau hasil turunannya digunakan atau dibagikan.

Halaman versi dataset:
https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate/dataset/1

Informasi pada halaman versi v1 mencantumkan tanggal pembuatan 15 Mei 2025, total 1.245 gambar, dengan pembagian train 1.085, valid 107, dan test 53. Halaman overview mencantumkan angka 534 gambar. Karena informasi jumlah pada halaman berbeda, jumlah aktual yang dipakai ALPRO harus ditentukan berdasarkan ZIP yang telah diunduh, file konfigurasi, dan hasil audit lokal.

Halaman versi juga mencantumkan preprocessing Auto-Orient, resize stretch ke 640×640, dan grayscale serta augmentasi tertentu. Konfigurasi yang benar-benar berlaku harus diverifikasi terhadap versi ZIP yang diunduh.

### Batasan dataset untuk OCR

Dataset object detection menyediakan anotasi lokasi plat (bounding box). Anotasi tersebut tidak otomatis menyediakan teks plat sebagai ground truth. Oleh karena itu:

- Dataset digunakan sebagai sumber utama untuk deteksi plat.
- Ketersediaan transkripsi teks harus diperiksa.
- Jika teks acuan tidak tersedia, tim perlu membuat anotasi manual pada subset evaluasi OCR.
- Evaluasi OCR kuantitatif hanya dilakukan pada sampel yang memiliki teks acuan tervalidasi.

Dataset disimpan secara lokal pada `data/raw/indonesian_license_plate/` setelah ZIP diekstrak. Dataset besar tidak dimasukkan ke Git. Catat tanggal unduh, versi, jumlah aktual, lisensi, dan konfigurasi dataset.

## 9. Metode evaluasi

| Komponen            | Metrik atau analisis                                                    |
| ------------------- | ----------------------------------------------------------------------- |
| Deteksi plat        | Precision, recall, IoU, mAP sesuai kebutuhan                            |
| Pengenalan teks     | Character Error Rate (CER), Word Error Rate (WER), Exact Match Accuracy |
| Kinerja aplikasi    | Waktu pemrosesan dan tingkat keberhasilan proses                        |
| Analisis kualitatif | Contoh hasil benar/salah dan kondisi penyebab kegagalan                 |

Evaluasi dilakukan pada data yang sesuai dan tidak digunakan untuk tuning berulang. Metrik OCR memerlukan teks ground truth. Pengujian aplikasi juga mencakup file tidak valid, gambar tanpa plat, gambar buram, gelap, miring, dan resolusi rendah jika tersedia.

## 10. Luaran proyek

1. Source code aplikasi ALPRO.
2. Aplikasi yang dapat dijalankan secara lokal.
3. Petunjuk instalasi dan penggunaan.
4. Dokumentasi sumber dan audit dataset.
5. Model deteksi dan konfigurasi eksperimen.
6. Hasil pengujian berupa tabel, grafik, dan contoh visual.
7. Laporan proyek akhir.
8. Draf artikel ilmiah.
9. Materi presentasi dan demonstrasi.

## 11. Prinsip pengembangan

- **Modularitas:** deteksi, preprocessing, OCR, evaluasi, dan UI dipisahkan.
- **Reproducibility:** versi dependency, dataset, parameter, dan split dicatat.
- **Testability:** logika inti dapat diuji tanpa menjalankan UI.
- **Traceability:** hasil eksperimen dapat ditelusuri ke konfigurasi dan data yang digunakan.
- **Responsible data use:** dataset digunakan sesuai lisensi dan atribusi.
- **Honest evaluation:** hasil, kesalahan, dan keterbatasan dilaporkan sesuai eksperimen.

## 12. Status keputusan proyek

| Keputusan                   | Status                                                                 |
| --------------------------- | ---------------------------------------------------------------------- |
| Nama proyek                 | ALPRO — Automatic License Plate Recognition Online                     |
| Dataset utama               | Indonesian_License_Plate dari Roboflow Universe; ZIP telah diunduh     |
| Format dataset              | ZIP; format anotasi dan struktur aktual diverifikasi saat audit        |
| Model deteksi               | YOLO menjadi pendekatan awal; versi final ditentukan saat implementasi |
| OCR                         | EasyOCR sebagai baseline awal                                          |
| Preprocessing               | OpenCV; teknik final ditentukan melalui eksperimen                     |
| UI                          | Streamlit                                                              |
| Input awal                  | Gambar statis                                                          |
| Ground truth OCR            | Perlu diperiksa; anotasi manual subset jika tidak tersedia             |
| Target performa             | Ditetapkan setelah baseline, bukan diasumsikan                         |
| Deployment publik/real-time | Di luar ruang lingkup awal                                             |

Dokumen ini menjadi sumber konteks bersama untuk diskusi dan implementasi ALPRO. Jika keputusan teknis berubah, perbarui bagian status ini dan dokumen perencanaan agar seluruh anggota kelompok menggunakan acuan yang sama.
