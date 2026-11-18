# Konteks Proyek ALPR

## 1. Identitas proyek

| Elemen | Keterangan |
|---|---|
| Nama sementara | ALPR — Automatic License Plate Recognition |
| Mata kuliah | Pengolahan Citra Digital |
| Program studi | Informatika |
| Tahun akademik | 2026/2027 |
| Bahasa pemrograman | Python |
| Bentuk aplikasi | Aplikasi desktop/web lokal berbasis Streamlit |
| Fokus pengerjaan saat ini | Perangkat lunak aplikasi |
| Domain masalah | Transportasi dan pengolahan citra |

Nama, anggota kelompok, dosen pengampu, dan detail administratif dapat ditambahkan setelah ditetapkan kelompok.

## 2. Latar belakang

Plat nomor kendaraan merupakan rangkaian karakter alfanumerik yang ditampilkan pada kendaraan. Membaca plat dari gambar secara manual membutuhkan perhatian pengguna dan dapat menjadi kurang praktis ketika gambar berjumlah banyak. Pengolahan citra digital dan computer vision dapat digunakan untuk membantu menemukan area plat dan membaca karakter di dalamnya.

Proyek ini mengembangkan prototipe aplikasi yang memproses gambar kendaraan melalui tahapan deteksi area plat, pemotongan citra, prapemrosesan, dan pengenalan teks menggunakan OCR.

## 3. Permasalahan yang ditangani

Permasalahan utama proyek adalah bagaimana membangun aplikasi Python yang dapat:
1. Menemukan lokasi plat nomor pada gambar kendaraan.
2. Menyiapkan area plat agar karakter lebih mudah dikenali.
3. Membaca karakter plat menggunakan OCR.
4. Menampilkan hasil proses secara transparan.
5. Mengukur kinerja deteksi dan pengenalan teks menggunakan data uji berlabel.

Kinerja sistem dapat dipengaruhi oleh pencahayaan, resolusi, sudut pengambilan gambar, blur, noise, kontras, occlusion, dan variasi bentuk/warna plat.

## 4. Tujuan

- Mengimplementasikan pipeline pengolahan citra untuk pengenalan plat nomor kendaraan.
- Menerapkan deteksi objek untuk menemukan area plat.
- Menerapkan operasi prapemrosesan citra dan mengamati dampaknya terhadap OCR.
- Mengintegrasikan OCR untuk memperoleh teks plat.
- Membangun antarmuka aplikasi yang dapat digunakan untuk menguji gambar.
- Mengevaluasi sistem secara kuantitatif dan kualitatif serta mendokumentasikan keterbatasannya.

## 5. Pengguna dan skenario penggunaan

**Pengguna sasaran:** mahasiswa, dosen, atau pengguna demonstrasi yang ingin menguji pembacaan plat kendaraan dari gambar.

**Skenario utama:**
1. Pengguna membuka aplikasi.
2. Pengguna mengunggah gambar kendaraan.
3. Aplikasi menampilkan gambar masukan.
4. Aplikasi mendeteksi area plat dan menampilkan bounding box.
5. Aplikasi memotong area plat.
6. Aplikasi menampilkan tahapan preprocessing.
7. OCR menghasilkan prediksi teks.
8. Aplikasi menampilkan hasil, confidence jika disediakan engine, dan status proses.

## 6. Ruang lingkup

### Termasuk dalam ruang lingkup awal
- Input gambar statis melalui upload.
- Deteksi area plat kendaraan.
- Cropping plat.
- Prapemrosesan citra menggunakan OpenCV.
- Pengenalan teks menggunakan OCR.
- Antarmuka berbasis Streamlit.
- Evaluasi pada dataset berlabel.
- Dokumentasi instalasi, penggunaan, eksperimen, dan hasil.

### Di luar ruang lingkup awal
- Pengenalan kendaraan dari CCTV secara real-time.
- Pelacakan kendaraan antar-frame.
- Integrasi ke sistem parkir, gerbang tol, atau sistem kepolisian.
- Penyimpanan identitas pemilik kendaraan.
- Pengambilan keputusan hukum atau penindakan otomatis.
- Jaminan akurasi untuk seluruh kondisi lapangan.

Fitur video atau kamera langsung dapat dipertimbangkan sebagai pengembangan lanjutan jika waktu dan sumber daya memungkinkan.

## 7. Rancangan solusi

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
Tampilan hasil dan evaluasi
```

### Pembagian tanggung jawab teknologi

| Teknologi | Peran |
|---|---|
| Python | Bahasa utama |
| OpenCV | Membaca, memotong, dan memproses citra |
| YOLO (kandidat) | Deteksi bounding box plat |
| EasyOCR (baseline) | Pengenalan karakter |
| Streamlit | Antarmuka pengguna |
| NumPy/Pandas | Manipulasi array dan data evaluasi |
| Matplotlib | Visualisasi hasil |
| scikit-learn / jiwer | Metrik evaluasi yang sesuai |
| pytest | Pengujian perangkat lunak |

Pemilihan versi dan konfigurasi final akan ditentukan setelah pemeriksaan dataset, eksperimen baseline, dan pemeriksaan kompatibilitas lingkungan.

## 8. Dataset

Dataset utama yang akan dieksplorasi adalah dataset plat Indonesia dari Roboflow Universe dan/atau Zenodo. Dataset UFPR-ALPR dapat dipertimbangkan sebagai sumber pembanding akademik, dengan memperhatikan bahwa datanya berasal dari Brasil dan ketentuan penggunaannya membatasi pemakaian serta redistribusi.

Sumber awal:
- Indonesian License Plate (Roboflow): https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate
- Indonesian License Plate Detection Dataset (Zenodo): https://zenodo.org/records/15605718
- UFPR-ALPR: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
- Ketentuan UFPR-ALPR: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/

**Hal yang wajib diverifikasi sebelum dataset dipakai:**
- Lisensi dan izin penggunaan.
- Jumlah gambar dan versi dataset aktual.
- Ketersediaan bounding box untuk deteksi.
- Ketersediaan transkripsi teks untuk evaluasi OCR.
- Distribusi kendaraan dan kondisi citra.
- Duplikasi dan kualitas anotasi.

Anotasi bounding box tidak sama dengan ground truth teks OCR. Jika teks plat tidak disediakan, tim perlu membuat anotasi teks pada subset yang digunakan untuk evaluasi OCR. Data harus dikelola sesuai lisensi dan pertimbangan privasi.

## 9. Metode evaluasi

| Bagian | Metrik/analisis |
|---|---|
| Deteksi plat | Precision, recall, IoU, mAP jika sesuai |
| Pengenalan teks | Character Error Rate (CER), Word Error Rate (WER), Exact Match Accuracy |
| Kinerja aplikasi | Waktu pemrosesan, tingkat keberhasilan proses |
| Analisis kualitatif | Contoh prediksi benar/salah dan kondisi citra yang menyebabkan kegagalan |

Pengujian akan mencakup citra jelas, gelap, buram, miring, resolusi rendah, tanpa plat, dan beberapa kendaraan bila dataset mendukungnya. Pembagian train/validation/test harus menghindari kebocoran data.

## 10. Luaran yang ditargetkan

1. Source code aplikasi Python.
2. Aplikasi yang dapat dijalankan secara lokal.
3. Petunjuk instalasi dan penggunaan.
4. Dokumentasi sumber dataset dan lisensi.
5. Hasil pengujian berupa tabel, grafik, dan contoh citra.
6. Laporan proyek akhir.
7. Draf artikel ilmiah.
8. Materi presentasi dan demonstrasi.

## 11. Prinsip pengembangan

- **Modularitas:** deteksi, preprocessing, OCR, evaluasi, dan UI dipisahkan.
- **Reproducibility:** versi dependency, dataset, parameter, dan pembagian data dicatat.
- **Testability:** logika inti dapat diuji tanpa menjalankan UI.
- **Traceability:** setiap hasil eksperimen dapat ditelusuri ke data dan konfigurasi yang digunakan.
- **Responsible data use:** dataset digunakan sesuai lisensi dan privasi.
- **Honest evaluation:** hasil, error, dan keterbatasan dilaporkan apa adanya.

## 12. Status dan keputusan yang belum ditetapkan

| Keputusan | Status |
|---|---|
| Dataset utama | Belum final; audit Roboflow dan Zenodo terlebih dahulu |
| Model deteksi final | Belum final; YOLO menjadi kandidat awal |
| Engine OCR final | EasyOCR menjadi baseline, dapat dibandingkan dengan alternatif |
| Jumlah data | Ditentukan setelah audit dataset |
| Rasio split | Rencana awal 70/15/15, disesuaikan dengan kondisi data |
| Target performa | Ditetapkan setelah baseline, bukan diasumsikan |
| Video/real-time | Di luar ruang lingkup awal |
| Bentuk UI | Streamlit sebagai pilihan awal |

Dokumen ini menjadi sumber konteks bersama untuk diskusi dan implementasi berikutnya. Setiap perubahan keputusan penting perlu diperbarui di sini agar seluruh anggota tim menggunakan acuan yang sama.
