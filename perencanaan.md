# Perencanaan Pengembangan Aplikasi ALPR

> **Nama proyek sementara:** ALPR — Automatic License Plate Recognition  
> **Domain:** Pengolahan Citra Digital  
> **Bahasa pemrograman:** Python  
> **Status dokumen:** Rencana awal; keputusan teknis dapat diperbarui setelah eksplorasi dataset dan eksperimen baseline.

## 1. Ringkasan proyek

Proyek ini bertujuan mengembangkan aplikasi pengolahan citra yang dapat menerima gambar kendaraan, menemukan area plat nomor, mempersiapkan citra plat, lalu mengenali karakter yang terdapat di dalamnya. Aplikasi ditujukan sebagai prototipe akademik untuk demonstrasi dan evaluasi, bukan sebagai sistem penegakan hukum atau identifikasi kendaraan yang tervalidasi untuk penggunaan operasional.

Alur konseptual:

1. Pengguna memasukkan gambar kendaraan.
2. Sistem memvalidasi dan membaca gambar.
3. Sistem mendeteksi bounding box plat nomor.
4. Area plat dipotong (*cropping*) dari gambar asli.
5. Citra plat dipraproses menggunakan OpenCV.
6. OCR membaca teks pada citra plat.
7. Sistem menampilkan teks hasil, confidence jika tersedia, dan visualisasi tahapan.
8. Hasil dibandingkan dengan ground truth untuk evaluasi.

## 2. Acuan tugas dan ruang lingkup

Dokumen tugas proyek akhir Pengolahan Citra Digital meminta aplikasi yang dapat dijalankan serta laporan proyek akhir, draf artikel ilmiah, dan dokumentasi presentasi/demo. Proyek harus menunjukkan proses pemilihan/akuisisi data, prapemrosesan, penerapan algoritma, serta evaluasi. Penggunaan pustaka tambahan harus dijelaskan.

Rencana ini berfokus pada **pengembangan perangkat lunak aplikasi Python**. Pembuatan perangkat keras, integrasi kamera CCTV secara langsung, deployment produksi, dan integrasi ke sistem kepolisian/parkir tidak menjadi target awal.

### Target fungsional

- Memuat gambar kendaraan dari file lokal.
- Menampilkan gambar masukan.
- Mendeteksi satu atau beberapa area plat (bergantung kemampuan model).
- Menampilkan bounding box dan hasil crop.
- Menjalankan prapemrosesan citra yang dapat diperiksa pengguna.
- Menjalankan OCR dan menampilkan teks hasil pembacaan.
- Menangani kasus gambar tidak valid, plat tidak ditemukan, atau OCR tidak menghasilkan teks.
- Menyediakan mode evaluasi pada kumpulan gambar berlabel.
- Menyimpan hasil pengujian dan ringkasan metrik.

### Batasan awal

- Input awal berupa gambar statis (JPG/JPEG/PNG); video dan kamera real-time dapat menjadi pengembangan lanjutan.
- Fokus data utama adalah plat kendaraan Indonesia.
- Sistem tidak menjamin pembacaan benar untuk semua kondisi, terutama gambar buram, gelap, miring ekstrem, tertutup, atau beresolusi rendah.
- Hasil OCR merupakan prediksi, bukan bukti identitas kendaraan.
- Dataset harus digunakan sesuai lisensi dan ketentuan privasi sumbernya.

## 3. Rancangan pendekatan teknis

### 3.1 Pipeline

| Tahap | Tanggung jawab | Kandidat teknologi |
|---|---|---|
| Input | Membaca dan memvalidasi file gambar | Streamlit, Pillow, OpenCV |
| Deteksi plat | Menghasilkan koordinat area plat | Model object detection YOLO |
| Cropping | Memotong area plat dari gambar | OpenCV |
| Prapemrosesan | Menyiapkan citra agar OCR lebih mudah membaca karakter | OpenCV |
| OCR | Mengubah karakter pada citra menjadi teks | EasyOCR sebagai baseline |
| Post-processing | Membersihkan hasil OCR secara hati-hati | Python, regex |
| Evaluasi | Membandingkan prediksi dengan label acuan | jiwer / scikit-learn / pandas |
| Antarmuka | Mengatur interaksi dan menampilkan hasil | Streamlit |

YOLO dan OCR menyelesaikan tugas yang berbeda. Model deteksi menemukan lokasi plat, sedangkan OCR membaca teks pada potongan plat. Pemilihan versi/model YOLO final dilakukan setelah memeriksa lisensi, format anotasi, kemampuan perangkat, dan hasil eksperimen.

### 3.2 Prapemrosesan yang akan diuji

Prapemrosesan tidak harus menerapkan semua operasi sekaligus. Setiap operasi akan diuji agar dampaknya terhadap OCR dapat diketahui.

1. **Resize:** menyesuaikan ukuran crop jika resolusinya terlalu kecil.
2. **Grayscale:** mengubah citra berwarna menjadi intensitas keabuan.
3. **Denoising:** mengurangi derau, misalnya dengan Gaussian atau median blur.
4. **Contrast enhancement:** menguji CLAHE atau normalisasi kontras.
5. **Thresholding:** membandingkan Otsu/adaptive threshold dengan citra grayscale biasa.
6. **Morphology:** menguji operasi opening/closing bila karakter terputus atau menyatu.
7. **Deskew (opsional):** mengoreksi kemiringan plat jika metode dan estimasi sudutnya cukup stabil.

Simpan atau tampilkan hasil antara agar perubahan citra dapat diamati. Jangan mengasumsikan satu konfigurasi terbaik untuk semua gambar.

### 3.3 Strategi baseline dan pengembangan

- **Baseline pertama:** gunakan model deteksi plat yang telah dilatih atau dataset beranotasi untuk melatih model deteksi sederhana, lalu gunakan EasyOCR pada crop plat.
- **Eksperimen prapemrosesan:** bandingkan OCR pada crop asli dengan beberapa variasi prapemrosesan.
- **Penyempurnaan:** evaluasi kesalahan deteksi dan OCR, lalu ubah threshold confidence, ukuran input, atau pipeline prapemrosesan berdasarkan bukti.
- **Opsional:** segmentasi karakter secara eksplisit jika dibutuhkan oleh tujuan eksperimen. OCR end-to-end tidak mensyaratkan segmentasi karakter manual.

## 4. Langkah pengembangan dari awal sampai akhir

### Tahap 1 — Memahami kebutuhan dan menetapkan batasan

**Aktivitas**
- Baca kembali instruksi tugas dan catat luaran wajib.
- Tetapkan pengguna sasaran: pengguna yang ingin membaca plat dari foto kendaraan.
- Tetapkan input, output, batasan, dan skenario gagal.
- Sepakati apakah target hanya gambar statis atau juga video.

**Luaran**
- Dokumen konteks proyek (`konteks.md`).
- Daftar kebutuhan fungsional dan nonfungsional.
- Batasan proyek yang disepakati kelompok.

**Kriteria selesai**
- Semua anggota memahami pipeline dan ruang lingkup yang sama.

### Tahap 2 — Studi pustaka dan penelusuran metode

**Aktivitas**
- Pelajari konsep deteksi objek, bounding box, cropping, transformasi grayscale, filtering, thresholding, morphology, OCR, dan evaluasi.
- Cari penelitian ALPR yang relevan, khususnya kondisi plat Indonesia.
- Catat alasan pemilihan metode dan keterbatasan tiap pendekatan.
- Prioritaskan artikel jurnal/prosiding lima tahun terakhir untuk memenuhi ketentuan referensi tugas; sumber dataset dan dokumentasi library dicatat terpisah sebagai referensi teknis.

**Luaran**
- Matriks studi pustaka: sumber, data, metode, hasil, kelebihan, keterbatasan, relevansi.
- Daftar pustaka awal.

**Kriteria selesai**
- Ada dasar ilmiah untuk pemilihan detektor, OCR, prapemrosesan, dan metrik.

### Tahap 3 — Akuisisi dan audit dataset

**Aktivitas**
1. Periksa sumber dataset yang disarankan pada bagian 5.
2. Baca lisensi, syarat penggunaan, atribusi, dan izin redistribusi sebelum mengunduh atau membagikan data.
3. Unduh versi dataset dan simpan catatan sumber, tanggal akses, versi, lisensi, serta jumlah file.
4. Audit struktur folder, format gambar, format anotasi, label kelas, dan kualitas bounding box.
5. Periksa gambar rusak, duplikat, label kosong, serta distribusi kondisi gambar.
6. Pastikan dataset deteksi memiliki bounding box plat. Untuk evaluasi OCR, siapkan transkripsi teks plat sebagai ground truth; anotasi bounding box saja tidak cukup.
7. Jika ground truth teks tidak tersedia, anotasi manual subset yang layak dengan prosedur pemeriksaan ulang oleh anggota lain.
8. Jika mengumpulkan foto sendiri, hindari menampilkan data pribadi yang tidak diperlukan, minta izin bila relevan, dan samarkan/kelola data sesuai aturan kampus serta ketentuan privasi.

**Luaran**
- Dataset mentah yang tidak diubah.
- Catatan dataset (`data/README.md` atau metadata).
- Anotasi tervalidasi dan daftar ground truth OCR.
- Laporan audit dataset.

**Kriteria selesai**
- Sumber, lisensi, jumlah citra, format anotasi, dan kelengkapan label dapat dijelaskan.
- Dataset dapat dipakai secara sah untuk eksperimen.

### Tahap 4 — Menyiapkan dataset dan pembagian data

**Aktivitas**
- Normalisasi format anotasi ke format yang diperlukan model.
- Periksa koordinat bounding box agar berada di dalam dimensi gambar.
- Pisahkan data train, validation, dan test.
- Hindari kebocoran data: foto dari kendaraan/rekaman yang sama sebaiknya tidak tersebar ke train dan test jika hal itu membuat pengujian terlalu mudah.
- Tetapkan seed dan catat rasio pembagian.
- Untuk OCR, pisahkan data berdasarkan gambar/sumber yang sama dan pastikan setiap sampel memiliki teks acuan.

**Rencana awal**
- Gunakan pembagian train/validation/test, misalnya 70/15/15, hanya jika ukuran dan struktur dataset memungkinkan.
- Jika dataset kecil, pertimbangkan skema yang lebih sesuai dan jelaskan alasannya.
- Jangan mengubah test set berulang kali untuk memilih parameter.

**Luaran**
- Dataset siap pakai.
- File konfigurasi dataset.
- Statistik jumlah data per split dan kelas.

### Tahap 5 — Menyiapkan lingkungan pengembangan

**Aktivitas**
- Buat repository Git.
- Buat virtual environment.
- Pasang dependency secara bertahap.
- Catat versi Python, library, OS, dan spesifikasi perangkat.
- Siapkan file konfigurasi agar path dan parameter tidak tersebar di kode.

**Dependency kandidat**
- `opencv-python`: pemrosesan citra.
- `ultralytics`: pelatihan/inferensi YOLO jika dipilih.
- `easyocr`: OCR baseline.
- `streamlit`: antarmuka aplikasi.
- `numpy`, `pandas`, `Pillow`: pengolahan data dan gambar.
- `matplotlib`, `seaborn`: visualisasi analisis.
- `scikit-learn`, `jiwer`: metrik yang relevan.
- `pytest`: pengujian unit/integrasi.

Versi final dicatat dalam `requirements.txt` atau file lock. Periksa kompatibilitas PyTorch/CUDA sebelum memasang paket GPU.

**Luaran**
- Environment yang dapat direproduksi.
- `requirements.txt`.
- README instalasi awal.

### Tahap 6 — Implementasi modul deteksi plat

**Aktivitas**
- Siapkan model deteksi dan konfigurasi bobot.
- Buat fungsi inferensi yang menerima gambar dan mengembalikan bounding box, confidence, dan kelas.
- Terapkan validasi koordinat dan penanganan ketika tidak ada plat terdeteksi.
- Crop plat dari gambar asli, bukan dari gambar anotasi.
- Gambar bounding box pada salinan gambar untuk visualisasi.

**Luaran**
- Modul detector.
- Hasil visual deteksi dan crop.
- Log confidence dan waktu inferensi.

**Kriteria selesai**
- Modul dapat dipanggil tanpa bergantung pada antarmuka Streamlit.

### Tahap 7 — Implementasi modul prapemrosesan

**Aktivitas**
- Buat fungsi-fungsi kecil untuk grayscale, resize, denoising, contrast enhancement, thresholding, dan morphology.
- Sediakan konfigurasi parameter.
- Simpan beberapa varian hasil untuk dibandingkan.
- Uji pada kondisi terang, gelap, miring, buram, dan kontras rendah.

**Luaran**
- Modul preprocessing.
- Contoh citra sebelum/sesudah tiap operasi.
- Catatan parameter yang diuji.

### Tahap 8 — Implementasi modul OCR dan post-processing

**Aktivitas**
- Jalankan OCR pada crop asli dan crop hasil prapemrosesan.
- Ambil teks dan confidence jika tersedia.
- Bersihkan spasi/karakter yang tidak relevan tanpa mengubah hasil secara spekulatif.
- Terapkan validasi pola plat Indonesia hanya sebagai pemeriksaan format opsional; format yang tidak cocok tidak otomatis berarti prediksi salah.
- Simpan prediksi mentah dan hasil normalisasi agar evaluasi transparan.

**Luaran**
- Modul OCR.
- Fungsi normalisasi teks.
- Contoh hasil benar dan gagal.

### Tahap 9 — Integrasi pipeline

**Aktivitas**
- Buat satu service/pipeline yang mengorkestrasi validasi input, deteksi, cropping, preprocessing, OCR, dan penyusunan hasil.
- Pisahkan logika pemrosesan dari kode UI.
- Definisikan format hasil yang konsisten, termasuk status, bounding box, teks, confidence, durasi, dan pesan error.
- Pastikan error pada satu gambar tidak membuat seluruh aplikasi berhenti.

**Luaran**
- Pipeline end-to-end yang dapat dipanggil dari Python.
- Pengujian integrasi.

### Tahap 10 — Pengembangan antarmuka aplikasi

**Aktivitas**
- Buat halaman upload gambar.
- Tampilkan gambar input.
- Tampilkan gambar dengan bounding box.
- Tampilkan crop plat dan hasil prapemrosesan.
- Tampilkan teks OCR, confidence bila tersedia, dan waktu pemrosesan.
- Sediakan pesan yang jelas jika plat tidak ditemukan atau OCR gagal.
- Tambahkan halaman evaluasi/riwayat eksperimen bila waktu memungkinkan.

**Luaran**
- Aplikasi Streamlit yang dapat dijalankan lokal.
- Petunjuk penggunaan.

### Tahap 11 — Pengujian dan evaluasi

Tetapkan skenario dan metrik sebelum menafsirkan hasil. Laporkan jumlah sampel dan kondisi pengujian.

**Evaluasi deteksi**
- Precision dan recall deteksi.
- IoU untuk membandingkan bounding box prediksi dengan ground truth.
- mAP jika sesuai dengan framework dan anotasi yang digunakan.

**Evaluasi OCR**
- Character Error Rate (CER).
- Word Error Rate (WER) atau Exact Match Accuracy untuk satu string plat.
- Laporkan normalisasi teks yang digunakan sebelum perbandingan.

**Evaluasi aplikasi**
- Waktu pemrosesan per gambar.
- Persentase gambar yang berhasil diproses.
- Uji file tidak valid, gambar tanpa kendaraan/plat, lebih dari satu kendaraan, gambar buram, gelap, dan miring.

**Catatan**
- Accuracy klasifikasi biasa tidak menggantikan metrik deteksi dan OCR.
- Jangan menyimpulkan model baik hanya dari beberapa contoh yang dipilih.
- Simpan hasil setiap eksperimen dan gunakan test set yang tidak dipakai untuk tuning.

**Luaran**
- Tabel metrik, grafik, contoh keberhasilan/kegagalan, dan log pengujian.
- Analisis kesalahan serta keterbatasan.

### Tahap 12 — Dokumentasi, laporan, dan demonstrasi

**Aktivitas**
- Dokumentasikan arsitektur, alur algoritma, library, parameter, sumber data, lisensi, dan cara menjalankan aplikasi.
- Siapkan screenshot input, bounding box, crop, preprocessing, dan output OCR.
- Susun laporan sesuai struktur pada instruksi tugas.
- Susun draf artikel ilmiah sesuai struktur tugas.
- Siapkan presentasi dan skenario demonstrasi.
- Cantumkan pembagian kontribusi anggota kelompok.

**Luaran**
- Source code dan README.
- Laporan proyek akhir.
- Draf artikel ilmiah.
- Dokumentasi presentasi/demo dan bukti evaluasi.

## 5. Cara memperoleh dataset

Dataset yang disarankan sebagai titik awal:

### 5.1 Indonesian License Plate — Roboflow Universe

Tautan: https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate

Halaman dataset yang ditemukan menampilkan 534 gambar dan lisensi CC BY 4.0. Dataset ini relevan untuk deteksi plat Indonesia. Periksa kembali versi dataset, jumlah aktual saat diunduh, anotasi, dan lisensi pada halaman versi yang dipilih.

**Langkah memperoleh:**
1. Buka tautan dataset.
2. Periksa tab dataset/version dan contoh anotasi.
3. Pastikan kelas yang tersedia benar-benar area plat.
4. Periksa lisensi dan ketentuan atribusi.
5. Pilih versi dan format ekspor yang sesuai, misalnya YOLO.
6. Unduh dataset melalui fasilitas ekspor yang disediakan.
7. Simpan metadata versi dan tanggal unduh.
8. Audit label sebelum pelatihan.

### 5.2 Indonesian License Plate Detection Dataset — Zenodo

Tautan: https://zenodo.org/records/15605718

Catatan pada halaman menyebut dataset berisi citra plat kendaraan Indonesia dengan anotasi kompatibel YOLOv11, satu kelas `license-plate`, dan berasal dari proyek Roboflow publik. Halaman Zenodo menyebut dataset ini tidak dimodifikasi setelah diunduh. Periksa file, metadata, dan lisensi/ketentuan yang tertera di record sebelum digunakan.

**Langkah memperoleh:**
1. Buka record Zenodo.
2. Periksa deskripsi, file, versi, DOI, dan lisensi.
3. Unduh arsip dataset.
4. Ekstrak ke direktori `data/raw/`.
5. Cocokkan pasangan gambar dan label serta cek konfigurasi kelas.
6. Catat DOI dan sumber asli Roboflow dalam dokumentasi dataset.

### 5.3 UFPR-ALPR — dataset pembanding

Informasi: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/  
Ketentuan: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/

UFPR-ALPR memuat 4.500 gambar beranotasi dari 150 kendaraan, termasuk lokasi plat dan karakter. Dataset ini berasal dari Brasil, sehingga tidak mewakili karakteristik plat Indonesia. Aksesnya untuk penelitian akademik nonkomersial; redistribusi, modifikasi, dan penggunaan komersial dibatasi tanpa izin. Halaman lisensi meminta permohonan melalui email institusi kepada penulis utama dan menyatakan proses dapat memerlukan 1–3 hari kerja.

Gunakan sebagai pembanding/rujukan metodologi hanya jika sesuai kebutuhan dan ketentuan. Jangan mengunggah ulang atau membagikan dataset ini.

### 5.4 Dataset OCR dan ground truth teks

Dataset deteksi berisi bounding box tidak otomatis menyediakan transkripsi teks plat. Untuk mengukur kualitas OCR, setiap crop evaluasi harus memiliki teks acuan yang benar.

Jika sumber utama tidak menyediakan label teks:
- Buat subset OCR yang terpisah dan terukur.
- Minta dua anotator menyalin teks secara independen, lalu selesaikan perbedaan melalui verifikasi.
- Simpan teks acuan dalam CSV dengan kolom seperti `image`, `plate_text`, `source`, dan `review_status`.
- Batasi akses ke data yang dapat mengidentifikasi pemilik kendaraan; gunakan data publik yang sah atau gambar yang dikumpulkan dengan izin.
- Pisahkan gambar yang sama/serupa antar split agar tidak terjadi kebocoran.

## 6. Rencana pengujian

| ID | Skenario | Hasil yang diharapkan |
|---|---|---|
| T01 | Gambar kendaraan jelas dan plat terlihat | Plat terdeteksi dan teks terbaca |
| T02 | Gambar kendaraan tanpa plat terlihat | Sistem menampilkan status plat tidak ditemukan |
| T03 | Gambar buram | Sistem tetap berjalan; kesalahan dicatat |
| T04 | Gambar gelap/kontras rendah | Sistem tetap berjalan dan hasil preprocessing dapat diamati |
| T05 | Plat miring | Sistem mencoba memproses; hasil dicatat |
| T06 | Lebih dari satu kendaraan | Semua deteksi yang didukung ditampilkan |
| T07 | File bukan gambar | Sistem menolak dengan pesan yang informatif |
| T08 | Resolusi sangat kecil/besar | Sistem menangani input tanpa crash |
| T09 | OCR gagal membaca | Sistem menampilkan status gagal, bukan teks rekaan |

## 7. Risiko dan mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Dataset Indonesia terbatas | Model kurang general | Gabungkan sumber yang lisensinya sesuai dan tambah data berizin |
| Tidak ada label teks OCR | Evaluasi OCR tidak dapat dilakukan | Anotasi ground truth subset secara manual |
| Dataset tidak seimbang | Model bias pada kondisi tertentu | Analisis distribusi dan tambah sampel yang kurang |
| Plat buram/tertutup | Deteksi/OCR gagal | Catat sebagai batasan; uji preprocessing |
| Dependency/GPU tidak kompatibel | Instalasi/inferensi gagal | Kunci versi dan sediakan mode CPU jika memungkinkan |
| Kebocoran data | Metrik terlalu optimistis | Pisahkan berdasarkan kendaraan/sesi/sumber bila metadata tersedia |
| Lisensi/privasi tidak jelas | Penggunaan data tidak sah | Audit lisensi dan dokumentasikan asal data |

## 8. Kriteria keberhasilan proyek

Proyek dinyatakan siap didemonstrasikan jika:
- Aplikasi dapat dijalankan dari instruksi instalasi yang terdokumentasi.
- Pengguna dapat mengunggah gambar dan melihat hasil pipeline.
- Proses deteksi, crop, preprocessing, dan OCR dapat dijelaskan.
- Dataset, jumlah data, sumber, lisensi, dan pembagian data terdokumentasi.
- Evaluasi menggunakan metrik yang sesuai dan menyertakan analisis kesalahan.
- Source code, laporan, draf artikel, dan bukti demonstrasi tersedia.

Target angka performa ditetapkan setelah baseline dan audit dataset, bukan ditentukan tanpa eksperimen.

## 9. Checklist pekerjaan

- [ ] Menyetujui ruang lingkup proyek dan pembagian tugas.
- [ ] Membaca studi pustaka dan menyusun matriks penelitian.
- [ ] Memilih dataset serta memeriksa lisensinya.
- [ ] Mengunduh dan mengaudit dataset.
- [ ] Menyiapkan ground truth teks untuk evaluasi OCR.
- [ ] Membuat environment dan memasang dependency.
- [ ] Mengimplementasikan modul deteksi.
- [ ] Mengimplementasikan modul preprocessing.
- [ ] Mengimplementasikan modul OCR.
- [ ] Mengintegrasikan pipeline.
- [ ] Membuat UI Streamlit.
- [ ] Menulis unit test dan integration test.
- [ ] Menjalankan evaluasi dan menganalisis error.
- [ ] Menyelesaikan README, laporan, artikel, dan demo.

## 10. Referensi awal teknis dan dataset

1. Roboflow Universe, *Indonesian_License_Plate*: https://universe.roboflow.com/dataset-pyufc/indonesian_license_plate
2. Zenodo, *Indonesian License Plate Detection Dataset (YOLOv11 Format)*: https://zenodo.org/records/15605718
3. UFPR VRI, *UFPR-ALPR Dataset*: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
4. UFPR VRI, *UFPR-ALPR License Agreement*: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/
5. Laroca et al. (2018), “A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector,” IJCNN. Referensi dataset UFPR-ALPR.

> Daftar ini adalah titik awal sumber dataset dan teknis, bukan pengganti ketentuan minimal 10 referensi (termasuk minimal 5 artikel jurnal/prosiding lima tahun terakhir) pada laporan akhir. Lengkapi studi pustaka dengan penelitian terbaru yang benar-benar dibaca dan relevan.
