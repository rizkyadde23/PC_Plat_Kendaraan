# Checklist Smoke UI Streamlit — ALPRO

Gunakan checklist ini untuk verifikasi manual UI Streamlit setelah redesign final.

## Persiapan
- [ ] Aktifkan `.venv`.
- [ ] Jalankan `streamlit run app.py`.
- [ ] Pastikan `configs/pipeline.yaml` menunjuk ke bobot detektor yang valid.
- [ ] Siapkan 3 sampel: gambar valid (plat jelas), kasus no-detection, file invalid/non-image.

## Navigasi & Layout
- [ ] **Home/Landing page** tampil sebagai halaman pertama.
- [ ] Tombol **Mulai Pembacaan** (hero) membuka halaman Plate Recognition.
- [ ] Tombol **Start Recognition** (bagian bawah Home) berada di tengah dan membuka Plate Recognition.
- [ ] Sidebar **tidak tampil**.
- [ ] Halaman Plate Recognition memiliki tombol **Kembali ke Home** dan **Configuration**.

## Configuration Dialog
- [ ] Dialog terbuka dari tombol **Configuration** di halaman Plate Recognition.
- [ ] Nilai default tampil sesuai config aktif.
- [ ] **Save Configuration** menyimpan perubahan dan diterapkan pada proses berikutnya.
- [ ] **Reset to Default** mengembalikan nilai default.
- [ ] **Cancel** menutup dialog tanpa menyimpan perubahan draft.

## Plate Recognition
- [ ] Upload gambar valid menampilkan preview/input dengan baik.
- [ ] Tombol proses aktif setelah gambar valid dipilih.
- [ ] Saat proses berjalan, indikator loading tampil.
- [ ] **Recognition Result** tampil sebagai panel paling dominan.
- [ ] Hasil OCR, status, OCR confidence, detection confidence, dan jumlah deteksi tampil.
- [ ] Gambar deteksi (bounding box) tampil proporsional.
- [ ] Crop plat tampil saat deteksi berhasil.
- [ ] Hasil preprocessing tampil; langkah intermediate dapat dilihat jika tersedia.
- [ ] Detail teknis (timing, detections JSON, OCR candidates) tersedia di expander.

## Error/Empty State
- [ ] **No detection**: status jelas, tidak crash (`IMG-20180108-WA0024...` / `WA0104...` adalah contoh known case).
- [ ] **Empty OCR**: status jelas, pesan informatif.
- [ ] **Invalid input**: error ramah, aplikasi tetap bisa digunakan.
- [ ] Tidak ada traceback mentah yang tampil ke user.

## Save Result (opsional)
- [ ] Jika opsi simpan aktif, hasil dapat disimpan ke `outputs/predictions/pipeline/<timestamp>/`.
- [ ] File JSON/gambar hasil tersimpan sesuai ekspektasi.

## Catatan
- Hasil OCR adalah prediksi model; **CER/WER/Exact Match belum diklaim** tanpa GT terverifikasi.
- `data/raw/` tidak boleh diubah.
- Ambil screenshot Home, Recognition Result, detection/crop/preprocess, dan kasus no-detection untuk dokumentasi/demo.
