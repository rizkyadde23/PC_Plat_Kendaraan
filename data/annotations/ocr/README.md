# Kerangka Anotasi Ground Truth OCR — ALPRO (Milestone 4)

Direktori ini menyiapkan **kerangka** anotasi teks plat nomor (ground truth OCR). 
**Tidak ada teks ground truth yang dibuat otomatis/asumtif** pada Milestone 4.

## Tujuan
- Menyediakan skema/format dan daftar kandidat untuk anotasi manual transkripsi plat.
- Evaluasi OCR kuantitatif (CER/WER/Exact Match) hanya valid pada sampel dengan `plate_text_gt` yang terisi dan **terverifikasi**.

## Berkas
- `ocr_annotations_template.csv`: template skema anotasi.
- `ocr_annotation_candidates.csv`: daftar kandidat anotasi (50 sampel, deterministik, seed=42, proporsional per split) dari gambar non-empty label sesuai manifest M4. Kolom teks **kosong** dan harus diisi manual.
- `ocr_annotations.csv`: (opsional, dibuat saat Anda mulai mengisi) salin dari kandidat/template dan isi hasil anotasi.

## Skema CSV (6.5 C.i)
Kolom: `image_path,split,plate_text_gt,unreadable,notes,verified`

- `image_path`: path relatif terhadap root proyek ke gambar sumber.
- `split`: `train` / `valid` / `test` (split bawaan).
- `plate_text_gt`: transkripsi teks plat **hasil anotasi manual**. Kosongkan jika belum dianotasi.
- `unreadable`: `true/false` (atau `1/0`) — `true` jika teks plat tidak terbaca/ambigu secara visual.
- `notes`: catatan anotator (opsional, mis. kondisi buram/tertutup/ganda).
- `verified`: `true/false` (atau `1/0`) — `true` jika sudah diperiksa/QA.

## Konvensi Anotasi (disarankan)
- Gunakan **huruf kapital** sesuai tampilan plat (A–Z, 0–9).
- Pertahankan **spasi** sebagaimana terbaca pada plat (jika ragu, catat di `notes`).
- Jangan menebak karakter yang tidak terbaca; tandai `unreadable=true` dan jelaskan di `notes`.
- Jangan menyalin hasil OCR model sebagai ground truth.
- Simpan hasil anotasi di direktori ini (`data/annotations/ocr/`), bukan di `data/raw/`.

## Catatan
- Subset kandidat dipilih dari gambar dengan label deteksi non-empty (sesuai manifest M4). Label kosong & duplikat telah didokumentasikan terpisah.
- Crop plat otomatis belum tersedia (direncanakan Milestone 6); anotasi dapat dilakukan berbasis gambar sumber (dan/atau crop manual yang disimpan di lokasi kerja turunan bila diperlukan).
