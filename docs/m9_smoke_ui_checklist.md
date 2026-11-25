# Checklist Smoke UI Streamlit — M9 ALPRO

Gunakan checklist ini untuk verifikasi manual UI (`streamlit run app.py`).  
Status M9: tidak ada klaim metrik OCR kuantitatif tanpa GT terverifikasi.

## Persiapan
- [ ] Aktifkan `.venv` dan jalankan `streamlit run app.py`.
- [ ] Pastikan `configs/pipeline.yaml` menunjuk ke bobot detektor yang valid.
- [ ] Siapkan 3 sampel: valid (plat jelas), no-detection known case, file invalid/non-image.

## Skenario Smoke
- [ ] **Upload valid**: gambar tampil, tombol proses aktif.
- [ ] **Deteksi**: annotated image tampil; bbox/confidence terlihat di detail deteksi.
- [ ] **Crop**: crop plat tampil saat deteksi ada.
- [ ] **Preprocessing**: hasil varian default (`original_crop`) tampil; intermediate bisa di-toggle.
- [ ] **OCR**: `text_raw`, `text_normalized`, confidence, status tampil.
- [ ] **No detection**: known case menampilkan status `no_detection` tanpa crash (`IMG-20180108-WA0024...` / `WA0104...`).
- [ ] **Invalid input**: file rusak/tidak terbaca menampilkan error ramah, app tetap berjalan.
- [ ] **Save opsional**: jika dicentang, artefak tersimpan ke `outputs/predictions/pipeline/<timestamp>/`.
- [ ] **Pipeline JSON**: expander menampilkan JSON tanpa image array.

## Bukti yang Disarankan
- [ ] Screenshot: input, annotated, crop, preprocessed, hasil OCR.
- [ ] Screenshot: kasus `no_detection`.
- [ ] Catat waktu proses/status jika relevan.

## Catatan
- Hasil OCR adalah prediksi model; **CER/WER/EMA belum diklaim** karena GT OCR terverifikasi belum tersedia.
- `data/raw/` tidak boleh diubah.