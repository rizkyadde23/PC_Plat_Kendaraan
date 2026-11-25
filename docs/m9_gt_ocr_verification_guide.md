# Panduan Verifikasi GT OCR — M9 ALPRO

Tujuan: menyiapkan ground truth teks plat yang **terverifikasi** agar evaluasi kuantitatif OCR (CER/WER/Exact Match) dapat diklaim secara valid.

## Berkas Terkait
- Kandidat: `data/annotations/ocr/ocr_annotation_candidates.csv` (50 baris, seed 42, proporsional split).
- Template: `data/annotations/ocr/ocr_annotations_template.csv`.
- Output anotasi kerja (opsional): salin ke `data/annotations/ocr/ocr_annotations.csv` lalu isi.

## Skema Kolom
`image_path,split,plate_text_gt,unreadable,notes,verified`

- `plate_text_gt`: isi teks plat terbaca; kosongkan jika belum dianotasi.
- `unreadable`: `true` jika tidak terbaca/ambigu.
- `verified`: `true` hanya setelah dicek ulang.
- `notes`: kondisi buram/tertutup/sudut/dua baris/dll.

## Aturan Anotasi
1. Gunakan huruf kapital A–Z dan angka 0–9; pertahankan spasi sesuai plat bila terbaca.
2. Jangan menebak karakter ambigu; tandai `unreadable=true` dan beri `notes`.
3. Jangan menyalin output OCR model sebagai GT.
4. Lakukan QA minimal 1 pass ulang sebelum `verified=true`.
5. Simpan perubahan di `data/annotations/ocr/`, **jangan ubah `data/raw/`**.

## Menjalankan Evaluasi Kuantitatif (setelah GT terisi)
```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe scripts/evaluate_ocr.py --experiment-id m9_ocr_gt_verified
```

Script `evaluate_ocr.py` hanya menghitung metrik pada baris yang memenuhi: `verified=true`, `unreadable!=true`, `plate_text_gt` non-empty.

## Kebijakan Klaim
- Tanpa GT terverifikasi: **tidak ada klaim CER/WER/Exact Match**.
- Dengan GT terverifikasi: laporkan jumlah sampel, split, dan kriteria inklusi secara eksplisit.