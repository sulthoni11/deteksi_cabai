# PRD — Aplikasi Deteksi Tanaman Cabai (YOLOv8) — v2

**Tanggal:** 29 Oktober 2026 | **Bentuk:** Web app (Streamlit) | **Status:** Dataset tersedia lokal (`dataset/`)

## 1. Keputusan Final

| Item | Keputusan |
|---|---|
| Tujuan | Deteksi/klasifikasi penyakit daun cabai dari foto, berbasis web |
| Arah Fase 1 | **Klasifikasi (yolov8n-cls)** — dataset siap tanpa anotasi; deteksi bbox = fase upgrade (Fase lanjutan) |
| Dataset | Mendeley `w9mr3vf56s` (DOI: 10.17632/w9mr3vf56s.1) — **sudah diunduh** di `dataset/` |
| Dataset kerja | `Chili Leaf Disease Original Dataset` — 1.856 img, 6 kelas: Healthy Leaf (458), Nutrition Deficiency (444), Curl Virus (423), Cercospora Leaf Spot (180), Bacterial Spot (156), White spot (195) |
| App | **Streamlit**, tema **hijau muda + putih**, nama aplikasi menyusul |
| Hardware | NVIDIA RTX 4050 Laptop 6GB (CUDA) + RAM 15,7 GB |
| Data uji lapangan | Opsional: foto sendiri ±100–200 img sebagai data uji generalisasi (kontribusi skripsi) |

## 2. Rencana Eksekusi

1. **Fase 0 — Dokumen:** tulis PRD v2 ini ✓
2. **Fase 1 — Environment:** venv + `ultralytics`, `streamlit`, `torch` (CUDA); verifikasi GPU terdeteksi torch
3. **Fase 2 — Data:** `prepare_data.py` — split 70/20/10 deterministik ke `data_cls/` (folder per kelas per split); augmentasi = bawaan Ultralytics (folder "Augmented" dari dataset TIDAK dipakai untuk val/test — hindari data leakage)
4. **Fase 3 — Training:** `yolov8n-cls.pt`, imgsz=224, epochs=50, batch=64 → confusion matrix + kurva acc/loss (bahan Bab IV)
5. **Fase 4 — Web App:** `app.py` + `.streamlit/config.toml` (tema hijau muda/putih) + 3 gambar contoh demo

## 3. Spesifikasi UI (Streamlit)

**Layout:** sidebar + kolom utama tunggal, bahasa Indonesia, responsif.

**Sidebar (hijau muda):**
- Logo/judul + badge "YOLOv8"
- Slider Confidence (default 0.25)
- Panel "Tentang Penyakit": accordion per kelas + gambar contoh + penanganan singkat
- Info versi + DOI dataset di bawah

**Kolom Utama:**
- **Upload:** dropzone "Seret gambar ke sini atau Klik Upload" (JPG/PNG ≤ 5 MB) + preview + 3 thumbnail contoh siap klik + tombol "Analisis"
- **Loading:** spinner "Menganalisis…"
- **Hasil:** 2 kolom — (kiri) gambar asli + label besar hasil + confidence bar; (kanan) kartu ringkasan, top-3 probabilitas (bar chart), tombol "Unduh Hasil (CSV)" & "Coba Gambar Lain"
- **Empty state:** dropzone menonjol; **Error state:** pesan inline merah (ukuran/format); **Prediksi tidak meyakinkan:** saran "foto lebih dekat/perjelas"

## 4. Metrik Keberhasilan

- Accuracy ≥ 85% pada test set; recall ≥ 80% untuk kelas kritis (Curl Virus) dan Healthy
- Inferensi < 1 detik (GPU) di app; maksimal 2 klik dari upload → hasil

## 5. Risiko & Catatan

- **Keamanan:** API key Roboflow ter-hardcode di `deteksi_cabai.py` → jangan ikut di-commit/kirim; pindah ke environment variable
- **Ketidakseimbangan kelas** (156–458 img) → monitor confusion matrix; augmentasi balanced di train
- **Fase lanjutan (upgrade deteksi):** anotasi bbox subset via Roboflow (akun sudah ada) → latih `yolov8n` detect → app menampilkan bounding box + counting
- Dataset "Augmented" (23.000+ img) tersedia jika perlu training tambahan; val/test tetap dari data Original

## 6. Milestone

| | Deliverable | Estimasi |
|---|---|---|
| M1 | Environment siap + GPU terverifikasi | 30 mnt |
| M2 | `data_cls/` + training selesai + confusion matrix | 1 jam |
| M3 | `app.py` jalan + tema + gambar demo | 2–3 jam |
| M4 | (Ops) foto lapangan + retrain | 1–2 hari |
| M5 | (Ops) upgrade ke deteksi (bbox + counting) | 3–5 hari |
