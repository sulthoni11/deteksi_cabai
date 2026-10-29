"""
Fase 4 - Web app Streamlit: Deteksi Penyakit Daun Cabai (YOLOv8n-cls).

Jalankan:
    venv\\Scripts\\streamlit run app.py

Butuh model hasil training di runs/cabai_cls/weights/best.pt
(dibuat oleh train.py).
"""

from __future__ import annotations

import csv
import io
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

APP_NAME = "Deteksi Penyakit Daun Cabai"
APP_SUBTITLE = "Klasifikasi citra daun cabai berbasis YOLOv8"
APP_VERSION = "1.0.0"
DATASET_DOI = "10.17632/w9mr3vf56s.1"

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
MODEL_CANDIDATES = [
    ROOT / "runs" / "cabai_cls" / "weights" / "best.pt",
    ROOT / "model" / "best.pt",
]
MAX_MB = 5
ALLOWED_EXT = {"jpg", "jpeg", "png"}

# ---------------------------------------------------------------------------
# Metadata kelas: nama model (folder dataset) -> tampilan & penanganan
# ---------------------------------------------------------------------------
KELAS: dict[str, dict[str, str]] = {
    "Healthy Leaf": {
        "nama": "Daun Sehat",
        "gambar": "daun_sehat.jpg",
        "gejala": "Daun hijau merata, tidak ada bercak, keriting, atau perubahan warna.",
        "penanganan": (
            "Tidak perlu tindakan khusus. Lanjutkan penyiraman rutin, pemupukan "
            "berimbang (NPK), dan sanitasi gulma agar tanaman tetap sehat."
        ),
    },
    "Bacterial Spot": {
        "nama": "Bercak Bakteri",
        "gambar": "bercak_bakteri.jpg",
        "gejala": "Bercak kecil berair yang berubah cokelat kehitaman, kadang berlubang.",
        "penanganan": (
            "Buang dan musnahkan daun terinfeksi, hindari penyiraman dari atas "
            "(percikan menyebarkan bakteri), aplikasikan bakterisida berbahan "
            "tembaga, dan lakukan rotasi tanaman."
        ),
    },
    "Cercospora Leaf Spot": {
        "nama": "Bercak Daun Cercospora",
        "gambar": "bercak_cercospora.jpg",
        "gejala": "Bercak bulat dengan pusat abu-abu terang dan tepi cokelat kemerahan.",
        "penanganan": (
            "Pangkas daun sakit, perbaiki sirkulasi udara antar tanaman, kurangi "
            "kelembapan kanopi, dan gunakan fungisida (mis. klorotalonil atau "
            "mancozeb) sesuai dosis anjuran."
        ),
    },
    "Curl Virus": {
        "nama": "Virus Keriting (Curl Virus)",
        "gambar": "virus_keriting.jpg",
        "gejala": "Daun mengeriting/menggulung, menebal, menguning, pertumbuhan kerdil.",
        "penanganan": (
            "Cabut dan musnahkan tanaman terinfeksi (tidak ada obat untuk virus), "
            "kendalikan kutu kebul sebagai vektor dengan perangkap kuning dan "
            "insektisida, serta gunakan varietas tahan."
        ),
    },
    "Nutrition Deficiency": {
        "nama": "Defisiensi Nutrisi",
        "gambar": "defisiensi_nutrisi.jpg",
        "gejala": "Daun menguning tidak merata, urat daun pucat, pertumbuhan lambat.",
        "penanganan": (
            "Lakukan uji tanah/daun untuk memastikan unsur yang kurang, berikan "
            "pupuk lengkap makro+ mikro sesuai dosis, dan perbaiki pH tanah agar "
            "unsur hara dapat diserap."
        ),
    },
    "White spot": {
        "nama": "Bercak Putih",
        "gambar": "bercak_putih.jpg",
        "gejala": "Bercak keputihan/berkapur pada permukaan daun, dapat meluas.",
        "penanganan": (
            "Buang daun bergejala, jaga kelembapan dan jarak tanam, serta "
            "aplikasikan fungisida bila serangan meluas."
        ),
    },
}

CUSTOM_CSS = """
<style>
    /* Palet: hijau muda + putih */
    .stApp { background: #FFFFFF; }
    [data-testid="stSidebar"] { background: #EAF7E9; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] * { color: #1B4332; }

    h1, h2, h3, h4 { color: #1B5E20 !important; }

    .app-title { font-size: 2rem; font-weight: 800; color: #1B5E20; margin-bottom: 0; }
    .app-sub { color: #4E7A5A; margin-top: -0.2rem; margin-bottom: 1rem; }

    .badge {
        display: inline-block; background: #4CAF50; color: #fff;
        padding: 2px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: .04em;
    }
    .card {
        background: #F6FBF4; border: 1px solid #CDE8C8;
        border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;
    }
    .empty-state {
        border: 2px dashed #A5D6A7; border-radius: 16px;
        background: #F7FCF6; padding: 34px 20px; text-align: center;
    }
    .empty-state h3 { margin: 0 0 .3rem 0; }
    .empty-state p { color: #55705C; margin: 0; }

    .hasil-label { font-size: 1.75rem; font-weight: 800; color: #1B5E20; line-height: 1.2; }
    .hasil-conf { color: #4E7A5A; font-size: .95rem; margin-bottom: .4rem; }
    .pill {
        display:inline-block; background:#DCEDC8; color:#33691E;
        border-radius:999px; padding:2px 10px; font-size:.75rem; font-weight:600;
    }

    /* Tombol utama hijau */
    .stButton > button, .stDownloadButton > button {
        background: #4CAF50; color: #FFFFFF; border: none;
        border-radius: 10px; font-weight: 600; padding: .45rem 1rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #43A047; color: #FFFFFF;
    }
    .stButton > button:focus, .stDownloadButton > button:focus { color: #FFFFFF; }
</style>
"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model…")
def load_model(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def find_model() -> Path | None:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def predict(image_bytes: bytes) -> dict:
    """Jalankan inferensi pada byte gambar, kembalikan probabilitas tiap kelas."""
    model_path = find_model()
    model = load_model(str(model_path))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    t0 = time.perf_counter()
    results = model.predict(image, imgsz=224, verbose=False)
    elapsed = time.perf_counter() - t0

    probs = results[0].probs
    names: dict[int, str] = results[0].names
    scores = {names[i]: float(v) for i, v in enumerate(probs.data.tolist())}
    top1_idx = int(probs.top1)

    return {
        "scores": scores,
        "top1": names[top1_idx],
        "confidence": float(probs.top1conf),
        "elapsed_ms": elapsed * 1000,
    }


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def display_name(class_key: str) -> str:
    return KELAS.get(class_key, {}).get("nama", class_key)


def validasi(up) -> str | None:
    """Kembalikan pesan error bila file tidak valid, None bila lolos."""
    ext = up.name.rsplit(".", 1)[-1].lower() if "." in up.name else ""
    if ext not in ALLOWED_EXT:
        return f"Format file .{ext or '?'} tidak didukung. Gunakan JPG atau PNG."
    if up.size > MAX_MB * 1024 * 1024:
        return f"Ukuran file {up.size / 1024 / 1024:.1f} MB melebihi batas {MAX_MB} MB."
    return None


def set_gambar(nama: str, data: bytes) -> None:
    st.session_state.gambar = {"nama": nama, "data": data}
    st.session_state.hasil = None


def reset() -> None:
    for key in ("gambar", "hasil", "error"):
        st.session_state.pop(key, None)
    st.session_state["last_upload_id"] = None


def buat_csv(hasil: dict, nama_file: str) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Nama File", nama_file])
    writer.writerow(["Waktu Analisis", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Model", "YOLOv8n-cls"])
    writer.writerow([])
    writer.writerow(["Peringkat", "Kelas", "Nama Tampilan", "Probabilitas"])
    for i, (kelas, skor) in enumerate(hasil["scores_top"], start=1):
        writer.writerow([i, kelas, display_name(kelas), f"{skor:.4f}"])
    return buf.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

for key, default in (("gambar", None), ("hasil", None), ("error", None),
                     ("last_upload_id", None)):
    st.session_state.setdefault(key, default)

model_path = find_model()

# ------------------------------ Sidebar -----------------------------------
with st.sidebar:
    st.markdown(
        f"<div class='app-title' style='font-size:1.35rem'>🌶️ {APP_NAME}</div>"
        "<div class='app-sub' style='font-size:.85rem'>Klasifikasi daun cabai</div>"
        "<span class='badge'>YOLOv8</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Prediksi di bawah nilai ini dianggap belum meyakinkan.",
    )

    st.markdown("### Tentang Penyakit")
    for key, info in KELAS.items():
        with st.expander(info["nama"]):
            gambar_kelas = ASSET_DIR / "penyakit" / info["gambar"]
            if gambar_kelas.exists():
                st.image(str(gambar_kelas), use_container_width=True)
            st.markdown(f"**Gejala:** {info['gejala']}")
            st.markdown(f"**Penanganan:** {info['penanganan']}")

    st.markdown("---")
    st.caption(f"Versi {APP_VERSION} · YOLOv8n-cls · imgsz 224")
    st.caption(f"Dataset: Mendeley · DOI {DATASET_DOI}")

# ------------------------------ Header ------------------------------------
st.markdown(f"<div class='app-title'>{APP_NAME}</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='app-sub'>{APP_SUBTITLE} — unggah foto daun cabai untuk "
    "mengetahui kondisi dan penanganannya.</div>",
    unsafe_allow_html=True,
)

if model_path is None:
    st.error(
        "Model belum tersedia. Jalankan `python train.py` terlebih dahulu "
        "sehingga `runs/cabai_cls/weights/best.pt` terbentuk."
    )
    st.stop()

# --------------------------- Input gambar ---------------------------------
up = st.file_uploader(
    "Seret gambar ke sini atau klik untuk upload",
    type=list(ALLOWED_EXT),
    help=f"JPG/PNG, maksimal {MAX_MB} MB",
)

if up is not None:
    fid = f"{up.name}:{up.size}"
    if st.session_state.get("last_upload_id") != fid:
        st.session_state["last_upload_id"] = fid
        err = validasi(up)
        if err:
            st.session_state["error"] = err
        else:
            st.session_state["error"] = None
            set_gambar(up.name, up.getvalue())
            st.session_state["last_upload_id"] = fid

if st.session_state.get("error"):
    st.markdown(
        f"<div class='card' style='border-color:#EF9A9A;background:#FFEBEE;"
        f"color:#B71C1C'>⚠️ {st.session_state['error']}</div>",
        unsafe_allow_html=True,
    )

# --------------------------- Empty state ----------------------------------
if st.session_state.get("gambar") is None:
    st.markdown(
        "<div class='empty-state'>"
        "<h3>Belum ada gambar</h3>"
        "<p>Seret foto daun cabai ke area di atas, atau pilih salah satu "
        "contoh di bawah.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Coba contoh gambar")
    contoh = [
        ("Daun Sehat", "contoh_daun_sehat.jpg"),
        ("Virus Keriting", "contoh_virus_keriting.jpg"),
        ("Bercak Cercospora", "contoh_bercak_cercospora.jpg"),
    ]
    cols = st.columns(3)
    for col, (label, file) in zip(cols, contoh):
        path = ASSET_DIR / "contoh" / file
        with col:
            if path.exists():
                st.image(str(path), use_container_width=True)
            if st.button(f"Gunakan: {label}", use_container_width=True):
                set_gambar(file, path.read_bytes())
                st.rerun()
    st.stop()

# ------------------------- Preview + Analisis ------------------------------
gambar = st.session_state["gambar"]
preview_col, action_col = st.columns([1.1, 1])

with preview_col:
    st.image(gambar["data"], caption=f"Pratinjau: {gambar['nama']}", use_container_width=True)

with action_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**File:** `{gambar['nama']}`")
    st.markdown(
        f"**Batas keyakinan:** {threshold:.0%}  \n"
        "Analisis akan membandingkan gambar dengan 6 kelas penyakit daun cabai."
    )
    analisis = st.button("🔍 Analisis", use_container_width=True, type="primary")
    if st.button("Coba Gambar Lain", use_container_width=True):
        reset()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if analisis:
    with st.spinner("Menganalisis…"):
        hasil = predict(gambar["data"])
    skor_urut = sorted(hasil["scores"].items(), key=lambda kv: kv[1], reverse=True)
    hasil["scores_top"] = skor_urut[:3]
    st.session_state["hasil"] = hasil
    st.session_state["hasil_nama"] = gambar["nama"]

# ------------------------------- Hasil ------------------------------------
hasil = st.session_state.get("hasil")
if hasil:
    st.markdown("---")
    top1 = hasil["top1"]
    conf = hasil["confidence"]
    info = KELAS.get(top1, {})
    yakin = conf >= threshold

    kiri, kanan = st.columns([1, 1])

    with kiri:
        st.image(gambar["data"], use_container_width=True)
        st.markdown(f"<div class='hasil-label'>{display_name(top1)}</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='hasil-conf'>Keyakinan model: {conf:.1%}</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(conf, 1.0))
        st.markdown(
            f"<span class='pill'>{'✓ Prediksi meyakinkan' if yakin else '⚠ Perlu verifikasi'}</span>",
            unsafe_allow_html=True,
        )

        if not yakin:
            st.warning(
                f"Prediksi di bawah ambang keyakinan ({threshold:.0%}). "
                "Coba foto lebih dekat, pastikan daun terlihat jelas dan "
                "pencahayaan cukup untuk hasil yang lebih baik."
            )

    with kanan:
        st.markdown(
            f"<div class='card'><b>Ringkasan</b><br>"
            f"<span style='font-size:1.05rem'>{display_name(top1)}</span><br>"
            f"Probabilitas tertinggi <b>{conf:.1%}</b><br>"
            f"Waktu inferensi <b>{hasil['elapsed_ms']:.0f} ms</b></div>",
            unsafe_allow_html=True,
        )
        if info:
            st.markdown(
                f"<div class='card'><b>Gejala umum</b><br>{info['gejala']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='card'><b>Saran penanganan</b><br>{info['penanganan']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Top-3 probabilitas")
        df = pd.DataFrame(
            {
                "Kelas": [display_name(k) for k, _ in hasil["scores_top"]],
                "Probabilitas": [round(v, 4) for _, v in hasil["scores_top"]],
            }
        ).set_index("Kelas")
        st.bar_chart(df, horizontal=True, color="#4CAF50")

        st.download_button(
            "⬇️ Unduh Hasil (CSV)",
            data=buat_csv(hasil, st.session_state.get("hasil_nama", gambar["nama"])),
            file_name=f"hasil_{Path(gambar['nama']).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )
