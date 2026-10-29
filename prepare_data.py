"""
Fase 2 - Persiapan data untuk YOLOv8 klasifikasi.

Membagi dataset asli (Original Dataset) menjadi train/val/test = 70/20/10
secara deterministik dan stratified per kelas, lalu menyalinnya ke data_cls/.

Struktur keluaran (format ImageFolder yang dipakai ultralytics):
    data_cls/
        train/<Kelas>/*.JPG
        val/<Kelas>/*.JPG
        test/<Kelas>/*.JPG

Catatan:
- Folder "Augmented Dataset" sengaja TIDAK dipakai, agar tidak terjadi
  data leakage (gambar augmentasi berasal dari gambar asli).
- Augmentasi training dilakukan otomatis oleh ultralytics.
- Split deterministik: seed tetap + sampel diurutkan, jadi hasilnya sama
  setiap kali script dijalankan ulang.

Pemakaian:
    python prepare_data.py
    python prepare_data.py --overwrite
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SRC_DIR = (
    ROOT
    / "dataset"
    / "Chili Leaf Disease Original Dataset"
    / "Chili Leaf Disease Original Dataset"
)
OUT_DIR = ROOT / "data_cls"
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.20, 0.10)  # train / val / test
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(src: Path) -> dict[str, list[Path]]:
    """Kumpulkan gambar per kelas (subfolder) dari dataset sumber."""
    if not src.is_dir():
        sys.exit(f"[ERROR] Folder dataset tidak ditemukan:\n  {src}")

    per_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        files = sorted(
            f
            for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        )
        if files:
            per_class[class_dir.name] = files

    if not per_class:
        sys.exit(f"[ERROR] Tidak ada gambar di {src}")
    return per_class


def split_counts(n: int, ratios: tuple[float, ...]) -> tuple[int, int, int]:
    """
    Hitung jumlah gambar per split untuk n gambar.

    Angka dibulatkan agar total tetap n, dan tiap split yang porsinya > 0
    minimal berisi 1 gambar (kelas kecil tetap terwakili di test).
    """
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    n_test = n - n_train - n_val

    # Jaga agar kelas kecil tetap punya sampel di val & test.
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        # Koreksi total jika pembulatan membuat jumlah melebihi n.
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            else:
                n_test -= 1
        while n_train + n_val + n_test < n:
            n_train += 1

    return n_train, n_val, n_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset cabai 70/20/10")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hapus data_cls/ lama sebelum membuat split baru",
    )
    args = parser.parse_args()

    if OUT_DIR.exists():
        if not args.overwrite:
            sys.exit(
                f"[ERROR] {OUT_DIR.name}/ sudah ada. "
                "Jalankan dengan --overwrite untuk membuat ulang."
            )
        print(f"[INFO] Menghapus {OUT_DIR} ...")
        shutil.rmtree(OUT_DIR)

    per_class = collect_images(SRC_DIR)
    rng = random.Random(SEED)

    print(f"[INFO] Sumber : {SRC_DIR}")
    print(f"[INFO] Tujuan : {OUT_DIR}")
    print(f"[INFO] Split  : train {RATIOS[0]:.0%} / val {RATIOS[1]:.0%} / test {RATIOS[2]:.0%}")
    print(f"[INFO] Seed   : {SEED}\n")

    totals = {s: 0 for s in SPLITS}
    header = f"{'Kelas':<24}{'Total':>7}{'Train':>7}{'Val':>6}{'Test':>6}"
    print(header)
    print("-" * len(header))

    for class_name, files in per_class.items():
        # Urut + acak dengan seed tetap -> deterministik.
        shuffled = files[:]
        rng.shuffle(shuffled)

        n_train, n_val, _n_test = split_counts(len(shuffled), RATIOS)
        partitions = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }

        for split, items in partitions.items():
            dest = OUT_DIR / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in items:
                shutil.copy2(img, dest / img.name)
            totals[split] += len(items)

        print(
            f"{class_name:<24}{len(shuffled):>7}"
            f"{len(partitions['train']):>7}"
            f"{len(partitions['val']):>6}"
            f"{len(partitions['test']):>6}"
        )

    print("-" * len(header))
    grand = sum(totals.values())
    print(
        f"{'TOTAL':<24}{grand:>7}{totals['train']:>7}"
        f"{totals['val']:>6}{totals['test']:>6}"
    )
    print(f"\n[OK] Selesai. {grand} gambar tersalin ke {OUT_DIR}")


if __name__ == "__main__":
    main()
