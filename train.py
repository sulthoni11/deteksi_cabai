"""
Fase 3 - Training + evaluasi YOLOv8n-cls untuk klasifikasi penyakit daun cabai.

Menghasilkan (untuk bahan Bab IV):
    runs/cabai_cls/weights/best.pt          -> model terbaik
    runs/cabai_cls/results.csv              -> log per epoch
    runs/cabai_cls/laporan/kurva_acc_loss.png
    runs/cabai_cls/laporan/confusion_matrix_test.png
    runs/cabai_cls/laporan/metrics_test.json
    runs/cabai_cls/laporan/per_class_test.csv

Pemakaian:
    python train.py                 # train sesuai PRD (50 epoch)
    python train.py --epochs 10     # uji cepat
    python train.py --eval-only     # evaluasi ulang model yang sudah ada
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_cls"
RUNS_DIR = ROOT / "runs"
RUN_NAME = "cabai_cls"
RUN_DIR = RUNS_DIR / RUN_NAME
REPORT_DIR = RUN_DIR / "laporan"
BEST = RUN_DIR / "weights" / "best.pt"

MODEL = "yolov8n-cls.pt"
IMGSZ = 224
EPOCHS = 50
BATCH = 64
SEED = 0


def plot_curves(results_csv: Path, out_png: Path) -> bool:
    """Kurva accuracy top-1 dan loss (train vs val) dari results.csv."""
    if not results_csv.exists():
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with results_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # ultralytics kadang menulis spasi di header
        reader.fieldnames = [c.strip() for c in reader.fieldnames or []]
        for row in reader:
            rows.append({(k or "").strip(): v for k, v in row.items()})
    if not rows:
        return False

    def series(key: str) -> list[float]:
        out = []
        for r in rows:
            try:
                out.append(float(r[key]))
            except (KeyError, TypeError, ValueError):
                out.append(float("nan"))
        return out

    epochs = [int(float(r.get("epoch", i + 1))) for i, r in enumerate(rows)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(epochs, series("metrics/accuracy_top1"), label="Val Top-1", linewidth=2)
    ax.plot(epochs, series("metrics/accuracy_top5"), label="Val Top-5", linewidth=2)
    ax.set_title("Akurasi Validasi")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Akurasi")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, series("train/loss"), label="Train Loss", linewidth=2)
    ax.plot(epochs, series("val/loss"), label="Val Loss", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Kurva Training YOLOv8n-cls - Penyakit Daun Cabai")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def evaluate(weights: Path) -> None:
    """Evaluasi pada test set 10% + simpan metrik dan confusion matrix."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA_DIR),
        split="test",
        imgsz=IMGSZ,
        workers=0,
        project=str(RUN_DIR),
        name="val",
        exist_ok=True,
        verbose=True,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Nama kelas tersimpan di model (dict index -> nama), bukan di objek metrics.
    names = [model.names[i] for i in sorted(model.names)]

    # Ringkasan metrik
    summary = {
        "model": weights.name,
        "dataset": "Chili Leaf Disease Original Dataset",
        "doi": "10.17632/w9mr3vf56s.1",
        "split": "test (10%)",
        "imgsz": IMGSZ,
        "top1": float(metrics.top1),
        "top5": float(metrics.top5),
        "classes": names,
    }

    # Confusion matrix ultralytics berindeks [prediksi, aktual]:
    #   recall    = diagonal / jumlah kolom (total aktual per kelas)
    #   precision = diagonal / jumlah baris (total prediksi per kelas)
    cm = metrics.confusion_matrix.matrix
    per_class = []
    for i, name in enumerate(names):
        support = float(cm[:, i].sum())
        predicted = float(cm[i].sum())
        recall = float(cm[i, i] / support) if support else 0.0
        precision = float(cm[i, i] / predicted) if predicted else 0.0
        summary.setdefault("per_class", {})[name] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "support": int(support),
        }
        per_class.append(
            {
                "kelas": name,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "support": int(support),
            }
        )

    (REPORT_DIR / "metrics_test.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (REPORT_DIR / "per_class_test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kelas", "recall", "precision", "support"])
        writer.writeheader()
        writer.writerows(per_class)

    # Salin confusion matrix yang dibuat ultralytics
    for cand in (RUN_DIR / "val" / "confusion_matrix.png",
                 RUN_DIR / "val" / "confusion_matrix_normalized.png"):
        if cand.exists():
            dst = REPORT_DIR / cand.name.replace("confusion_matrix", "confusion_matrix_test")
            dst.write_bytes(cand.read_bytes())

    print("\n=== Evaluasi Test Set ===")
    print(f"Top-1 Accuracy : {metrics.top1:.4f}")
    print(f"Top-5 Accuracy : {metrics.top5:.4f}")
    for row in per_class:
        print(f"  {row['kelas']:<24} recall={row['recall']:.3f} "
              f"precision={row['precision']:.3f} n={row['support']}")
    print(f"\nMetrik tersimpan di: {REPORT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training YOLOv8n-cls daun cabai")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--eval-only", action="store_true",
                        help="Lewati training, langsung evaluasi best.pt")
    args = parser.parse_args()

    if not DATA_DIR.is_dir():
        raise SystemExit("[ERROR] data_cls/ belum ada. Jalankan prepare_data.py dulu.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        from ultralytics import YOLO

        model = YOLO(MODEL)
        model.train(
            data=str(DATA_DIR),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=SEED,
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
            plots=True,
            workers=4,
        )
        plot_curves(RUN_DIR / "results.csv", REPORT_DIR / "kurva_acc_loss.png")

    if BEST.exists():
        evaluate(BEST)
    else:
        print(f"[WARNING] {BEST} tidak ditemukan, evaluasi dilewati.")


if __name__ == "__main__":
    main()
