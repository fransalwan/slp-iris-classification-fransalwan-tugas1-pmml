"""
Single Layer Perceptron (SLP) — Klasifikasi Iris Setosa vs Iris Versicolor.

Implementasi dari nol (NumPy + Pandas, tanpa scikit-learn) yang menghasilkan
angka identik dengan perhitungan manual di Google Sheets / spreadsheet.

Jalankan:
    python src/perceptron.py
    python src/perceptron.py --epochs 10 --lr 0.05 --verbose

Catatan implementasi yang harus dipertahankan agar cocok dengan spreadsheet:
  1. Bobot awal teta1-teta4 dan bias = 0.5, learning rate = 0.1.
  2. Threshold prediksi: g(z) > 0.5 -> 1, selainnya -> 0.
  3. Turunan pada baris training TERAKHIR dihitung tetapi tidak diterapkan.
     Di spreadsheet, bobot pada baris terakhir dipakai apa adanya untuk
     testing dan menjadi bobot awal epoch berikutnya.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FITUR = ["sepal_l", "sepal_w", "petal_l", "petal_w"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def baca_csv(*kandidat: str) -> pd.DataFrame:
    """Baca CSV pertama yang ditemukan dari beberapa kemungkinan nama file."""
    for nama in kandidat:
        if os.path.exists(nama):
            return pd.read_csv(nama)
    raise FileNotFoundError("File tidak ditemukan. Sudah dicoba: " + ", ".join(kandidat))


def muat_dataset(data_dir: str = DATA_DIR):
    df_train = baca_csv(
        os.path.join(data_dir, "Iris_Dataset_-_Training.csv"),
        os.path.join(data_dir, "Iris_Dataset - Training.csv"),
    )
    df_test = baca_csv(
        os.path.join(data_dir, "Iris_Dataset_-_Testing.csv"),
        os.path.join(data_dir, "Iris_Dataset - Testing.csv"),
    )
    siap = lambda df: (df[FITUR].values.astype(float), df["target"].values.astype(int))
    return siap(df_train), siap(df_test)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def sigmoid(z: float) -> float:
    """Sigmoid versi stabil numerik — identik dengan 1 / (1 + e^-z)."""
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (1.0 + ez)


@dataclass
class HasilPelatihan:
    bias: float
    teta: np.ndarray
    train_mse: list = field(default_factory=list)
    test_mse: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    test_acc: list = field(default_factory=list)
    tabel_train: list = field(default_factory=list)
    tabel_test: list = field(default_factory=list)


class SingleLayerPerceptron:
    def __init__(self, lr=0.1, epochs=5, bobot_awal=0.5, bias_awal=0.5, threshold=0.5):
        self.lr = lr
        self.epochs = epochs
        self.bobot_awal = bobot_awal
        self.bias_awal = bias_awal
        self.threshold = threshold

    def prediksi_kelas(self, gz: float) -> int:
        return 1 if gz > self.threshold else 0

    def fit(self, X_train, y_train, X_test, y_test) -> HasilPelatihan:
        teta = np.full(X_train.shape[1], self.bobot_awal, dtype=float)
        bias = float(self.bias_awal)
        n = len(X_train)
        h = HasilPelatihan(bias=bias, teta=teta)

        for epoch in range(1, self.epochs + 1):
            baris_train = []
            for i in range(n):
                teta_lama, bias_lama = teta.copy(), bias

                z = bias + np.dot(X_train[i], teta)
                gz = sigmoid(z)
                pred = self.prediksi_kelas(gz)
                error = gz - y_train[i]

                # Turunan mentah (belum dikali learning rate)
                dbias = 2 * error * (1 - gz) * gz
                dteta = dbias * X_train[i]

                baris_train.append(
                    {
                        "EPOCH": f"epoch {epoch}", "No": i + 1,
                        "X1": X_train[i][0], "X2": X_train[i][1],
                        "X3": X_train[i][2], "X4": X_train[i][3],
                        "TARGET": y_train[i],
                        "bias": bias_lama, "teta1": teta_lama[0], "teta2": teta_lama[1],
                        "teta3": teta_lama[2], "teta4": teta_lama[3],
                        "z": z, "g(z)": gz, "Prediksi": pred,
                        "error": error, "Sum Square Error": error ** 2,
                        "dbias": dbias, "dteta1": dteta[0], "dteta2": dteta[1],
                        "dteta3": dteta[2], "dteta4": dteta[3],
                        "Akurasi": int(pred == y_train[i]),
                    }
                )

                # Baris terakhir tidak di-update (lihat docstring modul)
                if i < n - 1:
                    teta = teta - self.lr * dteta
                    bias = bias - self.lr * dbias

            baris_test = []
            for i in range(len(X_test)):
                z = bias + np.dot(X_test[i], teta)
                gz = sigmoid(z)
                pred = self.prediksi_kelas(gz)
                error = gz - y_test[i]
                baris_test.append(
                    {
                        "EPOCH": "testing", "No": i + 1,
                        "X1": X_test[i][0], "X2": X_test[i][1],
                        "X3": X_test[i][2], "X4": X_test[i][3],
                        "TARGET": y_test[i],
                        "bias": bias, "teta1": teta[0], "teta2": teta[1],
                        "teta3": teta[2], "teta4": teta[3],
                        "z": z, "g(z)": gz, "Prediksi": pred,
                        "error": error, "Sum Square Error": error ** 2,
                        "Akurasi": int(pred == y_test[i]),
                    }
                )

            df_tr, df_te = pd.DataFrame(baris_train), pd.DataFrame(baris_test)
            h.tabel_train.append(df_tr)
            h.tabel_test.append(df_te)
            h.train_mse.append(df_tr["Sum Square Error"].mean())
            h.train_acc.append(df_tr["Akurasi"].mean())
            h.test_mse.append(df_te["Sum Square Error"].mean())
            h.test_acc.append(df_te["Akurasi"].mean())

        h.bias, h.teta = bias, teta
        return h


# --------------------------------------------------------------------------- #
# Evaluasi
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred):
    """Mengembalikan (TN, FP, FN, TP). Kelas positif = 1 (Versicolor)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return (
        int(((y_true == 0) & (y_pred == 0)).sum()),
        int(((y_true == 0) & (y_pred == 1)).sum()),
        int(((y_true == 1) & (y_pred == 0)).sum()),
        int(((y_true == 1) & (y_pred == 1)).sum()),
    )


def metrik(TN, FP, FN, TP):
    akurasi = (TP + TN) / (TP + TN + FP + FN)
    presisi = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * presisi * recall / (presisi + recall) if (presisi + recall) else 0.0
    return {"akurasi": akurasi, "presisi": presisi, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# Visualisasi
# --------------------------------------------------------------------------- #
def simpan_grafik(h: HasilPelatihan, out_dir: str = RESULTS_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ep = range(1, len(h.train_mse) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, h.train_mse, "o-", label="Training", color="#44306E", linewidth=2)
    plt.plot(ep, h.test_mse, "o-", label="Validation", color="#D9A441", linewidth=2)
    plt.title("Loss (MSE) per Epoch", fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.xticks(list(ep))
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_chart.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, [a * 100 for a in h.train_acc], "o-", label="Training",
             color="#44306E", linewidth=2)
    plt.plot(ep, [a * 100 for a in h.test_acc], "o-", label="Validation",
             color="#D9A441", linewidth=2)
    plt.title("Akurasi per Epoch", fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("Akurasi (%)")
    plt.ylim(0, 105); plt.xticks(list(ep))
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_chart.png"), dpi=150)
    plt.close()


def simpan_rekap(h: HasilPelatihan, out_dir: str = RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": range(1, len(h.train_mse) + 1),
            "train_mse": h.train_mse,
            "test_mse": h.test_mse,
            "train_acc": h.train_acc,
            "test_acc": h.test_acc,
        }
    ).to_csv(os.path.join(out_dir, "metrics_per_epoch.csv"), index=False)

    pd.DataFrame(
        [{"bias": h.bias, "teta1": h.teta[0], "teta2": h.teta[1],
          "teta3": h.teta[2], "teta4": h.teta[3]}]
    ).to_csv(os.path.join(out_dir, "final_weights.csv"), index=False)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="SLP untuk klasifikasi Iris")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bobot-awal", type=float, default=0.5)
    p.add_argument("--bias-awal", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true",
                   help="Cetak tabel per baris untuk setiap epoch")
    p.add_argument("--no-save", action="store_true",
                   help="Jangan tulis apa pun ke folder results/")
    args = p.parse_args()

    (X_train, y_train), (X_test, y_test) = muat_dataset()
    print(f"Data training : {len(X_train)} baris "
          f"(Setosa {int((y_train == 0).sum())}, Versicolor {int((y_train == 1).sum())})")
    print(f"Data testing  : {len(X_test)} baris "
          f"(Setosa {int((y_test == 0).sum())}, Versicolor {int((y_test == 1).sum())})")

    model = SingleLayerPerceptron(
        lr=args.lr, epochs=args.epochs,
        bobot_awal=args.bobot_awal, bias_awal=args.bias_awal,
    )
    h = model.fit(X_train, y_train, X_test, y_test)

    if args.verbose:
        pd.options.display.float_format = "{:.11f}".format
        for i in range(args.epochs):
            print(f"\n{'=' * 30} EPOCH {i + 1} — TRAINING {'=' * 30}")
            print(h.tabel_train[i].to_string(index=False))
            print(f"\n{'=' * 30} EPOCH {i + 1} — TESTING {'=' * 30}")
            print(h.tabel_test[i].to_string(index=False))

    print("\n--- Rekap per epoch ---")
    print(f"{'Epoch':>6} {'MSE train':>12} {'MSE test':>12} {'Acc train':>11} {'Acc test':>10}")
    for i in range(args.epochs):
        print(f"{i + 1:>6} {h.train_mse[i]:>12.8f} {h.test_mse[i]:>12.8f} "
              f"{h.train_acc[i]:>10.4%} {h.test_acc[i]:>9.4%}")

    print("\n--- Bobot akhir ---")
    print(f"bias  = {h.bias:.11f}")
    for j, t in enumerate(h.teta, start=1):
        print(f"teta{j} = {t:.11f}")

    df_akhir = h.tabel_test[-1]
    TN, FP, FN, TP = confusion_matrix(df_akhir["TARGET"], df_akhir["Prediksi"])
    m = metrik(TN, FP, FN, TP)
    print("\n--- Confusion matrix (testing, kelas positif = Versicolor) ---")
    print(f"              Pred 0  Pred 1")
    print(f"Aktual 0      {TN:>6}  {FP:>6}")
    print(f"Aktual 1      {FN:>6}  {TP:>6}")
    print(f"\nAkurasi {m['akurasi']:.2%} | Presisi {m['presisi']:.2%} | "
          f"Recall {m['recall']:.2%} | F1 {m['f1']:.2%}")

    if not args.no_save:
        simpan_grafik(h)
        simpan_rekap(h)
        print(f"\nGrafik dan rekap tersimpan di {RESULTS_DIR}")


if __name__ == "__main__":
    main()
