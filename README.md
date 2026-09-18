# Single Layer Perceptron — Klasifikasi Iris Setosa vs Iris Versicolor

Tugas 1 — Pembelajaran Mesin dan Machine Learning (PMML)

| | |
|---|---|
| **Nama** | Frans Alwan Purba |
| **NIM** | 25/563545/PPA/07116 |
| **Tugas** | Assignment 1 — Single Layer Perceptron |

Implementasi *single layer perceptron* dari nol untuk klasifikasi biner dua
kelas bunga Iris, dikerjakan dua kali secara paralel: perhitungan manual di
spreadsheet dan implementasi Python. Keluaran keduanya identik sampai digit
ke-11 — spreadsheet berfungsi sebagai *ground truth* untuk memverifikasi kode.

---

## Ringkasan Hasil

| Epoch | MSE Training | MSE Validation | Akurasi Training | Akurasi Validation |
|------:|-------------:|---------------:|-----------------:|-------------------:|
| 1 | 0.44715463 | 0.28755158 | 52.50% | 50.00% |
| 2 | 0.04486284 | 0.15501014 | 93.75% | 50.00% |
| 3 | 0.02381170 | 0.07774028 | 97.50% | 100.00% |
| 4 | 0.01511584 | 0.04165461 | 100.00% | 100.00% |
| 5 | 0.01083626 | 0.02509921 | 100.00% | 100.00% |

**Bobot akhir setelah 5 epoch**

```
bias  =  0.22708411127
teta1 = -0.37710744384   (SepalLength)
teta2 = -0.58487969769   (SepalWidth)
teta3 =  1.31082264837   (PetalLength)
teta4 =  0.93461775627   (PetalWidth)
```

**Evaluasi pada 20 data testing** (kelas positif = Versicolor)

|  | Prediksi Setosa | Prediksi Versicolor |
|---|---:|---:|
| **Aktual Setosa** | 10 | 0 |
| **Aktual Versicolor** | 0 | 10 |

Akurasi 100% · Presisi 100% · Recall 100% · F1-Score 100%

---

## Struktur Repositori

```
slp-iris-classification/
├── README.md
├── requirements.txt
├── Assignment1_SLP.pdf              # soal tugas
├── data/
│   ├── Iris_Dataset_-_Training.csv  # 80 baris (40 Setosa + 40 Versicolor)
│   └── Iris_Dataset_-_Testing.csv   # 20 baris (10 Setosa + 10 Versicolor)
├── notebook/
│   └── Frans_Alwan_Purba_Perceptron_PPML.ipynb
├── src/
│   └── perceptron.py                # versi script, bisa dijalankan dari CLI
├── spreadsheet/
│   └── Tugas1_PMML_Frans_Alwan_Purba.xlsx
├── slides/
│   └── Tugas1_PMML_SLP_Iris.pptx
└── results/
    ├── accuracy_chart.png
    ├── loss_chart.png
    ├── metrics_per_epoch.csv
    └── final_weights.csv
```

---

## Cara Menjalankan

```bash
git clone https://github.com/<username>/slp-iris-classification.git
cd slp-iris-classification
pip install -r requirements.txt
```

**Sebagai script:**

```bash
python src/perceptron.py                 # ringkasan + grafik ke results/
python src/perceptron.py --verbose       # cetak tabel per baris tiap epoch
python src/perceptron.py --epochs 10 --lr 0.05
```

**Sebagai notebook:**

```bash
jupyter notebook notebook/Frans_Alwan_Purba_Perceptron_PPML.ipynb
```

Notebook membaca CSV dari direktori kerjanya sendiri. Bila dijalankan di
Google Colab, unggah kedua file dari `data/` terlebih dahulu.

---

## Konfigurasi Model

| Parameter | Nilai |
|---|---|
| Neuron input | 4 (SepalLength, SepalWidth, PetalLength, PetalWidth) |
| Neuron output | 1 (binary encoding) |
| Target | Setosa = 0, Versicolor = 1 |
| Bobot awal θ₁–θ₄ | 0.5 |
| Bias awal | 0.5 |
| Learning rate | 0.1 |
| Fungsi aktivasi | Sigmoid, g(z) = 1 / (1 + e⁻ᶻ) |
| Fungsi loss | Sum Square Error, dirata-rata menjadi MSE per epoch |
| Metode update | Gradient Descent stokastik (per satu data) |
| Jumlah epoch | 5 |
| Threshold prediksi | g(z) > 0.5 → 1, selainnya → 0 |
| Urutan data training | Selang-seling blok 20 (0–1–0–1) |

**Rumus turunan**

```
z      = bias + θ₁X₁ + θ₂X₂ + θ₃X₃ + θ₄X₄
g(z)   = 1 / (1 + e^(-z))
error  = g(z) − TARGET
dθⱼ    = 2 · error · [1 − g(z)] · g(z) · Xⱼ
dbias  = 2 · error · [1 − g(z)] · g(z)
θⱼ ← θⱼ − LR · dθⱼ          bias ← bias − LR · dbias
```

---

## Tiga Detail yang Menentukan Kecocokan dengan Spreadsheet

Ketiganya tidak terlihat dari rumus, tetapi menentukan apakah angka Python
cocok dengan spreadsheet:

1. **Bobot awal 0.5, bukan nilai lain.** Implementasi awal memakai 0.2 dan
   seluruh angka meleset sejak baris pertama.
2. **Threshold `g(z) > 0.5`, bukan `>=`.** Berpengaruh pada baris yang
   nilai sigmoidnya persis 0.5.
3. **Turunan baris training ke-80 tidak pernah diterapkan.** Di spreadsheet,
   bobot pada baris terakhir dipakai apa adanya untuk testing dan menjadi
   bobot awal epoch berikutnya — tidak ada baris ke-81 yang menampung hasil
   update-nya. Tanpa penyesuaian ini MSE testing epoch 1 keluar 0.29054,
   bukan 0.28755.

---

## Catatan Analisis

- **Akurasi validasi tertahan di 50% selama dua epoch pertama.** Pada epoch 1
  dan 2 seluruh data testing masih diprediksi sebagai Versicolor, sehingga 10
  dari 20 kebetulan benar. Angka ini menyesatkan bila dibaca tanpa confusion
  matrix.
- **Urutan data berpengaruh besar.** Menyajikan 40 Setosa berurutan lalu 40
  Versicolor membuat bobot terlalu condong ke satu kelas sebelum melihat kelas
  lain. Perbandingannya ada di sheet `Trial without shuffled data`.
- **Akurasi 100% bukan bukti generalisasi.** Setosa dan Versicolor adalah dua
  kelas Iris yang paling mudah dipisahkan secara linear, dan data testing hanya
  20 baris. Menambahkan Iris Virginica akan menurunkan hasil ini secara
  signifikan.
- **θ₃ dan θ₄ (dimensi petal) bernilai positif besar**, sementara θ₁ dan θ₂
  (dimensi sepal) bernilai negatif — konsisten dengan fakta bahwa ukuran petal
  adalah pembeda utama kedua kelas.

---

## Referensi

1. Rosenblatt, F. (1958). *The perceptron: A probabilistic model for information
   storage and organization in the brain.* Psychological Review, 65(6), 386–408.
2. Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.*
   Annals of Eugenics, 7(2), 179–188.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
4. Dua, D., & Graff, C. (2019). *Iris Data Set.* UCI Machine Learning Repository.
   https://archive.ics.uci.edu/dataset/53/iris
5. Materi kuliah Pembelajaran Mesin dan Machine Learning — Single Layer
   Perceptron & Gradient Descent.
