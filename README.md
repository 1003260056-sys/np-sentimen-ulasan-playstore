# Analisis Sentimen Ulasan Aplikasi E-Commerce

Proyek ini adalah kode pendamping buku ajar *Analisis Sentimen Ulasan Pengguna Menggunakan Natural Language Processing Berbasis Algoritma Klasifikasi*, untuk Bagian I (Bab 1 sampai 5, materi UTS).

Data yang dipakai adalah ulasan asli pengguna Tokopedia (`com.tokopedia.tkpd`) dan Shopee (`com.shopee.id`) di Google Play Store, sekitar 750 ulasan per aplikasi. Setiap ulasan diberi label dari jumlah bintangnya:

- bintang 4 dan 5: Positif
- bintang 3: Netral
- bintang 1 dan 2: Negatif

Program ini mengambil ulasan, membersihkan teksnya, mengubahnya menjadi angka, melatih enam model, lalu memilih model dengan hasil terbaik untuk menebak sentimen kalimat baru.

## How to Run?

Jalankan semua perintah dari dalam folder `sentiment_analysis_project/`.

Siapkan lingkungan dulu:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Lalu jalankan seluruh tahap sekaligus:

```bash
python run_pipeline.py --step all
```

Untuk mencoba cepat dengan data lebih sedikit, tambahkan `--count 100`. Proses pengambilan data butuh internet, dan tahap pelatihan bisa memakan waktu beberapa menit.

### Menjalankan satu tahap saja

```bash
python run_pipeline.py --step check        # cek lingkungan
python run_pipeline.py --step ingest       # ambil data, lihat ringkasannya, bagi jadi train/val/test
python run_pipeline.py --step preprocess   # bersihkan teks
python run_pipeline.py --step features     # ubah teks jadi fitur angka
python run_pipeline.py --step train        # latih model dan pilih yang terbaik
python run_pipeline.py --step visualize    # buat 10 grafik
```

Kalau dijalankan satu per satu, urutannya harus: `ingest`, `preprocess`, `features`, `train`, `visualize`.

### Menebak sentimen kalimat baru

Bisa dilakukan setelah tahap `train` selesai:

```bash
python run_pipeline.py --step predict --text "Aplikasinya bagus banget, pengiriman cepat"
```

### Memakai data sendiri

Kalau tidak ingin mengambil data dari Play Store, siapkan file CSV dengan kolom seperti pada bagian "Format Data" di bawah, lalu:

```bash
python run_pipeline.py --step ingest --input-data data/raw/dataset_saya.csv
```

## Hubungan dengan Isi Buku

| Step | Bab | Kode di buku | File | Hasil |
|---|---|---|---|---|
| `check` | 1 | [KODE PYTHON 1.1] | `src/check_env.py` | Daftar versi pustaka di terminal |
| `ingest` | 2 | [KODE PYTHON 2.1] | `src/scrape_dataset.py` | `data/raw/ulasan_pengguna.csv` |
| `ingest` | 2 | [KODE PYTHON 2.2] | `src/eda_korpus.py` | Ringkasan data di terminal |
| `ingest` | 2 | [KODE PYTHON 2.3] | `src/split_dataset.py` | `train_raw.csv`, `val_raw.csv`, `test_raw.csv` di `data/processed/` |
| `preprocess` | 3 | [KODE PYTHON 3.1] | `src/slang_normalizer.py` | `data/external/kamus_slang_id.csv` |
| `preprocess` | 3 | [KODE PYTHON 3.2] | `src/stemming_demo.py` | Contoh proses pembersihan di terminal |
| `preprocess` | 3 | [KODE PYTHON 3.3] | `src/preprocessing_pipeline.py` | `train_clean.csv`, `val_clean.csv`, `test_clean.csv` di `data/processed/` |
| `features` | 4 | [KODE PYTHON 4.1] | `src/tfidf_features.py` | `models/tfidf_vectorizer.joblib` |
| `features` | 4 | [KODE PYTHON 4.2] | `src/word_embeddings.py` | `models/fasttext_ulasan.model`, `models/word2vec_ulasan.model` |
| `features` | 4 | [KODE PYTHON 4.3] | `src/feature_selection_smote.py` | `data/processed/distribusi_smote.csv` |
| `train` | 5 | [KODE PYTHON 5.1] | `src/train_nb_svm.py` | `multinomialnb_model.joblib`, `complementnb_model.joblib`, `linearsvc_model.joblib` di `models/` |
| `train` | 5 | [KODE PYTHON 5.2] | `src/train_logreg_rf_xgb.py` | `logisticregression_model.joblib`, `randomforest_model.joblib`, `xgboost_model.joblib`, `label_encoder.joblib` di `models/` |
| `train` | 5 | [KODE PYTHON 5.3] | `src/evaluate_and_export.py` | `data/processed/hasil_komparasi_model_uts.csv` dan `models/best_sentiment_pipeline.joblib` |
| `visualize` | 2 sampai 5 | - | `src/visualize.py` | 10 gambar PNG di `outputs/figures/` |

Satu catatan kecil: fungsi di `tfidf_features.py` punya nilai bawaan `min_df=3` seperti di buku, tetapi saat lewat `run_pipeline.py` yang dipakai adalah `min_df: 2` dari `config.yaml`.

## Format Data

File `data/raw/ulasan_pengguna.csv` harus punya 6 kolom ini:

| Kolom | Isi |
|---|---|
| `id_ulasan` | Nomor ulasan, contoh `REV_0001` |
| `tanggal` | Tanggal ulasan, format `YYYY-MM-DD` |
| `kategori_produk` | Nama aplikasi (`Tokopedia` atau `Shopee`) |
| `rating` | Jumlah bintang, 1 sampai 5 |
| `teks_ulasan` | Isi ulasan |
| `label_sentimen` | `Positif`, `Netral`, atau `Negatif` |

File `data/external/kamus_slang_id.csv` hanya punya 2 kolom: `slang` (kata tidak baku, misalnya `bgt`) dan `baku` (padanannya, misalnya `banget`).

## Pengaturan

Semua pengaturan seperti lokasi file, jumlah ulasan, dan rasio pembagian data ada di `config.yaml`. Ubah di sana kalau ingin menyesuaikan, tanpa perlu menyentuh kode.

Setiap file di `src/` juga bisa dijalankan sendiri, misalnya `python src/eda_korpus.py`. Pengaturan yang tidak diberikan lewat terminal akan diambil dari `config.yaml`.

## Menambah Tahap Baru

Di `run_pipeline.py`, setiap tahap adalah satu fungsi yang didaftarkan dengan `@daftarkan_step`. Untuk menambah tahap baru, tulis fungsinya dan daftarkan dengan cara yang sama. Nama tahap itu akan otomatis muncul di pilihan `--step`.
