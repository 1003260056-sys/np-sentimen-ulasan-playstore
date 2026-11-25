"""[KODE PYTHON 1.1] Verifikasi lingkungan dan baseline TF-IDF + MultinomialNB (Bab 1).

Mode 1: diimpor -> verify_env_and_run_baseline(...)
Mode 2: mandiri  -> python src/check_env.py [--filepath ...]
"""

# Mengimpor modul sys untuk menambah path dan membaca versi Python
import sys
# Mengimpor os untuk pengecekan berkas
import os
# Mengimpor argparse untuk argumen terminal
import argparse
# Mengimpor importlib untuk memuat pustaka secara dinamis
import importlib

# Menambahkan folder akar proyek ke sys.path agar "from src ..." berfungsi saat dijalankan mandiri
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pandas untuk tabel data
import pandas as pd
# Mengimpor fungsi pembagian data
from sklearn.model_selection import train_test_split
# Mengimpor vektorisasi TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Mengimpor pengklasifikasi Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Mengimpor metrik evaluasi
from sklearn.metrics import accuracy_score, f1_score
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek

# Kolom wajib skema dataset
KOLOM_WAJIB = ['id_ulasan', 'tanggal', 'kategori_produk', 'rating', 'teks_ulasan', 'label_sentimen']


def _buat_sampel_enam_ulasan():
    """Membuat 6 sampel ulasan Play Store (REV_001-REV_006) untuk validasi cepat."""
    # Daftar baris contoh: id, tanggal, kategori, rating, teks, label
    baris = [
        ('REV_001', '2024-01-05', 'Tokopedia', 5, 'Aplikasi bagus banget, pengiriman cepat dan barang ori', 'Positif'),
        ('REV_002', '2024-01-06', 'Shopee', 5, 'Mantap, promo banyak dan harga murah sekali', 'Positif'),
        ('REV_003', '2024-01-07', 'Tokopedia', 3, 'Biasa saja, kadang lancar kadang lemot', 'Netral'),
        ('REV_004', '2024-01-08', 'Shopee', 3, 'Lumayan, tetapi fitur chat kurang lengkap', 'Netral'),
        ('REV_005', '2024-01-09', 'Tokopedia', 1, 'Sangat kecewa, aplikasi sering error dan barang tidak sampai', 'Negatif'),
        ('REV_006', '2024-01-10', 'Shopee', 1, 'Penjual lambat, pesanan dibatalkan sepihak, buruk sekali', 'Negatif'),
    ]
    # Mengubah daftar baris menjadi DataFrame berkolom skema
    return pd.DataFrame(baris, columns=KOLOM_WAJIB)


def verify_env_and_run_baseline(filepath='data/raw/ulasan_pengguna.csv', test_size=0.3, random_state=42):
    """Mencetak versi pustaka lalu menjalankan baseline TF-IDF + MultinomialNB."""
    # Mencetak judul bagian
    print('=' * 70)
    print('[BAB 1] VERIFIKASI LINGKUNGAN PYTHON')
    print('=' * 70)
    # Mencetak versi Python
    print(f'Python               : {sys.version.split()[0]}')
    # Daftar pustaka: (nama tampilan, nama modul)
    pustaka = [('pandas', 'pandas'), ('numpy', 'numpy'), ('scikit-learn', 'sklearn'),
               ('google_play_scraper', 'google_play_scraper'), ('Sastrawi', 'Sastrawi'),
               ('gensim', 'gensim'), ('xgboost', 'xgboost'), ('matplotlib', 'matplotlib')]
    # Mengulang setiap pustaka
    for nama, modul in pustaka:
        try:
            # Mencoba mengimpor modul
            obj = importlib.import_module(modul)
            # Mengambil atribut versi bila tersedia
            versi = getattr(obj, '__version__', 'terpasang (versi tidak dilaporkan)')
            # Mencetak versi
            print(f'{nama:<21}: {versi}')
        except ImportError:
            # Memberi petunjuk instalasi jika pustaka belum ada
            print(f'{nama:<21}: BELUM TERPASANG -> jalankan: pip install -r requirements.txt')

    # Menentukan path absolut dataset
    lokasi = path_proyek(filepath)
    # Memeriksa apakah dataset ada
    if os.path.exists(lokasi):
        # Membaca dataset nyata
        df = pd.read_csv(lokasi, encoding='utf-8')
        print(f'\nDataset ditemukan: {lokasi} ({len(df)} baris)')
    else:
        # Membuat 6 sampel sementara (tidak disimpan ke disk)
        df = _buat_sampel_enam_ulasan()
        print(f'\nDataset belum ada di {lokasi}. Memakai 6 sampel sementara (REV_001-REV_006).')

    # Memvalidasi skema kolom
    kurang = [k for k in KOLOM_WAJIB if k not in df.columns]
    # Menghentikan jika ada kolom yang hilang
    if kurang:
        raise ValueError(f'Kolom wajib tidak ditemukan: {kurang}')
    print('Validasi skema 6 kolom: LOLOS')
    # Membuang baris kosong/duplikat agar baseline valid
    df = df.dropna(subset=['teks_ulasan', 'label_sentimen']).drop_duplicates(subset=['teks_ulasan'])
    # Memisahkan fitur dan target
    X = df['teks_ulasan'].astype(str)
    y = df['label_sentimen']
    # Stratifikasi hanya bila setiap kelas punya cukup anggota
    stratifikasi = y if y.value_counts().min() >= 2 and len(df) >= 30 else None
    # Membagi data latih dan uji
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratifikasi)
    # Membuat vektorisasi TF-IDF
    vektor = TfidfVectorizer()
    # Mempelajari kosakata hanya dari data latih
    A_tr = vektor.fit_transform(X_tr)
    # Mentransformasi data uji
    A_te = vektor.transform(X_te)
    # Melatih Naive Bayes
    model = MultinomialNB().fit(A_tr, y_tr)
    # Memprediksi data uji
    pred = model.predict(A_te)
    # Mencetak hasil baseline
    print(f'Baseline TF-IDF + MultinomialNB -> Akurasi: {accuracy_score(y_te, pred):.4f} | '
          f'Macro-F1: {f1_score(y_te, pred, average="macro", zero_division=0):.4f}')
    # Mengembalikan hasil ringkas
    return {'akurasi': accuracy_score(y_te, pred), 'n_data': len(df)}


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Verifikasi lingkungan & baseline (Bab 1)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--filepath', default=None, help='Path dataset CSV')
    parser.add_argument('--test-size', type=float, default=0.3, help='Proporsi data uji')
    parser.add_argument('--seed', type=int, default=None, help='Random state')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    # Memanggil fungsi utama dengan fallback ke config
    verify_env_and_run_baseline(
        filepath=args.filepath or cfg['paths']['raw_dataset'],
        test_size=args.test_size,
        random_state=args.seed if args.seed is not None else cfg['scraping']['random_state'])
