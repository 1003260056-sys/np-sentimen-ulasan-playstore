"""[KODE PYTHON 2.2] Analisis Data Eksploratif korpus (Bab 2).

Mode 1: diimpor -> run_corpus_eda(...)
Mode 2: mandiri  -> python src/eda_korpus.py [--filepath ...]
"""

# Mengimpor modul dasar
import sys
import os
import re
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pandas dan numpy
import pandas as pd
import numpy as np
# Mengimpor penghitung n-gram
from sklearn.feature_extraction.text import CountVectorizer
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek


def _top_ngram(teks, n, top_n):
    """Menghitung n-gram teratas (unigram bila n=1, bigram bila n=2) dari teks mentah."""
    # Membuat penghitung n-gram dengan huruf kecil
    cv = CountVectorizer(ngram_range=(n, n), lowercase=True, token_pattern=r'(?u)\b\w\w+\b')
    # Membentuk matriks hitungan
    matriks = cv.fit_transform(teks)
    # Menjumlahkan frekuensi tiap n-gram
    frekuensi = np.asarray(matriks.sum(axis=0)).ravel()
    # Mengambil nama fitur
    nama = cv.get_feature_names_out()
    # Mengurutkan dari frekuensi terbesar
    urut = frekuensi.argsort()[::-1][:top_n]
    # Mengembalikan pasangan (n-gram, frekuensi)
    return [(nama[i], int(frekuensi[i])) for i in urut]


def run_corpus_eda(filepath='data/raw/ulasan_pengguna.csv', top_n=10):
    """Mengaudit dan merangkum korpus ulasan mentah."""
    # Membaca dataset mentah
    df = pd.read_csv(path_proyek(filepath), encoding='utf-8')
    print('=' * 70)
    print('[BAB 2] AUDIT DAN EDA KORPUS')
    print('=' * 70)
    # Audit data mentah
    print(f'Jumlah baris mentah        : {len(df)}')
    print('\nJumlah nilai kosong per kolom:')
    print(df.isnull().sum())
    print(f"\nJumlah duplikat (teks_ulasan): {df.duplicated(subset=['teks_ulasan']).sum()}")
    # Pembersihan in-memory
    df = df.dropna(subset=['teks_ulasan']).drop_duplicates(subset=['teks_ulasan']).reset_index(drop=True)
    print(f'Jumlah baris setelah dropna & drop_duplicates: {len(df)}')
    # Distribusi kelas
    print('\nDistribusi kelas sentimen:')
    dist = df['label_sentimen'].value_counts()
    print(pd.DataFrame({'jumlah': dist, 'persen': (dist / dist.sum() * 100).round(2)}))
    # Tabulasi silang kategori vs sentimen
    print('\nCrosstab kategori_produk vs label_sentimen:')
    print(pd.crosstab(df['kategori_produk'], df['label_sentimen']))
    # Statistik panjang teks
    teks = df['teks_ulasan'].astype(str)
    panjang_char = teks.str.len()
    jumlah_kata = teks.str.split().str.len()
    print('\nStatistik panjang ulasan:')
    for nama, seri in [('Jumlah karakter', panjang_char), ('Jumlah kata', jumlah_kata)]:
        print(f'  {nama:<16}: mean={seri.mean():.2f} | median={seri.median():.2f} | min={seri.min()} | max={seri.max()}')
    # Type-Token Ratio: TTR = (V / N) * 100%
    token = re.findall(r'\w+', ' '.join(teks).lower())
    N = len(token)
    V = len(set(token))
    ttr = (V / N) * 100 if N else 0.0
    print(f'\nType-Token Ratio: V={V}, N={N}, TTR={ttr:.2f}%')
    # N-gram teratas
    unigram = _top_ngram(teks, 1, top_n)
    bigram = _top_ngram(teks, 2, top_n)
    print(f'\nTop-{top_n} unigram mentah:')
    for kata, f in unigram:
        print(f'  {kata:<20} {f}')
    print(f'\nTop-{top_n} bigram mentah:')
    for kata, f in bigram:
        print(f'  {kata:<25} {f}')
    # Mengembalikan data bersih dan statistik
    return {'data_bersih': df, 'ttr': ttr, 'unigram': unigram, 'bigram': bigram}


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='EDA korpus ulasan (Bab 2)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--filepath', default=None, help='Path dataset CSV')
    parser.add_argument('--top-n', type=int, default=10, help='Jumlah n-gram teratas')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    run_corpus_eda(filepath=args.filepath or cfg['paths']['raw_dataset'], top_n=args.top_n)
