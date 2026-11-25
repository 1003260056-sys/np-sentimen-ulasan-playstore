"""[KODE PYTHON 4.1] Ekstraksi fitur TF-IDF (Bab 4).

Mode 1: diimpor -> extract_tfidf_features(...)
Mode 2: mandiri  -> python src/tfidf_features.py [--min-df 3 ...]
"""

# Mengimpor modul dasar
import sys
import os
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor numpy, pandas, joblib
import numpy as np
import pandas as pd
import joblib
# Mengimpor vektorisasi TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek, siapkan_folder_file


def extract_tfidf_features(processed_dir='data/processed', models_dir='models', text_col='teks_bersih',
                           target_col='label_sentimen', classes=('Positif', 'Netral', 'Negatif'),
                           ngram_range=(1, 2), min_df=3, max_df=0.90, sublinear_tf=True, norm='l2', top_n=10):
    """Fit TF-IDF pada data latih saja, transform val/test, dan tampilkan statistik."""
    print('=' * 70)
    print('[BAB 4] EKSTRAKSI FITUR TF-IDF')
    print('=' * 70)
    folder = path_proyek(processed_dir)
    # Membaca ketiga bagian data bersih
    data = {b: pd.read_csv(os.path.join(folder, f'{b}_clean.csv'), encoding='utf-8') for b in ['train', 'val', 'test']}
    for df in data.values():
        df[text_col] = df[text_col].fillna('').astype(str)
    # Membuat vektorisasi
    vektor = TfidfVectorizer(ngram_range=tuple(ngram_range), min_df=min_df, max_df=max_df,
                             sublinear_tf=sublinear_tf, norm=norm)
    # fit_transform HANYA pada data latih (mencegah kebocoran data)
    X_train = vektor.fit_transform(data['train'][text_col])
    # transform pada validasi dan uji
    X_val = vektor.transform(data['val'][text_col])
    X_test = vektor.transform(data['test'][text_col])
    # Statistik DTM
    baris, kolom = X_train.shape
    nnz = X_train.nnz
    sparsity = (1 - nnz / (baris * kolom)) * 100
    print(f'Dimensi DTM latih : {baris} x {kolom}')
    print(f'Dimensi DTM val   : {X_val.shape[0]} x {X_val.shape[1]}')
    print(f'Dimensi DTM uji   : {X_test.shape[0]} x {X_test.shape[1]}')
    print(f'Elemen non-nol    : {nnz}')
    print(f'Sparsity          : {sparsity:.2f}%')
    # Top fitur per kelas berdasarkan rata-rata bobot TF-IDF
    nama_fitur = vektor.get_feature_names_out()
    y_train = data['train'][target_col].to_numpy()
    top_per_kelas = {}
    for kelas in classes:
        # Memilih baris milik kelas
        mask = y_train == kelas
        if mask.sum() == 0:
            continue
        # Menghitung rata-rata bobot
        rata = np.asarray(X_train[mask].mean(axis=0)).ravel()
        idx = rata.argsort()[::-1][:top_n]
        top_per_kelas[kelas] = [(nama_fitur[i], float(rata[i])) for i in idx]
        print(f'\nTop-{top_n} fitur kelas {kelas}:')
        for f, b in top_per_kelas[kelas]:
            print(f'  {f:<25} {b:.4f}')
    # Menyimpan vektorisasi untuk pemakaian ulang
    lokasi = os.path.join(path_proyek(models_dir), 'tfidf_vectorizer.joblib')
    siapkan_folder_file(lokasi)
    joblib.dump(vektor, lokasi)
    print(f'\nVektorisasi tersimpan: {lokasi}')
    return {'vectorizer': vektor, 'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'sparsity': sparsity, 'top_per_kelas': top_per_kelas}


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Ekstraksi TF-IDF (Bab 4)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--processed-dir', default=None, help='Folder data/processed')
    parser.add_argument('--min-df', type=int, default=None, help='Nilai min_df')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    f, s = cfg['features'], cfg['schema']
    extract_tfidf_features(
        processed_dir=args.processed_dir or cfg['paths']['processed_dir'], models_dir=cfg['paths']['models_dir'],
        text_col=s['clean_text_col'], target_col=s['target_col'], classes=s['classes'],
        ngram_range=f['ngram_range'], min_df=args.min_df if args.min_df is not None else f['min_df'],
        max_df=f['max_df'], sublinear_tf=f['sublinear_tf'], norm=f['norm'])
