"""[KODE PYTHON 4.3] Seleksi fitur Chi-Square, reduksi SVD, dan SMOTE (Bab 4).

Mode 1: diimpor -> run_feature_selection_and_smote(...)
Mode 2: mandiri  -> python src/feature_selection_smote.py [--k 500 --svd 100]
"""

# Mengimpor modul dasar
import sys
import os
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pandas
import pandas as pd
# Mengimpor TF-IDF, seleksi fitur, dan SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
# Mengimpor SMOTE
from imblearn.over_sampling import SMOTE
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek


def run_feature_selection_and_smote(processed_dir='data/processed', text_col='teks_bersih',
                                    target_col='label_sentimen', classes=('Positif', 'Netral', 'Negatif'),
                                    ngram_range=(1, 2), min_df=2, max_df=0.90, chi2_k=500,
                                    svd_components=100, random_state=42):
    """TF-IDF -> SelectKBest(chi2) -> TruncatedSVD -> SMOTE (hanya data latih)."""
    print('=' * 70)
    print('[BAB 4] SELEKSI FITUR, REDUKSI DIMENSI, DAN SMOTE')
    print('=' * 70)
    folder = path_proyek(processed_dir)
    # Membaca data latih
    train = pd.read_csv(os.path.join(folder, 'train_clean.csv'), encoding='utf-8')
    train[text_col] = train[text_col].fillna('').astype(str)
    y = train[target_col]
    # Ekstraksi TF-IDF
    tfidf = TfidfVectorizer(ngram_range=tuple(ngram_range), min_df=min_df, max_df=max_df, sublinear_tf=True)
    X = tfidf.fit_transform(train[text_col])
    print(f'TF-IDF awal          : {X.shape}')
    # Seleksi fitur (k tidak boleh melebihi jumlah fitur)
    k = min(chi2_k, X.shape[1])
    seleksi = SelectKBest(chi2, k=k)
    X_sel = seleksi.fit_transform(X, y)
    print(f'Setelah SelectKBest  : {X_sel.shape} (k={k})')
    # Reduksi dimensi (komponen harus lebih kecil dari jumlah fitur)
    n_komp = max(2, min(svd_components, X_sel.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_komp, random_state=random_state)
    X_svd = svd.fit_transform(X_sel)
    print(f'Setelah TruncatedSVD : {X_svd.shape} (variansi terjelaskan {svd.explained_variance_ratio_.sum() * 100:.2f}%)')
    # Distribusi sebelum SMOTE
    sebelum = y.value_counts().reindex(list(classes)).fillna(0).astype(int)
    # k_neighbors disesuaikan dengan ukuran kelas terkecil
    k_nn = max(1, min(5, int(sebelum[sebelum > 0].min()) - 1))
    # Menerapkan SMOTE khusus data latih
    smote = SMOTE(random_state=random_state, k_neighbors=k_nn)
    X_res, y_res = smote.fit_resample(X_svd, y)
    sesudah = pd.Series(y_res).value_counts().reindex(list(classes)).fillna(0).astype(int)
    print('\nDistribusi kelas sebelum vs sesudah SMOTE:')
    tabel = pd.DataFrame({'Sebelum SMOTE': sebelum, 'Sesudah SMOTE': sesudah})
    print(tabel)
    # Menyimpan distribusi untuk grafik 08
    lokasi = os.path.join(folder, 'distribusi_smote.csv')
    os.makedirs(folder, exist_ok=True)
    tabel.rename_axis('kelas').reset_index().to_csv(lokasi, index=False, encoding='utf-8')
    print(f'Tabel distribusi tersimpan: {lokasi}')
    return {'X_resampled': X_res, 'y_resampled': y_res, 'distribusi': tabel}


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Seleksi fitur & SMOTE (Bab 4)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--processed-dir', default=None, help='Folder data/processed')
    parser.add_argument('--k', type=int, default=None, help='Jumlah fitur chi2')
    parser.add_argument('--svd', type=int, default=None, help='Jumlah komponen SVD')
    parser.add_argument('--seed', type=int, default=None, help='Random state')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    f, s = cfg['features'], cfg['schema']
    run_feature_selection_and_smote(
        processed_dir=args.processed_dir or cfg['paths']['processed_dir'],
        text_col=s['clean_text_col'], target_col=s['target_col'], classes=s['classes'],
        ngram_range=f['ngram_range'], min_df=f['min_df'], max_df=f['max_df'],
        chi2_k=args.k or f['chi2_k'], svd_components=args.svd or f['svd_components'],
        random_state=args.seed if args.seed is not None else cfg['scraping']['random_state'])
