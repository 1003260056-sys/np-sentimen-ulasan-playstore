"""[KODE PYTHON 3.3] Transformer preprocessing kompatibel scikit-learn dan eksekusi pipeline (Bab 3).

Mode 1: diimpor -> IndonesianTextPreprocessor, execute_preprocessing_pipeline(...)
Mode 2: mandiri  -> python src/preprocessing_pipeline.py [--processed-dir ...]
"""

# Mengimpor modul dasar
import sys
import os
import time
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pandas
import pandas as pd
# Mengimpor kelas dasar estimator scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin
# Mengimpor preprocessor lanjutan
from src.stemming_demo import AdvancedTextPreprocessor
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek


class IndonesianTextPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer scikit-learn: mengubah daftar teks mentah menjadi teks bersih."""

    def __init__(self, use_stemmer=True, negation_whitelist=None):
        # Parameter hanya disimpan apa adanya (aturan scikit-learn)
        self.use_stemmer = use_stemmer
        self.negation_whitelist = negation_whitelist

    def _siapkan(self):
        # Membuat preprocessor internal sekali saja (lazy)
        if getattr(self, '_prep', None) is None:
            self._prep = AdvancedTextPreprocessor(use_stemmer=self.use_stemmer,
                                                  negation_whitelist=self.negation_whitelist)

    def fit(self, X, y=None):
        # Tidak ada yang dipelajari; hanya menyiapkan sumber daya
        self._siapkan()
        return self

    def transform_one(self, teks):
        """Membersihkan satu kalimat."""
        self._siapkan()
        return self._prep.process(teks)

    def transform(self, X):
        """Membersihkan banyak kalimat dan mengembalikan list string."""
        self._siapkan()
        return [self._prep.process('' if pd.isna(t) else t) for t in X]


def _ukuran_kosakata(seri):
    """Menghitung jumlah kata unik pada sebuah Series teks."""
    himpunan = set()
    for teks in seri.fillna('').astype(str):
        himpunan.update(teks.lower().split())
    return len(himpunan)


def execute_preprocessing_pipeline(processed_dir='data/processed', text_col='teks_ulasan',
                                   clean_col='teks_bersih', use_stemmer=True, negation_whitelist=None):
    """Membersihkan train/val/test raw lalu menyimpan versi *_clean.csv."""
    print('=' * 70)
    print('[BAB 3] PIPELINE PREPROCESSING')
    print('=' * 70)
    folder = path_proyek(processed_dir)
    # Membuat folder secara defensif
    os.makedirs(folder, exist_ok=True)
    # Membuat transformer
    prep = IndonesianTextPreprocessor(use_stemmer=use_stemmer, negation_whitelist=negation_whitelist).fit(None)
    hasil = {}
    for bagian in ['train', 'val', 'test']:
        # Membaca berkas raw
        df = pd.read_csv(os.path.join(folder, f'{bagian}_raw.csv'), encoding='utf-8')
        df[text_col] = df[text_col].fillna('').astype(str)
        # Mengukur waktu eksekusi
        mulai = time.perf_counter()
        df[clean_col] = prep.transform(df[text_col])
        durasi = time.perf_counter() - mulai
        # Menghitung reduksi kosakata
        sebelum = _ukuran_kosakata(df[text_col])
        sesudah = _ukuran_kosakata(df[clean_col])
        reduksi = (1 - sesudah / sebelum) * 100 if sebelum else 0.0
        print(f'{bagian:<5}: {len(df)} baris | waktu {durasi:.2f} dtk | '
              f'vokabulari {sebelum} -> {sesudah} (reduksi {reduksi:.2f}%)')
        # Menyimpan berkas clean
        df.to_csv(os.path.join(folder, f'{bagian}_clean.csv'), index=False, encoding='utf-8')
        hasil[bagian] = df
    return hasil


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Pipeline preprocessing (Bab 3)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--processed-dir', default=None, help='Folder data/processed')
    parser.add_argument('--no-stemmer', action='store_true', help='Matikan stemming')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    execute_preprocessing_pipeline(
        processed_dir=args.processed_dir or cfg['paths']['processed_dir'],
        text_col=cfg['schema']['text_col'], clean_col=cfg['schema']['clean_text_col'],
        use_stemmer=False if args.no_stemmer else cfg['preprocessing']['use_stemmer'],
        negation_whitelist=cfg['preprocessing']['negation_whitelist'])
