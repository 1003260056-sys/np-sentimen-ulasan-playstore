"""[KODE PYTHON 2.3] Validasi anotasi (Cohen's Kappa) dan pembagian dataset (Bab 2).

Mode 1: diimpor -> validate_and_split_dataset(...)
Mode 2: mandiri  -> python src/split_dataset.py [--filepath ...]
"""

# Mengimpor modul dasar
import sys
import os
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pandas dan numpy
import pandas as pd
import numpy as np
# Mengimpor pembagi data dan metrik kappa
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek


def _interpretasi_kappa(k):
    """Interpretasi skala Landis & Koch (1977)."""
    # Kappa negatif berarti lebih buruk dari kebetulan
    if k < 0:
        return 'Poor (buruk)'
    if k <= 0.20:
        return 'Slight (sangat rendah)'
    if k <= 0.40:
        return 'Fair (cukup)'
    if k <= 0.60:
        return 'Moderate (sedang)'
    if k <= 0.80:
        return 'Substantial (kuat)'
    return 'Almost Perfect (hampir sempurna)'


def validate_and_split_dataset(filepath='data/raw/ulasan_pengguna.csv', output_dir='data/processed',
                               train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Menghitung Cohen's Kappa dua anotator, lalu membagi data secara bertingkat."""
    # Memeriksa total rasio
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio harus sama dengan 1.0')
    # Membaca dan membersihkan data
    df = pd.read_csv(path_proyek(filepath), encoding='utf-8')
    df = df.dropna(subset=['teks_ulasan', 'label_sentimen']).drop_duplicates(subset=['teks_ulasan']).reset_index(drop=True)
    print('=' * 70)
    print('[BAB 2] VALIDASI ANOTASI DAN PEMBAGIAN DATASET')
    print('=' * 70)
    # Anotator 1 = label dari pemetaan rating; anotator 2 disimulasikan dengan 8% ketidaksepakatan acak
    rng = np.random.default_rng(random_state)
    anotator_1 = df['label_sentimen'].to_numpy()
    anotator_2 = anotator_1.copy()
    kelas = np.unique(anotator_1)
    # Memilih baris yang "tidak disepakati"
    idx = rng.choice(len(df), size=int(0.08 * len(df)), replace=False)
    for i in idx:
        # Mengganti dengan kelas lain secara acak
        pilihan = [c for c in kelas if c != anotator_1[i]]
        anotator_2[i] = rng.choice(pilihan)
    # Menghitung kappa
    kappa = cohen_kappa_score(anotator_1, anotator_2)
    print(f"Cohen's Kappa (simulasi 2 anotator): {kappa:.4f} -> {_interpretasi_kappa(kappa)}")
    # Pembagian tahap 1: memisahkan data uji
    sisa, uji = train_test_split(df, test_size=test_ratio, stratify=df['label_sentimen'], random_state=random_state)
    # Pembagian tahap 2: memisahkan validasi dari sisa
    rel_val = val_ratio / (train_ratio + val_ratio)
    latih, val = train_test_split(sisa, test_size=rel_val, stratify=sisa['label_sentimen'], random_state=random_state)
    # Membuat folder keluaran secara defensif
    folder = path_proyek(output_dir)
    os.makedirs(folder, exist_ok=True)
    # Menyimpan masing-masing bagian
    for nama, bagian in [('train_raw.csv', latih), ('val_raw.csv', val), ('test_raw.csv', uji)]:
        bagian.to_csv(os.path.join(folder, nama), index=False, encoding='utf-8')
        print(f'{nama:<14}: {len(bagian)} baris ({len(bagian) / len(df) * 100:.1f}%)')
        print('   ', bagian['label_sentimen'].value_counts(normalize=True).round(3).to_dict())
    # Mengembalikan ketiga bagian dan kappa
    return {'train': latih, 'val': val, 'test': uji, 'kappa': kappa}


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Validasi & split dataset (Bab 2)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--filepath', default=None, help='Path dataset CSV')
    parser.add_argument('--output-dir', default=None, help='Folder keluaran')
    parser.add_argument('--seed', type=int, default=None, help='Random state')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    sp = cfg['split']
    validate_and_split_dataset(
        filepath=args.filepath or cfg['paths']['raw_dataset'],
        output_dir=args.output_dir or cfg['paths']['processed_dir'],
        train_ratio=sp['train_ratio'], val_ratio=sp['val_ratio'], test_ratio=sp['test_ratio'],
        random_state=args.seed if args.seed is not None else cfg['scraping']['random_state'])
