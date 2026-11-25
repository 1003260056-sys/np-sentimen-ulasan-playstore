"""[KODE PYTHON 4.2] Word embeddings Word2Vec & FastText (Bab 4).

Mode 1: diimpor -> EmbeddingAggregator, train_and_evaluate_embeddings(...)
Mode 2: mandiri  -> python src/word_embeddings.py [--dim 100 --window 5]
"""

# Mengimpor modul dasar
import sys
import os
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor numpy dan pandas
import numpy as np
import pandas as pd
# Mengimpor model embedding gensim
from gensim.models import Word2Vec, FastText
# Mengimpor TF-IDF, regresi logistik, dan metrik
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek, siapkan_folder_file

# Kata kunci domain untuk uji kemiripan (bentuk dasar hasil stemming)
KATA_KUNCI = ['kirim', 'barang', 'aplikasi', 'harga', 'lambat', 'bagus', 'kurir', 'promo']
# Kata OOV/typo untuk uji ketahanan FastText
KATA_OOV = ['chargerr', 'bateray', 'hapeee', 'lemottt']


class EmbeddingAggregator:
    """Mengubah dokumen (daftar token) menjadi satu vektor dengan agregasi vektor kata."""

    def __init__(self, wv, dim):
        # Menyimpan KeyedVectors dan dimensinya
        self.wv = wv
        self.dim = dim

    def mean_pooling(self, dokumen):
        """Rata-rata vektor kata pada tiap dokumen."""
        hasil = []
        for token in dokumen:
            # Mengambil vektor hanya untuk kata yang dikenal
            vek = [self.wv[t] for t in token if t in self.wv.key_to_index]
            # Dokumen tanpa kata dikenal menjadi vektor nol
            hasil.append(np.mean(vek, axis=0) if vek else np.zeros(self.dim))
        return np.vstack(hasil)

    def tfidf_weighted_pooling(self, dokumen, idf):
        """Rata-rata berbobot vektor kata dengan bobot IDF."""
        hasil = []
        for token in dokumen:
            vek, bobot = [], []
            for t in token:
                if t in self.wv.key_to_index:
                    vek.append(self.wv[t])
                    # Kata tanpa IDF diberi bobot 1.0
                    bobot.append(idf.get(t, 1.0))
            if vek:
                hasil.append(np.average(vek, axis=0, weights=bobot))
            else:
                hasil.append(np.zeros(self.dim))
        return np.vstack(hasil)


def train_and_evaluate_embeddings(processed_dir='data/processed', models_dir='models', text_col='teks_bersih',
                                  target_col='label_sentimen', vector_size=100, window=5, min_count=2,
                                  random_state=42):
    """Melatih Word2Vec dan FastText, lalu mengevaluasi agregasi dokumen dengan LogReg."""
    print('=' * 70)
    print('[BAB 4] WORD EMBEDDINGS: WORD2VEC & FASTTEXT')
    print('=' * 70)
    folder = path_proyek(processed_dir)
    # Membaca data latih dan validasi
    train = pd.read_csv(os.path.join(folder, 'train_clean.csv'), encoding='utf-8')
    val = pd.read_csv(os.path.join(folder, 'val_clean.csv'), encoding='utf-8')
    for df in (train, val):
        df[text_col] = df[text_col].fillna('').astype(str)
    # Tokenisasi sederhana berdasarkan spasi
    tok_train = [t.split() for t in train[text_col]]
    tok_val = [t.split() for t in val[text_col]]
    # Melatih Word2Vec Skip-gram
    w2v = Word2Vec(sentences=tok_train, vector_size=vector_size, window=window, min_count=min_count,
                   sg=1, seed=random_state, workers=1, epochs=30)
    # Melatih FastText dengan n-gram karakter 3-6
    ft = FastText(sentences=tok_train, vector_size=vector_size, window=window, min_count=min_count,
                  sg=1, min_n=3, max_n=6, seed=random_state, workers=1, epochs=30)
    # Menyimpan model (folder dibuat defensif)
    path_ft = os.path.join(path_proyek(models_dir), 'fasttext_ulasan.model')
    path_w2v = os.path.join(path_proyek(models_dir), 'word2vec_ulasan.model')
    siapkan_folder_file(path_ft)
    ft.save(path_ft)
    w2v.save(path_w2v)
    print(f'Kosakata Word2Vec: {len(w2v.wv)} | FastText: {len(ft.wv)}')
    print(f'Model tersimpan: {path_ft}')
    # Kata-kata terdekat untuk kata kunci domain
    for nama, model in [('Word2Vec', w2v), ('FastText', ft)]:
        print(f'\n[{nama}] most_similar kata kunci domain:')
        for kata in KATA_KUNCI:
            if kata in model.wv.key_to_index:
                mirip = ', '.join(f'{k}({s:.2f})' for k, s in model.wv.most_similar(kata, topn=5))
                print(f'  {kata:<10} -> {mirip}')
    # Uji ketahanan OOV/typo
    print('\nUji ketahanan kata OOV/typo:')
    for kata in KATA_OOV:
        status_w2v = 'ada' if kata in w2v.wv.key_to_index else 'OOV (KeyError)'
        mirip = ', '.join(f'{k}({s:.2f})' for k, s in ft.wv.most_similar(kata, topn=3))
        print(f'  {kata:<10} Word2Vec: {status_w2v:<15} FastText -> {mirip}')
    # Menyiapkan bobot IDF untuk pooling berbobot
    tf = TfidfVectorizer(token_pattern=r'\S+').fit(train[text_col])
    idf = dict(zip(tf.get_feature_names_out(), tf.idf_))
    # Evaluasi tiap kombinasi model dan metode agregasi
    print('\nEvaluasi LogReg pada validasi (Macro-F1):')
    hasil = {}
    for nama, model in [('Word2Vec', w2v), ('FastText', ft)]:
        agg = EmbeddingAggregator(model.wv, vector_size)
        for metode in ['mean', 'tfidf']:
            if metode == 'mean':
                A_tr, A_va = agg.mean_pooling(tok_train), agg.mean_pooling(tok_val)
            else:
                A_tr, A_va = agg.tfidf_weighted_pooling(tok_train, idf), agg.tfidf_weighted_pooling(tok_val, idf)
            # Melatih pengklasifikasi sederhana
            clf = LogisticRegression(max_iter=1000, random_state=random_state).fit(A_tr, train[target_col])
            pred = clf.predict(A_va)
            f1 = f1_score(val[target_col], pred, average='macro', zero_division=0)
            hasil[f'{nama}-{metode}'] = {'akurasi': accuracy_score(val[target_col], pred), 'macro_f1': f1}
            print(f'  {nama:<9} pooling={metode:<6} Akurasi={hasil[f"{nama}-{metode}"]["akurasi"]:.4f}  Macro-F1={f1:.4f}')
    return {'word2vec': w2v, 'fasttext': ft, 'evaluasi': hasil}


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Word embeddings (Bab 4)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--processed-dir', default=None, help='Folder data/processed')
    parser.add_argument('--dim', type=int, default=None, help='Dimensi vektor')
    parser.add_argument('--window', type=int, default=None, help='Ukuran jendela konteks')
    parser.add_argument('--seed', type=int, default=None, help='Random state')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    f = cfg['features']
    train_and_evaluate_embeddings(
        processed_dir=args.processed_dir or cfg['paths']['processed_dir'], models_dir=cfg['paths']['models_dir'],
        text_col=cfg['schema']['clean_text_col'], target_col=cfg['schema']['target_col'],
        vector_size=args.dim or f['embedding_dim'], window=args.window or f['embedding_window'],
        random_state=args.seed if args.seed is not None else cfg['scraping']['random_state'])
