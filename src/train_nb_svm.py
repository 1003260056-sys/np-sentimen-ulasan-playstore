"""[KODE PYTHON 5.1] Pelatihan MultinomialNB, ComplementNB, dan LinearSVC dengan GridSearchCV (Bab 5).

Mode 1: diimpor -> train_and_tune_nb_svm(...)
Mode 2: mandiri  -> python src/train_nb_svm.py [--cv-folds 5]
"""

# Mengimpor modul dasar
import sys
import os
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pandas dan joblib
import pandas as pd
import joblib
# Mengimpor komponen scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek, siapkan_folder_file


def train_and_tune_nb_svm(processed_dir='data/processed', models_dir='models', text_col='teks_bersih',
                          target_col='label_sentimen', cv_folds=5, scoring='f1_macro', random_state=42):
    """Melatih dan menyetel tiga model; menyimpan model terbaik masing-masing."""
    print('=' * 70)
    print('[BAB 5] PELATIHAN MULTINOMIALNB, COMPLEMENTNB, LINEARSVC')
    print('=' * 70)
    folder = path_proyek(processed_dir)
    # Membaca data latih dan validasi
    train = pd.read_csv(os.path.join(folder, 'train_clean.csv'), encoding='utf-8')
    val = pd.read_csv(os.path.join(folder, 'val_clean.csv'), encoding='utf-8')
    for df in (train, val):
        df[text_col] = df[text_col].fillna('').astype(str)
    # Kandidat: (estimator, grid hiperparameter)
    kandidat = {
        'MultinomialNB': (MultinomialNB(), {'clf__alpha': [0.1, 0.5, 1.0]}),
        'ComplementNB': (ComplementNB(), {'clf__alpha': [0.1, 0.5, 1.0]}),
        'LinearSVC': (LinearSVC(random_state=random_state, max_iter=2000), {'clf__C': [0.1, 1.0, 10.0]}),
    }
    # Skema validasi silang bertingkat
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    hasil = {}
    for nama, (estimator, grid) in kandidat.items():
        # Membangun pipeline TF-IDF + pengklasifikasi
        pipa = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                         ('clf', estimator)])
        # Pencarian grid
        gs = GridSearchCV(pipa, grid, cv=cv, scoring=scoring, n_jobs=-1)
        gs.fit(train[text_col], train[target_col])
        # Evaluasi pada validasi
        pred = gs.best_estimator_.predict(val[text_col])
        f1 = f1_score(val[target_col], pred, average='macro', zero_division=0)
        print(f'\n--- {nama} ---')
        print(f'Parameter terbaik : {gs.best_params_}')
        print(f'CV {scoring}      : {gs.best_score_:.4f}')
        print(f'Validasi Akurasi  : {accuracy_score(val[target_col], pred):.4f} | Macro-F1: {f1:.4f}')
        print(classification_report(val[target_col], pred, zero_division=0))
        # Menyimpan model terbaik
        lokasi = os.path.join(path_proyek(models_dir), f'{nama.lower()}_model.joblib')
        siapkan_folder_file(lokasi)
        joblib.dump(gs.best_estimator_, lokasi)
        print(f'Model tersimpan   : {lokasi}')
        hasil[nama] = {'best_params': gs.best_params_, 'cv_score': gs.best_score_, 'val_macro_f1': f1}
    return hasil


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Latih NB & SVM (Bab 5)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--processed-dir', default=None, help='Folder data/processed')
    parser.add_argument('--cv-folds', type=int, default=None, help='Jumlah lipatan CV')
    parser.add_argument('--seed', type=int, default=None, help='Random state')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    train_and_tune_nb_svm(
        processed_dir=args.processed_dir or cfg['paths']['processed_dir'], models_dir=cfg['paths']['models_dir'],
        text_col=cfg['schema']['clean_text_col'], target_col=cfg['schema']['target_col'],
        cv_folds=args.cv_folds or cfg['training']['cv_folds'], scoring=cfg['training']['scoring'],
        random_state=args.seed if args.seed is not None else cfg['scraping']['random_state'])
