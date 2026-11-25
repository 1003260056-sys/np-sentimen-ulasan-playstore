"""[KODE PYTHON 5.2] Pelatihan LogisticRegression, RandomForest, dan XGBoost (Bab 5).

Mode 1: diimpor -> train_and_tune_logreg_rf_xgb(...)
Mode 2: mandiri  -> python src/train_logreg_rf_xgb.py [--cv-folds 5]
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Mengimpor XGBoost
from xgboost import XGBClassifier
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek, siapkan_folder_file


def train_and_tune_logreg_rf_xgb(processed_dir='data/processed', models_dir='models', text_col='teks_bersih',
                                 target_col='label_sentimen', cv_folds=5, scoring='f1_macro',
                                 random_state=42, top_n=10):
    """Melatih tiga model dengan label terenkode dan menyimpan LabelEncoder."""
    print('=' * 70)
    print('[BAB 5] PELATIHAN LOGREG, RANDOM FOREST, XGBOOST')
    print('=' * 70)
    folder = path_proyek(processed_dir)
    # Membaca data latih dan validasi
    train = pd.read_csv(os.path.join(folder, 'train_clean.csv'), encoding='utf-8')
    val = pd.read_csv(os.path.join(folder, 'val_clean.csv'), encoding='utf-8')
    for df in (train, val):
        df[text_col] = df[text_col].fillna('').astype(str)
    # Mengubah label teks menjadi angka
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train[target_col])
    y_val = encoder.transform(val[target_col])
    # Menyimpan LabelEncoder
    path_enc = os.path.join(path_proyek(models_dir), 'label_encoder.joblib')
    siapkan_folder_file(path_enc)
    joblib.dump(encoder, path_enc)
    print(f'LabelEncoder tersimpan: {path_enc} (kelas: {list(encoder.classes_)})')
    # Kandidat model dan grid hiperparameter
    kandidat = {
        'LogisticRegression': (LogisticRegression(random_state=random_state, max_iter=1000),
                               {'clf__C': [0.1, 1.0, 10.0]}),
        'RandomForest': (RandomForestClassifier(random_state=random_state),
                         {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 30]}),
        'XGBoost': (XGBClassifier(random_state=random_state, eval_metric='mlogloss', n_jobs=1),
                    {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 6], 'clf__learning_rate': [0.1]}),
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    hasil = {}
    for nama, (estimator, grid) in kandidat.items():
        pipa = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                         ('clf', estimator)])
        gs = GridSearchCV(pipa, grid, cv=cv, scoring=scoring, n_jobs=-1)
        gs.fit(train[text_col], y_train)
        pred = gs.best_estimator_.predict(val[text_col])
        f1 = f1_score(y_val, pred, average='macro', zero_division=0)
        print(f'\n--- {nama} ---')
        print(f'Parameter terbaik : {gs.best_params_}')
        print(f'CV {scoring}      : {gs.best_score_:.4f}')
        print(f'Validasi Akurasi  : {accuracy_score(y_val, pred):.4f} | Macro-F1: {f1:.4f}')
        print(classification_report(y_val, pred, target_names=encoder.classes_, zero_division=0))
        lokasi = os.path.join(path_proyek(models_dir), f'{nama.lower()}_model.joblib')
        siapkan_folder_file(lokasi)
        joblib.dump(gs.best_estimator_, lokasi)
        print(f'Model tersimpan   : {lokasi}')
        hasil[nama] = {'best_params': gs.best_params_, 'cv_score': gs.best_score_, 'val_macro_f1': f1}
        # Mengekstrak koefisien teratas per kelas khusus LogisticRegression
        if nama == 'LogisticRegression':
            fitur = gs.best_estimator_.named_steps['tfidf'].get_feature_names_out()
            koef = gs.best_estimator_.named_steps['clf'].coef_
            for i, kelas in enumerate(encoder.classes_):
                idx = koef[i].argsort()[::-1][:top_n]
                print(f'\nTop-{top_n} fitur kelas {kelas} (koefisien tertinggi):')
                for j in idx:
                    print(f'  {fitur[j]:<25} {koef[i][j]:.4f}')
    return hasil


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Latih LogReg, RF, XGBoost (Bab 5)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--processed-dir', default=None, help='Folder data/processed')
    parser.add_argument('--cv-folds', type=int, default=None, help='Jumlah lipatan CV')
    parser.add_argument('--seed', type=int, default=None, help='Random state')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    train_and_tune_logreg_rf_xgb(
        processed_dir=args.processed_dir or cfg['paths']['processed_dir'], models_dir=cfg['paths']['models_dir'],
        text_col=cfg['schema']['clean_text_col'], target_col=cfg['schema']['target_col'],
        cv_folds=args.cv_folds or cfg['training']['cv_folds'], scoring=cfg['training']['scoring'],
        random_state=args.seed if args.seed is not None else cfg['scraping']['random_state'])
