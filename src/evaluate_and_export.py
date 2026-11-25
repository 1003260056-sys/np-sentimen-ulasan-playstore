"""[KODE PYTHON 5.3] Evaluasi komprehensif pada data uji dan ekspor model juara (Bab 5).

Mode 1: diimpor -> run_comprehensive_evaluation(...)
Mode 2: mandiri  -> python src/evaluate_and_export.py
"""

# Mengimpor modul dasar
import sys
import os
import time
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor numpy, pandas, joblib
import numpy as np
import pandas as pd
import joblib
# Mengimpor baseline dan metrik
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Mengimpor preprocessor dan fungsi bantu proyek
from src.preprocessing_pipeline import IndonesianTextPreprocessor
from src import muat_config, path_proyek, siapkan_folder_file, prediksi_label

# Nama model -> berkas joblib
BERKAS_MODEL = {
    'MultinomialNB': 'multinomialnb_model.joblib', 'ComplementNB': 'complementnb_model.joblib',
    'LinearSVC': 'linearsvc_model.joblib', 'LogisticRegression': 'logisticregression_model.joblib',
    'RandomForest': 'randomforest_model.joblib', 'XGBoost': 'xgboost_model.joblib',
}
# Model yang dilatih dengan label terenkode angka
MODEL_TERENKODE = {'LogisticRegression', 'RandomForest', 'XGBoost'}

# Kalimat ulasan baru untuk demonstrasi
KALIMAT_DEMO = [
    'Aplikasinya bagus banget, barang ori dan pengiriman cepat!',
    'Biasa saja sih, kadang lancar kadang lemot',
    'Kecewa berat, pesanan tidak sampai dan seller tidak ramah',
]


def run_comprehensive_evaluation(processed_dir='data/processed', models_dir='models', text_col='teks_bersih',
                                 target_col='label_sentimen', comparison_csv='data/processed/hasil_komparasi_model_uts.csv',
                                 best_model_path='models/best_sentiment_pipeline.joblib',
                                 use_stemmer=True, negation_whitelist=None):
    """Mengevaluasi baseline + 6 model pada data uji, menyimpan peringkat dan model juara."""
    print('=' * 70)
    print('[BAB 5] EVALUASI KOMPREHENSIF PADA DATA UJI')
    print('=' * 70)
    folder = path_proyek(processed_dir)
    # Membaca data latih (untuk baseline) dan uji
    train = pd.read_csv(os.path.join(folder, 'train_clean.csv'), encoding='utf-8')
    test = pd.read_csv(os.path.join(folder, 'test_clean.csv'), encoding='utf-8')
    for df in (train, test):
        df[text_col] = df[text_col].fillna('').astype(str)
    y_test = test[target_col].to_numpy()
    X_test = test[text_col].tolist()
    baris = []

    def _hitung(nama, y_pred, latensi):
        # Menghitung seluruh metrik makro
        return {'Model': nama, 'Accuracy': accuracy_score(y_test, y_pred),
                'Macro Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'Macro Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'Macro F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'Latency (ms)': latensi}

    # Baseline: selalu memprediksi kelas mayoritas
    dummy = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train), 1)), train[target_col])
    mulai = time.perf_counter()
    pred_dummy = dummy.predict(np.zeros((len(test), 1)))
    baris.append(_hitung('Dummy (most_frequent)', pred_dummy, (time.perf_counter() - mulai) / len(test) * 1000))
    # Mengevaluasi tiap model terlatih
    model_dimuat = {}
    for nama, berkas in BERKAS_MODEL.items():
        lokasi = os.path.join(path_proyek(models_dir), berkas)
        if not os.path.exists(lokasi):
            print(f'[PERINGATAN] Model {nama} tidak ditemukan ({lokasi}); dilewati.')
            continue
        model = joblib.load(lokasi)
        model_dimuat[nama] = model
        # Mengukur waktu prediksi seluruh data uji
        mulai = time.perf_counter()
        pred = model.predict(X_test)
        latensi = (time.perf_counter() - mulai) / len(X_test) * 1000
        # Mengubah label angka kembali ke teks bila perlu
        if nama in MODEL_TERENKODE:
            encoder = joblib.load(os.path.join(path_proyek(models_dir), 'label_encoder.joblib'))
            pred = encoder.inverse_transform(np.asarray(pred))
        baris.append(_hitung(nama, pred, latensi))
    # Menyusun tabel peringkat berdasarkan Macro F1
    tabel = pd.DataFrame(baris).sort_values('Macro F1-Score', ascending=False).reset_index(drop=True)
    tabel.insert(0, 'Peringkat', range(1, len(tabel) + 1))
    print(tabel.round(4).to_string(index=False))
    # Menyimpan tabel
    lokasi_csv = path_proyek(comparison_csv)
    siapkan_folder_file(lokasi_csv)
    tabel.to_csv(lokasi_csv, index=False, encoding='utf-8')
    print(f'\nTabel komparasi tersimpan: {lokasi_csv}')
    # Memilih juara (selain baseline)
    kandidat = tabel[tabel['Model'].isin(model_dimuat.keys())]
    if kandidat.empty:
        raise RuntimeError('Tidak ada model terlatih yang ditemukan. Jalankan --step train terlebih dahulu.')
    juara = kandidat.iloc[0]['Model']
    lokasi_best = path_proyek(best_model_path)
    siapkan_folder_file(lokasi_best)
    joblib.dump(model_dimuat[juara], lokasi_best)
    print(f'Model juara: {juara} -> {lokasi_best}')
    # Demonstrasi prediksi pada kalimat baru
    prep = IndonesianTextPreprocessor(use_stemmer=use_stemmer, negation_whitelist=negation_whitelist).fit(None)
    bersih = prep.transform(KALIMAT_DEMO)
    prediksi = prediksi_label(model_dimuat[juara], bersih, os.path.join(models_dir, 'label_encoder.joblib'))
    print('\nDemonstrasi prediksi:')
    for mentah, b, p in zip(KALIMAT_DEMO, bersih, prediksi):
        print(f'  Mentah : {mentah}\n  Bersih : {b}\n  Prediksi: {p}\n')
    return {'tabel': tabel, 'juara': juara}


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Evaluasi & ekspor model juara (Bab 5)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--processed-dir', default=None, help='Folder data/processed')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    p = cfg['paths']
    run_comprehensive_evaluation(
        processed_dir=args.processed_dir or p['processed_dir'], models_dir=p['models_dir'],
        text_col=cfg['schema']['clean_text_col'], target_col=cfg['schema']['target_col'],
        comparison_csv=p['comparison_csv'], best_model_path=p['best_model'],
        use_stemmer=cfg['preprocessing']['use_stemmer'],
        negation_whitelist=cfg['preprocessing']['negation_whitelist'])
