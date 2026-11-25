"""Orkestrator CLI proyek Analisis Sentimen Ulasan Pengguna.

Tahapan: check -> ingest -> preprocess -> features -> train -> visualize (+ predict).

Contoh:
    python run_pipeline.py --step all
    python run_pipeline.py --step ingest --input-data data/raw/punya_saya.csv
    python run_pipeline.py --step predict --text "barangnya bagus banget"

Cara menambah step baru (mis. deep_learning, transformer, error_analysis, absa, serve):
    1. Tulis fungsi  def step_nama(ctx): ...  (ctx = objek Konteks berisi cfg, seed, dsb.).
    2. Daftarkan dengan dekorator  @daftarkan_step('nama', help='...', dalam_all=True/False).
    Daftar --step, teks help, dan urutan --step all dibangun otomatis dari registri.
"""

# Mengimpor modul dasar
import os
import sys
import argparse
from collections import OrderedDict

# Memastikan folder akar proyek ada di sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek, prediksi_label

# Kolom wajib dataset
KOLOM_WAJIB = ['id_ulasan', 'tanggal', 'kategori_produk', 'rating', 'teks_ulasan', 'label_sentimen']

# Registri step: nama -> {fungsi, help, dalam_all}; urutan pendaftaran = urutan eksekusi
REGISTRI = OrderedDict()


class Konteks:
    """Menyimpan konfigurasi dan argumen CLI yang dibagikan ke semua step."""

    def __init__(self, cfg, args):
        # Menyimpan konfigurasi dan argumen
        self.cfg = cfg
        self.args = args
        # Menentukan seed (override CLI > config)
        self.seed = args.seed if args.seed is not None else cfg['scraping']['random_state']
        # Menentukan sumber dataset (kustom atau bawaan config)
        self.raw_path = args.input_data or cfg['paths']['raw_dataset']
        # Penanda apakah memakai dataset kustom
        self.input_kustom = bool(args.input_data)


def daftarkan_step(nama, help='', dalam_all=True):
    """Dekorator untuk mendaftarkan sebuah step ke registri."""
    def pembungkus(fungsi):
        # Menyimpan metadata step
        REGISTRI[nama] = {'fungsi': fungsi, 'help': help, 'dalam_all': dalam_all}
        return fungsi
    return pembungkus


def log(pesan):
    """Mencetak pesan log tahapan pipeline."""
    print(f'\n##### [PIPELINE] {pesan} #####')


@daftarkan_step('check', help='verifikasi lingkungan & baseline TF-IDF + MultinomialNB (Bab 1)')
def step_check(ctx):
    # Mengimpor fungsi tahap check
    from src.check_env import verify_env_and_run_baseline
    log('Step "check": verifikasi lingkungan')
    verify_env_and_run_baseline(filepath=ctx.raw_path, test_size=0.3, random_state=ctx.seed)


@daftarkan_step('ingest', help='pengambilan data (scraping), EDA korpus, dan pembagian train/val/test (Bab 2)')
def step_ingest(ctx):
    # Mengimpor fungsi tahap ingest
    from src.eda_korpus import run_corpus_eda
    from src.split_dataset import validate_and_split_dataset
    log('Step "ingest": scraping/EDA/split data')
    sc, sp = ctx.cfg['scraping'], ctx.cfg['split']
    # Scraping dilewati bila dataset kustom diberikan
    if not ctx.input_kustom:
        from src.scrape_dataset import scrape_playstore_reviews
        apps = [dict(a) for a in sc['apps']]
        if ctx.args.count is not None:
            for a in apps:
                a['count'] = ctx.args.count
        scrape_playstore_reviews(apps=apps, lang=sc['lang'], country=sc['country'], output_path=ctx.raw_path,
                                 missing_rate=sc['missing_noise_rate'], duplicate_rows=sc['duplicate_noise_rows'],
                                 random_seed=ctx.seed)
    else:
        print(f'Memakai dataset kustom: {ctx.raw_path} (scraping dilewati)')
    run_corpus_eda(filepath=ctx.raw_path, top_n=10)
    validate_and_split_dataset(filepath=ctx.raw_path, output_dir=ctx.cfg['paths']['processed_dir'],
                               train_ratio=sp['train_ratio'], val_ratio=sp['val_ratio'],
                               test_ratio=sp['test_ratio'], random_state=ctx.seed)


@daftarkan_step('preprocess', help='kamus slang, demo stemming, dan pipeline preprocessing teks (Bab 3)')
def step_preprocess(ctx):
    # Mengimpor fungsi tahap preprocess
    from src.slang_normalizer import create_slang_csv
    from src.stemming_demo import run_stemming_demo
    from src.preprocessing_pipeline import execute_preprocessing_pipeline
    log('Step "preprocess": normalisasi & pembersihan teks')
    pre, s = ctx.cfg['preprocessing'], ctx.cfg['schema']
    create_slang_csv(ctx.cfg['paths']['slang_dict'])
    run_stemming_demo(use_stemmer=pre['use_stemmer'], negation_whitelist=pre['negation_whitelist'])
    execute_preprocessing_pipeline(processed_dir=ctx.cfg['paths']['processed_dir'], text_col=s['text_col'],
                                   clean_col=s['clean_text_col'], use_stemmer=pre['use_stemmer'],
                                   negation_whitelist=pre['negation_whitelist'])


@daftarkan_step('features', help='ekstraksi fitur TF-IDF, word embeddings, seleksi fitur & SMOTE (Bab 4)')
def step_features(ctx):
    # Mengimpor fungsi tahap features
    from src.tfidf_features import extract_tfidf_features
    from src.word_embeddings import train_and_evaluate_embeddings
    from src.feature_selection_smote import run_feature_selection_and_smote
    log('Step "features": rekayasa fitur')
    f, s, p = ctx.cfg['features'], ctx.cfg['schema'], ctx.cfg['paths']
    extract_tfidf_features(processed_dir=p['processed_dir'], models_dir=p['models_dir'],
                           text_col=s['clean_text_col'], target_col=s['target_col'], classes=s['classes'],
                           ngram_range=f['ngram_range'], min_df=f['min_df'], max_df=f['max_df'],
                           sublinear_tf=f['sublinear_tf'], norm=f['norm'])
    train_and_evaluate_embeddings(processed_dir=p['processed_dir'], models_dir=p['models_dir'],
                                  text_col=s['clean_text_col'], target_col=s['target_col'],
                                  vector_size=f['embedding_dim'], window=f['embedding_window'],
                                  random_state=ctx.seed)
    run_feature_selection_and_smote(processed_dir=p['processed_dir'], text_col=s['clean_text_col'],
                                    target_col=s['target_col'], classes=s['classes'],
                                    ngram_range=f['ngram_range'], min_df=f['min_df'], max_df=f['max_df'],
                                    chi2_k=f['chi2_k'], svd_components=f['svd_components'],
                                    random_state=ctx.seed)


@daftarkan_step('train', help='pelatihan & tuning 6 model, evaluasi data uji, ekspor model juara (Bab 5)')
def step_train(ctx):
    # Mengimpor fungsi tahap train
    from src.train_nb_svm import train_and_tune_nb_svm
    from src.train_logreg_rf_xgb import train_and_tune_logreg_rf_xgb
    from src.evaluate_and_export import run_comprehensive_evaluation
    log('Step "train": pelatihan, evaluasi, dan ekspor model')
    p, s, t, pre = ctx.cfg['paths'], ctx.cfg['schema'], ctx.cfg['training'], ctx.cfg['preprocessing']
    train_and_tune_nb_svm(processed_dir=p['processed_dir'], models_dir=p['models_dir'],
                          text_col=s['clean_text_col'], target_col=s['target_col'],
                          cv_folds=t['cv_folds'], scoring=t['scoring'], random_state=ctx.seed)
    train_and_tune_logreg_rf_xgb(processed_dir=p['processed_dir'], models_dir=p['models_dir'],
                                 text_col=s['clean_text_col'], target_col=s['target_col'],
                                 cv_folds=t['cv_folds'], scoring=t['scoring'], random_state=ctx.seed)
    run_comprehensive_evaluation(processed_dir=p['processed_dir'], models_dir=p['models_dir'],
                                 text_col=s['clean_text_col'], target_col=s['target_col'],
                                 comparison_csv=p['comparison_csv'], best_model_path=p['best_model'],
                                 use_stemmer=pre['use_stemmer'], negation_whitelist=pre['negation_whitelist'])


@daftarkan_step('visualize', help='membuat 10 grafik PNG 300 DPI di outputs/figures/')
def step_visualize(ctx):
    # Mengimpor fungsi tahap visualize
    from src.visualize import generate_all_visualizations
    log('Step "visualize": membuat grafik')
    generate_all_visualizations(ctx.cfg)


@daftarkan_step('predict', help='inferensi sentimen untuk kalimat pada --text memakai model juara', dalam_all=False)
def step_predict(ctx):
    # Teks wajib diisi untuk step predict
    if not ctx.args.text:
        raise SystemExit('Argumen --text wajib diisi untuk --step predict.')
    # Mengimpor dependensi inferensi
    from src.preprocessing_pipeline import IndonesianTextPreprocessor
    import joblib
    log('Step "predict": inferensi sentimen')
    pre, p = ctx.cfg['preprocessing'], ctx.cfg['paths']
    lokasi = path_proyek(p['best_model'])
    # Memastikan model juara tersedia
    if not os.path.exists(lokasi):
        raise SystemExit(f'Model juara belum ada: {lokasi}. Jalankan "python run_pipeline.py --step train" dahulu.')
    # Membersihkan teks input
    prep = IndonesianTextPreprocessor(use_stemmer=pre['use_stemmer'],
                                      negation_whitelist=pre['negation_whitelist']).fit(None)
    bersih = prep.transform_one(ctx.args.text)
    # Memuat model dan memprediksi (LabelEncoder ditangani otomatis)
    model = joblib.load(lokasi)
    hasil = prediksi_label(model, [bersih], os.path.join(p['models_dir'], 'label_encoder.joblib'))[0]
    print('=' * 70)
    print(f'Teks mentah : {ctx.args.text}')
    print(f'Teks bersih : {bersih}')
    print(f'Prediksi    : {hasil}')
    print('=' * 70)


def bangun_parser():
    """Membangun parser argumen; pilihan --step diambil dari registri."""
    parser = argparse.ArgumentParser(description='Orkestrator pipeline analisis sentimen ulasan pengguna')
    # Menyusun teks help dari registri
    daftar_help = '; '.join(f'{n}: {m["help"]}' for n, m in REGISTRI.items())
    parser.add_argument('--config', default='config.yaml', help='Path berkas konfigurasi')
    parser.add_argument('--step', default='all', choices=['all'] + list(REGISTRI.keys()),
                        help='Tahapan yang dijalankan. all: check -> ingest -> preprocess -> features -> '
                             'train -> visualize. ' + daftar_help)
    parser.add_argument('--input-data', default=None, help='CSV kustom (melewati scraping pada step ingest)')
    parser.add_argument('--text', default=None, help='Kalimat uji untuk --step predict')
    parser.add_argument('--count', type=int, default=None, help='Override jumlah ulasan per aplikasi (step ingest)')
    parser.add_argument('--seed', type=int, default=None, help='Override random_state')
    return parser


def main():
    # Membaca argumen CLI
    args = bangun_parser().parse_args()
    # Memuat konfigurasi dan membuat konteks
    ctx = Konteks(muat_config(args.config), args)
    # Memvalidasi skema dataset kustom
    if ctx.input_kustom:
        import pandas as pd
        kolom = pd.read_csv(path_proyek(ctx.raw_path), nrows=1, encoding='utf-8').columns
        kurang = [k for k in KOLOM_WAJIB if k not in kolom]
        if kurang:
            raise SystemExit(f'Dataset kustom kehilangan kolom wajib: {kurang}')
    # Menentukan daftar step yang dijalankan
    if args.step == 'all':
        urutan = [n for n, m in REGISTRI.items() if m['dalam_all']]
    else:
        urutan = [args.step]
    # Menjalankan step satu per satu
    for nama in urutan:
        REGISTRI[nama]['fungsi'](ctx)


if __name__ == '__main__':
    main()
