"""Generator 10 grafik PNG resolusi 300 DPI untuk buku ajar (Bab 2-5).

Mode 1: diimpor -> generate_all_visualizations(config)
Mode 2: mandiri  -> python src/visualize.py [--dpi 300]
"""

# Mengimpor modul dasar
import sys
import os
import argparse
import traceback

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor matplotlib dengan backend tanpa layar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Mengimpor seaborn, numpy, pandas, joblib
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
# Mengimpor komponen scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek, prediksi_label


class _Pembuat:
    """Menyimpan konfigurasi dan data bersama untuk seluruh grafik."""

    def __init__(self, cfg):
        # Menyimpan konfigurasi dan pintasan kolom
        self.cfg = cfg
        s = cfg['schema']
        self.teks, self.bersih, self.target = s['text_col'], s['clean_text_col'], s['target_col']
        self.kelas = s['classes']
        self.dpi = cfg['visualization']['dpi']
        self.folder_data = path_proyek(cfg['paths']['processed_dir'])
        self.folder_model = path_proyek(cfg['paths']['models_dir'])
        self.folder_gambar = path_proyek(cfg['paths']['figures_dir'])
        # Membuat folder gambar secara defensif
        os.makedirs(self.folder_gambar, exist_ok=True)
        # Menerapkan gaya visual
        try:
            plt.style.use(cfg['visualization']['style'])
        except OSError:
            sns.set_style('whitegrid')
        # Palet warna tetap per kelas
        self.warna = {'Positif': '#2E8B57', 'Netral': '#F0A30A', 'Negatif': '#C0392B'}

    def baca(self, nama):
        # Membaca berkas CSV dari data/processed
        return pd.read_csv(os.path.join(self.folder_data, nama), encoding='utf-8')

    def semua_raw(self):
        # Menggabungkan train/val/test raw yang sudah bersih
        return pd.concat([self.baca(f'{b}_raw.csv') for b in ['train', 'val', 'test']], ignore_index=True)

    def simpan(self, fig, nama):
        # Menyimpan figur dengan DPI dari konfigurasi
        lokasi = os.path.join(self.folder_gambar, nama)
        os.makedirs(os.path.dirname(lokasi), exist_ok=True)
        fig.savefig(lokasi, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] {lokasi}')
        return lokasi

    # ---------------- Grafik 1 ----------------
    def g01(self):
        df = self.semua_raw()
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        sns.countplot(data=df, x=self.target, order=self.kelas, palette=self.warna, ax=ax[0])
        ax[0].set_title('Distribusi Sentimen Keseluruhan')
        ax[0].set_xlabel('Label Sentimen')
        ax[0].set_ylabel('Jumlah Ulasan')
        sns.countplot(data=df, x=self.target, hue='kategori_produk', order=self.kelas, ax=ax[1])
        ax[1].set_title('Distribusi Sentimen per Aplikasi')
        ax[1].set_xlabel('Label Sentimen')
        ax[1].set_ylabel('Jumlah Ulasan')
        return self.simpan(fig, '01_distribusi_sentimen_dan_aplikasi.png')

    # ---------------- Grafik 2 ----------------
    def g02(self):
        df = self.semua_raw()
        tab = pd.crosstab(df['rating'], df[self.target]).reindex(columns=self.kelas, fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('Heatmap Rating vs Label Sentimen (Rating-Text Mismatch)')
        ax.set_xlabel('Label Sentimen')
        ax.set_ylabel('Rating (Bintang)')
        return self.simpan(fig, '02_heatmap_rating_vs_sentimen.png')

    # ---------------- Grafik 3 ----------------
    def g03(self):
        df = self.semua_raw()
        df['bulan'] = pd.to_datetime(df['tanggal'], errors='coerce').dt.to_period('M').dt.to_timestamp()
        tren = df.dropna(subset=['bulan']).groupby(['bulan', self.target]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 5))
        for k in self.kelas:
            if k in tren.columns:
                ax.plot(tren.index, tren[k], marker='o', label=k, color=self.warna[k])
        ax.set_title('Tren Jumlah Ulasan per Kelas Sentimen')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Jumlah Ulasan')
        ax.legend(title='Sentimen')
        fig.autofmt_xdate()
        return self.simpan(fig, '03_tren_sentimen_temporal.png')

    # ---------------- Grafik 4 ----------------
    def g04(self):
        df = self.semua_raw()
        df['word_count'] = df[self.teks].astype(str).str.split().str.len()
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for k in self.kelas:
            sub = df[df[self.target] == k]['word_count']
            if len(sub) > 1:
                sns.histplot(sub, kde=True, stat='density', label=k, color=self.warna[k], alpha=0.35, ax=ax[0])
        ax[0].set_title('Histogram + KDE Jumlah Kata per Kelas')
        ax[0].set_xlabel('Jumlah Kata')
        ax[0].legend(title='Sentimen')
        sns.boxplot(data=df, x=self.target, y='word_count', order=self.kelas, palette=self.warna, ax=ax[1])
        ax[1].set_title('Boxplot Jumlah Kata per Kelas')
        ax[1].set_xlabel('Label Sentimen')
        ax[1].set_ylabel('Jumlah Kata')
        return self.simpan(fig, '04_distribusi_panjang_ulasan.png')

    # ---------------- Grafik 5 ----------------
    def g05(self):
        from wordcloud import WordCloud
        raw, clean = self.baca('train_raw.csv'), self.baca('train_clean.csv')
        def vocab(seri):
            h = set()
            for t in seri.fillna('').astype(str):
                h.update(t.lower().split())
            return len(h)
        sebelum, sesudah = vocab(raw[self.teks]), vocab(clean[self.bersih])
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        ax0 = fig.add_subplot(gs[0, :])
        ax0.bar(['Sebelum Preprocessing', 'Sesudah Preprocessing'], [sebelum, sesudah], color=['#7f8c8d', '#2980b9'])
        for i, v in enumerate([sebelum, sesudah]):
            ax0.text(i, v, str(v), ha='center', va='bottom')
        ax0.set_title(f'Jumlah Kata Unik (reduksi {(1 - sesudah / sebelum) * 100:.1f}%)')
        ax0.set_ylabel('Kata Unik')
        for i, k in enumerate(self.kelas):
            ax = fig.add_subplot(gs[1, i])
            teks = ' '.join(clean[clean[self.target] == k][self.bersih].fillna('').astype(str))
            if teks.strip():
                wc = WordCloud(width=600, height=400, background_color='white', colormap='viridis').generate(teks)
                ax.imshow(wc, interpolation='bilinear')
            ax.set_title(f'WordCloud {k}')
            ax.axis('off')
        return self.simpan(fig, '05_reduksi_vokabulari_dan_wordcloud.png')

    # ---------------- Grafik 6 ----------------
    def g06(self):
        f = self.cfg['features']
        clean = self.baca('train_clean.csv')
        clean[self.bersih] = clean[self.bersih].fillna('').astype(str)
        tf = TfidfVectorizer(ngram_range=tuple(f['ngram_range']), min_df=f['min_df'], max_df=f['max_df'], sublinear_tf=True)
        X = tf.fit_transform(clean[self.bersih])
        nama = tf.get_feature_names_out()
        y = clean[self.target].to_numpy()
        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        for a, k in zip(ax, self.kelas):
            rata = np.asarray(X[y == k].mean(axis=0)).ravel()
            idx = rata.argsort()[::-1][:10][::-1]
            a.barh([nama[i] for i in idx], rata[idx], color=self.warna[k])
            a.set_title(f'Top-10 TF-IDF: {k}')
            a.set_xlabel('Rata-rata Bobot TF-IDF')
        return self.simpan(fig, '06_top_tfidf_per_kelas.png')

    # ---------------- Grafik 7 ----------------
    def g07(self):
        from gensim.models import FastText
        ft = FastText.load(os.path.join(self.folder_model, 'fasttext_ulasan.model'))
        kunci = ['kirim', 'barang', 'aplikasi', 'harga', 'lambat', 'bagus', 'kurir', 'promo', 'cepat', 'murah', 'rusak', 'kecewa']
        kata, grup = [], []
        for k in kunci:
            if k in ft.wv.key_to_index:
                if k not in kata:
                    kata.append(k)
                    grup.append('kata kunci')
                for t, _ in ft.wv.most_similar(k, topn=4):
                    if t not in kata:
                        kata.append(t)
                        grup.append('tetangga')
        if len(kata) < 2:
            raise ValueError('Kata domain tidak cukup pada model FastText.')
        koord = PCA(n_components=2).fit_transform(np.vstack([ft.wv[k] for k in kata]))
        fig, ax = plt.subplots(figsize=(11, 8))
        for g, warna in [('kata kunci', '#C0392B'), ('tetangga', '#2980B9')]:
            idx = [i for i, x in enumerate(grup) if x == g]
            ax.scatter(koord[idx, 0], koord[idx, 1], c=warna, label=g, s=45)
        for i, k in enumerate(kata):
            ax.annotate(k, (koord[i, 0], koord[i, 1]), fontsize=8, xytext=(3, 3), textcoords='offset points')
        ax.set_title('Proyeksi PCA 2D Vektor FastText Kata Domain Ulasan')
        ax.set_xlabel('Komponen Utama 1')
        ax.set_ylabel('Komponen Utama 2')
        ax.legend()
        return self.simpan(fig, '07_proyeksi_embeddings_2d.png')

    # ---------------- Grafik 8 ----------------
    def g08(self):
        d = self.baca('distribusi_smote.csv').melt(id_vars='kelas', var_name='Tahap', value_name='Jumlah')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=d, x='kelas', y='Jumlah', hue='Tahap', order=self.kelas, ax=ax)
        for c in ax.containers:
            ax.bar_label(c)
        ax.set_title('Distribusi Kelas Data Latih: Sebelum vs Sesudah SMOTE')
        ax.set_xlabel('Kelas Sentimen')
        return self.simpan(fig, '08_distribusi_smote_before_after.png')

    # ---------------- Grafik 9 ----------------
    def g09(self):
        model = joblib.load(os.path.join(self.folder_model, 'logisticregression_model.joblib'))
        encoder = joblib.load(os.path.join(self.folder_model, 'label_encoder.joblib'))
        fitur = model.named_steps['tfidf'].get_feature_names_out()
        koef = model.named_steps['clf'].coef_
        kelas_model = list(encoder.inverse_transform(model.named_steps['clf'].classes_))
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        for a, k in zip(ax, ['Positif', 'Negatif']):
            i = kelas_model.index(k)
            idx = koef[i].argsort()[::-1][:10][::-1]
            a.barh([fitur[j] for j in idx], koef[i][idx], color=self.warna[k])
            a.set_title(f'Top-10 Koefisien LogReg: {k}')
            a.set_xlabel('Koefisien (coef_)')
        return self.simpan(fig, '09_koefisien_fitur_logreg.png')

    # ---------------- Grafik 10 ----------------
    def g10(self):
        p = self.cfg['paths']
        tabel = pd.read_csv(path_proyek(p['comparison_csv']), encoding='utf-8')
        metrik = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']
        panjang = tabel.melt(id_vars='Model', value_vars=metrik, var_name='Metrik', value_name='Skor')
        model = joblib.load(path_proyek(p['best_model']))
        uji = self.baca('test_clean.csv')
        uji[self.bersih] = uji[self.bersih].fillna('').astype(str)
        pred = prediksi_label(model, uji[self.bersih].tolist(), os.path.join(p['models_dir'], 'label_encoder.joblib'))
        cm = confusion_matrix(uji[self.target], pred, labels=self.kelas)
        fig, ax = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1.6, 1]})
        sns.barplot(data=panjang, x='Model', y='Skor', hue='Metrik', ax=ax[0])
        ax[0].set_title('Komparasi Metrik Model (Data Uji)')
        ax[0].set_ylim(0, 1)
        ax[0].tick_params(axis='x', rotation=30)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.kelas, yticklabels=self.kelas, ax=ax[1])
        ax[1].set_title('Confusion Matrix Model Juara')
        ax[1].set_xlabel('Prediksi')
        ax[1].set_ylabel('Aktual')
        return self.simpan(fig, '10_komparasi_model_dan_confusion_matrix.png')


def generate_all_visualizations(config=None):
    """Membuat 10 grafik PNG; kegagalan satu grafik tidak menghentikan yang lain."""
    print('=' * 70)
    print('[VISUALISASI] MEMBUAT 10 GRAFIK PNG (300 DPI)')
    print('=' * 70)
    # Memuat konfigurasi bila tidak diberikan
    cfg = config if isinstance(config, dict) else muat_config(config or 'config.yaml')
    pembuat = _Pembuat(cfg)
    berhasil, gagal = [], []
    for nomor in range(1, 11):
        fungsi = getattr(pembuat, f'g{nomor:02d}')
        try:
            berhasil.append(fungsi())
        except Exception as galat:
            gagal.append((nomor, galat))
            print(f'  [GAGAL] Grafik {nomor:02d}: {type(galat).__name__}: {galat}')
            print('          Pastikan langkah bab sebelumnya sudah dijalankan (--step all).')
    print(f'\nSelesai: {len(berhasil)}/10 grafik berhasil dibuat di {pembuat.folder_gambar}')
    return berhasil


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Generator 10 grafik PNG')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--dpi', type=int, default=None, help='Override DPI')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    if args.dpi:
        cfg['visualization']['dpi'] = args.dpi
    generate_all_visualizations(cfg)
