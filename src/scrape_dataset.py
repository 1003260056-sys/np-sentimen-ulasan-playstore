"""[KODE PYTHON 2.1] Pengambilan ulasan riil Google Play Store (Bab 2).

Mode 1: diimpor -> scrape_playstore_reviews(...)
Mode 2: mandiri  -> python src/scrape_dataset.py [--count 750 ...]
"""

# Mengimpor sys dan os untuk path
import sys
import os
# Mengimpor argparse untuk argumen terminal
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor numpy untuk nilai NaN dan acak
import numpy as np
# Mengimpor pandas untuk tabel
import pandas as pd
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek, siapkan_folder_file

# Pemetaan app_id ke nama kategori produk
NAMA_APLIKASI = {'com.tokopedia.tkpd': 'Tokopedia', 'com.shopee.id': 'Shopee'}


def _petakan_label(skor):
    """Memetakan bintang ke label: >=4 Positif, ==3 Netral, <=2 Negatif."""
    # Skor 4 atau 5 -> Positif
    if skor >= 4:
        return 'Positif'
    # Skor 3 -> Netral
    if skor == 3:
        return 'Netral'
    # Skor 1 atau 2 -> Negatif
    return 'Negatif'


def scrape_playstore_reviews(apps=None, lang='id', country='id', output_path='data/raw/ulasan_pengguna.csv',
                             missing_rate=0.02, duplicate_rows=15, random_seed=42):
    """Mengunduh ulasan riil lalu menyimpan CSV 6 kolom (dengan derau terkontrol)."""
    # Mengimpor pustaka scraper di sini agar mode lain tidak wajib memasangnya
    from google_play_scraper import reviews, Sort
    # Nilai bawaan aplikasi bila tidak diberikan
    if apps is None:
        apps = [{'app_id': 'com.tokopedia.tkpd', 'count': 750}, {'app_id': 'com.shopee.id', 'count': 750}]
    # Daftar baris hasil unduhan
    baris = []
    # Mengulang setiap aplikasi
    for app in apps:
        app_id = app['app_id']
        jumlah = int(app['count'])
        print(f'Mengunduh {jumlah} ulasan dari {app_id} ...')
        try:
            # Memanggil API scraper (mengembalikan tuple hasil, token)
            hasil, _ = reviews(app_id, lang=lang, country=country, sort=Sort.MOST_RELEVANT, count=jumlah)
        except Exception as galat:
            # Pesan diagnostik yang jelas bila jaringan/API bermasalah
            print('\n[GAGAL] Tidak dapat mengunduh ulasan.')
            print(f'  Aplikasi : {app_id}')
            print(f'  Penyebab : {type(galat).__name__}: {galat}')
            print('  Saran    : 1) Periksa koneksi internet/proxy/VPN Anda.')
            print('             2) Pastikan app_id benar dan aplikasi tersedia di negara tersebut.')
            print('             3) Coba ulang beberapa menit lagi (pembatasan laju Google).')
            print('             4) Atau gunakan dataset sendiri: --input-data path/ke/berkas.csv')
            raise RuntimeError(f'Scraping gagal untuk {app_id}') from galat
        # Mengubah setiap ulasan menjadi baris tabel
        for r in hasil:
            baris.append({
                'tanggal': r['at'].strftime('%Y-%m-%d'),
                'kategori_produk': NAMA_APLIKASI.get(app_id, app_id),
                'rating': int(r['score']),
                'teks_ulasan': r['content'],
                'label_sentimen': _petakan_label(int(r['score'])),
            })
    # Membuat DataFrame dari seluruh baris
    df = pd.DataFrame(baris)
    # Menghentikan jika tidak ada data
    if df.empty:
        raise RuntimeError('Tidak ada ulasan yang berhasil diunduh.')
    # Membuat generator acak dengan seed tetap
    rng = np.random.default_rng(random_seed)
    # Menghitung jumlah baris yang akan dikosongkan
    n_kosong = int(round(len(df) * missing_rate))
    # Memilih indeks acak untuk dikosongkan
    idx_kosong = rng.choice(len(df), size=n_kosong, replace=False)
    # Menyuntikkan NaN pada kolom teks
    df['teks_ulasan'] = df['teks_ulasan'].astype(object)
    df.loc[df.index[idx_kosong], 'teks_ulasan'] = np.nan
    # Memilih baris bertek valid untuk diduplikasi
    kandidat = df[df['teks_ulasan'].notna()]
    # Membatasi jumlah duplikat sesuai ketersediaan data
    n_dup = min(int(duplicate_rows), len(kandidat))
    # Mengambil sampel baris untuk digandakan
    duplikat = kandidat.sample(n=n_dup, random_state=random_seed)
    # Menambahkan baris duplikat kontrol di akhir tabel
    df = pd.concat([df, duplikat], ignore_index=True)
    # Membuat kolom id berformat REV_0001
    df.insert(0, 'id_ulasan', [f'REV_{i:04d}' for i in range(1, len(df) + 1)])
    # Menyusun urutan kolom akhir
    df = df[['id_ulasan', 'tanggal', 'kategori_produk', 'rating', 'teks_ulasan', 'label_sentimen']]
    # Menentukan path absolut keluaran
    lokasi = path_proyek(output_path)
    # Membuat folder keluaran secara defensif
    os.makedirs(os.path.dirname(lokasi), exist_ok=True)
    # Menyimpan CSV
    df.to_csv(lokasi, index=False, encoding='utf-8')
    print(f'Tersimpan: {lokasi} ({len(df)} baris, {n_kosong} teks kosong, {n_dup} duplikat kontrol)')
    # Mengembalikan DataFrame
    return df


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Scraping ulasan Google Play Store (Bab 2)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--count', type=int, default=None, help='Jumlah ulasan per aplikasi')
    parser.add_argument('--output', default=None, help='Path CSV keluaran')
    parser.add_argument('--seed', type=int, default=None, help='Random state')
    args = parser.parse_args()
    # Memuat konfigurasi
    cfg = muat_config(args.config)
    sc = cfg['scraping']
    # Menyalin daftar aplikasi lalu menimpa jumlah bila diminta
    apps = [dict(a) for a in sc['apps']]
    if args.count is not None:
        for a in apps:
            a['count'] = args.count
    # Menjalankan fungsi utama
    scrape_playstore_reviews(
        apps=apps, lang=sc['lang'], country=sc['country'],
        output_path=args.output or cfg['paths']['raw_dataset'],
        missing_rate=sc['missing_noise_rate'], duplicate_rows=sc['duplicate_noise_rows'],
        random_seed=args.seed if args.seed is not None else sc['random_state'])
