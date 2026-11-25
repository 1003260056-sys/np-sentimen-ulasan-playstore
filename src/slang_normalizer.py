"""[KODE PYTHON 3.1] Normalisasi slang, emoji, dan pembersihan teks (Bab 3).

Mode 1: diimpor -> SLANG_DICTIONARY, EMOJI_MAP, create_slang_csv, TextNormalizer
Mode 2: mandiri  -> python src/slang_normalizer.py [--output ...] [--text "..."]
"""

# Mengimpor modul dasar
import sys
import os
import re
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pandas untuk menulis CSV
import pandas as pd
# Mengimpor fungsi bantu proyek
from src import muat_config, path_proyek

# Kamus slang -> kata baku (lebih dari 60 entri khas ulasan aplikasi & e-commerce)
SLANG_DICTIONARY = {
    'bgt': 'banget', 'bngt': 'banget', 'bgd': 'banget',
    'gk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'tdk': 'tidak', 'tak': 'tidak', 'bkn': 'bukan', 'jgn': 'jangan', 'blm': 'belum',
    'brg': 'barang', 'brng': 'barang', 'smp': 'sampai', 'sampe': 'sampai',
    'ori': 'asli', 'mantul': 'mantap', 'rekomen': 'rekomendasi', 'recommended': 'rekomendasi',
    'lemot': 'lambat', 'hape': 'ponsel', 'hp': 'ponsel', 'tws': 'earphone', 'casan': 'charger',
    'seller': 'penjual', 'drop': 'turun', 'ongkir': 'ongkos kirim', 'apk': 'aplikasi', 'app': 'aplikasi',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena', 'karna': 'karena',
    'jd': 'jadi', 'jg': 'juga', 'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah',
    'bs': 'bisa', 'bsa': 'bisa', 'sy': 'saya', 'gue': 'saya', 'gw': 'saya', 'aku': 'saya',
    'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau', 'tp': 'tapi', 'dr': 'dari', 'pd': 'pada',
    'trs': 'terus', 'trus': 'terus', 'sm': 'sama', 'aja': 'saja', 'ajah': 'saja', 'doang': 'saja',
    'org': 'orang', 'thx': 'terima kasih', 'makasih': 'terima kasih', 'tks': 'terima kasih',
    'trims': 'terima kasih', 'pake': 'pakai', 'pakek': 'pakai', 'cepet': 'cepat', 'bgs': 'bagus',
    'dpt': 'dapat', 'dapet': 'dapat', 'lg': 'lagi', 'bener': 'benar', 'emg': 'memang',
    'nyesel': 'menyesal', 'ngecewain': 'mengecewakan', 'mls': 'malas', 'top': 'bagus', 'ok': 'oke',
}

# Pemetaan emoji ke kata Indonesia yang membawa makna sentimen
EMOJI_MAP = {
    '👍': 'bagus', '👎': 'buruk', '😍': 'suka', '🥰': 'suka', '😊': 'senang', '😀': 'senang',
    '😁': 'senang', '😂': 'lucu', '🤣': 'lucu', '🔥': 'keren', '❤': 'suka', '💕': 'suka',
    '😡': 'marah', '😠': 'marah', '🤬': 'marah', '😭': 'sedih', '😢': 'sedih', '😞': 'kecewa',
    '😤': 'kesal', '🤮': 'jijik', '⭐': 'bintang', '🌟': 'bintang', '🙏': 'terima kasih',
    '👏': 'bagus', '💯': 'sempurna', '😅': 'canggung', '🤩': 'kagum', '😔': 'kecewa',
}


def create_slang_csv(filepath='data/external/kamus_slang_id.csv'):
    """Menulis SLANG_DICTIONARY ke CSV 2 kolom: slang, baku."""
    # Menentukan path absolut
    lokasi = path_proyek(filepath)
    # Membuat folder secara defensif
    os.makedirs(os.path.dirname(lokasi), exist_ok=True)
    # Menyusun DataFrame dua kolom
    df = pd.DataFrame(sorted(SLANG_DICTIONARY.items()), columns=['slang', 'baku'])
    # Menyimpan CSV
    df.to_csv(lokasi, index=False, encoding='utf-8')
    print(f'Kamus slang tersimpan: {lokasi} ({len(df)} entri)')
    # Mengembalikan DataFrame
    return df


class TextNormalizer:
    """Pembersih teks: demojize, case folding, buang URL/HTML/mention, reduksi huruf berulang, substitusi slang."""

    def __init__(self, slang_dict=None, emoji_map=None):
        # Memakai kamus bawaan bila tidak diberikan
        self.slang_dict = dict(SLANG_DICTIONARY if slang_dict is None else slang_dict)
        # Memakai peta emoji bawaan bila tidak diberikan
        self.emoji_map = dict(EMOJI_MAP if emoji_map is None else emoji_map)
        # Pola regex URL
        self.pola_url = re.compile(r'https?://\S+|www\.\S+')
        # Pola tag HTML
        self.pola_html = re.compile(r'<[^>]+>')
        # Pola mention
        self.pola_mention = re.compile(r'@\w+')
        # Pola karakter berulang tiga kali atau lebih
        self.pola_ulang = re.compile(r'(.)\1{2,}')
        # Pola karakter selain huruf dan spasi
        self.pola_non_huruf = re.compile(r'[^a-z\s]')

    def demojize(self, teks):
        """Mengganti emoji dengan kata Indonesia (dipisahkan spasi)."""
        # Mengulang tiap emoji pada peta
        for emoji, kata in self.emoji_map.items():
            teks = teks.replace(emoji, f' {kata} ')
        # Menghapus penanda variasi emoji yang tersisa
        return teks.replace('\ufe0f', '')

    def clean_text(self, teks):
        """Menjalankan seluruh tahap pembersihan dan mengembalikan teks bersih."""
        # Mengubah ke string agar aman terhadap tipe lain
        teks = str(teks)
        # Mengganti emoji menjadi kata
        teks = self.demojize(teks)
        # Case folding
        teks = teks.lower()
        # Menghapus URL, HTML, dan mention
        teks = self.pola_url.sub(' ', teks)
        teks = self.pola_html.sub(' ', teks)
        teks = self.pola_mention.sub(' ', teks)
        # Mereduksi huruf berulang (mantuuul -> mantul)
        teks = self.pola_ulang.sub(r'\1', teks)
        # Menghapus angka dan tanda baca
        teks = self.pola_non_huruf.sub(' ', teks)
        # Memecah menjadi token berdasarkan spasi
        token = teks.split()
        # Substitusi slang berbasis lookup token (tidak merusak substring kata baku)
        token = [self.slang_dict.get(t, t) for t in token]
        # Menggabungkan kembali dan merapikan spasi
        return ' '.join(' '.join(token).split())


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Kamus slang & normalisasi teks (Bab 3)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--output', default=None, help='Path CSV kamus slang')
    parser.add_argument('--text', default='Barangnya ori bgt, pengiriman cepeettt 👍😍 gak nyesel!!!', help='Kalimat uji')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    # Membuat CSV kamus
    create_slang_csv(args.output or cfg['paths']['slang_dict'])
    # Mendemonstrasikan normalisasi
    norm = TextNormalizer()
    print('Teks mentah :', args.text)
    print('Teks bersih :', norm.clean_text(args.text))
