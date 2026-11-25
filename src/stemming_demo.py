"""[KODE PYTHON 3.2] Stopword, stemming Nazief-Adriani, dan jejak preprocessing (Bab 3).

Mode 1: diimpor -> NEGATION_WHITELIST, AdvancedTextPreprocessor, run_stemming_demo()
Mode 2: mandiri  -> python src/stemming_demo.py [--text "..."]
"""

# Mengimpor modul dasar
import sys
import os
import argparse

# Menambahkan akar proyek ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mengimpor pabrik stopword dan stemmer Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# Mengimpor normalizer teks
from src.slang_normalizer import TextNormalizer
# Mengimpor fungsi bantu proyek
from src import muat_config

# Kata negasi yang tidak boleh dihapus karena membalik polaritas sentimen
NEGATION_WHITELIST = {'tidak', 'bukan', 'kurang', 'belum', 'jangan', 'tanpa'}

# Empat kalimat uji studi kasus
KALIMAT_UJI = [
    'Aplikasinya bgt lemot, gak bisa dibuka pas checkout!!! 😡',
    'Barang ori, pengirimannya cepeettt banget 👍😍 mantul',
    'Pelayanan seller kurang ramah dan pesanan belum sampai juga',
    'Tidak mengecewakan, harga murah tanpa ongkir. Recommended!',
]


class AdvancedTextPreprocessor:
    """Normalisasi -> hapus stopword (kecuali negasi) -> stemming Sastrawi."""

    def __init__(self, use_stemmer=True, negation_whitelist=None, normalizer=None):
        # Menyimpan pilihan stemming
        self.use_stemmer = use_stemmer
        # Menyimpan daftar negasi sebagai himpunan
        self.negation_whitelist = set(NEGATION_WHITELIST if negation_whitelist is None else negation_whitelist)
        # Memakai normalizer yang diberikan atau membuat baru
        self.normalizer = normalizer if normalizer is not None else TextNormalizer()
        # Mengambil daftar stopword Sastrawi
        self.stopwords = set(StopWordRemoverFactory().get_stop_words())
        # Membuat stemmer hanya bila dibutuhkan (proses pembuatan cukup lama)
        self.stemmer = StemmerFactory().create_stemmer() if use_stemmer else None
        # Cache hasil stem agar kata yang sama tidak diproses ulang
        self._cache = {}

    def normalize(self, teks):
        """Tahap 1: pembersihan dan normalisasi slang/emoji."""
        return self.normalizer.clean_text(teks)

    def remove_stopwords(self, teks):
        """Tahap 2: hapus stopword, namun pertahankan kata negasi."""
        # Menyaring token: simpan jika negasi atau bukan stopword
        token = [t for t in teks.split() if t in self.negation_whitelist or t not in self.stopwords]
        return ' '.join(token)

    def stem(self, teks):
        """Tahap 3: stemming Nazief-Adriani (negasi tidak distem)."""
        # Bila stemming dimatikan, kembalikan teks apa adanya
        if not self.use_stemmer:
            return teks
        hasil = []
        for t in teks.split():
            # Negasi dilewati agar tetap utuh
            if t in self.negation_whitelist:
                hasil.append(t)
                continue
            # Mengisi cache bila token belum pernah distem
            if t not in self._cache:
                self._cache[t] = self.stemmer.stem(t)
            hasil.append(self._cache[t])
        return ' '.join(hasil)

    def process(self, teks):
        """Menjalankan seluruh tahap dan mengembalikan teks akhir."""
        return self.stem(self.remove_stopwords(self.normalize(teks)))

    def process_with_trace(self, teks):
        """Menjalankan seluruh tahap dan mencetak jejak perubahan bertahap."""
        # Menghitung hasil tiap tahap
        normalized = self.normalize(teks)
        stopword_removed = self.remove_stopwords(normalized)
        stemmed = self.stem(stopword_removed)
        # Menyusun jejak
        jejak = {'raw': teks, 'normalized': normalized, 'stopword_removed': stopword_removed, 'stemmed': stemmed}
        # Mencetak jejak
        for kunci, nilai in jejak.items():
            print(f'  {kunci:<17}: {nilai}')
        return jejak


def run_stemming_demo(use_stemmer=True, negation_whitelist=None, kalimat=None):
    """Mendemonstrasikan jejak preprocessing pada 4 kalimat uji."""
    print('=' * 70)
    print('[BAB 3] DEMO STOPWORD & STEMMING (JEJAK BERTAHAP)')
    print('=' * 70)
    # Membuat preprocessor
    prep = AdvancedTextPreprocessor(use_stemmer=use_stemmer, negation_whitelist=negation_whitelist)
    hasil = []
    # Memproses tiap kalimat uji
    for i, k in enumerate(kalimat or KALIMAT_UJI, start=1):
        print(f'\nKalimat {i}:')
        hasil.append(prep.process_with_trace(k))
    return hasil


if __name__ == '__main__':
    # Membuat parser argumen
    parser = argparse.ArgumentParser(description='Demo stopword & stemming (Bab 3)')
    parser.add_argument('--config', default='config.yaml', help='Path config.yaml')
    parser.add_argument('--text', default=None, help='Kalimat uji tunggal (opsional)')
    parser.add_argument('--no-stemmer', action='store_true', help='Matikan stemming')
    args = parser.parse_args()
    # Memuat konfigurasi sebagai fallback
    cfg = muat_config(args.config)
    pre = cfg['preprocessing']
    run_stemming_demo(use_stemmer=False if args.no_stemmer else pre['use_stemmer'],
                      negation_whitelist=pre['negation_whitelist'],
                      kalimat=[args.text] if args.text else None)
