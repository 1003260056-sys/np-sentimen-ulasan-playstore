"""Paket src: kumpulan modul pipeline analisis sentimen ulasan pengguna (Bab 1-5).

Berkas ini juga memuat fungsi bantu bersama yang dipakai oleh semua modul.
"""

# Mengimpor modul os untuk operasi path dan folder
import os

# Mengimpor joblib untuk memuat model/encoder yang tersimpan
import joblib

# Mengimpor numpy untuk pengecekan tipe data hasil prediksi
import numpy as np

# Mengimpor yaml untuk membaca config.yaml
import yaml

# Menentukan folder akar proyek (satu tingkat di atas folder src)
ROOT_PROYEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path_proyek(path):
    """Mengubah path relatif menjadi path absolut berdasarkan folder akar proyek."""
    # Jika path sudah absolut, kembalikan apa adanya
    if os.path.isabs(path):
        return path
    # Jika tidak, gabungkan dengan folder akar proyek
    return os.path.join(ROOT_PROYEK, path)


def muat_config(path='config.yaml'):
    """Membaca berkas konfigurasi YAML dan mengembalikannya sebagai dictionary."""
    # Jika berkas ada relatif terhadap direktori kerja, pakai itu
    if os.path.exists(path):
        lokasi = os.path.abspath(path)
    else:
        # Jika tidak, cari relatif terhadap akar proyek
        lokasi = path_proyek(path)
    # Membuka berkas konfigurasi dengan encoding UTF-8
    with open(lokasi, 'r', encoding='utf-8') as berkas:
        # Mengurai isi YAML menjadi dictionary Python
        return yaml.safe_load(berkas)


def siapkan_folder_file(path):
    """Membuat folder induk dari sebuah path berkas secara defensif."""
    # Mengambil nama folder dari path berkas
    folder = os.path.dirname(path)
    # Membuat folder jika belum ada (tidak error bila sudah ada)
    os.makedirs(folder if folder else '.', exist_ok=True)


def prediksi_label(model, teks_list, path_encoder='models/label_encoder.joblib'):
    """Memprediksi label teks; otomatis mengubah label numerik menjadi string
    memakai LabelEncoder (diperlukan untuk model XGBoost/LogReg/RF yang dilatih
    dengan label terenkode)."""
    # Menjalankan prediksi dari pipeline (TF-IDF + pengklasifikasi)
    hasil = np.asarray(model.predict(list(teks_list)))
    # Jika hasil berupa bilangan bulat, berarti label terenkode
    if np.issubdtype(hasil.dtype, np.integer):
        # Memuat LabelEncoder yang tersimpan saat pelatihan
        encoder = joblib.load(path_proyek(path_encoder))
        # Mengubah angka kembali menjadi nama kelas
        return encoder.inverse_transform(hasil)
    # Jika sudah berupa string, kembalikan langsung
    return hasil
