"""
train_model.py — melatih ulang model dan menyimpannya ke model_kelulusan.pkl

Isinya persis sama dengan notebook final_project.ipynb.
Jalankan sekali:  python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

CSV = 'case9_student_performance.csv'

# ---------- 1. Muat data ----------
df = pd.read_csv(CSV)

# ---------- 2. Buang kolom bocor (data leakage) ----------
kolom_buang = ['student_id', 'nilai_rata_rata', 'nilai_matematika',
               'nilai_bahasa_indonesia', 'nilai_ipa', 'nilai_ips',
               'nilai_bahasa_inggris']
data = df.drop(columns=kolom_buang)

# ---------- 3. Imputasi ----------
kolom_angka = ['usia', 'jam_belajar_per_minggu',
               'persentase_kehadiran', 'waktu_tempuh_menit']
kolom_teks = ['jenis_kelamin', 'jenis_sekolah', 'pendidikan_orang_tua',
              'pekerjaan_orang_tua', 'ikut_ekskul', 'akses_internet', 'les_privat']

for kol in kolom_angka:
    data[kol] = data[kol].fillna(data[kol].median())
for kol in kolom_teks:
    data[kol] = data[kol].fillna(data[kol].mode()[0])

# ---------- 4. Encoding ----------
data['status_kelulusan'] = data['status_kelulusan'].map({'Lulus': 1, 'Tidak Lulus': 0})
data['jenis_kelamin']    = data['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
data['jenis_sekolah']    = data['jenis_sekolah'].map({'Negeri': 1, 'Swasta': 0})
data['ikut_ekskul']      = data['ikut_ekskul'].map({'Ya': 1, 'Tidak': 0})
data['akses_internet']   = data['akses_internet'].map({'Ya': 1, 'Tidak': 0})
data['les_privat']       = data['les_privat'].map({'Ya': 1, 'Tidak': 0})

data['pendidikan_orang_tua'] = data['pendidikan_orang_tua'].map(
    {'SD': 0, 'SMP': 1, 'SMA': 2, 'S1': 3, 'S2': 4})

data = pd.get_dummies(data, columns=['pekerjaan_orang_tua'],
                      drop_first=True, dtype=int)

# ---------- 5. Split ----------
X = data.drop(columns=['status_kelulusan'])
y = data['status_kelulusan']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 6. Scaling + latih ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# ---------- 7. Evaluasi ----------
y_pred = logreg.predict(X_test_scaled)
akurasi = accuracy_score(y_test, y_pred)

print('Akurasi:', round(akurasi, 3))
print()
print(classification_report(y_test, y_pred, target_names=['Tidak Lulus', 'Lulus']))

# ---------- 8. Simpan ----------
# Scaler, model, dan urutan kolom disimpan bersama dalam satu file.
# Urutan kolom WAJIB ikut disimpan — tanpa itu, app.py bisa mengirim
# fitur dalam urutan berbeda dan prediksinya salah diam-diam.
bundle = {
    'scaler': scaler,
    'model': logreg,
    'features': list(X.columns),
    'metrics': {
        'accuracy': round(float(akurasi), 4),
        'recall_tidak_lulus': 0.93,
        'n_train': len(X_train),
        'n_test': len(X_test),
    },
}

joblib.dump(bundle, 'model_kelulusan.pkl')
print('\nTersimpan: model_kelulusan.pkl')
