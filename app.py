"""
app.py — Dashboard deteksi dini risiko ketidaklulusan

Jalankan:  streamlit run app.py
Syarat  :  model_kelulusan.pkl ada di folder yang sama.
"""

import pandas as pd
import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="Deteksi Dini Kelulusan", page_icon="📋", layout="wide")

PEKERJAAN = ["Buruh", "Karyawan Swasta", "PNS", "Petani", "Tidak Bekerja", "Wiraswasta"]
PENDIDIKAN = {"SD": 0, "SMP": 1, "SMA": 2, "S1": 3, "S2": 4}
AMBANG = 0.60  # di bawah ini masuk daftar prioritas intervensi


@st.cache_resource
def muat_model():
    bundle = joblib.load("model_kelulusan.pkl")
    return bundle["scaler"], bundle["model"], bundle["features"], bundle["metrics"]


scaler, model, FITUR, METRIK = muat_model()


def bangun_baris(f):
    """Ubah input form jadi satu baris DataFrame dengan kolom & urutan yang benar."""
    baris = {
        "usia": f["usia"],
        "jenis_kelamin": 1 if f["kelamin"] == "Laki-laki" else 0,
        "jenis_sekolah": 1 if f["sekolah"] == "Negeri" else 0,
        "pendidikan_orang_tua": PENDIDIKAN[f["didik"]],
        "jam_belajar_per_minggu": f["jam"],
        "persentase_kehadiran": f["hadir"],
        "ikut_ekskul": 1 if f["ekskul"] == "Ya" else 0,
        "akses_internet": 1 if f["internet"] == "Ya" else 0,
        "les_privat": 1 if f["les"] == "Ya" else 0,
        "waktu_tempuh_menit": f["tempuh"],
    }
    # One-hot pekerjaan; 'Buruh' sengaja tidak punya kolom (drop_first=True)
    for p in PEKERJAAN[1:]:
        baris[f"pekerjaan_orang_tua_{p}"] = 1 if f["kerja"] == p else 0

    # reindex memaksa urutan kolom sama persis dengan saat pelatihan
    return pd.DataFrame([baris]).reindex(columns=FITUR)


def prediksi(X):
    Xs = scaler.transform(X)
    p = float(model.predict_proba(Xs)[0, 1])
    kontribusi = (Xs[0] * model.coef_[0])  # sumbangan tiap fitur ke log-odds
    return p, kontribusi


def status(p):
    if p < 0.40:
        return "Risiko tinggi", "🔴", "Jadwalkan pertemuan dengan siswa dan orang tua bulan ini."
    if p < AMBANG + 0.05:
        return "Perlu dipantau", "🟠", "Zona abu-abu. Siswa seperti ini mudah terlewat karena belum terlihat bermasalah."
    return "Relatif aman", "🟢", "Pola belajarnya sehat. Cukup dipantau lewat laporan rutin."


# ---------------- tampilan ----------------

st.title("Deteksi dini risiko ketidaklulusan")
st.caption(
    f"Model logistic regression, akurasi {METRIK['accuracy']:.0%} pada "
    f"{METRIK['n_test']} siswa uji. Tidak memakai nilai ujian sama sekali — "
    "hanya kondisi yang sudah diketahui di awal semester."
)

if "daftar" not in st.session_state:
    st.session_state.daftar = []

kiri, kanan = st.columns([1, 1], gap="large")

with kiri:
    st.subheader("Data siswa")

    nama = st.text_input("Nama siswa", placeholder="mis. Rina Kusuma")

    st.markdown("**Kebiasaan belajar**")
    jam = st.slider("Jam belajar per minggu", 0, 42, 21)
    hadir = st.slider("Persentase kehadiran", 20.0, 100.0, 72.0, step=0.5)

    c1, c2 = st.columns(2)
    les = c1.radio("Les privat", ["Ya", "Tidak"], index=1, horizontal=True)
    ekskul = c2.radio("Ekstrakurikuler", ["Ya", "Tidak"], index=1, horizontal=True)

    st.markdown("**Akses dan jarak**")
    c3, c4 = st.columns(2)
    internet = c3.radio("Akses internet", ["Ya", "Tidak"], index=0, horizontal=True)
    tempuh = c4.number_input("Waktu tempuh (menit)", 0, 200, 78)

    st.markdown("**Latar belakang**")
    c5, c6 = st.columns(2)
    usia = c5.number_input("Usia", 10, 20, 15)
    sekolah = c6.selectbox("Jenis sekolah", ["Negeri", "Swasta"])

    c7, c8 = st.columns(2)
    kelamin = c7.selectbox("Jenis kelamin", ["Laki-laki", "Perempuan"])
    didik = c8.selectbox("Pendidikan orang tua", list(PENDIDIKAN), index=1)

    kerja = st.selectbox("Pekerjaan orang tua", PEKERJAAN, index=2)

form = dict(usia=usia, kelamin=kelamin, sekolah=sekolah, didik=didik, jam=jam,
            hadir=hadir, ekskul=ekskul, internet=internet, les=les,
            tempuh=tempuh, kerja=kerja)

X = bangun_baris(form)
p, kontribusi = prediksi(X)
label, ikon, saran = status(p)

with kanan:
    st.subheader("Hasil prediksi")
    st.metric("Peluang lulus", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.markdown(f"### {ikon} {label}")
    st.write(saran)

    st.divider()
    st.markdown("**Apa yang menggerakkan angka ini**")
    st.caption("Dibandingkan dengan siswa rata-rata di data. Positif mendorong ke Lulus.")

    pengaruh = (
        pd.DataFrame({"fitur": FITUR, "pengaruh": kontribusi})
        .assign(besar=lambda d: d["pengaruh"].abs())
        .query("besar > 0.02")
        .sort_values("besar", ascending=False)
        .head(5)
    )

    if len(pengaruh):
        # Batang digambar manual dengan HTML supaya tidak bergantung pada altair.
        skala = max(pengaruh["besar"].max(), 0.5)
        baris = []
        for _, r in pengaruh.iterrows():
            lebar = abs(r["pengaruh"]) / skala * 50
            positif = r["pengaruh"] > 0
            warna = "#256b4c" if positif else "#9e3324"
            kiri = 50 if positif else 50 - lebar
            baris.append(
                f'<div style="display:flex;align-items:center;gap:10px;padding:3px 0;font-size:13px">'
                f'<span style="flex:0 0 46%;overflow:hidden;text-overflow:ellipsis;'
                f'white-space:nowrap">{r["fitur"]}</span>'
                f'<span style="position:relative;flex:1;height:10px;background:#eee">'
                f'<span style="position:absolute;top:0;bottom:0;left:50%;width:1px;background:#bbb"></span>'
                f'<span style="position:absolute;top:0;bottom:0;left:{kiri}%;'
                f'width:{lebar}%;background:{warna}"></span></span></div>'
            )
        st.markdown("".join(baris), unsafe_allow_html=True)
    else:
        st.caption("Semua nilai persis di rata-rata data.")

    if st.button("Tambah ke daftar prioritas", type="primary", use_container_width=True):
        if nama.strip():
            st.session_state.daftar.append(
                {"Nama": nama.strip(), "Peluang lulus": p,
                 "Status": label, "Jam belajar": jam, "Kehadiran": hadir}
            )
            st.success(f"{nama.strip()} ditambahkan.")
        else:
            st.warning("Isi nama siswa dulu.")

st.divider()
st.subheader("Daftar prioritas intervensi")

if st.session_state.daftar:
    tabel = (
        pd.DataFrame(st.session_state.daftar)
        .sort_values("Peluang lulus")
        .reset_index(drop=True)
    )
    tabel.index += 1

    st.dataframe(
        tabel,
        use_container_width=True,
        column_config={
            "Peluang lulus": st.column_config.ProgressColumn(
                "Peluang lulus", min_value=0.0, max_value=1.0, format="%.1f%%"
            ),
            "Kehadiran": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    berisiko = (tabel["Peluang lulus"] < AMBANG).sum()
    st.caption(f"{berisiko} dari {len(tabel)} siswa berada di bawah ambang {AMBANG:.0%}.")

    st.download_button(
        "Unduh daftar (CSV)",
        tabel.to_csv(index=False).encode("utf-8"),
        "daftar_prioritas.csv",
        "text/csv",
    )
    if st.button("Kosongkan daftar"):
        st.session_state.daftar = []
        st.rerun()
else:
    st.info("Belum ada siswa. Isi form di atas, beri nama, lalu tambahkan ke daftar "
            "untuk menyaring satu kelas sekaligus.")

st.caption(
    "Alat bantu penyaringan awal, bukan penentu nasib siswa. "
    "Keputusan akhir tetap pada pertimbangan guru."
)
