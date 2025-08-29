import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ==========================
# Load Model & Encoder
# ==========================
with open("model/xgb_stunting_model1.pkl", "rb") as f:
    saved = pickle.load(f)

best_model = saved["model"]
X_train = saved["columns"]
label_encoder = saved["label_encoder"]

# ==========================
# Load WHO Reference Data
# ==========================
pb_l = pd.read_excel("rumus/Panjang_Laki-laki_usia_0-2-tahun_z-score.xlsx")
pb_p = pd.read_excel("rumus/Panjang_Perempuan_usia_0-2-tahun_z-score-Panjang.xlsx")
tb_l = pd.read_excel("rumus/Tinggi_Laki-laki_usia_2-5-tahun_z-score.xlsx")
tb_p = pd.read_excel("rumus/Tinggi_Perempuan_usia_2-5-tahun_z-score.xlsx")

pb_l.columns = pb_l.columns.str.strip()
tb_l.columns = tb_l.columns.str.strip()
pb_p.columns = pb_p.columns.str.strip()
tb_p.columns = tb_p.columns.str.strip()

# ==========================
# Fungsi Bantu
# ==========================
def calculate_age_months_days(birth_date, test_date):
    """Hitung umur dalam bulan + sisa hari"""
    if not isinstance(birth_date, pd.Timestamp):
        birth_date = pd.to_datetime(birth_date)
    if not isinstance(test_date, pd.Timestamp):
        test_date = pd.to_datetime(test_date)

    months = (test_date.year - birth_date.year) * 12 + (test_date.month - birth_date.month)
    day_diff = test_date.day - birth_date.day

    if day_diff < 0:
        months -= 1
        prev_month = test_date.month - 1 if test_date.month > 1 else 12
        prev_year = test_date.year if test_date.month > 1 else test_date.year - 1
        days_in_prev_month = (pd.Timestamp(datetime(prev_year, prev_month % 12 + 1, 1)) - pd.Timedelta(days=1)).day
        day_diff = days_in_prev_month + day_diff

    return months, day_diff, months + (day_diff / 30)


def get_who_row(who_df, umur_bulan):
    # Cari baris WHO terdekat
    closest_age = who_df['Month'].iloc[(who_df['Month'] - umur_bulan).abs().argsort()[0]]
    return who_df[who_df["Month"] == closest_age].iloc[0]

def calculate_zscore(measurement, L, M, S):
    measurement = float(measurement)
    L, M, S = float(L), float(M), float(S)
    if S == 0 or abs(S) < 1e-6:
        return (measurement - M) / (L + 1e-6) if L != 0 else 0.0
    else:
        if L != 0:
            return (((measurement / M)**L) - 1) / (L * S)
        else:
            return (np.log(measurement / M)) / S

def stunting_status(z_score):
    if z_score < -3:
        return "Sangat Pendek"
    elif -3 <= z_score < -2:
        return "Pendek"
    elif -2 <= z_score <= 3:
        return "Normal"
    else:
        return "Tinggi"

# ==========================
# Streamlit UI
# ==========================
col1, col2, col3 = st.columns([1,1,1])
with col2:
    st.image("img/logo1.jpg", width=400)

st.markdown("<h1 style='text-align: center;'>Prediksi Risiko Stunting</h1>", unsafe_allow_html=True)

# Input pengguna
jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
jk_val = 1 if jenis_kelamin == "Laki-laki" else 0

tgl_lahir = st.date_input("Tanggal Lahir")
tgl_tes   = st.date_input("Tanggal Tes")

tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=0.0, step=0.1, format="%.1f")
cara_ukur = st.radio("Cara Ukur", ["STANDING", "LYING DOWN"])

# ==========================
# Prediksi
# ==========================
if st.button("Prediksi"):

    # Hitung umur
    umur_bulan, umur_hari, umur_bulan_desimal = calculate_age_months_days(tgl_lahir, tgl_tes)

    # Tentukan WHO reference
    if jk_val == 1:
        who_df_new = pb_l if umur_bulan_desimal < 24 else tb_l
    else:
        who_df_new = pb_p if umur_bulan_desimal < 24 else tb_p

    who_row_new = get_who_row(who_df_new, umur_bulan_desimal)

    # Koreksi tinggi badan
    height_corrected = tinggi_badan
    if cara_ukur == "LYING DOWN" and umur_bulan_desimal >= 24:
        height_corrected -= 0.7
    elif cara_ukur == "STANDING" and umur_bulan_desimal < 24:
        height_corrected += 0.7

    # Hitung Z-Score
    z_haz = calculate_zscore(height_corrected, who_row_new["L"], who_row_new["M"], who_row_new["S"])
    status_who = stunting_status(z_haz)

    # Buat DataFrame baru untuk prediksi ML
    data_baru = pd.DataFrame([{
        'Jenis Kelamin': jk_val,
        'Tinggi Badan (cm)': tinggi_badan,
        'Cara Ukur': cara_ukur,
        'Umur (bulan)': umur_bulan_desimal,
        'Z-Score HAZ': z_haz
    }])

    data_baru_processed = pd.get_dummies(data_baru, columns=['Cara Ukur'], drop_first=True)
    data_baru_processed = data_baru_processed.reindex(columns=X_train, fill_value=0)

    y_pred_baru = best_model.predict(data_baru_processed)
    predicted_label = label_encoder.inverse_transform(y_pred_baru)[0]

    # ==========================
    # Tentukan hasil & saran
    # ==========================
    if status_who == "Sangat Pendek":
        hasil = "Sangat Pendek (Severely Stunted)"
        saran = """Segera bawa anak ke tenaga kesehatan untuk pemeriksaan dan kemungkinan suplementasi. 
        Berikan makanan tinggi protein hewani (daging, ikan, telur) dan nabati (tahu, tempe, kacang-kacangan). 
        Jika <6 bulan berikan ASI eksklusif, jika >6 bulan berikan MPASI bergizi. 
        Jaga kebersihan agar terhindar dari infeksi, serta pantau pertumbuhan tiap bulan."""
    elif status_who == "Pendek":
        hasil = "Pendek (Stunted)"
        saran = """Perbaiki pola makan dengan makanan bergizi tinggi energi dan protein, berikan tiga kali makan utama 
        dan dua kali selingan sehat. Lengkapi imunisasi dan berikan stimulasi tumbuh kembang seperti bermain, berbicara, 
        dan bernyanyi. Konsultasikan kebutuhan gizi ke tenaga kesehatan dan pantau pertumbuhan secara rutin."""
    elif status_who == "Normal":
        hasil = "Normal (Tidak Stunting)"
        saran = """Pertahankan pola makan bergizi seimbang sesuai pedoman 'Isi Piringku'. 
        Lakukan pemantauan rutin di posyandu, jaga kebersihan lingkungan dan air minum, 
        serta dorong aktivitas fisik dan stimulasi perkembangan. 
        Berikan kasih sayang dan perhatian agar tumbuh kembang anak tetap optimal."""
    else:
        hasil = "Tinggi (Tall)"
        saran = """Anak memiliki tinggi badan di atas rata-rata. 
        Tetap jaga pola makan bergizi seimbang, dukung aktivitas fisik, 
        serta terus lakukan pemantauan rutin agar pertumbuhan anak tetap sehat dan optimal."""

    # ==========================
    # Output Hasil Prediksi
    # ==========================
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>Hasil Prediksi</h1>", unsafe_allow_html=True)

    st.markdown(f"**Umur:** {umur_bulan} bulan {umur_hari} hari")
    st.markdown(f"**Jenis Kelamin:** {jenis_kelamin}")
    st.markdown(f"**Tinggi Badan (koreksi):** {round(height_corrected,1)} cm")
    st.markdown(f"<h4 style='text-align: center; font-weight: bold;'>Z-Score HAZ</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; font-weight: bold;'>{round(z_haz, 2)}</h4>", unsafe_allow_html=True)

    if "Sangat Pendek" in hasil:  
        st.markdown(
            f"""<div style='padding:15px; border-radius:10px; background-color:#dc3545; color:white; text-align:center; font-weight:bold;'> 
                Hasil Prediksi :{hasil}
            </div>""",
            unsafe_allow_html=True
        )
    elif "Pendek" in hasil:  
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#FFD700; color:black; text-align:center; font-weight:bold;">
                Hasil Prediksi: {hasil}
            </div>
            """,
            unsafe_allow_html=True
        )
    elif "Normal" in hasil:  
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#28a745; color:white; text-align:center; font-weight:bold;">
                Hasil Prediksi: {hasil}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:  
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#007BFF; color:white; text-align:center; font-weight:bold;">
                Hasil Prediksi: {hasil}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(saran)