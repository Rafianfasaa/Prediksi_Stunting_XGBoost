import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
def get_who_row(who_df, umur_bulan):
    return who_df[who_df["Month"] == umur_bulan].iloc[0]

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
    elif -2 <= z_score <= 3:  # Bisa pakai <= 2 untuk ketat, atau <= 3 sesuai WHO
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

jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
jk_val = 1 if jenis_kelamin == "Laki-laki" else 0

tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=0.0, step=0.1, format="%.1f")
cara_ukur = st.radio("Cara Ukur", ["STANDING", "LYING DOWN"])
umur_bulan = st.number_input("Umur (bulan)", min_value=0, max_value=60, step=1)

# ==========================
# Prediksi
# ==========================
if st.button("Prediksi"):

    # Tentukan WHO reference sesuai umur & jenis kelamin
    if jk_val == 1: 
        who_df_new = pb_l if umur_bulan < 24 else tb_l
    else:  
        who_df_new = pb_p if umur_bulan < 24 else tb_p

    who_row_new = get_who_row(who_df_new, umur_bulan)

    # Koreksi tinggi badan jika cara ukur tidak sesuai
    height_corrected = tinggi_badan
    if cara_ukur == "LYING DOWN" and umur_bulan >= 24:
        height_corrected += 0.7
    elif cara_ukur == "STANDING" and umur_bulan < 24:
        height_corrected -= 0.7

    # Hitung Z-Score
    z_haz = calculate_zscore(height_corrected, who_row_new["L"], who_row_new["M"], who_row_new["S"])
    status_who = stunting_status(z_haz)

    # Buat DataFrame baru untuk prediksi
    data_baru = pd.DataFrame([{
        'Jenis Kelamin': jk_val,
        'Tinggi Badan (cm)': tinggi_badan,
        'Cara Ukur': cara_ukur,
        'Umur (bulan)': umur_bulan,
        'Z-Score HAZ': z_haz
    }])

    # Preprocessing sama dengan data training
    data_baru_processed = pd.get_dummies(data_baru, columns=['Cara Ukur'], drop_first=True)
    data_baru_processed = data_baru_processed.reindex(columns=X_train, fill_value=0)

    # Prediksi
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
    elif status_who  == "Normal":
        hasil = "Normal (Tidak Stunting)"
        saran = """Pertahankan pola makan bergizi seimbang sesuai pedoman 'Isi Piringku'. 
        Lakukan pemantauan rutin di posyandu, jaga kebersihan lingkungan dan air minum, 
        serta dorong aktivitas fisik dan stimulasi perkembangan. 
        Berikan kasih sayang dan perhatian agar tumbuh kembang anak tetap optimal."""
    else:  # Tinggi
        hasil = "Tinggi (Tall)"
        saran = """Anak memiliki tinggi badan di atas rata-rata. 
        Tetap jaga pola makan bergizi seimbang, dukung aktivitas fisik, 
        serta terus lakukan pemantauan rutin agar pertumbuhan anak tetap sehat dan optimal."""

    # ==========================
    # Output Hasil Prediksi
    # ==========================
    st.markdown("<h1 class='center-title' style='text-align: center; font-weight: bold;'>Hasil Prediksi</h1>", unsafe_allow_html=True)

    st.markdown(f"<h4 style='text-align: center; font-weight: bold;'>Z-Score HAZ</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; font-weight: bold;'>{round(z_haz, 2)}</h4>", unsafe_allow_html=True)

    if "Sangat Pendek" in hasil:  
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#dc3545; color:white; text-align:center; font-weight:bold;">
                Hasil Prediksi: {hasil}
            </div>
            """,
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
    else:  # Tinggi
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:#007BFF; color:white; text-align:center; font-weight:bold;">
                Hasil Prediksi: {hasil}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(saran)