import streamlit as st
import pandas as pd
import numpy as np

import pickle
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = pickle.load(open('model_xgb.pkl','rb'))

# Title
st.title('Prediksi Resiko Terkena Obesitas')
st.write('Obesitas adalah kondisi di mana tubuh memiliki kelebihan lemak yang dapat berdampak negatif pada kesehatan')
st.write('Obesitas bisa menyebabkan berbagai masalah kesehatan, seperti: Penyakit jantung dan stroke, Diabetes tipe 2, Gangguan pernapasan dan pencernaan, Nyeri sendi, Masalah mental dan bahakan beresiko Kanker.')
st.write('Mari cegah resiko obesitas dengen mengetahui sejak awal')
st.write('Silahkan isi form berikut: ')
col1, col2 = st.columns(2)

# Input


with col1:
     # jenis_kelamin = st.number_input("Masukan Jenis Kelamin", value=0)
   Gender_options = {
    1 : "Laki-laki",
    0 : "Perempuan"
    }
   Jenis_Kelamin = st.selectbox("Pilih Jenis Kelamin", options=list(Gender_options.keys()), format_func=lambda x: Gender_options[x])
   selected_value = Gender_options[Jenis_Kelamin]

with col1:
   Age = st.number_input("Masukan umur", value=0)

with col1:
   Height = st.number_input("Masukan Tinggi Badan", value=0)

with col1:
   Weight = st.number_input("Masukan Berat Badan", value=0)

with col1:
     # Riwayat_Obesitas_Keluarga = st.number_input("Riwayat Obesitas Keluarga", value=0)
   family_history_with_overweight = {
    1 : "Ada",
    0 : "Tidak Ada"
    }
   Riwayat_Obesitas_Keluarga = st.selectbox("Riwayat Obesitas Keluarga", options=list(family_history_with_overweight.keys()), format_func=lambda x: family_history_with_overweight[x])
   selected_value = family_history_with_overweight[Riwayat_Obesitas_Keluarga]

with col1:
     # Sering Konsumsi Makakanan Berkalori = st.number_input("Sering Konsumsi Makanan Berkalori", value=0)
   FAVC = {
    1 : "Ya",
    0 : "Tidak"
    }
   Sering_Konsumsi_Makanan_Berkalori = st.selectbox("Sering Konsumsi Makanan Berkalori", options=list(FAVC.keys()), format_func=lambda x: FAVC[x])
   selected_value = FAVC[Sering_Konsumsi_Makanan_Berkalori]

with col2:
     # Konsumsi _Makanan_ Ringan= st.number_input("Konsumsi Makanan Ringan", value=0)
   CAEC = {
    3 : "Tidak",
    2 : "Kadang-Kadang",
    1 : "Sering",
    0 : "Selalu"
    }
   Konsumsi_Makanan_Ringan = st.selectbox("Konsumsi Makanan Ringan", options=list(CAEC.keys()), format_func=lambda x: CAEC[x])
   selected_value = CAEC[Konsumsi_Makanan_Ringan]

with col2:
   FCVC = st.number_input("Rate Konsumsi Sayuran (Skala 1.0 - 3.0)",value=0.0, step=0.1, max_value = 3.0)

with col2:
   NCP = st.number_input("Jumlah makan perhari (Skala 1.0 - 3.0)",value=0.0, step=0.1, max_value = 3.0)

with col2:
   CH2O = st.number_input("Rate Konsumsi Air Minum Perhari (Skala 1.0 - 3.0)",value=0.0, step=0.1, max_value = 3.0)

with col2:
   FAF = st.number_input("Rate Aktivitas Fisik Perhari (Skala 1.0 - 3.0)",value=0.0, step=0.1, max_value = 3.0)




# Pastikan semua variabel yang digunakan dalam prediksi berupa angka (int/float)
if st.button('Prediksi'):
    pred_Obesitas = model.predict([[
        Jenis_Kelamin,  # Sudah berupa integer dari selectbox
        Age, 
        Height, 
        Weight, 
        Riwayat_Obesitas_Keluarga,  # Sudah berupa integer dari selectbox
        Sering_Konsumsi_Makanan_Berkalori,  # Sudah berupa integer dari selectbox
        FCVC, 
        NCP, 
        Konsumsi_Makanan_Ringan,  # Sudah berupa integer dari selectbox
        CH2O, 
        FAF
    ]])

    st.write(f'Hasil Prediksi: {pred_Obesitas[0]}')  # Pastikan pred_Obesitas adalah array

    if pred_Obesitas == 0 :
        st.error('Insufficient Weight (Berat Badan Kurang)')
    elif pred_Obesitas == 1 :
        st.success('Normal Weight')
    elif pred_Obesitas == 2 :
        st.warning('Obesity Type I') 
    elif pred_Obesitas == 3 :
        st.warning('Obesity Type II')  
    elif pred_Obesitas == 4 :
        st.error('Obesity Type III')
    elif pred_Obesitas == 5 :
        st.error('Overweight Level I') 
    elif pred_Obesitas == 6 :
        st.error('Overweight Level II')        
    else:
        st.error('Tidak Terprediksi')

