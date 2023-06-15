import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from sklearn.preprocessing import MinMaxScaler

Home, Preprocessing, Modelling, Implementasi = st.tabs(['Home','Preprocessing','Modelling','Implementasi'])
with Home:
   st.title("""Time Series PT. Bank Mandiri""")
   st.subheader('Nama Kelompok')
   st.text("""
            1. Zuni Amanda Dewi 200411100051
            2. Abd. Hanif Azhari 200411100101""")
   st.subheader('Data')
   data=pd.read_csv('BMRI.JK.csv')
   data
   st.subheader('Deskripsi Data')
   st.write("""Harga Open atau harga pembukaan adalah harga yang dipasang pada transaksi pertama kali dilakukan pada hari itu.""")
   st.write("""Harga High (tertinggi) dan harga low (terendah) merupakan kisaran harga pergerakan harian dari saham tersebut dimana pemodal memiliki keberanian atau rasionalitas untuk melakukan posisi beli atau posisi jual.""")
   st.write("""Harga Close (penutupan) dan Adj Close digunakan untuk menentukan signal beli atau signal jual dalam berbagai indikator teknikal, berbagai alat analisis teknikal.""")
   st.write("""Dalam pengamatan Time Series adalah fungsi waktu, setiap data sesuai dengan contoh waktu, jadi ada hubungan antara titik data yang berbeda dari kumpulan data, kasus khusus deret waktu adalah deret waktu univariat di mana hanya memiliki satu fitur untuk ditangani""")
   st.write("""Dalam sistem ini, kami mengambil data yang diperoleh dari finance.yahoo.com dengan link berikut :
   https://finance.yahoo.com/quote/BMRI.JK/history?p=BMRI.JK""")
   st.write("""Dimana Fitur yang ada di dalam data tersebut diantaranya :""")
   st.text("""
            1) Harga Open
            2) Harga High dan Low
            3) Harga Close
            4) Adj Close
            5) Volume""")
  
# with Preprocessing:


# with Modelling :



# with Implementasi:
#    st.title("""Implementasi Data""")

#    suhu1 = st.number_input("Temprature Ruangan 1 Jam Sebelumnya", 5.35, 36.5, step=0.01)
#    suhu2 = st.number_input("Temprature Ruangan 2 Jam Sebelumnya", 5.35, 36.5, step=0.01)
#    suhu3 = st.number_input("Temprature Ruangan 3 Jam Sebelumnya", 5.35, 36.5, step=0.01)

#    def submit():
#       #load save model
#       model = pickle.load(open('model_knn.sav', 'rb'))
#       data=pd.read_csv('MLTempDataset.csv', index_col=0)
#       temp=data["DAYTON_MW"]
#       n=len(temp)
#       sizeTrain=(round(n*0.8))
#       data_Train=pd.DataFrame(temp[:sizeTrain])

#       scaler=MinMaxScaler()
#       scaled = scaler.fit_transform(data_Train)
      
#       inputs = np.array([[suhu3, suhu2, suhu1]])
#       st.write("Data Input :",inputs)
#       x = scaler.transform(inputs.reshape(-1,1))
#       st.write("Hasil Normalisasi :",x)
#     #   test=np.array(x)
#       test=x.reshape(1,3)
#       st.write(test)
#       y_pred = model.predict(test) 
#       st.write("Hasil Prediksi :",y_pred)
#       x=scaler.inverse_transform(y_pred.reshape(-1,1))
#       st.success(f"Suhu ruang diprediksi sebesar : {x[0][0]}")
      
#    all = st.button("Submit")
#    if all :
#       st.balloons()
#       submit()