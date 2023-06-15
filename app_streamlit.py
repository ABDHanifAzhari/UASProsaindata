import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

data=pd.read_csv('BMRI.JK.csv')
databersih=data.dropna()

temp=databersih["Close"]
n=len(temp)
sizeTrain=(round(n*0.8))
data_Train=pd.DataFrame(temp[:sizeTrain])
data_Test=pd.DataFrame(temp[sizeTrain:])

#mengambil nama kolom
judul = data_Test.columns.copy() 

#menghitung hasil normalisasi + menampilkan
scaler = MinMaxScaler()
train = scaler.fit_transform(data_Train)
test = scaler.fit_transform(data_Test)
hasil_train = pd.DataFrame(train,columns=judul)
hasil_test = pd.DataFrame(test,columns=judul)

st.title("""Time Series PT. Bank Mandiri""")
Home, Preprocessing, Modelling, Implementasi = st.tabs(['Home','Preprocessing','Modelling','Implementasi'])
with Home:
   st.subheader('Nama Kelompok')
   st.text("""
            1. Zuni Amanda Dewi 200411100051
            2. Abd. Hanif Azhari 200411100101""")
   st.subheader('Data')
   data
   st.subheader('Deskripsi Data')
   st.write("""Saham (stock) merupakan salah satu instrumen pasar keuangan yang paling popular. Menerbitkan saham merupakan salah satu pilihan perusahaan ketika memutuskan untuk pendanaan perusahaan. Pada sisi yang lain, saham merupakan instrument investasi yang banyak dipilih para investor karena saham mampu memberikan tingkat keuntungan yang menarik""")
   st.write("""Dalam sistem ini kami menggunakan data saham dari PT. Bank Mandiri""")
   st.write("""Data tersebut kami peroleh dari finance.yahoo.com dengan link berikut :
   https://finance.yahoo.com/quote/BMRI.JK/history?p=BMRI.JK""")
   st.write("""Dimana Fitur yang ada di dalam data tersebut diantaranya :""")
   st.text("""
            1) Harga Open
            2) Harga High dan Low
            3) Harga Close
            4) Adj Close
            5) Volume""")
   st.write("""Harga Open atau harga pembukaan adalah harga yang dipasang pada transaksi pertama kali dilakukan pada hari itu.""")
   st.write("""Harga High (tertinggi) dan harga low (terendah) merupakan kisaran harga pergerakan harian dari saham tersebut dimana pemodal memiliki keberanian atau rasionalitas untuk melakukan posisi beli atau posisi jual.""")
   st.write("""Harga Close (penutupan) dan Adj Close digunakan untuk menentukan signal beli atau signal jual dalam berbagai indikator teknikal, berbagai alat analisis teknikal.""")
   st.write("""Dalam pengamatan Time Series adalah fungsi waktu, setiap data sesuai dengan contoh waktu, jadi ada hubungan antara titik data yang berbeda dari kumpulan data, kasus khusus deret waktu adalah deret waktu univariat di mana hanya memiliki satu fitur untuk ditangani""")
   
   
  
with Preprocessing:
   st.title("Preprocessing")
   pilih = st.radio(
    "Apa Yang Ingin Anda Lakukan",
    ('Min Max Scaler',))
   # met1 = st.checkbox("Min Max Scaler")
   if pilih == 'Min Max Scaler' :
      st.subheader("Normalisasi Data Train")
      st.dataframe(hasil_train)
      st.subheader("Normalisasi Data Test")
      st.dataframe(hasil_test)


with Modelling :
   st.title("Modelling")
   model1 = pickle.load(open('model_knn.sav', 'rb'))
   model2 = pickle.load(open('model_svm.sav', 'rb'))
   model3 = pickle.load(open('model_dt.sav', 'rb'))
   
   # split a univariate sequence into samples
   def split_sequence(sequence, n_steps):
      X, y = list(), list()
      for i in range(len(sequence)):
         # find the end of this pattern
         end_ix = i + n_steps
         # check if we are beyond the sequence
         if end_ix > len(sequence)-1:
            break
         # gather input and output parts of the pattern
         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
         X.append(seq_x)
         y.append(seq_y)
      return array(X), array(y)
   
   # transform to a supervised learning problem
   X1, y1 = split_sequence(hasil_train['Close'], 3)
   X2, y2 = split_sequence(hasil_test['Close'], 3)
   dfX1 = pd.DataFrame(X1, columns=["Xt-3","Xt-2", "Xt-1"])
   dfy1 = pd.DataFrame(y1, columns=["Xt"])
   dfX2 = pd.DataFrame(X2, columns=["Xt-3","Xt-2", "Xt-1"])
   dfy2 = pd.DataFrame(y2, columns=["Xt"])

   df_train = pd.concat((dfX1, dfy1), axis = 1)
   df_test = pd.concat((dfX2, dfy2), axis = 1)

   # ambil data
   X_test = df_test.drop(columns="Xt")
   y_test = df_test.Xt
   # X_test

   st.write ("Pilih metode yang ingin anda gunakan :")
   met1 = st.checkbox("KNN")
   met2 = st.checkbox("SVM")
   met3 = st.checkbox("Decision Tree")
   submit2 = st.button("Pilih")

   if submit2:      
      if met1 :
         st.subheader("Akurasi")
         y_pred = model1.predict(X_test)
         y_actual = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
         y_prediksi = scaler.inverse_transform(y_pred.reshape(-1, 1))
         mape=mean_absolute_percentage_error(y_prediksi, y_actual)
         st.write("Nilai MAPE Menggunakan KNN sebesar : ", (mape))
         st.write("Metode yang Anda gunakan Adalah KNN")

      elif met2:
         st.subheader("Akurasi")
         y_pred = model2.predict(X_test)
         y_actual = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
         y_prediksi = scaler.inverse_transform(y_pred.reshape(-1, 1))
         mape=mean_absolute_percentage_error(y_prediksi, y_actual)
         st.write("Nilai MAPE Menggunakan SVM sebesar : ", (mape))
         st.write("Metode yang Anda gunakan Adalah SVM")

      elif met3 :
         st.subheader("Akurasi")
         y_pred = model3.predict(X_test)
         y_actual = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
         y_prediksi = scaler.inverse_transform(y_pred.reshape(-1, 1))
         mape=mean_absolute_percentage_error(y_prediksi, y_actual)
         st.write("Nilai MAPE Menggunakan Decision Tree sebesar : ", (mape))
         st.write("Metode yang Anda gunakan Adalah Decision Tree")
      else :
         st.write("Anda Belum Memilih Metode")

with Implementasi:
   def submit3():
      inputs = np.array([[saham3, saham2, saham1]])
      st.write("Data Input :",inputs)
      scaler = MinMaxScaler()
      train = scaler.fit_transform(data_Train)
      x = scaler.transform(inputs.reshape(-1,1))
      st.write("Data Normalisasi",x)
      test=x.reshape(1,3)
      st.write(test)

      # create output
      if met1:
         y_pred1 = model1.predict(test)
         x=scaler.inverse_transform(y_pred1.reshape(-1,1))
      # st.success(f"Suhu ruang diprediksi sebesar : {x[0][0]}")
         st.title("k-nearest neighbors")
         st.success(f"Saham Diprediksi Sebesar : {x[0][0]}")
         
      elif met2:
         y_pred2 = model2.predict(test)
         x=scaler.inverse_transform(y_pred2.reshape(-1,1))
         st.title("Support Vector Machine")
         st.success(f"Saham Diprediksi Sebesar : {x[0][0]}")

      elif met3:
         y_pred3 = model3.predict(test)
         x=scaler.inverse_transform(y_pred3.reshape(-1,1))
         st.title("Decision Tree")
         st.success(f"Saham Diprediksi Sebesar : {x[0][0]}")

      else :
         st.write("Metode yang Anda Pilih Belum Ada, Silahkan Kembali ke Tabs Modelling Untuk memilih Metode")

   st.title("Prediksi Saham PT. Bank Mandiri")
   # saham1 = st.input("Harga Saham 1 Bulan Sebelumnya")
   saham1 = st.number_input("Harga Saham Sebelumnya (Xt-1)",3587.5,10225.0, step=0.1)
   saham2 = st.number_input("Harga Saham Sebelumnya (Xt-2)",3587.5,10225.0, step=0.1)
   saham3 = st.number_input("Harga Saham Sebelumnya (Xt-3)",3587.5,10225.0, step=0.1)

   # create button submit
   submitted = st.button("Cek")
   if submitted:
      submit3()
      st.balloons()
