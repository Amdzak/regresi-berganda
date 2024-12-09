import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Membaca data
data = pd.read_csv('ebay_mens_perfume.csv')

# Mengonversi kolom 'sold' menjadi numerik jika perlu, dan menghapus data yang tidak relevan
data['sold'] = pd.to_numeric(data['sold'], errors='coerce')
data = data[data['sold'] < 50.0]  # filter data jika diperlukan

# Mengonversi kolom kategorikal (misalnya 'brand' dan 'type') menjadi numerik
encoder = LabelEncoder()
data['brand'] = encoder.fit_transform(data['brand'])
data['type'] = encoder.fit_transform(data['type'])

# Memilih beberapa kolom sebagai fitur (X) dan satu kolom sebagai target (y)
X = data[['price', 'available', 'brand', 'type']]  # Fitur: price, available, brand, type
y = data['sold']  # Target: sold

# Imputasi nilai yang hilang (jika ada)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)  # Imputasi pada fitur
y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()  # Imputasi pada target jika perlu

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear berganda
model = LinearRegression()
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Menampilkan hasil evaluasi
st.title('Prediksi Penjualan Parfum')
# st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
# st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
st.write(f"**Koefisien regresi:** {model.coef_}")
st.write(f"**Intercept regresi:** {model.intercept_}")

# Menampilkan grafik
st.subheader('Visualisasi Prediksi Penjualan')
fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], y_test, color='blue', label='Data Uji')
ax.scatter(X_test[:, 0], y_pred, color='red', label='Prediksi')
ax.set_xlabel('Harga')
ax.set_ylabel('Penjualan')
ax.set_title('Perbandingan Data Uji dan Prediksi Penjualan')
ax.legend()
st.pyplot(fig)

# Input data untuk prediksi
st.subheader('Masukkan Data untuk Prediksi:')
price = st.number_input('Harga', min_value=0.0, value=30.99)
available = st.number_input('Tersedia', min_value=0, value=9)
brand = st.number_input('Brand (0 untuk Unbranded, 1 untuk AS SHOW, dst)', min_value=0, value=3)
type = st.number_input('Type (0 untuk Eau de Toilette, 1 untuk Eau de Parfum, dst)', min_value=0, value=1)

# Prediksi jika tombol dipencet
if st.button('Prediksi Penjualan'):
    prediction = model.predict([[price, available, brand, type]])
    st.write(f"**Prediksi Penjualan:** {prediction[0]:.2f}")


