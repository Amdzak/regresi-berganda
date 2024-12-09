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

# Praproses data
data['lastUpdated'] = pd.to_datetime('now')
data['available'] = np.where(data['available'] == 'nan','0', data['available'])
data['sold'] = np.where(data['sold'] == 'NaN','0', data['sold'])

# encoder
encoder_brand = LabelEncoder()
encoder_type = LabelEncoder()

# Encode kolom 'brand' dan 'type'
data['brand_encoded'] = encoder_brand.fit_transform(data['brand'])
data['type_encoded'] = encoder_type.fit_transform(data['type'])

# Menyimpan mapping label asli untuk 'brand' dan 'type'
brand_mapping_df = pd.DataFrame({
    'Brand': encoder_brand.classes_,
    'Encoded Value': encoder_brand.transform(encoder_brand.classes_)
})

type_mapping_df = pd.DataFrame({
    'Type': encoder_type.classes_,
    'Encoded Value': encoder_type.transform(encoder_type.classes_)
})

# Menampilkan mapping di Streamlit
st.title('Prediksi Harga Parfum')
st.subheader("Dataset")
st.dataframe(data)

st.subheader("Mapping Brand dan Type")
st.write("**Daftar Brand (Tabel):**")
st.dataframe(brand_mapping_df)

st.write("**Daftar Type (Tabel):**")
st.dataframe(type_mapping_df)

# Memilih beberapa kolom sebagai fitur (X) dan satu kolom sebagai target (y)
X = data[['available', 'brand_encoded', 'type_encoded', 'sold']]  # Fitur: available, brand_encoded, type_encoded, sold
y = data['price']  # Target: price

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

# Pastikan y_test dan y_pred adalah array satu dimensi
y_test = y_test.ravel()
y_pred = y_pred.ravel()

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)  # Menghitung MSE
rmse = np.sqrt(mse)  # Menghitung RMSE secara manual

# Menampilkan hasil evaluasi
st.subheader("Regresi Linear Berganda")
st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
st.write(f"**Persamaan regresi:** y = a + b1x1 + b2x2 + ... + bnxn")
st.write(f"**Persamaan regresi:** y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Tersedia + {model.coef_[1]:.2f} * Brand + {model.coef_[2]:.2f} * Type + {model.coef_[3]:.2f} * Terjual")
st.write(f"**Koefisien regresi:** {model.coef_}")
st.write(f"**Intercept regresi:** {model.intercept_}")

# Menampilkan grafik
st.subheader('Visualisasi Prediksi Harga')
fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], y_test, color='blue', label='Data Uji')
ax.scatter(X_test[:, 0], y_pred, color='red', label='Prediksi')
ax.set_xlabel('Tersedia')
ax.set_ylabel('Harga')
ax.set_title('Perbandingan Data Uji dan Prediksi Harga')
ax.legend()
st.pyplot(fig)

# Input data untuk prediksi
st.subheader('Masukkan Data untuk Prediksi:')
available = st.number_input('Tersedia', min_value=0, value=9)
brand = st.number_input('Brand (0 untuk AS PHOTOS, 1 untuk AS PICTURE SHOWN, dst)', min_value=0, value=3)
type = st.number_input('Type (0 untuk Aftershave, 1 untuk Assorted, dst)', min_value=0, value=1)
sold = st.number_input('Terjual', min_value=0, value=25)

# Prediksi jika tombol dipencet
if st.button('Prediksi Harga'):
    prediction = model.predict([[available, brand, type, sold]])
    st.write(f"**Prediksi Harga:** ${prediction[0]:.2f}")
