# -*- coding: utf-8 -*-
"""UAS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xULKb4bNE3HVVKTSlWquL0R2_yAC0POQ

# **MENS PERFUME**

## **Praproses Data**

1. Setup Kebutuhan Hehe
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

"""2. Buah list x dan y kemudian plot dalam diagram titik"""

data = pd.read_csv('ebay_mens_perfume.csv')
data.head()

data.isnull().sum()

data['lastUpdated'] = pd.to_datetime('now')

data['available'] = np.where(data['available'] == 'NaN','0', data['available'])
data['sold'] = np.where(data['sold'] == 'NaN','0', data['sold'])
# data['type'] = np.where(data['type'] == 'NaN','/', data['type'])
# data['brand'] = np.where(data['brand'] == 'NaN','-', data['brand'])

# print(data['type'].unique())
# data.isnull().sum()

for kolom in data.columns :
  if data[kolom].dtype == 'object' :
    data[kolom].fillna('-', inplace=True)

data.isnull().sum()

data.head()

"""## **Klasterisasi**"""

# Assuming your data is a list of tuples like [(price, sold), ...]
# First, convert the list to a Pandas DataFrame
data = pd.DataFrame(data, columns=['price', 'sold'])

# Convert 'sold' column to numeric, handling errors
data['sold'] = pd.to_numeric(data['sold'], errors='coerce')

# Now, you can perform the filtering and plotting operations:
data = data[data['sold'] < 50.0] # Now this line works since data is a dataframe

x=data['price'].to_numpy()
y=data['sold'].to_numpy()
plt.scatter(x,y)
plt.show()

"""3. Gabungkan data x dan y dalam menggunakan zip():"""

data = list(zip(x,y))
data

"""4. Import library sklearn dan modul KMeans. Lakukan
pengelompokan menjadi 2 klaster dan plot dalam diagram
titik:
"""

from sklearn.cluster import KMeans
kmeans = KMeans (n_clusters = 2)
kmeans.fit(data)
plt.scatter(x,y, c=kmeans.labels_)
plt.show()

"""5. Buat dataframe yang menampilkan nilai x dan y dan hasil
klaster dari k-means!
"""

data2cluster = pd.DataFrame(data, columns = ['x','y'])
data2cluster ['Cluster'] = kmeans.labels_
data2cluster

"""6. Tampilkan diagram boxplot hasil pengelompokan data
berdasarkan nilai x:
"""

sns.boxplot(x = 'Cluster', y = 'x', data = data2cluster).set_title('Pengelompokan data berdasarkan x')
plt.show()

"""7. Tampilkan diagram boxplot hasil pengelompokan data
berdasarkan nilai y:
"""

sns.boxplot(x = 'Cluster', y = 'y', data = data2cluster).set_title('Pengelompokan data berdasarkan y')
plt.show()

"""8. Lakukan pengelompokan kembali menjadi 3 klaster dan
plot dalam diagram titik:
"""

kmeans = KMeans (n_clusters= 3)
kmeans.fit(data)
plt.scatter(x,y, c=kmeans.labels_)
plt.show()

"""9. Buat dataframe yang menampilkan nilai x dan y dan hasil
klaster dari k-means!
"""

data3cluster=pd.DataFrame(data,columns=['x','y'])
data3cluster['Cluster']=kmeans.labels_
data3cluster

"""10. Tampilkan diagram boxplot hasil pengelompokan data
berdasarkan nilai x:
"""

sns.boxplot(x='Cluster',y='x',data=data3cluster).set_title('Pengelompokan data berdasarkan x')
plt.show()

"""11. Tampilkan diagram boxplot hasil pengelompokan data
berdasarkan nilai y:
"""

sns.boxplot(x='Cluster',y='y',data=data3cluster).set_title('Pengelompokan data berdasarkan y')
plt.show()

"""12. Lakukan pengelompokan kembali menjadi 4 klaster dan
plot dalam diagram titik:
"""

kmeans =KMeans (n_clusters= 4)
kmeans.fit(data)
plt.scatter(x,y,c=kmeans.labels_)
plt.show()

"""13. Buat dataframe yang menampilkan nilai x dan y dan hasil
klaster dari k-means!
"""

data4cluster=pd.DataFrame(data, columns=['x','y'])
data4cluster ['Cluster']=kmeans.labels_
data4cluster

"""14. Tampilkan diagram boxplot hasil pengelompokan data
berdasarkan nilai x:
"""

sns.boxplot(x='Cluster',y='x',data=data4cluster).set_title('Pengelompokan data berdasarkan x')
plt.show()

"""15. Tampilkan diagram boxplot hasil pengelompokan data
berdasarkan nilai y:
"""

sns.boxplot(x='Cluster',y='y',data=data4cluster).set_title('Pengelompokan data berdasarkan y')
plt.show()

"""17. Hitung nilai inertia dari setiap hasil klaster. Tampilkan nilai
inertia dalam diagram garis:
"""

inertias = []

for i in range (1,11):
  kmeans = KMeans (n_clusters= i)
  kmeans.fit(data)
  inertias.append(kmeans.inertia_)

plt.plot(range (1,11), inertias, marker = 'o')
plt.title('Metode Elbow')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.show()

"""## **Regresi (Prediksi)**"""



"""2. Plot data ke dalam diagram scatter"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

# Assuming original_data holds the full data before KMeans operations
# Load or retrieve the original data
original_data = pd.read_csv('ebay_mens_perfume.csv')  # Or whatever your original data source was

# Filter and convert as needed
original_data['sold'] = pd.to_numeric(original_data['sold'], errors='coerce')
original_data = original_data[original_data['sold'] < 50.0]

# Create the DataFrame
data = pd.DataFrame(original_data, columns=['price', 'sold', 'available', 'brand', 'type'])

# Select 3 columns for plotting
x1 = data[['price', 'available', 'sold']]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D
ax.scatter3D(x1['price'], x1['available'], x1['sold'], c='r', marker='o')

# Adding labels to the axes
ax.set_xlabel('Price')
ax.set_ylabel('Available')
ax.set_zlabel('Sold')

# Show plot
plt.show()

"""3. Ambil nilai volume dan weight sebagai nilai x1 dan x2"""

X = data[['price', 'available', 'brand', 'type']]
X.head()

"""4. Ambil nilai CO2 sebagai nilai y"""

y = data['sold']
y.head()

"""5. Hitung nilai regresi berganda dan tampilkan

encoder data
"""

encoder = LabelEncoder()
X['brand'] = encoder.fit_transform(data['brand'])
X['type'] = encoder.fit_transform(data['type'])
# X.head()
# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean') # or strategy='median', 'most_frequent', 'constant'
X = imputer.fit_transform(X)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear berganda
model = LinearRegression()
model.fit(X_train, y_train)

# Melakukan prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
print(regr.intercept_)

"""6. Gunakan model regresi di no 5 untuk memprediksi sebuah
data!
"""

y_pred = regr.predict([[30.99	, 9.0, 0, 0]])
print(y_pred)

