import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
data = pd.read_csv("housing.csv")

# Menampilkan 5 data pertama
print(data.head())

print("\nJumlah data dan kolom:")
print(data.shape)

print("\nInformasi dataset:")
data.info()

print("\nStatistik dataset:")
print(data.describe())

# =============================
# Visualisasi Distribusi Harga
# =============================

print("\nVisualisasi distribusi harga rumah")

plt.figure(figsize=(8,5))
sns.histplot(data["median_house_value"], bins=50)
plt.title("Distribusi Harga Rumah")
plt.show()

# =============================
# Correlation Heatmap
# =============================

print("\nAnalisis Korelasi Fitur")

correlation = data.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nJumlah Missing Value:")
print(data.isnull().sum())

print("\nMengisi Missing Value dengan Median")

median_value = data["total_bedrooms"].median()
data["total_bedrooms"] = data["total_bedrooms"].fillna(median_value)

print("\nCek Missing Value setelah diperbaiki:")
print(data.isnull().sum())

print("\nEncoding Kolom Kategori: ocean_proximity")

data_encoded = pd.get_dummies(data, columns=["ocean_proximity"])

print("\n5 Data Setelah Encoding:")
print(data_encoded.head())

print("\nMenentukan Feature dan Target")

X = data_encoded.drop("median_house_value", axis=1)
y = data_encoded["median_house_value"]

print("Jumlah fitur:", X.shape)
print("Jumlah target:", y.shape)

from sklearn.model_selection import train_test_split

print("\nMembagi Data Training dan Testing")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Jumlah data training:", X_train.shape)
print("Jumlah data testing:", X_test.shape)

from sklearn.linear_model import LinearRegression

print("\nMembuat Model Linear Regression")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model berhasil dilatih")

import matplotlib.pyplot as plt
import pandas as pd

print("\nFeature Importance")

importance = pd.Series(model.coef_, index=X.columns)

importance = importance.sort_values(ascending=False)

print(importance)

plt.figure(figsize=(10,6))
importance.plot(kind='bar')
plt.title("Feature Importance")
plt.ylabel("Coefficient")
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\nEvaluasi Model")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nSimulasi Prediksi Rumah Baru")

contoh_rumah = X_test.iloc[0:1]

prediksi = model.predict(contoh_rumah)

print("Harga rumah yang diprediksi:", prediksi[0])

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai model.pkl")