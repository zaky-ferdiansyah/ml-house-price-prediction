import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset

data = pd.read_csv("housing.csv")

# Mengisi missing value

data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

# Encoding kategori

data = pd.get_dummies(data, columns=["ocean_proximity"])

# Feature dan target

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Train model

import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== Streamlit App =====

st.title("Prediksi Harga Rumah California")

st.write("Masukkan data rumah untuk memprediksi harga.")

# Input user

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=40.0)
total_rooms = st.number_input("Total Rooms", value=1000.0)
total_bedrooms = st.number_input("Total Bedrooms", value=200.0)
population = st.number_input("Population", value=300.0)
households = st.number_input("Households", value=150.0)
median_income = st.number_input("Median Income", value=3.0)

# Tombol prediksi

if st.button("Prediksi Harga Rumah"):

    input_dict = {col:0 for col in X.columns}

    input_dict["longitude"] = longitude
    input_dict["latitude"] = latitude
    input_dict["housing_median_age"] = housing_median_age
    input_dict["total_rooms"] = total_rooms
    input_dict["total_bedrooms"] = total_bedrooms
    input_dict["population"] = population
    input_dict["households"] = households
    input_dict["median_income"] = median_income

    input_data = pd.DataFrame([input_dict])

    prediction = model.predict(input_data)


    st.success(f"Perkiraan harga rumah: ${prediction[0]:,.2f}")
