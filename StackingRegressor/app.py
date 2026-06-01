import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/stacking_regressor.pkl"

if not os.path.exists(MODEL_PATH):

    df = pd.read_csv(
        "data/processed/cleaned_housing.csv"
    )

    X = df.drop("MedInc", axis=1)

    y = df["MedInc"]

    estimators = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=20,
                random_state=42
            )
        ),
        (
            "ada",
            AdaBoostRegressor(
                n_estimators=20,
                random_state=42
            )
        )
    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression()
    )

    model.fit(X, y)

    joblib.dump(
        model,
        MODEL_PATH
    )

else:
    model = joblib.load(MODEL_PATH)

st.title("Stacking Regressor Housing Prediction")

house_age = st.number_input("House Age")

ave_rooms = st.number_input("Average Rooms")

ave_bedrooms = st.number_input("Average Bedrooms")

population = st.number_input("Population")

ave_occup = st.number_input("Average Occupancy")

latitude = st.number_input("Latitude")

longitude = st.number_input("Longitude")

if st.button("Predict"):

    input_data = np.array([[

        house_age,
        ave_rooms,
        ave_bedrooms,
        population,
        ave_occup,
        latitude,
        longitude

    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Median Income: {prediction[0]:.2f}"
    )