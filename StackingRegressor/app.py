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

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
try:
    os.makedirs(MODELS_DIR, exist_ok=True)
except Exception:
    MODELS_DIR = None

MODEL_PATH = os.path.join(MODELS_DIR, "stacking_regressor.pkl") if MODELS_DIR else None

processed_csv = os.path.join(BASE_DIR, "data", "processed", "cleaned_housing.csv")
raw_csv = os.path.join(BASE_DIR, "data", "raw", "housing.csv")

# Try to load existing model; if unavailable, train in-memory and attempt to save.
model = None
if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

if model is None:
    # Load processed CSV if available, otherwise fall back to raw CSV
    if os.path.exists(processed_csv):
        df = pd.read_csv(processed_csv)
    elif os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
    else:
        raise FileNotFoundError(
            f"Could not find dataset. Checked: {processed_csv} and {raw_csv}"
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

    if MODEL_PATH:
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not save model file: {e}")

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