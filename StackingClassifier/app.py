import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
try:
    os.makedirs(MODELS_DIR, exist_ok=True)
except Exception:
    MODELS_DIR = None

MODEL_PATH = os.path.join(MODELS_DIR, "stacking_classifier.pkl") if MODELS_DIR else None
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl") if MODELS_DIR else None

processed_csv = os.path.join(BASE_DIR, "data", "processed", "cleaned_iris.csv")
raw_csv = os.path.join(BASE_DIR, "data", "raw", "Iris.csv")

# Try loading pretrained model; if loading fails, train in-memory and attempt to save.
model = None
encoder = None
if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except Exception:
        model = None
        encoder = None

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

    encoder = LabelEncoder()
    df["Species"] = encoder.fit_transform(df["Species"])
    X = df.drop("Species", axis=1)
    y = df["Species"]

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=20, random_state=42)),
        ("ada", AdaBoostClassifier(n_estimators=20, random_state=42)),
    ]

    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    model.fit(X, y)

    # Attempt to save model and encoder; if saving fails (e.g., read-only filesystem), continue without failing.
    if MODEL_PATH and ENCODER_PATH:
        try:
            joblib.dump(model, MODEL_PATH)
            joblib.dump(encoder, ENCODER_PATH)
        except Exception as e:
            st.warning(f"Could not save model files: {e}")

st.title(
    "Iris Flower Stacking Classifier"
)

st.write(
    "Predict Iris Species using Ensemble Learning"
)

id_value = st.number_input(
    "Id",
    min_value=1
)

sepal_length = st.number_input(
    "Sepal Length"
)

sepal_width = st.number_input(
    "Sepal Width"
)

petal_length = st.number_input(
    "Petal Length"
)

petal_width = st.number_input(
    "Petal Width"
)

if st.button("Predict Species"):

    input_data = np.array([[

        id_value,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width

    ]])

    prediction = model.predict(
        input_data
    )

    species = encoder.inverse_transform(
        prediction
    )

    st.success(
        f"Predicted Species: {species[0]}"
    )