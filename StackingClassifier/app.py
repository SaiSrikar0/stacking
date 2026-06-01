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

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/stacking_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

if not os.path.exists(MODEL_PATH):

    df = pd.read_csv(
        "data/processed/cleaned_iris.csv"
    )

    encoder = LabelEncoder()

    df["Species"] = encoder.fit_transform(
        df["Species"]
    )

    X = df.drop("Species", axis=1)

    y = df["Species"]

    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=20,
                random_state=42
            )
        ),
        (
            "ada",
            AdaBoostClassifier(
                n_estimators=20,
                random_state=42
            )
        )
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression()
    )

    model.fit(X, y)

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

else:
    model = joblib.load(MODEL_PATH)

    encoder = joblib.load(
        ENCODER_PATH
    )

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