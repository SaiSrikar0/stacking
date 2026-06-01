import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

processed_csv = os.path.join(BASE_DIR, "data", "processed", "cleaned_housing.csv")
raw_csv = os.path.join(BASE_DIR, "data", "raw", "housing.csv")

if os.path.exists(processed_csv):
    df = pd.read_csv(processed_csv)
elif os.path.exists(raw_csv):
    df = pd.read_csv(raw_csv)
else:
    raise FileNotFoundError(f"Could not find dataset. Checked: {processed_csv} and {raw_csv}")

X = df.drop("MedInc", axis=1)

y = df["MedInc"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

estimators = [
    (
        "rf",
        RandomForestRegressor(
            n_estimators=50,
            random_state=42
        )
    ),
    (
        "ada",
        AdaBoostRegressor(
            n_estimators=50,
            random_state=42
        )
    )
]

model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(
    "MAE:",
    mean_absolute_error(y_test, predictions)
)

print(
    "R2 Score:",
    r2_score(y_test, predictions)
)

joblib.dump(model, os.path.join(MODELS_DIR, "stacking_regressor.pkl"))

print("Model saved successfully!")