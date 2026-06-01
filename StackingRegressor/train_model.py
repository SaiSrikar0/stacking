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

os.makedirs("models", exist_ok=True)

df = pd.read_csv(
    "data/processed/cleaned_housing.csv"
)

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

joblib.dump(
    model,
    "models/stacking_regressor.pkl"
)

print("Model saved successfully!")