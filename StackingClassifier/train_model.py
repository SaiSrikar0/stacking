import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)

df = pd.read_csv(
    "data/processed/cleaned_iris.csv"
)

encoder = LabelEncoder()

df["Species"] = encoder.fit_transform(
    df["Species"]
)

X = df.drop("Species", axis=1)

y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

estimators = [
    (
        "rf",
        RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )
    ),
    (
        "ada",
        AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        )
    )
]

model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        predictions
    )
)

print(
    classification_report(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    "models/stacking_classifier.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("Model saved successfully!")
