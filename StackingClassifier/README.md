# Stacking Classifier - Iris Flower Classification

Deployed app : https://stacking-classi.streamlit.app/

## Overview

This project implements a Stacking Classifier to predict Iris flower species using ensemble learning techniques. The model combines multiple machine learning algorithms to improve classification performance and prediction accuracy.

The application is deployed using Streamlit, allowing users to enter flower measurements and receive real-time species predictions.

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

## Machine Learning Workflow

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Engineering
5. Ensemble Learning using Stacking Classifier
6. Model Evaluation
7. Streamlit Deployment

---

## Model Architecture

### Base Models
- Random Forest Classifier
- AdaBoost Classifier

### Meta Model
- Logistic Regression

The predictions from the base models are combined and passed to the meta model to generate the final classification.

---

## Evaluation Metrics

The model performance is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report

---

## Dataset

Dataset Used: Iris Flower Dataset

Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target:
- Species

Classes:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

## Project Structure

```text
StackingClassifier/
│
├── app.py
├── eda.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── iris.csv
│   │
│   └── processed/
│       └── cleaned_iris.csv
│
├── models/
│
└── images/
```

---

## Exploratory Data Analysis

The project includes:

- Dataset Inspection
- Missing Value Analysis
- Duplicate Removal
- Correlation Heatmap
- Pairplot Visualization

Generated visualizations are stored in the `images` folder.

---

## Features

- Ensemble Learning using Stacking Classifier
- Interactive Streamlit Interface
- Real-time Predictions
- Automated Model Training
- Data Visualization
- Modular Project Structure

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Anusreereddysama/StackingClassifier.git
```

Move into the project directory:

```bash
cd StackingClassifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run EDA

```bash
python eda.py
```

---

## Train Model

```bash
python train_model.py
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```
