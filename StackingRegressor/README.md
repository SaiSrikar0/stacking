# Stacking Regressor Housing Prediction

Deployed app : https://stacking-regre.streamlit.app/

## Overview

This project implements a Stacking Regressor model to predict Median Income using California Housing data. The model combines multiple machine learning algorithms to improve prediction performance through ensemble learning.

The application is deployed using Streamlit, allowing users to provide housing-related inputs and obtain real-time predictions.

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
4. Model Building
5. Ensemble Learning using Stacking Regressor
6. Model Evaluation
7. Streamlit Deployment

---

## Model Architecture

### Base Models
- Random Forest Regressor
- AdaBoost Regressor

### Meta Model
- Linear Regression

The predictions from the base models are combined and passed to the meta model to generate the final prediction.

---

## Evaluation Metrics

The model performance is evaluated using:

- Mean Absolute Error (MAE)
- R² Score

---

## Project Structure

```text
StackingRegressor/
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
│   │   └── housing.csv
│   │
│   └── processed/
│       └── cleaned_housing.csv
│
├── models/
│
└── images/
```

---

## Exploratory Data Analysis

The project includes:

- Data inspection
- Missing value analysis
- Duplicate removal
- Correlation Heatmap

Generated visualizations are stored in the `images` folder.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Anusreereddysama/StackingRegressor.git
```

Move into the project directory:

```bash
cd StackingRegressor
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

---

## Features

- Ensemble Learning using Stacking Regressor
- Interactive Streamlit Interface
- Real-time Predictions
- Automated Model Training
- Data Visualization
- Modular Project Structure


