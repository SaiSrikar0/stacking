import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("images", exist_ok=True)

df = pd.read_csv("data/raw/housing.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.drop_duplicates()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("images/heatmap.png")

df.to_csv(
    "data/processed/cleaned_housing.csv",
    index=False
)

print("Processed dataset saved successfully!")