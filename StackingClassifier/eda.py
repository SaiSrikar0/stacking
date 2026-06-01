import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("images", exist_ok=True)

df = pd.read_csv("data/raw/iris.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.drop_duplicates()

plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")

pairplot = sns.pairplot(
    df,
    hue="Species"
)

pairplot.savefig("images/pairplot.png")

df.to_csv(
    "data/processed/cleaned_iris.csv",
    index=False
)

print("Processed dataset saved successfully!")