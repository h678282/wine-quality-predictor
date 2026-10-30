
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# finds the project directory and the data files
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# loads the red wine and the white wine datasets
red_wine = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
white_wine = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")

# adds a feature that identifies the wine type
red_wine["wine_type"] = "red"
white_wine["wine_type"] = "white"

# combines the datasets 
wine = pd.concat([red_wine, white_wine], ignore_index=True)

print("=== DATASET OVERVIEW ===")
print(f"Number of observations: {len(wine)}")
print(f"Number of features including wine_type: {wine.shape[1] - 1}")
print(f"Number of columns including target: {wine.shape[1]}")
print(f"Red wines: {len(red_wine)}")
print(f"White wines: {len(white_wine)}")

print("\n=== FIRST FIVE ROWS ===")
print(wine.head())

print("\n=== DATA TYPES ===")
print(wine.dtypes)

print("\n=== MISSING VALUES ===")
print(wine.isnull().sum())

print("\n=== NUMERICAL SUMMARY ===")
print(wine.describe().round(2))

print("\n=== QUALITY DISTRIBUTION ===")
print(wine["quality"].value_counts().sort_index())

print("\n=== QUALITY BY WINE TYPE ===")
print(wine.groupby("wine_type")["quality"].agg(["count", "mean", "median"]))

# create a folder for plots
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# plot the distribution of quality scores
plt.figure(figsize=(9, 5))
sns.countplot(data=wine, x="quality", hue="wine_type")
plt.title("Wine Quality Distribution")
plt.xlabel("Quality score")
plt.ylabel("Number of wines")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "quality_distribution.png", dpi=150)
plt.close()

# plot correlations between numerical variables
plt.figure(figsize=(11, 8))
correlation = wine.drop(columns="wine_type").corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

print("\nPlots saved in the outputs folder.")
