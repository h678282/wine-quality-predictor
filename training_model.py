
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# load in necessary data

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")

red["wine_type"] = "red"
white["wine_type"] = "white"

wine = pd.concat([red, white], ignore_index=True)

X = wine.drop(columns="quality")
y = wine["quality"]

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = ["wine_type"]


# split the data in training part and test part (80 - 20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("=== DATA SPLIT ===")
print(f"Training: {len(X_train)}")
print(f"Test: {len(X_test)}")


# preprocessing

linear_preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

tree_preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_features),
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])


# defining the different models to test (random forest and linear regression, baseline used for comparison)

models = {
    "Baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": Pipeline([
        ("preprocessor", linear_preprocessor),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
}


# cross validation

cv_results = []

print("\n=== 5-FOLD CROSS-VALIDATION ===")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=1
    )

    mae_scores = -scores

    cv_results.append({
        "Model": name,
        "Mean CV MAE": mae_scores.mean(),
        "Std CV MAE": mae_scores.std()
    })

    print(f"\n{name}")
    print(f"Mean MAE: {mae_scores.mean():.4f}")
    print(f"Std MAE:  {mae_scores.std():.4f}")


cv_df = pd.DataFrame(cv_results).sort_values("Mean CV MAE")

print("\n=== CROSS-VALIDATION COMPARISON ===")
print(cv_df.to_string(index=False))


# selecting the best model and training it

best_model_name = cv_df.iloc[0]["Model"]
best_model = models[best_model_name]

print(f"\nSelected model: {best_model_name}")

best_model.fit(X_train, y_train)


# final evaluation on test data

predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n=== FINAL TEST RESULTS ===")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# saving the model

joblib.dump(best_model, MODEL_DIR / "wine_model.joblib")

print("\nModel saved to models/wine_model.joblib")


# permutation importance

importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
}).sort_values("Importance", ascending=False)

print("\n=== PERMUTATION IMPORTANCE ===")
print(importance_df.to_string(index=False))

# visualizing feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    xerr=importance_df["Std"]
)

plt.xlabel("Increase in MAE after permutation")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=150
)

plt.close()

print("\nFeature importance plot saved.")


# error analysis

# combining actual values and predicted values
error_analysis = X_test.copy()

error_analysis["actual_quality"] = y_test
error_analysis["predicted_quality"] = predictions
error_analysis["absolute_error"] = (
    error_analysis["actual_quality"]
    - error_analysis["predicted_quality"]
).abs()

error_analysis["error"] = (
    error_analysis["predicted_quality"]
    - error_analysis["actual_quality"]
)

# error by wine type (red / white)
print("\n=== ERROR BY WINE TYPE ===")

print(
    error_analysis.groupby("wine_type")["absolute_error"]
    .agg(["count", "mean", "median"])
    .round(4)
)

# error by actual quality
print("\n=== ERROR BY ACTUAL QUALITY ===")

print(
    error_analysis.groupby("actual_quality")["absolute_error"]
    .agg(["count", "mean", "median"])
    .round(4)
)

# plot actual quality vs predicted quality
plt.figure(figsize=(7, 6))

plt.scatter(
    error_analysis["actual_quality"],
    error_analysis["predicted_quality"],
    alpha=0.25
)

plt.plot([3, 9], [3, 9], "r--", label="Perfect predictions")

plt.xlabel("Actual quality")
plt.ylabel("Predicted quality")
plt.title("Actual vs Predicted Wine Quality")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=150
)

plt.close()

print("\nError analysis plot saved.")
