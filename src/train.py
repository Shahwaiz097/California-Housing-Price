from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
DATA_FILE = DATA_DIR / "housing.csv"
SAMPLE_FILE = DATA_DIR / "housing_sample.csv"


def load_data():
    """Load full data if available, otherwise download it, otherwise use sample data."""
    if DATA_FILE.exists():
        print("Loading data/housing.csv")
        return pd.read_csv(DATA_FILE)

    try:
        print("Downloading California housing dataset...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_FILE, index=False)
        print("Saved full dataset to data/housing.csv")
        return df
    except Exception as error:
        print("Could not download full dataset.")
        print("Using data/housing_sample.csv instead.")
        print("Error:", error)
        return pd.read_csv(SAMPLE_FILE)


def build_model(regressor):
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    return model


def evaluate_model(model, X_test, y_test, model_name, prediction_file, plot_file):
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })
    results.to_csv(OUTPUTS_DIR / prediction_file, index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.35)
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value])
    plt.xlabel("Actual house value")
    plt.ylabel("Predicted house value")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / plot_file, dpi=150)
    plt.close()

    return {
        "model": model_name,
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2_score": round(r2, 4)
    }


if __name__ == "__main__":
    df = load_data()

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear_model = build_model(LinearRegression())
    random_forest_model = build_model(
        RandomForestRegressor(n_estimators=200, random_state=42)
    )

    linear_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    metrics = []
    metrics.append(evaluate_model(
        linear_model,
        X_test,
        y_test,
        "Linear Regression",
        "housing_predictions_linear_regression.csv",
        "actual_vs_predicted_linear_regression.png"
    ))

    metrics.append(evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest Regression",
        "housing_predictions_random_forest.csv",
        "actual_vs_predicted_random_forest_regression.png"
    ))

    pd.DataFrame(metrics).to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)

    joblib.dump(linear_model, MODELS_DIR / "linear_regression_model.pkl")
    joblib.dump(random_forest_model, MODELS_DIR / "random_forest_model.pkl")

    print("Training complete.")
    print(pd.DataFrame(metrics))
