
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


def generate_dataset(n_samples=120, noise=15, pattern="Curved"):
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(-3, 3, n_samples)).reshape(-1, 1)

    if pattern == "Linear":
        y = 4 * X.ravel() + 5
    elif pattern == "Curved":
        y = 2 * X.ravel() ** 2 + 3 * X.ravel() + 5
    else:
        y = 8 * np.sin(X.ravel()) + 2 * X.ravel() + 5

    y = y + rng.normal(0, noise / 10, n_samples)

    return X, y


def build_model(model_name, degree=3, alpha=1.0):
    if model_name == "Linear Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

    steps = [
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler())
    ]

    if model_name == "Polynomial Regression":
        steps.append(("regressor", LinearRegression()))
    else:
        steps.append(("regressor", Ridge(alpha=alpha)))

    return Pipeline(steps)


def evaluate_model(model_name, X_train, X_test, y_train, y_test,
                   degree=3, alpha=1.0):
    model = build_model(model_name, degree, alpha)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result = {
        "Model": model_name,
        "Train MSE": mean_squared_error(y_train, train_pred),
        "Test MSE": mean_squared_error(y_test, test_pred),
        "Train R²": r2_score(y_train, train_pred),
        "Test R²": r2_score(y_test, test_pred),
    }

    return model, result


def run_comparison(X, y, degree=5, alpha=1.0, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model_names = [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression"
    ]

    models = {}
    results = []

    for name in model_names:
        model, result = evaluate_model(
            name, X_train, X_test, y_train, y_test,
            degree=degree, alpha=alpha
        )
        models[name] = model
        results.append(result)

    metrics = pd.DataFrame(results)

    # Sort the X coordinates so the predicted curves plot smoothly.
    X_curve = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    return {
        "models": models,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_curve": X_curve,
    }
