import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

MODELS_WITH_SCALER = {
    "Logistic Regression", "Linear Regression", "Ridge", "Lasso",
    "SVM", "SVR", "KNN", "KNN Regressor",
}

MODEL_MAP = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "SVM": LinearSVC,
    "KNN": KNeighborsClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Linear Regression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "Random Forest Regressor": RandomForestRegressor,
    "Decision Tree Regressor": DecisionTreeRegressor,
    "SVR": LinearSVR,
    "KNN Regressor": KNeighborsRegressor,
    "Gradient Boosting Regressor": GradientBoostingRegressor,
}

PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"],
    },
    "Random Forest": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [3, 5, 10, 15, 20, None],
    },
    "Decision Tree": {
        "model__max_depth": [2, 3, 5, 7, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10, 20],
    },
    "SVM": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__kernel": ["linear", "rbf"],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 10, 15, 20],
        "model__weights": ["uniform", "distance"],
    },
    "Gradient Boosting": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    },
    "Linear Regression": {},
    "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "Random Forest Regressor": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [3, 5, 10, 15, 20, None],
    },
    "Decision Tree Regressor": {
        "model__max_depth": [2, 3, 5, 7, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10, 20],
    },
    "SVR": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__kernel": ["linear", "rbf"],
    },
    "KNN Regressor": {
        "model__n_neighbors": [3, 5, 7, 10, 15, 20],
        "model__weights": ["uniform", "distance"],
    },
    "Gradient Boosting Regressor": {
        "model__n_estimators": [50, 100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    },
}


def build_pipeline(model_name: str, scaler: str = "standard") -> Pipeline:
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}"
        )

    model_class = MODEL_MAP[model_name]

    if model_name in MODELS_WITH_SCALER:
        if scaler == "standard":
            return Pipeline([("scaler", StandardScaler()), ("model", model_class())])
        else:
            return Pipeline([("model", model_class())])

    return Pipeline([("model", model_class())])


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def run_grid_search(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv: int = 5,
    search_type: str = "grid",
    n_iter: int = 20,
):
    if search_type == "random":
        search = RandomizedSearchCV(
            pipeline, param_grid, cv=cv, n_iter=n_iter,
            scoring="accuracy", n_jobs=-1, random_state=42,
        )
    else:
        search = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring="accuracy", n_jobs=-1,
        )
    search.fit(X_train, y_train)
    return search


def get_param_grid(model_name: str) -> dict:
    if model_name not in PARAM_GRIDS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(PARAM_GRIDS.keys())}"
        )
    return PARAM_GRIDS[model_name].copy()
