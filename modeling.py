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
    "SVM", "SVR", "KNN", "K-Nearest Neighbors", "KNN Regressor",
}

MODEL_MAP = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "SVM": LinearSVC,
    "KNN": KNeighborsClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
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
        "model__penalty": ["l2"],
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
    "K-Nearest Neighbors": {
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


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.inspection import partial_dependence as sklearn_partial_dependence


def run_grid_search(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv: int = 5,
    search_type: str = "grid",
    n_iter: int = 20,
    scoring: str = None,
):
    if scoring is None:
        scoring = "accuracy"

    if search_type == "random":
        search = RandomizedSearchCV(
            pipeline, param_grid, cv=cv, n_iter=n_iter,
            scoring=scoring, n_jobs=-1, random_state=42,
        )
    else:
        search = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring=scoring, n_jobs=-1,
        )
    search.fit(X_train, y_train)
    return search


def run_random_search(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv: int = 5,
    n_iter: int = 20,
    scoring: str = None,
):
    return run_grid_search(
        pipeline, X_train, y_train, param_grid,
        cv=cv, search_type="random", n_iter=n_iter, scoring=scoring,
    )


def compute_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: np.ndarray = None,
    scoring: str = None,
) -> dict:
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    if scoring is None:
        scoring = "accuracy"

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes,
        scoring=scoring, n_jobs=-1,
    )

    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores": train_scores.tolist(),
        "val_scores": val_scores.tolist(),
        "train_mean": np.mean(train_scores, axis=1).tolist(),
        "train_std": np.std(train_scores, axis=1).tolist(),
        "cv_mean": np.mean(val_scores, axis=1).tolist(),
        "cv_std": np.std(val_scores, axis=1).tolist(),
    }


def compute_validation_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_name: str,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: str = None,
) -> dict:
    if scoring is None:
        scoring = "accuracy"

    train_scores, val_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1,
    )

    return {
        "param_range": list(param_range),
        "train_scores": train_scores.tolist(),
        "val_scores": val_scores.tolist(),
    }


def permutation_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    result = sklearn_permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    df = pd.DataFrame({
        "feature": X_test.columns.tolist(),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def get_model_coefficients(model, feature_names: list) -> pd.DataFrame | None:
    try:
        named_steps = dict(model.steps)
    except AttributeError:
        return None

    estimator = named_steps.get("model", model)
    if not hasattr(estimator, "coef_"):
        return None

    coefs = estimator.coef_
    if coefs.ndim > 1:
        coefs = coefs[0]

    df = pd.DataFrame({
        "feature": feature_names[: len(coefs)],
        "coefficient": coefs,
    })

    return df.sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)


def partial_dependence_data(
    model,
    X: pd.DataFrame,
    features: list,
    grid_resolution: int = 50,
) -> dict:
    X_float = X.copy()
    for col in X_float.select_dtypes(include=[np.integer]).columns:
        X_float[col] = X_float[col].astype(float)

    result = sklearn_partial_dependence(
        model, X_float, features=features, grid_resolution=grid_resolution
    )

    if len(features) == 1:
        return {
            "feature": features[0],
            "grid_values": result["grid_values"][0].tolist(),
            "average": result["average"][0].tolist(),
        }
    else:
        return {
            "feature": features,
            "grid_values": [gv.tolist() for gv in result["grid_values"]],
            "average": result["average"].tolist(),
        }


def diagnose_model(
    train_metric: float,
    test_metric: float,
    cv_mean: float,
    cv_std: float,
) -> dict:
    gap = train_metric - test_metric

    if cv_std > 0.1:
        verdict = "High variance"
        explanation = (
            "Cross-validation scores vary significantly across folds, "
            "indicating the model's performance is unstable."
        )
        recommendation = (
            "Try collecting more training data, reducing model complexity, "
            "or increasing regularization to improve consistency."
        )
    elif train_metric < 0.6 and test_metric < 0.6:
        verdict = "Underfitting"
        explanation = (
            "Both training and test scores are low, suggesting the model "
            "cannot capture the underlying pattern in the data."
        )
        recommendation = (
            "Try a more complex model, add more features, reduce "
            "regularization, or engineer new features."
        )
    elif gap > 0.15:
        verdict = "Possible overfitting"
        explanation = (
            f"Training score ({train_metric:.3f}) is much higher than "
            f"test score ({test_metric:.3f}), indicating the model may be "
            "memorizing training data rather than generalizing."
        )
        recommendation = (
            "Try increasing regularization, reducing model complexity, "
            "removing noisy features, or adding more training data."
        )
    else:
        verdict = "Good fit"
        explanation = (
            f"The model generalizes well. Training score ({train_metric:.3f}) "
            f"and test score ({test_metric:.3f}) are close, and cross-validation "
            f"mean ({cv_mean:.3f}) is stable (std={cv_std:.3f})."
        )
        recommendation = (
            "Consider fine-tuning hyperparameters or trying feature "
            "engineering for marginal improvements."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "recommendation": recommendation,
    }


def explain_prediction(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    row_idx: int,
    feature_names: list,
) -> pd.DataFrame:
    try:
        named_steps = dict(model.steps)
    except AttributeError:
        named_steps = {}

    estimator = named_steps.get("model", model)

    if hasattr(estimator, "coef_"):
        coefs = estimator.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        intercept = getattr(estimator, "intercept_", 0)
        if hasattr(intercept, "__len__"):
            intercept = intercept[0]
        values = X_test.iloc[row_idx].values[: len(coefs)]
        contributions = coefs * values
        contributions_with_intercept = np.append(intercept, contributions)
        labels = ["intercept"] + feature_names[: len(coefs)]

        df = pd.DataFrame({
            "feature": labels,
            "value": [intercept] + values.tolist(),
            "contribution": contributions_with_intercept,
        })
        return df.sort_values("contribution", key=abs, ascending=False).reset_index(drop=True)
    else:
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            importances = np.ones(len(feature_names)) / len(feature_names)

        values = X_test.iloc[row_idx].values[: len(importances)]

        df = pd.DataFrame({
            "feature": feature_names[: len(importances)],
            "value": values,
            "contribution": importances,
        })
        return df.sort_values("contribution", ascending=False).reset_index(drop=True)


def get_param_grid(model_name: str) -> dict:
    if model_name not in PARAM_GRIDS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(PARAM_GRIDS.keys())}"
        )
    return PARAM_GRIDS[model_name].copy()
