import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import modeling


def test_build_pipeline_logistic_regression_has_scaler():
    pipe = modeling.build_pipeline("Logistic Regression")
    assert isinstance(pipe, Pipeline)
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "model" in step_names
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["model"], LogisticRegression)


def test_build_pipeline_random_forest_no_scaler():
    pipe = modeling.build_pipeline("Random Forest")
    assert isinstance(pipe, Pipeline)
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" not in step_names
    assert "model" in step_names
    assert isinstance(pipe.named_steps["model"], RandomForestClassifier)


def test_build_pipeline_decision_tree_no_scaler():
    pipe = modeling.build_pipeline("Decision Tree")
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" not in step_names
    assert isinstance(pipe.named_steps["model"], DecisionTreeClassifier)


def test_build_pipeline_svm_has_scaler():
    pipe = modeling.build_pipeline("SVM")
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert isinstance(pipe.named_steps["model"], LinearSVC)


def test_build_pipeline_knn_has_scaler():
    pipe = modeling.build_pipeline("KNN")
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert isinstance(pipe.named_steps["model"], KNeighborsClassifier)


def test_build_pipeline_gradient_boosting_no_scaler():
    pipe = modeling.build_pipeline("Gradient Boosting")
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" not in step_names
    assert isinstance(pipe.named_steps["model"], GradientBoostingClassifier)


def test_build_pipeline_regression_models():
    for name, expected_type in [
        ("Linear Regression", LinearRegression),
        ("Random Forest Regressor", RandomForestRegressor),
        ("Decision Tree Regressor", DecisionTreeRegressor),
        ("SVR", LinearSVR),
        ("KNN Regressor", KNeighborsRegressor),
        ("Gradient Boosting Regressor", GradientBoostingRegressor),
    ]:
        pipe = modeling.build_pipeline(name)
        assert isinstance(pipe.named_steps["model"], expected_type), f"Failed for {name}"


def test_build_pipeline_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        modeling.build_pipeline("Nonexistent Model")


def test_get_param_grid_logistic_regression():
    grid = modeling.get_param_grid("Logistic Regression")
    assert "model__C" in grid
    assert "model__penalty" in grid
    assert isinstance(grid["model__C"], list)
    assert len(grid["model__C"]) > 0


def test_get_param_grid_random_forest():
    grid = modeling.get_param_grid("Random Forest")
    assert "model__n_estimators" in grid
    assert "model__max_depth" in grid


def test_get_param_grid_decision_tree():
    grid = modeling.get_param_grid("Decision Tree")
    assert "model__max_depth" in grid
    assert "model__min_samples_split" in grid


def test_get_param_grid_svm():
    grid = modeling.get_param_grid("SVM")
    assert "model__C" in grid
    assert "model__kernel" in grid


def test_get_param_grid_knn():
    grid = modeling.get_param_grid("KNN")
    assert "model__n_neighbors" in grid
    assert "model__weights" in grid


def test_get_param_grid_gradient_boosting():
    grid = modeling.get_param_grid("Gradient Boosting")
    assert "model__n_estimators" in grid
    assert "model__learning_rate" in grid


def test_get_param_grid_regression_models():
    for name in ["Linear Regression", "Random Forest Regressor",
                  "Decision Tree Regressor", "SVR", "KNN Regressor",
                  "Gradient Boosting Regressor"]:
        grid = modeling.get_param_grid(name)
        assert isinstance(grid, dict), f"Failed for {name}"


def test_get_param_grid_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        modeling.get_param_grid("Nonexistent Model")


def test_run_grid_search_returns_grid_search_cv():
    from sklearn.model_selection import GridSearchCV
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pipe = modeling.build_pipeline("Logistic Regression")
    param_grid = {"model__C": [0.1, 1.0, 10.0]}
    result = modeling.run_grid_search(pipe, X_train=X, y_train=y, param_grid=param_grid, cv=3)
    assert isinstance(result, GridSearchCV)
    assert hasattr(result, "best_params_")
    assert "model__C" in result.best_params_


def test_run_random_search_returns_randomized_search_cv():
    from sklearn.model_selection import RandomizedSearchCV
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pipe = modeling.build_pipeline("Logistic Regression")
    param_grid = {"model__C": [0.1, 1.0, 10.0]}
    result = modeling.run_grid_search(
        pipe, X_train=X, y_train=y, param_grid=param_grid, cv=3,
        search_type="random", n_iter=2
    )
    assert isinstance(result, RandomizedSearchCV)
    assert hasattr(result, "best_params_")


def test_compute_learning_curve_returns_expected_keys():
    X = pd.DataFrame({"a": range(100), "b": range(100)})
    y = pd.Series([0] * 50 + [1] * 50)
    pipe = modeling.build_pipeline("Logistic Regression")
    pipe.fit(X, y)
    result = modeling.compute_learning_curve(pipe, X, y, cv=3)
    assert "train_sizes" in result
    assert "train_scores" in result
    assert "val_scores" in result
    assert len(result["train_sizes"]) == len(result["train_scores"])
    assert len(result["train_sizes"]) == len(result["val_scores"])


def test_compute_learning_curve_custom_train_sizes():
    X = pd.DataFrame({"a": range(100), "b": range(100)})
    y = pd.Series([0] * 50 + [1] * 50)
    pipe = modeling.build_pipeline("Logistic Regression")
    pipe.fit(X, y)
    custom_sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    result = modeling.compute_learning_curve(
        pipe, X, y, cv=3, train_sizes=custom_sizes
    )
    assert len(result["train_sizes"]) == 5


def test_compute_validation_curve_returns_expected_keys():
    X = pd.DataFrame({"a": range(100), "b": range(100)})
    y = pd.Series([0] * 50 + [1] * 50)
    pipe = modeling.build_pipeline("Logistic Regression")
    pipe.fit(X, y)
    result = modeling.compute_validation_curve(
        pipe, X, y, param_name="model__C",
        param_range=np.array([0.01, 0.1, 1.0, 10.0]), cv=3
    )
    assert "param_range" in result
    assert "train_scores" in result
    assert "val_scores" in result
    assert len(result["param_range"]) == 4
