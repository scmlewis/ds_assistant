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
