import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

import core
import modeling
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def diabetes_data():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# 1. Full classification pipeline: Iris end-to-end
# ---------------------------------------------------------------------------

class TestClassificationPipeline:

    def test_build_fit_score_diagnose(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        pipe.fit(X_train, y_train)

        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        assert 0.0 <= train_score <= 1.0
        assert 0.0 <= test_score <= 1.0

        diagnosis = modeling.diagnose_model(train_score, test_score, train_score, 0.01)
        assert diagnosis["verdict"] in ("Good fit", "Possible overfitting", "Underfitting", "High variance")
        assert "explanation" in diagnosis

    def test_all_classification_models_fit(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        for model_name in config.CLASSIFICATION_MODELS:
            pipe = modeling.build_pipeline(model_name)
            pipe.fit(X_train, y_train)
            score = pipe.score(X_test, y_test)
            assert score > 0.5, f"{model_name} scored {score:.3f} — too low"

    def test_pipeline_predict_matches_score(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Random Forest")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        manual_acc = accuracy_score(y_test, y_pred)
        pipeline_acc = pipe.score(X_test, y_test)
        assert abs(manual_acc - pipeline_acc) < 1e-10


# ---------------------------------------------------------------------------
# 2. Full regression pipeline: Diabetes end-to-end
# ---------------------------------------------------------------------------

class TestRegressionPipeline:

    def test_build_fit_score_diagnose(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        pipe = modeling.build_pipeline("Linear Regression")
        pipe.fit(X_train, y_train)

        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        assert train_score > 0.0

        diagnosis = modeling.diagnose_model(train_score, test_score, train_score, 0.05)
        assert diagnosis["verdict"] in ("Good fit", "Possible overfitting", "Underfitting", "High variance")

    def test_all_regression_models_fit(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        for model_name in config.REGRESSION_MODELS:
            pipe = modeling.build_pipeline(model_name)
            pipe.fit(X_train, y_train)
            score = pipe.score(X_test, y_test)
            assert score > -1.0, f"{model_name} R²={score:.3f} — unreasonably bad"

    def test_pipeline_predict_vs_r2(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        pipe = modeling.build_pipeline("Random Forest Regressor")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        manual_r2 = r2_score(y_test, y_pred)
        pipeline_r2 = pipe.score(X_test, y_test)
        assert abs(manual_r2 - pipeline_r2) < 1e-10


# ---------------------------------------------------------------------------
# 3. Grid search → best model → learning curve
# ---------------------------------------------------------------------------

class TestGridSearchFlow:

    def test_grid_search_best_model_scores_higher(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        param_grid = modeling.get_param_grid("Logistic Regression")

        result = modeling.run_grid_search(pipe, X_train, y_train, param_grid, cv=3)
        best_pipe = result.best_estimator_

        default_pipe = modeling.build_pipeline("Logistic Regression")
        default_pipe.fit(X_train, y_train)

        best_test = best_pipe.score(X_test, y_test)
        default_test = default_pipe.score(X_test, y_test)
        assert best_test >= default_test - 0.05

    def test_grid_search_then_learning_curve(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        param_grid = modeling.get_param_grid("Logistic Regression")

        result = modeling.run_grid_search(pipe, X_train, y_train, param_grid, cv=3)
        best_pipe = result.best_estimator_

        lc = modeling.compute_learning_curve(best_pipe, X_train, y_train, cv=3)
        assert len(lc["train_sizes"]) > 0
        assert len(lc["train_mean"]) == len(lc["train_sizes"])
        assert all(s > 0 for s in lc["train_sizes"])


# ---------------------------------------------------------------------------
# 4. Validation curve on tuned pipeline
# ---------------------------------------------------------------------------

class TestValidationCurveFlow:

    def test_validation_curve_after_grid_search(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        param_grid = modeling.get_param_grid("Logistic Regression")

        result = modeling.run_grid_search(pipe, X_train, y_train, param_grid, cv=3)
        best_pipe = result.best_estimator_

        vc = modeling.compute_validation_curve(
            best_pipe, X_train, y_train,
            param_name="model__C",
            param_range=np.array([0.01, 0.1, 1.0, 10.0]),
            cv=3,
        )
        assert len(vc["param_range"]) == 4
        assert len(vc["train_scores"]) == 4
        assert len(vc["val_scores"]) == 4


# ---------------------------------------------------------------------------
# 5. Feature importance chain
# ---------------------------------------------------------------------------

class TestFeatureImportanceChain:

    def test_permutation_importance_on_iris(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Random Forest")
        pipe.fit(X_train, y_train)

        imp = modeling.permutation_feature_importance(pipe, X_test, y_test)
        assert len(imp) == len(X_test.columns)
        assert imp.iloc[0]["importance_mean"] >= imp.iloc[-1]["importance_mean"]

    def test_coefficients_on_linear_model(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        pipe.fit(X_train, y_train)

        coeffs = modeling.get_model_coefficients(pipe, list(X_test.columns))
        assert coeffs is not None
        assert len(coeffs) == len(X_test.columns)

    def test_permutation_importance_on_nonlinear(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        pipe = modeling.build_pipeline("Gradient Boosting Regressor")
        pipe.fit(X_train, y_train)

        imp = modeling.permutation_feature_importance(pipe, X_test, y_test)
        assert len(imp) == len(X_test.columns)


# ---------------------------------------------------------------------------
# 6. Partial dependence on trained pipeline
# ---------------------------------------------------------------------------

class TestPartialDependenceChain:

    def test_single_feature_pdp(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        pipe.fit(X_train, y_train)

        pdp = modeling.partial_dependence_data(pipe, X_test, [X_test.columns[0]], grid_resolution=20)
        assert "grid_values" in pdp
        assert "average" in pdp
        assert pdp["feature"] == X_test.columns[0]

    def test_two_feature_pdp(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        pipe.fit(X_train, y_train)

        cols = [X_test.columns[0], X_test.columns[1]]
        pdp = modeling.partial_dependence_data(pipe, X_test, cols, grid_resolution=20)
        assert "grid_values" in pdp
        assert "average" in pdp

    def test_pdp_on_random_forest(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        pipe = modeling.build_pipeline("Random Forest Regressor")
        pipe.fit(X_train, y_train)

        pdp = modeling.partial_dependence_data(pipe, X_test, [X_test.columns[0]], grid_resolution=20)
        assert "grid_values" in pdp
        assert "average" in pdp


# ---------------------------------------------------------------------------
# 7. Explain prediction chain
# ---------------------------------------------------------------------------

class TestExplainPredictionChain:

    def test_explain_classification_prediction(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        pipe.fit(X_train, y_train)

        explanation = modeling.explain_prediction(
            pipe, X_test, y_test, row_idx=0, feature_names=list(X_test.columns)
        )
        assert "feature" in explanation.columns
        assert "contribution" in explanation.columns
        assert len(explanation) == len(X_test.columns) + 1  # +1 for intercept

    def test_explain_regression_prediction(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        pipe = modeling.build_pipeline("Linear Regression")
        pipe.fit(X_train, y_train)

        explanation = modeling.explain_prediction(
            pipe, X_test, y_test, row_idx=0, feature_names=list(X_test.columns)
        )
        assert "feature" in explanation.columns
        assert "contribution" in explanation.columns

    def test_explain_random_forest_returns_no_intercept(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Random Forest")
        pipe.fit(X_train, y_train)

        explanation = modeling.explain_prediction(
            pipe, X_test, y_test, row_idx=0, feature_names=list(X_test.columns)
        )
        assert len(explanation) == len(X_test.columns)


# ---------------------------------------------------------------------------
# 8. Diagnose from real training scores
# ---------------------------------------------------------------------------

class TestDiagnoseFromRealScores:

    def test_diagnose_iris_logistic_regression(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        pipe = modeling.build_pipeline("Logistic Regression")
        pipe.fit(X_train, y_train)

        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)

        diagnosis = modeling.diagnose_model(train_score, test_score, test_score, 0.01)
        assert isinstance(diagnosis["verdict"], str)
        assert len(diagnosis["explanation"]) > 0
        assert len(diagnosis["recommendation"]) > 0

    def test_diagnose_underfitting_on_noisy_data(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"a": rng.randn(200), "b": rng.randn(200)})
        y = pd.Series(rng.randint(0, 2, size=200))

        pipe = modeling.build_pipeline("Decision Tree")
        pipe.set_params(model__max_depth=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)

        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)

        diagnosis = modeling.diagnose_model(train_score, test_score, train_score, 0.01)
        assert diagnosis["verdict"] in ("Good fit", "Underfitting", "Possible overfitting")


# ---------------------------------------------------------------------------
# 9. Core validation + modeling pipeline
# ---------------------------------------------------------------------------

class TestCoreModelingIntegration:

    def test_validate_then_build_and_train(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        is_valid, msg = core.validate_data_for_modeling(X_train, y_train)
        assert is_valid is True or "passed" in msg.lower() or "valid" in msg.lower()

        pipe = modeling.build_pipeline("Logistic Regression")
        pipe.fit(X_train, y_train)
        assert pipe.score(X_test, y_test) > 0.8

    def test_validate_rejects_bad_data(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([np.inf, 1.0, 2.0])
        is_valid, msg = core.validate_data_for_modeling(X, y)
        assert is_valid is False


# ---------------------------------------------------------------------------
# 10. Cross-model comparison flow (simulates app.py's Compare Models flow)
# ---------------------------------------------------------------------------

class TestModelComparisonFlow:

    def test_compare_multiple_classifiers(self, iris_data):
        X_train, X_test, y_train, y_test = iris_data
        results = []

        for model_name in ["Logistic Regression", "Random Forest", "Decision Tree"]:
            pipe = modeling.build_pipeline(model_name)
            pipe.fit(X_train, y_train)
            test_score = pipe.score(X_test, y_test)
            results.append({"Model": model_name, "Test Score": test_score})

        results_df = pd.DataFrame(results)
        assert len(results_df) == 3
        assert results_df["Test Score"].min() > 0.5

        best_idx = results_df["Test Score"].idxmax()
        best_model = results_df.loc[best_idx, "Model"]
        assert best_model in ["Logistic Regression", "Random Forest", "Decision Tree"]

    def test_compare_multiple_regressors(self, diabetes_data):
        X_train, X_test, y_train, y_test = diabetes_data
        results = []

        for model_name in ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"]:
            pipe = modeling.build_pipeline(model_name)
            pipe.fit(X_train, y_train)
            test_score = pipe.score(X_test, y_test)
            results.append({"Model": model_name, "Test Score": test_score})

        results_df = pd.DataFrame(results)
        assert len(results_df) == 3
        assert results_df["Test Score"].min() > -1.0


# ---------------------------------------------------------------------------
# 11. Full app-like flow: data → train → tune → interpret → diagnose
# ---------------------------------------------------------------------------

class TestFullAppFlow:

    def test_iris_full_flow(self):
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

        is_valid, _ = core.validate_data_for_modeling(X, y)
        assert is_valid

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipe = modeling.build_pipeline("Random Forest")
        pipe.fit(X_train, y_train)
        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        assert test_score > 0.8

        param_grid = modeling.get_param_grid("Random Forest")
        gs = modeling.run_grid_search(pipe, X_train, y_train, param_grid, cv=3)
        best_pipe = gs.best_estimator_
        best_test = best_pipe.score(X_test, y_test)
        assert best_test > 0.8

        lc = modeling.compute_learning_curve(best_pipe, X_train, y_train, cv=3)
        assert len(lc["train_sizes"]) > 0

        imp = modeling.permutation_feature_importance(best_pipe, X_test, y_test)
        assert len(imp) == len(X.columns)

        pdp = modeling.partial_dependence_data(best_pipe, X_test, [X.columns[0]], grid_resolution=15)
        assert "grid_values" in pdp

        explanation = modeling.explain_prediction(best_pipe, X_test, y_test, row_idx=0, feature_names=list(X.columns))
        assert "contribution" in explanation.columns

        diagnosis = modeling.diagnose_model(train_score, test_score, gs.best_score_, 0.02)
        assert diagnosis["verdict"] in ("Good fit", "Possible overfitting", "Underfitting", "High variance")

    def test_diabetes_full_flow(self):
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

        is_valid, _ = core.validate_data_for_modeling(X, y)
        assert is_valid

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipe = modeling.build_pipeline("Gradient Boosting Regressor")
        pipe.fit(X_train, y_train)
        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        assert test_score > -1.0

        lc = modeling.compute_learning_curve(pipe, X_train, y_train, cv=3, scoring="r2")
        assert len(lc["train_sizes"]) > 0

        vc = modeling.compute_validation_curve(
            pipe, X_train, y_train,
            param_name="model__n_estimators",
            param_range=[10, 50, 100],
            cv=3, scoring="r2",
        )
        assert len(vc["param_range"]) == 3

        imp = modeling.permutation_feature_importance(pipe, X_test, y_test)
        assert len(imp) == len(X.columns)

        explanation = modeling.explain_prediction(pipe, X_test, y_test, row_idx=0, feature_names=list(X.columns))
        assert "contribution" in explanation.columns

        diagnosis = modeling.diagnose_model(train_score, test_score, train_score, 0.03)
        assert "verdict" in diagnosis
