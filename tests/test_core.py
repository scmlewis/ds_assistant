import numpy as np
import pandas as pd

import core


def test_validate_data_for_modeling_all_good():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series(["cat", "dog", "cat"], name="target")
    ok, msg = core.validate_data_for_modeling(X, y)
    assert ok is True
    assert "passed" in msg.lower()


def test_validate_data_for_modeling_inf_in_numeric_target():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, np.inf, 3.0], name="target")
    ok, msg = core.validate_data_for_modeling(X, y)
    assert ok is False
    assert "infinite" in msg.lower()


def test_calculate_correlation_significance_handles_missing_and_constant():
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "z": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    corr_df, pval_df = core.calculate_correlation_significance(df, ["x", "y", "z"])

    assert corr_df.shape == (3, 3)
    assert pval_df.shape == (3, 3)
    assert corr_df.loc["x", "x"] == 1.0
    assert pval_df.loc["x", "x"] == 0.0
    assert np.isnan(corr_df.loc["x", "z"]) or np.isnan(corr_df.loc["z", "x"])


def test_perform_hypothesis_test_insufficient_data():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    result = core.perform_hypothesis_test(df, "a", "b", test_type="pearson")
    assert result["Significant"] == "N/A"


def test_get_statistical_summary():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    summary = core.get_statistical_summary(df)
    assert summary is not None
    assert "IQR" in summary.columns
    assert "Range" in summary.columns
