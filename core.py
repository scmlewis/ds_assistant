import numpy as np
import pandas as pd
from scipy import stats


def validate_data_for_modeling(X, y):
    """Validate data before model training."""
    if X.isna().any().any():
        return False, "❌ Features contain missing values. Please clean data first."

    if y.isna().any():
        return False, "❌ Target contains missing values. Please clean data first."

    numeric_X = X.select_dtypes(include=[np.number])
    if not numeric_X.empty and np.isinf(numeric_X).any().any():
        return False, "❌ Features contain infinite values. Please clean data first."

    if pd.api.types.is_numeric_dtype(y) and np.isinf(y).any():
        return False, "❌ Target contains infinite values. Please clean data first."

    if len(X) == 0 or len(y) == 0:
        return False, "❌ No data available for training."

    if len(X) != len(y):
        return False, "❌ Feature and target length mismatch."

    return True, "✅ Data validation passed"


def calculate_correlation_significance(df, numeric_cols):
    """
    Calculate Pearson correlation with p-values for statistical significance.
    Returns DataFrame with correlations and p-values.
    """
    n_vars = len(numeric_cols)
    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))

    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                paired = df[[col1, col2]].dropna()
                if len(paired) < 2 or paired[col1].nunique() < 2 or paired[col2].nunique() < 2:
                    corr, pval = np.nan, np.nan
                else:
                    try:
                        corr, pval = stats.pearsonr(paired[col1], paired[col2])
                    except Exception:
                        corr, pval = np.nan, np.nan
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval

    corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)
    pval_df = pd.DataFrame(pval_matrix, index=numeric_cols, columns=numeric_cols)

    return corr_df, pval_df


def perform_hypothesis_test(df, col1, col2, test_type='pearson'):
    """
    Perform hypothesis test on two variables.
    """
    if test_type in {'pearson', 'spearman'}:
        paired = df[[col1, col2]].dropna()
        if len(paired) < 2 or paired[col1].nunique() < 2 or paired[col2].nunique() < 2:
            result = {
                'Test': 'Pearson Correlation' if test_type == 'pearson' else 'Spearman Correlation',
                'Correlation': np.nan,
                'P-value': np.nan,
                'Significant': 'N/A',
                'Sample Size': len(paired)
            }
        else:
            try:
                if test_type == 'pearson':
                    corr, pval = stats.pearsonr(paired[col1], paired[col2])
                    test_label = 'Pearson Correlation'
                else:
                    corr, pval = stats.spearmanr(paired[col1], paired[col2])
                    test_label = 'Spearman Correlation'
                result = {
                    'Test': test_label,
                    'Correlation': corr,
                    'P-value': pval,
                    'Significant': 'Yes ✓' if pval < 0.05 else 'No ✗',
                    'Sample Size': len(paired)
                }
            except Exception:
                result = {
                    'Test': 'Pearson Correlation' if test_type == 'pearson' else 'Spearman Correlation',
                    'Correlation': np.nan,
                    'P-value': np.nan,
                    'Significant': 'N/A',
                    'Sample Size': len(paired)
                }
    elif test_type == 'ttest':
        valid_data_1 = df[col1].dropna()
        valid_data_2 = df[col2].dropna()
        if len(valid_data_1) < 2 or len(valid_data_2) < 2:
            result = {
                'Test': 'Independent T-Test',
                'T-Statistic': np.nan,
                'P-value': np.nan,
                'Significant': 'N/A',
                'Sample Size 1': len(valid_data_1),
                'Sample Size 2': len(valid_data_2)
            }
        else:
            stat, pval = stats.ttest_ind(valid_data_1, valid_data_2)
            result = {
                'Test': 'Independent T-Test',
                'T-Statistic': stat,
                'P-value': pval,
                'Significant': 'Yes ✓' if pval < 0.05 else 'No ✗',
                'Sample Size 1': len(valid_data_1),
                'Sample Size 2': len(valid_data_2)
            }
    else:
        result = {}

    return pd.Series(result)


def get_statistical_summary(df):
    """Generate statistical summary table for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) == 0:
        return None

    summary = numeric_df.describe().T
    summary['IQR'] = summary['75%'] - summary['25%']
    summary['Range'] = summary['max'] - summary['min']

    return summary[['count', 'mean', '50%', 'std', 'min', '25%', '75%', 'max', 'IQR', 'Range']]
