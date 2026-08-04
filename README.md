# AI Data Science Assistant

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.3+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.23+-010187?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end data science workflow tool built with Streamlit. Upload a dataset, profile it, clean it, explore it visually, and train machine learning models -- all through a point-and-click interface with no code required.

---

## Features

### Data Upload & Profiling
- CSV file upload or built-in sample datasets (Iris for classification, Diabetes for regression)
- Data quality report with row/column counts, duplicates, and missing data percentage
- Memory usage analysis, completeness metrics, and column-level statistics
- Schema viewer and summary statistics with a missing data heatmap

### Data Cleaning
- Standardize column names, remove duplicates, drop or fill missing values
- Outlier removal using the IQR method
- Label encoding for categorical columns and feature scaling (StandardScaler or MinMaxScaler)
- Live before/after preview with Apply, Revert, and Download controls

### Feature Engineering
- Polynomial feature generation
- Interaction terms between features
- Binning for continuous variables

### Visualization
- **Correlations**: Heatmap with Pearson p-values and hypothesis testing (Pearson, Spearman, T-test)
- **Charts**: Histogram, Boxplot, Scatter, Bar, Column, and Pie charts
- **Distributions**: Histogram with KDE, KDE-only, and Violin plots
- **Pair Plot**: Scatter/histogram matrix across all numeric features

### ML Lab

The ML Lab provides four sub-tabs for a complete modeling workflow:

**Configure & Train** -- Select target column and features, auto-detects classification vs regression, trains up to 6 models per mode with cross-validation scores, confusion matrices, ROC curves, residual plots, and sample predictions.

**Tune** -- Hyperparameter optimization via Grid Search or Randomized Search with configurable cross-validation folds, learning curves, and validation curves.

**Interpret** -- Permutation importance, model coefficient extraction, partial dependence plots (1D and 2D), and single-prediction explanation with a contribution waterfall.

**Diagnose** -- Bias-variance analysis with fit classification (Good fit / Overfitting / Underfitting / High variance), train-test gap, confusion matrix, ROC curve, classification report, and regression error metrics (MAE, MSE, RSE).

### Export
- Cleaned datasets as CSV
- Trained models as pickle files
- Full HTML reports with metrics, plots, and diagnostics

---

## Supported Models

| Type | Models |
|------|--------|
| **Classification** | Logistic Regression, Random Forest, Decision Tree, SVM, K-Nearest Neighbors, Gradient Boosting |
| **Regression** | Linear Regression, Random Forest, Decision Tree, SVR, K-Nearest Neighbors, Gradient Boosting |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

```bash
git clone <repository-url>
cd ds_assistant
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n ds_assistant python=3.12
conda activate ds_assistant
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Workflow

1. **Upload Data** -- Load a CSV or select a sample dataset
2. **Clean Data** -- Apply cleaning operations with a live preview
3. **Visualize** -- Create charts, explore correlations, and run hypothesis tests
4. **ML Lab** -- Configure, train, tune, interpret, and diagnose models

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Framework | Streamlit |
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Statistics | SciPy |
| Visualization | Matplotlib, Seaborn |
| UI Components | streamlit-option-menu |
| Font | Google Fonts (Outfit) |

---

## Project Structure

```
ds_assistant/
  app.py              # Main Streamlit application
  config.py           # Configuration and constants
  core.py             # Data validation, correlation, and hypothesis tests
  modeling.py         # ML pipelines, tuning, interpretation, and diagnosis
  requirements.txt    # Python dependencies
  .streamlit/         # Streamlit server and theme configuration
```

---

## Configuration

All configuration is managed in `config.py`:

- **Workflow Steps** -- Navigation menu items and help text
- **Model Definitions** -- Classification and regression model mappings
- **Data Processing** -- IQR multiplier, train/test split ratio, cross-validation folds, random state
- **Chart Types** -- Available visualization types with column requirements
- **UI Settings** -- Theme colors, chart dimensions, preview row count
- **Sample Datasets** -- Built-in datasets for quick testing

---

## License

MIT License -- see [LICENSE](LICENSE) for details.
