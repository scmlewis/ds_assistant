# AI Data Science Assistant

An interactive Streamlit application for data science workflows including data cleaning, visualization, and machine learning model training/comparison.

## Features

- **📤 Upload & Explore** - Load CSV files or use built-in datasets (Iris, Diabetes)
- **🧹 Smart Data Cleaning** - Remove duplicates, handle missing values, standardize columns, scale features
- **📊 Powerful Visualizations** - Create correlation matrices, histograms, scatter plots, bar charts, and more
- **🤖 Model Training** - Train and compare classification or regression models side-by-side
- **💾 Export Results** - Download cleaned data and trained models

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ds_assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda create -n ds_assistant python=3.12
conda activate ds_assistant
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Workflow

1. **Welcome** - Get started with the application
2. **Upload Data** - Load a CSV file or use sample datasets
3. **Clean Data** - Apply cleaning operations with live preview
4. **Visualize** - Create charts and explore correlations
5. **Model** - Train ML models and compare performance

## Supported Models

### Classification
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

### Regression
- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor

## Project Structure

```
ds_assistant/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and constants
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── .env                        # API keys (not committed)
├── README.md                   # This file
└── APP_SPECIFICATION.md        # Complete technical specification
```

## Configuration

All configuration is managed in `config.py`:

- **Workflow Steps** - Define navigation menu items
- **Model Definitions** - Classification and regression models
- **Data Processing** - IQR multiplier, train/test split, cross-validation folds
- **UI Settings** - Colors, fonts, preview rows, chart sizes
- **Sample Datasets** - Available built-in datasets

## Data Cleaning Options

- **Standardize column names** - Convert to lowercase with underscores
- **Remove duplicates** - Drop exact duplicate rows
- **Drop missing rows** - Remove rows with any NaN values
- **Fill missing values** - Impute NaN with specified value
- **Remove outliers** - IQR method (1.5 × IQR by default)
- **Encode categorical** - Label encode object columns
- **Scale features** - Normalize using StandardScaler or MinMaxScaler

## Visualization Types

- **Correlation Matrix** - Heatmap of feature correlations
- **Histogram** - Distribution of single numeric column
- **Scatter Plot** - Relationship between two numeric columns
- **Boxplot** - Distribution across multiple columns
- **Bar Chart** - Counts by category (vertical)
- **Column Chart** - Counts by category (horizontal)
- **Pie Chart** - Proportions by category

## Model Training

The app provides:
- **Train/Test Split** - 80/20 split with random state for reproducibility
- **Cross-Validation** - 5-fold cross-validation for generalization assessment
- **Performance Metrics**:
  - Classification: Accuracy + CV scores
  - Regression: R² score + CV scores
- **Model Diagnostics**:
  - Confusion matrix for classification
  - Residuals plot for regression
- **Model Export** - Download trained models as pickle files

## Styling

The app uses a dark theme with:
- **Primary Color**: #667eea (purple-blue)
- **Secondary Color**: #764ba2 (darker purple)
- **Accent Color**: #10b981 (green)
- **Background**: #181d29 (dark gray)

Responsive design adapts to different screen sizes.

## Technical Specifications

For complete technical details, see `APP_SPECIFICATION.md`.

## Future Enhancements

- AI-assisted model recommendations (Groq integration)
- Hyperparameter tuning UI
- Feature importance plots
- Statistical tests
- Advanced data profiling
- Time series support

## Requirements

- **Streamlit** 1.44.1
- **Pandas** 2.3.0+
- **Scikit-learn** 1.3.0+
- **Matplotlib** 3.8.0+
- **Seaborn** 0.13.0+
- **NumPy** 1.23.0+

## License

MIT License - see LICENSE file for details

## Support

For issues or questions, please open an issue in the repository.

---

**Created with Streamlit** | **Python 3.12.7** | **Version 1.0**
