# Configuration and constants for AI Data Science Assistant

# Workflow Steps Definition
STEPS = {
    0: {"name": "Welcome", "help": "Start by uploading your dataset or loading a sample dataset."},
    1: {"name": "Upload Data", "help": "Upload a CSV file or load a sample dataset (Iris for classification, Diabetes for regression)."},
    2: {"name": "Clean Data", "help": "Apply data cleaning operations: remove duplicates, handle missing values, standardize columns, and scale features."},
    3: {"name": "Visualize", "help": "Create visualizations including correlation heatmaps and custom charts to explore your data."},
    4: {"name": "Model", "help": "Train and compare machine learning models for classification or regression tasks."},
}

STEP_NAMES = [step["name"] for step in STEPS.values()]
HELP_TEXTS = [step["help"] for step in STEPS.values()]
NUM_STEPS = 5

# Model Definitions
CLASSIFICATION_MODELS = {
    "Logistic Regression": "LogisticRegression",
    "Random Forest": "RandomForestClassifier",
    "Decision Tree": "DecisionTreeClassifier",
}

REGRESSION_MODELS = {
    "Linear Regression": "LinearRegression",
    "Random Forest": "RandomForestRegressor",
    "Decision Tree": "DecisionTreeRegressor",
}

# Data Processing Parameters
OUTLIER_IQR_MULTIPLIER = 1.5
TRAIN_TEST_SPLIT_SIZE = 0.2
CROSS_VAL_FOLDS = 5
RANDOM_STATE = 42

# Sample Datasets
SAMPLE_DATASETS = {
    "Iris (Classification)": "iris",
    "Diabetes (Regression)": "diabetes",
}

# Chart Types Configuration
CHART_TYPES = {
    "Histogram": {"min_numeric": 1, "max_numeric": 1},
    "Boxplot": {"min_numeric": 1, "max_numeric": None},
    "Scatter": {"min_numeric": 2, "max_numeric": 2},
    "Bar": {"min_numeric": 0, "max_numeric": 0},
    "Column": {"min_numeric": 0, "max_numeric": 0},
    "Pie": {"min_numeric": 0, "max_numeric": 0},
}

# UI Configuration
PAGE_TITLE = "AI Data Science Assistant"
CHART_HEIGHT = 400
DATA_PREVIEW_ROWS = 5
HEATMAP_SIZE = (12, 10)
CHART_SIZE = (10, 6)

# Theme Colors
PRIMARY_COLOR = "#6b8aed"
SECONDARY_COLOR = "#4e6bc2"
BACKGROUND_DARK = "#111318"
SIDEBAR_BACKGROUND = "#181b22"
TEXT_COLOR = "#dfe2e8"
HOVER_COLOR = "#252830"
ACCENT_COLOR = "#4ade80"

# Disabled AI Configuration (for future use)
# AI_MODELS = ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]
# AI_MAX_TOKENS = 800
# AI_TEMPERATURE = 0.7
# AI_TOP_P = 0.9
# AI_CHAT_HISTORY_LIMIT = 8
