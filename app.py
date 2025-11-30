import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, roc_curve, auc, roc_auc_score
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
from scipy import stats
import io
import pickle
from streamlit_option_menu import option_menu
import config

# Page Configuration
st.set_page_config(page_title=config.PAGE_TITLE, layout="wide")

# Initialize Session State
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "df" not in st.session_state:
    st.session_state.df = None
if "pending_clean_options" not in st.session_state:
    st.session_state.pending_clean_options = {}
if "pending_df" not in st.session_state:
    st.session_state.pending_df = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Custom CSS Styling - Modern Dark Theme
def apply_custom_styling():
    custom_css = """
    <style>
    /* Modern Dark Theme Colors */
    :root {
        --dark-bg: #0F1419;
        --dark-bg-secondary: #1A1F2E;
        --dark-bg-tertiary: #252D3D;
        --accent-blue: #5B7FFF;
        --accent-blue-light: #7B9FFF;
        --text-primary: #E8EAED;
        --text-secondary: #9CA3AF;
        --border-color: #2D3748;
        --hover-bg: #1F2937;
        --success-color: #34D399;
        --error-color: #F87171;
        --warning-color: #FBBF24;
    }

    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--dark-bg);
        color: var(--text-primary);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--dark-bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--dark-bg-secondary) !important;
    }

    /* Main content area */
    [data-testid="stAppViewContainer"] > section {
        padding: 2rem 2rem;
        max-width: 100%;
    }

    /* App header */
    .app-header {
        background: linear-gradient(135deg, #1A2A4A 0%, #0F1419 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin: 0 0 2.5rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .app-title {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.5px;
    }

    /* Headings */
    h1 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 2rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.5px;
    }

    h2 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.4rem;
        margin: 2rem 0 1.5rem 0;
        letter-spacing: -0.3px;
        border-bottom: 2px solid var(--accent-blue);
        padding-bottom: 1rem;
    }

    h3 {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 1.1rem;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* Help/Info Box */
    .help-box {
        background-color: var(--dark-bg-tertiary);
        border-left: 4px solid var(--accent-blue);
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: var(--text-secondary);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Stat Card */
    .stat-card {
        background: linear-gradient(135deg, var(--dark-bg-tertiary) 0%, var(--dark-bg-secondary) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-blue);
        box-shadow: 0 8px 24px rgba(91, 127, 255, 0.2);
    }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-blue);
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem;
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        margin: 2rem 0;
        background-color: var(--dark-bg-tertiary);
        transition: all 0.3s ease;
    }

    .empty-state:hover {
        border-color: var(--accent-blue);
        background-color: var(--dark-bg-secondary);
    }

    .empty-state-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }

    .empty-state-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1rem 0;
    }

    .empty-state-message {
        font-size: 0.95rem;
        color: var(--text-secondary);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-blue-light) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(91, 127, 255, 0.3);
        text-transform: none;
        letter-spacing: 0.3px;
        font-size: 0.95rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(91, 127, 255, 0.4);
        background: linear-gradient(135deg, var(--accent-blue-light) 0%, #8FA3FF 100%);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Input Fields */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] > div > div,
    [data-testid="stTextInput"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background-color: var(--dark-bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        transition: all 0.3s ease;
    }

    [data-testid="stSelectbox"] > div > div:focus-within,
    [data-testid="stNumberInput"] > div > div:focus-within,
    [data-testid="stTextInput"] > div > div:focus-within,
    [data-testid="stMultiSelect"] > div > div:focus-within {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 3px rgba(91, 127, 255, 0.15);
        background-color: var(--dark-bg-secondary);
    }

    /* Checkbox and Radio */
    [data-testid="stCheckbox"] label {
        color: var(--text-primary);
    }

    [data-testid="stCheckbox"] {
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    [data-testid="stCheckbox"]:hover {
        background-color: var(--dark-bg-tertiary);
    }

    [data-testid="stRadio"] label {
        color: var(--text-primary);
    }

    /* Messages */
    .stSuccess {
        background-color: rgba(52, 211, 153, 0.1) !important;
        border-left: 4px solid var(--success-color) !important;
        color: #A7F3D0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .stError {
        background-color: rgba(248, 113, 113, 0.1) !important;
        border-left: 4px solid var(--error-color) !important;
        color: #FECACA !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .stWarning {
        background-color: rgba(251, 191, 36, 0.1) !important;
        border-left: 4px solid var(--warning-color) !important;
        color: #FCD34D !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .stInfo {
        background-color: rgba(91, 127, 255, 0.1) !important;
        border-left: 4px solid var(--accent-blue) !important;
        color: #BFDBFE !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    /* Expander */
    [data-testid="stExpander"] > div > button {
        background-color: var(--dark-bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        font-weight: 600;
        padding: 1rem;
        transition: all 0.2s ease;
    }

    [data-testid="stExpander"] > div > button:hover {
        background-color: var(--dark-bg-secondary);
        border-color: var(--accent-blue);
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
    }

    /* Links */
    a {
        color: var(--accent-blue);
        text-decoration: none;
        transition: all 0.2s ease;
    }

    a:hover {
        color: var(--accent-blue-light);
        text-decoration: underline;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div {
        background-color: var(--border-color);
    }

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-blue-light) 100%);
    }

    /* Metric */
    [data-testid="stMetric"] {
        background-color: var(--dark-bg-tertiary);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }

    /* Responsive Design */
    @media (max-width: 800px) {
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        
        .app-title {
            font-size: 1.8rem;
        }
        
        h1 {
            font-size: 1.5rem;
        }

        [data-testid="stAppViewContainer"] > section {
            padding: 1rem 1.5rem;
        }
    }

    /* Smooth animations */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    /* Increase sidebar width */
    [data-testid="stSidebar"] {
        min-width: 350px;
        width: 350px;
    }

    [data-testid="stSidebarNav"] {
        padding-right: 1rem;
    }

    /* Help box in sidebar - reduce padding */
    [data-testid="stSidebar"] .help-box {
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_styling()

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def validate_data_for_modeling(X, y):
    """Validate data before model training."""
    # Check for NaN in features
    if X.isna().any().any():
        return False, "❌ Features contain missing values. Please clean data first."
    
    # Check for NaN in target
    if y.isna().any():
        return False, "❌ Target contains missing values. Please clean data first."
    
    # Check for infinite values in features
    if np.isinf(X.select_dtypes(include=[np.number])).any().any():
        return False, "❌ Features contain infinite values. Please clean data first."
    
    # Check for infinite values in target
    if np.isinf(y).any():
        return False, "❌ Target contains infinite values. Please clean data first."
    
    # Check if data is empty
    if len(X) == 0 or len(y) == 0:
        return False, "❌ No data available for training."
    
    # Check if lengths match
    if len(X) != len(y):
        return False, "❌ Feature and target length mismatch."
    
    return True, "✅ Data validation passed"

def load_sample_dataset(dataset_name):
    """Load built-in datasets from sklearn."""
    if dataset_name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df
    elif dataset_name == "diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

def data_quality_report(df):
    """Display comprehensive data metrics."""
    if df is None or df.empty:
        return
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df):,}</div>
            <div class="stat-label">Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df.columns)}</div>
            <div class="stat-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        duplicates = df.duplicated().sum()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{duplicates}</div>
            <div class="stat-label">Duplicates</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{missing_pct:.1f}%</div>
            <div class="stat-label">Missing Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Missing data breakdown
    missing_data = df.isna().sum()
    if missing_data.sum() > 0:
        st.subheader("Missing Data Breakdown")
        missing_table = pd.DataFrame({
            "Column": missing_data.index,
            "Missing Count": missing_data.values,
            "Percentage": (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_table[missing_table["Missing Count"] > 0], use_container_width=True)

def empty_state(icon, title, message):
    """Display user-friendly empty state UI."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-message">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def generate_ai_system_prompt(df=None):
    """Placeholder for AI prompt generation."""
    pass

def get_feature_importance(model, feature_names):
    """Extract feature importance from trained model."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            return importance_df
    except Exception:
        pass
    return None

def plot_feature_importance(importance_df, title="Feature Importance"):
    """Create feature importance visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1A1F2E')
    ax.set_facecolor('#252D3D')
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='#2D3748')
    
    ax.set_xlabel('Importance Score', color='#E8EAED', fontweight='600')
    ax.set_ylabel('Features', color='#E8EAED', fontweight='600')
    ax.set_title(title, color='#E8EAED', fontweight='600', pad=15)
    ax.tick_params(colors='#E8EAED')
    
    for spine in ax.spines.values():
        spine.set_color('#2D3748')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{width:.3f}', ha='left', va='center', color='#E8EAED', fontsize=9, fontweight='500')
    
    plt.tight_layout()
    return fig

def generate_data_profile(df):
    """Generate comprehensive data profiling report."""
    profile = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Memory Usage (MB)': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'Duplicate Rows': df.duplicated().sum(),
        'Complete Rows': len(df.dropna()),
        'Completeness %': ((len(df) - df.isna().any(axis=1).sum()) / len(df) * 100),
    }
    
    column_profile = []
    for col in df.columns:
        col_info = {
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null': df[col].notna().sum(),
            'Null': df[col].isna().sum(),
            'Unique': df[col].nunique(),
            'Duplicates': len(df[col]) - df[col].nunique()
        }
        column_profile.append(col_info)
    
    return profile, pd.DataFrame(column_profile)

def plot_roc_curve(y_test, y_pred_proba, model_name="Model"):
    """Create ROC curve visualization for binary classification."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1A1F2E')
    ax.set_facecolor('#252D3D')
    
    ax.plot(fpr, tpr, color='#5B7FFF', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#FF6B6B', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', color='#E8EAED', fontweight='600')
    ax.set_ylabel('True Positive Rate', color='#E8EAED', fontweight='600')
    ax.set_title(f'ROC Curve - {model_name}', color='#E8EAED', fontweight='600', pad=15)
    ax.tick_params(colors='#E8EAED')
    ax.legend(loc='lower right', facecolor='#252D3D', edgecolor='#2D3748', labelcolor='#E8EAED')
    
    for spine in ax.spines.values():
        spine.set_color('#2D3748')
    
    ax.grid(True, alpha=0.1, color='#2D3748')
    plt.tight_layout()
    return fig, roc_auc

def get_missing_value_heatmap(df):
    """Create missing value heatmap."""
    missing_matrix = df.isna().astype(int)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1A1F2E')
    ax.set_facecolor('#252D3D')
    
    heatmap = sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar=True, ax=ax, 
                         cbar_kws={'label': 'Missing (1) vs Present (0)'},
                         linewidths=0.2, linecolor='#1A1F2E')
    
    ax.set_xlabel('Row Index', color='#E8EAED', fontweight='600')
    ax.set_ylabel('Columns', color='#E8EAED', fontweight='600')
    ax.set_title('Missing Data Pattern Heatmap', color='#E8EAED', fontweight='600', pad=15)
    ax.tick_params(colors='#E8EAED')
    
    cbar = heatmap.collections[0].colorbar
    if cbar:
        cbar.set_label('Missing (1) vs Present (0)', color='#E8EAED')
        cbar.ax.tick_params(colors='#E8EAED')
    
    plt.tight_layout()
    return fig

def get_statistical_summary(df):
    """Generate statistical summary table for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        return None
    
    summary = numeric_df.describe().T
    summary['IQR'] = summary['75%'] - summary['25%']
    summary['Range'] = summary['max'] - summary['min']
    
    return summary[['count', 'mean', '50%', 'std', 'min', '25%', '75%', 'max', 'IQR', 'Range']]

def generate_html_report(df, profile, stat_summary, trained_models=None, mode=None):
    """Generate comprehensive HTML report with all analysis."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Science Analysis Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #0F1419 0%, #1A1F2E 100%);
                color: #E8EAED;
                line-height: 1.6;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: #1A1F2E;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                overflow: hidden;
            }}
            header {{
                background: linear-gradient(135deg, #5B7FFF 0%, #7B9FFF 100%);
                padding: 40px 20px;
                text-align: center;
            }}
            header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                color: white;
            }}
            header p {{
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1em;
            }}
            .content {{
                padding: 40px;
            }}
            section {{
                margin-bottom: 40px;
                border-bottom: 2px solid #2D3748;
                padding-bottom: 30px;
            }}
            section:last-child {{
                border-bottom: none;
            }}
            h2 {{
                color: #5B7FFF;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-left: 4px solid #5B7FFF;
                padding-left: 15px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: #252D3D;
                border-radius: 8px;
                overflow: hidden;
            }}
            th {{
                background: #5B7FFF;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #2D3748;
            }}
            tr:hover {{
                background: #2D3748;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #252D3D;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #5B7FFF;
            }}
            .metric-label {{
                color: #9CA3AF;
                font-size: 0.9em;
                margin-bottom: 5px;
            }}
            .metric-value {{
                color: #5B7FFF;
                font-size: 1.8em;
                font-weight: 700;
            }}
            footer {{
                background: #0F1419;
                padding: 20px;
                text-align: center;
                color: #9CA3AF;
                font-size: 0.9em;
            }}
            .info-box {{
                background: #252D3D;
                border-left: 4px solid #5B7FFF;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>✨ Data Science Analysis Report</h1>
                <p>Comprehensive analysis and insights from your dataset</p>
            </header>
            
            <div class="content">
                <!-- Dataset Overview -->
                <section>
                    <h2>📊 Dataset Overview</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-label">Total Rows</div>
                            <div class="metric-value">{profile['Total Rows']:,}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Total Columns</div>
                            <div class="metric-value">{profile['Total Columns']}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Memory Usage</div>
                            <div class="metric-value">{profile['Memory Usage (MB)']:.2f} MB</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Completeness</div>
                            <div class="metric-value">{profile['Completeness %']:.1f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Duplicate Rows</div>
                            <div class="metric-value">{profile['Duplicate Rows']}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Complete Rows</div>
                            <div class="metric-value">{profile['Complete Rows']:,}</div>
                        </div>
                    </div>
                </section>
                
                <!-- Statistical Summary -->
                {f'''
                <section>
                    <h2>📈 Statistical Summary</h2>
                    {stat_summary.to_html() if stat_summary is not None else '<p>No numeric columns found.</p>'}
                </section>
                ''' if stat_summary is not None else ''}
                
                <!-- Model Results -->
                {f'''
                <section>
                    <h2>🤖 Model Training Results</h2>
                    <div class="info-box">
                        <strong>Mode:</strong> {mode}
                    </div>
                </section>
                ''' if trained_models else ''}
                
            </div>
            
            <footer>
                <p>Generated by AI Data Science Assistant • Report created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return html_content

def plot_pair_plot(df, numeric_cols=None, sample_size=500):
    """Create pair plot for numeric columns."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    # Sample data if too large for performance
    if len(df) > sample_size:
        df_sample = df[numeric_cols].sample(n=sample_size, random_state=42)
    else:
        df_sample = df[numeric_cols]
    
    fig = plt.figure(figsize=(max(10, len(numeric_cols) * 2), max(10, len(numeric_cols) * 2)))
    fig.patch.set_facecolor('#1A1F2E')
    
    # Create pair plot manually with dark theme
    n_cols = len(numeric_cols)
    for i, col_x in enumerate(numeric_cols):
        for j, col_y in enumerate(numeric_cols):
            ax = plt.subplot(n_cols, n_cols, i * n_cols + j + 1)
            ax.set_facecolor('#252D3D')
            
            if i == j:
                # Diagonal: histogram
                ax.hist(df_sample[col_x], bins=20, color='#5B7FFF', alpha=0.7, edgecolor='#2D3748')
            else:
                # Off-diagonal: scatter plot
                ax.scatter(df_sample[col_x], df_sample[col_y], alpha=0.5, color='#5B7FFF', s=20, edgecolors='#2D3748')
            
            ax.tick_params(colors='#E8EAED', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#2D3748')
            
            if i == n_cols - 1:
                ax.set_xlabel(col_x, color='#E8EAED', fontsize=9, fontweight='600')
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(col_y, color='#E8EAED', fontsize=9, fontweight='600')
            else:
                ax.set_yticklabels([])
    
    plt.tight_layout()
    return fig

def plot_box_plots(df, numeric_cols=None):
    """Create box plots for numeric columns with outlier detection."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.patch.set_facecolor('#1A1F2E')
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.set_facecolor('#252D3D')
        
        # Create box plot
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#5B7FFF', alpha=0.7),
                        whiskerprops=dict(color='#E8EAED'),
                        capprops=dict(color='#E8EAED'),
                        medianprops=dict(color='#FF6B6B', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='#FF6B6B', markersize=6, alpha=0.5))
        
        ax.set_title(col, color='#E8EAED', fontweight='600', fontsize=11)
        ax.set_ylabel('Value', color='#E8EAED', fontweight='600')
        ax.tick_params(colors='#E8EAED')
        for spine in ax.spines.values():
            spine.set_color('#2D3748')
        ax.grid(True, alpha=0.1, color='#2D3748', axis='y')
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_categorical_distributions(df, categorical_cols=None, max_categories=20):
    """Create bar charts for categorical columns."""
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) == 0:
        return None
    
    n_cols = min(2, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.patch.set_facecolor('#1A1F2E')
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        ax.set_facecolor('#252D3D')
        
        # Count values
        value_counts = df[col].value_counts().head(max_categories)
        
        # Create bar chart
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='#5B7FFF', 
                      edgecolor='#2D3748', alpha=0.8)
        
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels([str(v)[:15] for v in value_counts.index], rotation=45, ha='right', 
                           color='#E8EAED', fontsize=9)
        ax.set_ylabel('Count', color='#E8EAED', fontweight='600')
        ax.set_title(f'{col} Distribution', color='#E8EAED', fontweight='600', fontsize=11)
        ax.tick_params(colors='#E8EAED')
        for spine in ax.spines.values():
            spine.set_color('#2D3748')
        ax.grid(True, alpha=0.1, color='#2D3748', axis='y')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', color='#E8EAED', fontsize=8)
    
    # Hide extra subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# ============================================================================
# ADVANCED FEATURES - HYPERPARAMETER TUNING
# ============================================================================

def tune_hyperparameters(X_train, y_train, X_test, y_test, model_name, mode):
    """
    Perform hyperparameter tuning using GridSearchCV.
    Returns best model, best params, and comparison results.
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [200, 500, 1000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    }
    
    # Select model
    if mode == 'Classification':
        model_map = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
    else:
        model_map = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
    
    base_model = model_map.get(model_name)
    if base_model is None:
        return None, None, None
    
    param_grid = param_grids.get(model_name, {})
    if not param_grid:
        return None, None, None
    
    # Scoring metric
    scoring = 'accuracy' if mode == 'Classification' else 'r2'
    
    # GridSearchCV
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=5, 
        scoring=scoring, 
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate
    if mode == 'Classification':
        baseline_score = accuracy_score(y_test, base_model.fit(X_train, y_train).predict(X_test))
        best_score = accuracy_score(y_test, best_model.predict(X_test))
        metric_name = 'Accuracy'
    else:
        baseline_score = r2_score(y_test, base_model.fit(X_train, y_train).predict(X_test))
        best_score = r2_score(y_test, best_model.predict(X_test))
        metric_name = 'R² Score'
    
    comparison = {
        'Baseline': baseline_score,
        'Optimized': best_score,
        'Improvement': best_score - baseline_score,
        'Improvement %': ((best_score - baseline_score) / abs(baseline_score) * 100) if baseline_score != 0 else 0,
        'Metric': metric_name
    }
    
    return best_model, best_params, comparison

# ============================================================================
# ADVANCED FEATURES - FEATURE ENGINEERING
# ============================================================================

def engineer_features(df, numeric_cols, feature_type='polynomial', degree=2, interaction_cols=None):
    """
    Create engineered features from numeric columns.
    """
    df_engineered = df.copy()
    
    if feature_type == 'polynomial':
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = df[numeric_cols]
        X_poly = poly.fit_transform(X)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(numeric_cols)
        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        # Add new features (exclude original)
        new_features = [col for col in df_poly.columns if col not in numeric_cols]
        for col in new_features:
            df_engineered[col] = df_poly[col]
        
        return df_engineered, new_features
    
    elif feature_type == 'interaction' and interaction_cols:
        # Create interaction features
        new_features = []
        for i, col1 in enumerate(interaction_cols):
            for col2 in interaction_cols[i+1:]:
                new_col = f'{col1}_x_{col2}'
                df_engineered[new_col] = df[col1] * df[col2]
                new_features.append(new_col)
        
        return df_engineered, new_features
    
    elif feature_type == 'binning':
        # Create binned features
        new_features = []
        for col in numeric_cols:
            new_col = f'{col}_binned'
            df_engineered[new_col] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
            new_features.append(new_col)
        
        return df_engineered, new_features
    
    return df_engineered, []

# ============================================================================
# ADVANCED FEATURES - SHAP EXPLAINABILITY
# ============================================================================

def plot_shap_summary(model, X_train, X_test, model_type='tree'):
    """
    Generate SHAP summary plot for model explainability.
    """
    if not HAS_SHAP:
        st.error("SHAP not installed. Install with: pip install shap")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    try:
        # Create explainer based on model type
        if model_type == 'tree' and hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        elif model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback to KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict, X_train.sample(min(50, len(X_train))))
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')
        
        # Set dark background
        fig = plt.gcf()
        fig.set_facecolor('#0F1419')
        
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.gcf().set_facecolor('#0F1419')
        ax = plt.gca()
        ax.set_facecolor('#1A1F2E')
        
        return fig
    except Exception as e:
        st.error(f"SHAP visualization error: {str(e)}")
        return None

def plot_shap_force(model, X_train, X_test, instance_idx=0, model_type='tree'):
    """
    Generate SHAP force plot for individual prediction explanation.
    """
    if not HAS_SHAP:
        return None
    
    try:
        # Create explainer
        if model_type == 'tree' and hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        elif model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_train.sample(min(50, len(X_train))))
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values_instance = shap_values[0][instance_idx]
        else:
            shap_values_instance = shap_values[instance_idx]
        
        # Return SHAP values for streamlit display
        return {
            'shap_values': shap_values_instance,
            'features': X_test.iloc[instance_idx],
            'base_value': explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]
        }
    except Exception as e:
        st.error(f"SHAP force plot error: {str(e)}")
        return None

# ============================================================================
# ADVANCED FEATURES - STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

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
                corr, pval = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval
    
    corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)
    pval_df = pd.DataFrame(pval_matrix, index=numeric_cols, columns=numeric_cols)
    
    return corr_df, pval_df

def plot_correlation_with_significance(df, numeric_cols):
    """
    Plot correlation matrix with significance annotations.
    """
    corr_df, pval_df = calculate_correlation_significance(df, numeric_cols)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.set_facecolor('#0F1419')
    ax.set_facecolor('#1A1F2E')
    
    # Create heatmap
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, cbar=True, vmin=-1, vmax=1)
    
    ax.set_title('Correlation Matrix with Pearson r', color='#E8EAED', fontweight='600', fontsize=12, pad=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color='#E8EAED')
    plt.setp(ax.get_yticklabels(), rotation=0, color='#E8EAED')
    
    # Add significance stars for p-values < 0.05
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            if i != j and pval_df.iloc[i, j] < 0.05:
                ax.text(j+0.5, i+0.7, '*', ha='center', va='center', 
                       color='#FFD700', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig, corr_df, pval_df

def perform_hypothesis_test(df, col1, col2, test_type='pearson'):
    """
    Perform hypothesis test on two variables.
    """
    valid_data_1 = df[col1].dropna()
    valid_data_2 = df[col2].dropna()
    
    if test_type == 'pearson':
        corr, pval = stats.pearsonr(valid_data_1, valid_data_2)
        result = {
            'Test': 'Pearson Correlation',
            'Correlation': corr,
            'P-value': pval,
            'Significant': 'Yes ✓' if pval < 0.05 else 'No ✗',
            'Sample Size': min(len(valid_data_1), len(valid_data_2))
        }
    elif test_type == 'spearman':
        corr, pval = stats.spearmanr(valid_data_1, valid_data_2)
        result = {
            'Test': 'Spearman Correlation',
            'Correlation': corr,
            'P-value': pval,
            'Significant': 'Yes ✓' if pval < 0.05 else 'No ✗',
            'Sample Size': min(len(valid_data_1), len(valid_data_2))
        }
    elif test_type == 'ttest':
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

def navigation_buttons():
    """Legacy navigation (pass-through)."""
    pass

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def landing_page():
    """Landing page with welcome message."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">✨ AI Data Science Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 0.5rem 0 1.5rem 0;">
        <p style="color: #9CA3AF; font-size: 1.1rem; margin: 0;">
            Transform your data into actionable insights with powerful ML workflows
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Features Section
    st.markdown("<h2 style='margin-top: 3rem; margin-bottom: 1.5rem; padding-bottom: 1rem;'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    features_left = [
        {
            "icon": "📥",
            "title": "Upload & Explore",
            "desc": "Load CSV files or use built-in datasets with instant analysis"
        },
        {
            "icon": "🧹",
            "title": "Smart Cleaning",
            "desc": "Remove duplicates, handle missing values, scale features with live preview"
        },
        {
            "icon": "📊",
            "title": "Visualize",
            "desc": "Create correlation matrices, distributions, and custom charts"
        }
    ]
    
    features_right = [
        {
            "icon": "🤖",
            "title": "ML Models",
            "desc": "Train and compare classification/regression models side-by-side"
        },
        {
            "icon": "🎯",
            "title": "Feature Analysis",
            "desc": "Understand feature importance and data profiling insights"
        },
        {
            "icon": "💾",
            "title": "Export Results",
            "desc": "Download cleaned data and trained models for production use"
        }
    ]
    
    with col1:
        for feature in features_left:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #252D3D 0%, #1F2833 100%);
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                        border: 1px solid #2D3748; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                        transition: all 0.3s ease; cursor: pointer;"
                 onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 8px 24px rgba(91, 127, 255, 0.2)';"
                 onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.3)';">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{feature['icon']}</div>
                <h3 style="color: #E8EAED; font-weight: 600; margin: 0.5rem 0; font-size: 1.1rem;">
                    {feature['title']}
                </h3>
                <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem; line-height: 1.5;">
                    {feature['desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        for feature in features_right:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #252D3D 0%, #1F2833 100%);
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                        border: 1px solid #2D3748; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                        transition: all 0.3s ease; cursor: pointer;"
                 onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 8px 24px rgba(91, 127, 255, 0.2)';"
                 onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.3)';">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{feature['icon']}</div>
                <h3 style="color: #E8EAED; font-weight: 600; margin: 0.5rem 0; font-size: 1.1rem;">
                    {feature['title']}
                </h3>
                <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem; line-height: 1.5;">
                    {feature['desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick Start Section
    st.markdown("<h2 style='margin-top: 3rem; margin-bottom: 1.5rem; padding-bottom: 1rem;'>🚀 Quick Start Guide</h2>", unsafe_allow_html=True)
    
    st.write("""
    **1. Upload Data** — Click the 📤 Upload Data menu to load your CSV or select a sample dataset
    
    **2. Explore & Profile** — View automatic data quality metrics, statistics, and column profiling
    
    **3. Clean & Prepare** — Apply cleaning operations with live preview before committing changes
    
    **4. Analyze & Visualize** — Create distributions, correlations, and custom charts to understand patterns
    
    **5. Build Models** — Train ML models and see feature importance, diagnostics, and performance metrics
    
    **6. Export & Deploy** — Download cleaned data and trained models for production use
    """)
    
    # Stats/Highlights Section
    
    # Tips Section (Integrated with Advanced Features)
    st.markdown("<h2 style='margin-top: 3rem; margin-bottom: 1.5rem; padding-bottom: 1rem;'>💡 Pro Tips</h2>", unsafe_allow_html=True)
    
    tips = [
        "🎯 Start with Sample Data — Load Iris or Diabetes datasets to explore features without your own data",
        "📊 Use Live Preview — All cleaning operations show before/after comparison before you commit",
        "🔗 Data Persists — Navigate freely between pages - your data and models are always preserved",
        "🤖 Compare Models — Train multiple models together to find the best performer for your task",
        "📥 Export Everything — Download cleaned data, trained models, and analysis reports",
        "⚡ Try Feature Engineering — Clean Data page: Polynomial features, interactions, and binning to improve models",
        "🔧 Optimize Models — Model Training page: Use hyperparameter tuning to boost accuracy by 10-20%",
        "🔬 Explain Predictions — Model Training page: SHAP explainability shows why models make predictions",
        "📈 Statistical Testing — Visualize Data page: Check p-values and significance of relationships"
    ]
    
    for tip in tips:
        st.markdown(f"""
        <div style="background: #252D3D; padding: 1.25rem; border-radius: 8px; border-left: 4px solid #5B7FFF; margin-bottom: 1rem;">
            <p style="color: #E8EAED; margin: 0; font-size: 0.95rem; line-height: 1.6;">{tip}</p>
        </div>
        """, unsafe_allow_html=True)

def upload_and_schema():
    """Upload data and schema inspection page."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📤 Upload Data</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[1]}
    </div>
    """, unsafe_allow_html=True)
    
    # Sample Datasets Section
    st.subheader("📚 Sample Datasets")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌸 Load Iris Dataset", use_container_width=True, key="iris_btn"):
            df = load_sample_dataset("iris")
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.pending_clean_options = {}
            st.success("✅ Iris dataset loaded successfully!")
            st.rerun()
    
    with col2:
        if st.button("🏥 Load Diabetes Dataset", use_container_width=True, key="diabetes_btn"):
            df = load_sample_dataset("diabetes")
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.pending_clean_options = {}
            st.success("✅ Diabetes dataset loaded successfully!")
            st.rerun()
    
    # CSV File Upload
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.pending_clean_options = {}
            st.success("✅ CSV file uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Show data if loaded
    if st.session_state.df is not None:
        st.subheader("📈 Data Quality Report")
        data_quality_report(st.session_state.df)
        
        # Data Profiling Dashboard
        st.subheader("📊 Data Profiling Report")
        profile, col_profile = generate_data_profile(st.session_state.df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💾 Memory Usage", f"{profile['Memory Usage (MB)']:.2f} MB")
        with col2:
            st.metric("✅ Completeness", f"{profile['Completeness %']:.1f}%")
        with col3:
            st.metric("👥 Complete Rows", f"{profile['Complete Rows']:,}")
        
        st.write("**Column-wise Profile:**")
        st.dataframe(col_profile, use_container_width=True)
        
        st.subheader("👁️ Data Preview")
        st.dataframe(st.session_state.df.head(config.DATA_PREVIEW_ROWS), use_container_width=True)
        
        st.subheader("📊 Summary Statistics")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        st.subheader("📋 Schema Information")
        schema_info = pd.DataFrame({
            "Column": st.session_state.df.columns,
            "Type": st.session_state.df.dtypes.astype(str),
            "Missing": st.session_state.df.isna().sum(),
            "Unique": st.session_state.df.nunique()
        })
        st.dataframe(schema_info, use_container_width=True)
        
        # Missing Value Heatmap
        st.subheader("🔍 Missing Data Pattern")
        if st.session_state.df.isna().sum().sum() > 0:
            missing_fig = get_missing_value_heatmap(st.session_state.df)
            st.pyplot(missing_fig, use_container_width=True)
        else:
            st.info("✅ No missing values detected in the dataset!")
        
        # Statistical Summary
        st.subheader("📊 Statistical Summary")
        stat_summary = get_statistical_summary(st.session_state.df)
        if stat_summary is not None:
            st.dataframe(stat_summary, use_container_width=True)
        else:
            st.info("ℹ️ No numeric columns found for statistical summary.")
    else:
        empty_state("📦", "No Data Yet", "Upload a CSV file or load a sample dataset to get started.")

def clean_data():
    """Data cleaning page."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🧹 Clean Data</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[2]}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        empty_state("⚠️", "No Data Available", "Please upload or load a dataset first.")
        return
    
    # Status banner
    rows = len(st.session_state.df)
    cols = len(st.session_state.df.columns)
    size_kb = st.session_state.df.memory_usage(deep=True).sum() / 1024
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #252D3D 0%, #1A1F2E 100%); 
                padding: 1.5rem; border-radius: 10px; border: 1px solid #2D3748; 
                margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);">
        <div style="color: #5B7FFF; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">📊 CURRENT DATA STATUS</div>
        <div style="color: #E8EAED; font-size: 1.1rem; font-weight: 500;">
            Rows: <span style="color: #5B7FFF; font-weight: 700;">{rows:,}</span> | 
            Columns: <span style="color: #5B7FFF; font-weight: 700;">{cols}</span> | 
            Size: <span style="color: #5B7FFF; font-weight: 700;">{size_kb:.2f} KB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Original Data Preview
    with st.expander("📊 Original Data Preview", expanded=False):
        st.dataframe(st.session_state.original_df.head(config.DATA_PREVIEW_ROWS), use_container_width=True)
    
    st.subheader("🔧 Cleaning Options")
    
    # Create two columns for cleaning options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Operations**")
        
        st.session_state.pending_clean_options["standardize_columns"] = st.checkbox(
            "✓ Standardize column names",
            value=st.session_state.pending_clean_options.get("standardize_columns", False),
            help="Convert to lowercase with underscores"
        )
        
        st.session_state.pending_clean_options["remove_duplicates"] = st.checkbox(
            "✓ Remove duplicates",
            value=st.session_state.pending_clean_options.get("remove_duplicates", False)
        )
        
        st.session_state.pending_clean_options["drop_missing"] = st.checkbox(
            "✓ Drop rows with missing values",
            value=st.session_state.pending_clean_options.get("drop_missing", False)
        )
        
        fill_missing = st.checkbox(
            "✓ Fill missing values",
            value=st.session_state.pending_clean_options.get("fill_missing", False)
        )
        st.session_state.pending_clean_options["fill_missing"] = fill_missing
        
        if fill_missing:
            fill_columns = st.multiselect(
                "Select columns to fill",
                st.session_state.df.columns,
                default=st.session_state.pending_clean_options.get("fill_columns", [])
            )
            st.session_state.pending_clean_options["fill_columns"] = fill_columns
            
            fill_value = st.number_input(
                "Fill value",
                value=float(st.session_state.pending_clean_options.get("fill_value", 0))
            )
            st.session_state.pending_clean_options["fill_value"] = fill_value
    
    with col2:
        st.write("**Advanced Operations**")
        
        st.session_state.pending_clean_options["remove_outliers"] = st.checkbox(
            "✓ Remove outliers (IQR method)",
            value=st.session_state.pending_clean_options.get("remove_outliers", False)
        )
        
        st.session_state.pending_clean_options["encode_categorical"] = st.checkbox(
            "✓ Encode categorical columns",
            value=st.session_state.pending_clean_options.get("encode_categorical", False)
        )
        
        scale_features = st.checkbox(
            "✓ Scale features",
            value=st.session_state.pending_clean_options.get("scale_features", False)
        )
        st.session_state.pending_clean_options["scale_features"] = scale_features
        
        if scale_features:
            scaler_type = st.radio(
                "Choose scaler",
                ["StandardScaler", "MinMaxScaler"],
                index=0 if st.session_state.pending_clean_options.get("scaler_type", "StandardScaler") == "StandardScaler" else 1
            )
            st.session_state.pending_clean_options["scaler_type"] = scaler_type
    
    # Apply cleaning and show live preview
    st.subheader("👁️ Live Preview")
    
    try:
        preview_df = st.session_state.original_df.copy()
        
        if st.session_state.pending_clean_options.get("standardize_columns", False):
            preview_df.columns = preview_df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
        
        if st.session_state.pending_clean_options.get("remove_duplicates", False):
            preview_df = preview_df.drop_duplicates()
        
        if st.session_state.pending_clean_options.get("drop_missing", False):
            preview_df = preview_df.dropna()
        
        if st.session_state.pending_clean_options.get("fill_missing", False):
            fill_columns = st.session_state.pending_clean_options.get("fill_columns", [])
            fill_value = st.session_state.pending_clean_options.get("fill_value", 0)
            if fill_columns:
                preview_df[fill_columns] = preview_df[fill_columns].fillna(fill_value)
        
        if st.session_state.pending_clean_options.get("remove_outliers", False):
            numeric_cols = preview_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = preview_df[col].quantile(0.25)
                Q3 = preview_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - config.OUTLIER_IQR_MULTIPLIER * IQR
                upper_bound = Q3 + config.OUTLIER_IQR_MULTIPLIER * IQR
                preview_df = preview_df[(preview_df[col] >= lower_bound) & (preview_df[col] <= upper_bound)]
        
        if st.session_state.pending_clean_options.get("encode_categorical", False):
            for col in preview_df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                preview_df[col] = le.fit_transform(preview_df[col])
        
        if st.session_state.pending_clean_options.get("scale_features", False):
            numeric_cols = preview_df.select_dtypes(include=[np.number]).columns
            scaler_type = st.session_state.pending_clean_options.get("scaler_type", "StandardScaler")
            scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
            preview_df[numeric_cols] = scaler.fit_transform(preview_df[numeric_cols])
        
        st.session_state.pending_df = preview_df.copy()
        
        original_shape = st.session_state.original_df.shape
        preview_shape = preview_df.shape
        st.success(f"✅ Original: {original_shape[0]}×{original_shape[1]} → Cleaned: {preview_shape[0]}×{preview_shape[1]}")
        
        st.dataframe(preview_df.head(config.DATA_PREVIEW_ROWS), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error during cleaning: {str(e)}")
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✔️ Apply", use_container_width=True, key="apply_clean"):
            st.session_state.df = st.session_state.pending_df.copy()
            st.balloons()
            st.success("✅ Cleaning applied successfully! Your data has been updated.")
            st.rerun()
    
    with col2:
        if st.button("↩️ Revert", use_container_width=True, key="revert_clean"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.pending_clean_options = {}
            st.info("↩️ Data reverted to original state.")
            st.rerun()
    
    with col3:
        if st.button("💾 Download", use_container_width=True, key="download_clean"):
            csv = st.session_state.pending_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Suggested Workflow Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2A3A4A 0%, #1A2A3A 100%); padding: 1.5rem; 
                border-radius: 10px; border-left: 4px solid #5B7FFF; margin-bottom: 2rem;">
        <div style="color: #5B7FFF; font-weight: 600; margin-bottom: 0.5rem;">🎯 Suggested Next Steps:</div>
        <p style="color: #E8EAED; margin: 0; font-size: 0.95rem; line-height: 1.6;">
            1. ✅ <strong>Clean Data</strong> (current step) → 2. Engineer Features Here → 3. Go to Visualize Data to check significance → 4. Train Models with new features
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Feature Engineering
    st.subheader("⚡ Feature Engineering")
    
    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        feat_col1, feat_col2 = st.columns([1, 1])
        
        with feat_col1:
            feature_type = st.selectbox(
                "Feature Engineering Type",
                ["Polynomial Features", "Interaction Terms", "Binning"],
                key="feat_type"
            )
            
            if feature_type == "Polynomial Features":
                degree = st.slider("Polynomial Degree", 2, 5, 2, key="poly_degree")
                selected_numeric = st.multiselect(
                    "Select columns for polynomial features",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))],
                    key="poly_cols"
                )
                
                if st.button("✨ Generate Polynomial Features", use_container_width=True, key="gen_poly"):
                    df_engineered, new_features = engineer_features(
                        st.session_state.df, selected_numeric, "polynomial", degree
                    )
                    st.session_state.df = df_engineered
                    st.success(f"✅ Created {len(new_features)} polynomial features!")
                    st.write(f"**New Features:** {', '.join(new_features[:5])}{'...' if len(new_features) > 5 else ''}")
            
            elif feature_type == "Interaction Terms":
                interaction_cols = st.multiselect(
                    "Select columns for interactions",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))],
                    key="interact_cols"
                )
                
                if st.button("✨ Generate Interactions", use_container_width=True, key="gen_interact"):
                    if len(interaction_cols) >= 2:
                        df_engineered, new_features = engineer_features(
                            st.session_state.df, None, "interaction", interaction_cols=interaction_cols
                        )
                        st.session_state.df = df_engineered
                        st.success(f"✅ Created {len(new_features)} interaction features!")
                        st.write(f"**New Features:** {', '.join(new_features)}")
                    else:
                        st.warning("⚠️ Please select at least 2 columns for interactions.")
            
            else:  # Binning
                binning_cols = st.multiselect(
                    "Select columns for binning",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))],
                    key="bin_cols"
                )
                
                if st.button("✨ Generate Binned Features", use_container_width=True, key="gen_bin"):
                    df_engineered, new_features = engineer_features(
                        st.session_state.df, binning_cols, "binning"
                    )
                    st.session_state.df = df_engineered
                    st.success(f"✅ Created {len(new_features)} binned features!")
        
        with feat_col2:
            st.info("💡 **Feature Engineering Tips:**\n\n"
                   "• **Polynomial Features**: Creates new columns (e.g., x², y², x×y). Original features are kept.\n"
                   "• **Interactions**: Combines features (e.g., height × weight). Shows joint effects on target.\n"
                   "• **Binning**: Divides values into 5 equal groups (quintiles). Creates ordinal categories.\n\n"
                   "**When to use:**\n"
                   "- Polynomial: When relationships are non-linear\n"
                   "- Interactions: When features influence each other\n"
                   "- Binning: For tree models or when you want to discretize continuous values")
    else:
        st.info("ℹ️ Feature engineering requires numeric columns.")

def visualize_data():
    """Data visualization page."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📊 Visualize Data</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[3]}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        empty_state("⚠️", "No Data Available", "Please upload or load a dataset first.")
        return
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Correlations", "📈 Charts", "📉 Distributions", "🔀 Pair Plot"])
    
    # TAB 1: CORRELATIONS & STATISTICAL TESTS
    with tab1:
        st.subheader("Correlation Matrix with Statistical Tests")
        
        numeric_df = st.session_state.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) >= 2:
            # Plot correlation with significance
            corr_fig, corr_df, pval_df = plot_correlation_with_significance(st.session_state.df, numeric_df.columns.tolist())
            st.pyplot(corr_fig, use_container_width=True)
            
            st.markdown("**Legend:** Gold stars (*) = statistically significant correlations (p < 0.05)")
            st.info("💡 Expand sections below to view detailed statistics and run hypothesis tests!")
            
            # Option to view detailed correlation statistics
            with st.expander("📊 Detailed Correlation Statistics", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Correlation Coefficients (r):**")
                    st.dataframe(corr_df, use_container_width=True)
                
                with col2:
                    st.write("**P-Values:**")
                    st.dataframe(pval_df, use_container_width=True)
            
            # Hypothesis testing with guidance
            with st.expander("🔬 Hypothesis Testing", expanded=False):
                st.markdown("""
                **Choose the right test:**
                - **Pearson**: Measures linear relationship (best for normally distributed data)
                - **Spearman**: Rank-based, robust to outliers (better for skewed data)
                - **T-test**: Compares if two variables have significantly different means
                """)
                
                test_col1, test_col2, test_col3 = st.columns(3)
                
                with test_col1:
                    var1 = st.selectbox("Variable 1", numeric_df.columns, key="test_var1")
                
                with test_col2:
                    var2 = st.selectbox("Variable 2", numeric_df.columns, key="test_var2")
                
                with test_col3:
                    test_type = st.selectbox("Test Type", ["pearson", "spearman", "ttest"], key="test_type")
                
                if st.button("🔍 Run Test", use_container_width=True, key="run_test"):
                    if var1 != var2:
                        with st.spinner("Running statistical test..."):
                            result = perform_hypothesis_test(st.session_state.df, var1, var2, test_type)
                        st.dataframe(result.to_frame().T, use_container_width=True)
                        
                        # Interpretation
                        if result.get('P-value', 1) < 0.05:
                            st.success("✓ **Result:** Statistically significant relationship found (p < 0.05)\n\nThis means the relationship is unlikely due to random chance!")
                        else:
                            st.info("ℹ️ **Result:** No statistically significant relationship (p ≥ 0.05)\n\nCould be due to random variation in the data.")
                    else:
                        st.warning("⚠️ Please select different variables.")
        else:
            st.info("ℹ️ Correlation matrix requires at least 2 numeric columns.")
    
    # TAB 2: CUSTOM CHARTS
    with tab2:
        st.subheader("Custom Chart Creation")
        
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox("Chart Type", list(config.CHART_TYPES.keys()), key="chart_type")
        with col2:
            selected_columns = st.multiselect("Select Columns", st.session_state.df.columns, key="chart_cols")
        
        if selected_columns and chart_type:
            try:
                fig, ax = plt.subplots(figsize=config.CHART_SIZE)
                fig.patch.set_facecolor('#1A1F2E')
                ax.set_facecolor('#252D3D')
                
                if chart_type == "Histogram":
                    st.session_state.df[selected_columns[0]].hist(bins=30, ax=ax, color='#5B7FFF', edgecolor='#2D3748', alpha=0.8)
                    ax.set_title(f'Histogram: {selected_columns[0]}', fontweight='600', color='#E8EAED', pad=15)
                    ax.set_xlabel(selected_columns[0], color='#E8EAED')
                    ax.set_ylabel('Frequency', color='#E8EAED')
                
                elif chart_type == "Boxplot":
                    st.session_state.df[selected_columns].boxplot(ax=ax, patch_artist=True)
                    for patch in ax.artists:
                        patch.set_facecolor('#5B7FFF')
                        patch.set_edgecolor('#7B9FFF')
                    ax.set_title('Boxplot', fontweight='600', color='#E8EAED', pad=15)
                    ax.set_ylabel('Value', color='#E8EAED')
                
                elif chart_type == "Scatter":
                    ax.scatter(st.session_state.df[selected_columns[0]], st.session_state.df[selected_columns[1]], 
                              alpha=0.6, color='#5B7FFF', s=50, edgecolors='#2D3748', linewidth=0.5)
                    ax.set_xlabel(selected_columns[0], color='#E8EAED')
                    ax.set_ylabel(selected_columns[1], color='#E8EAED')
                    ax.set_title(f'Scatter: {selected_columns[0]} vs {selected_columns[1]}', fontweight='600', color='#E8EAED', pad=15)
                
                elif chart_type == "Bar":
                    st.session_state.df[selected_columns[0]].value_counts().plot(kind='bar', ax=ax, color='#5B7FFF', edgecolor='#2D3748')
                    ax.set_title(f'Bar Chart: {selected_columns[0]}', fontweight='600', color='#E8EAED', pad=15)
                    ax.set_ylabel('Count', color='#E8EAED')
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='#E8EAED')
                
                elif chart_type == "Column":
                    st.session_state.df[selected_columns[0]].value_counts().plot(kind='barh', ax=ax, color='#5B7FFF', edgecolor='#2D3748')
                    ax.set_title(f'Column Chart: {selected_columns[0]}', fontweight='600', color='#E8EAED', pad=15)
                    ax.set_xlabel('Count', color='#E8EAED')
                
                elif chart_type == "Pie":
                    colors = ['#5B7FFF', '#7B9FFF', '#4A6FFF', '#3D5BFF', '#2A47FF']
                    ax.pie(st.session_state.df[selected_columns[0]].value_counts(), 
                          labels=st.session_state.df[selected_columns[0]].value_counts().index,
                          autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': '#E8EAED'})
                    ax.set_title(f'Pie Chart: {selected_columns[0]}', fontweight='600', color='#E8EAED', pad=15)
                
                ax.tick_params(colors='#E8EAED')
                for spine in ax.spines.values():
                    spine.set_color('#2D3748')
                    spine.set_linewidth(0.5)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    # TAB 3: DISTRIBUTION ANALYSIS
    with tab3:
        st.subheader("Distribution Analysis")
        
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            dist_col = st.selectbox("Select column for distribution analysis", numeric_cols, key="dist_col")
            dist_type = st.radio("Distribution plot type", ["Histogram with KDE", "KDE Plot", "Violin Plot"], 
                                horizontal=True, key="dist_type")
            
            if dist_col:
                fig, ax = plt.subplots(figsize=config.CHART_SIZE)
                fig.patch.set_facecolor('#1A1F2E')
                ax.set_facecolor('#252D3D')
                
                if dist_type == "Histogram with KDE":
                    st.session_state.df[dist_col].hist(bins=30, ax=ax, color='#5B7FFF', 
                                                        alpha=0.6, edgecolor='#2D3748', density=True)
                    st.session_state.df[dist_col].plot(kind='kde', ax=ax, color='#FF6B6B', linewidth=2.5)
                    ax.set_title(f'Distribution: {dist_col} (Histogram + KDE)', fontweight='600', color='#E8EAED', pad=15)
                
                elif dist_type == "KDE Plot":
                    st.session_state.df[dist_col].plot(kind='kde', ax=ax, color='#5B7FFF', linewidth=3)
                    ax.fill_between(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata(), 
                                   alpha=0.3, color='#5B7FFF')
                    ax.set_title(f'KDE Plot: {dist_col}', fontweight='600', color='#E8EAED', pad=15)
                
                elif dist_type == "Violin Plot":
                    parts = ax.violinplot([st.session_state.df[dist_col].dropna()], 
                                         positions=[0], showmeans=True, showmedians=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor('#5B7FFF')
                        pc.set_alpha(0.7)
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                        if partname in parts:
                            vp = parts[partname]
                            vp.set_edgecolor('#E8EAED')
                            vp.set_linewidth(2)
                    ax.set_xticks([0])
                    ax.set_xticklabels([dist_col])
                    ax.set_title(f'Violin Plot: {dist_col}', fontweight='600', color='#E8EAED', pad=15)
                
                ax.set_ylabel('Density' if 'KDE' in dist_type or 'Histogram' in dist_type else 'Frequency', 
                             color='#E8EAED', fontweight='600')
                ax.set_xlabel(dist_col, color='#E8EAED', fontweight='600')
                ax.tick_params(colors='#E8EAED')
                
                for spine in ax.spines.values():
                    spine.set_color('#2D3748')
                    spine.set_linewidth(0.5)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("ℹ️ Distribution analysis requires at least one numeric column.")
    
    # TAB 4: PAIR PLOT
    with tab4:
        st.subheader("Pair Plot Analysis")
        
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            if st.button("📊 Generate Pair Plot", help="Shows pairwise relationships between numeric features", 
                        use_container_width=True, key="pair_plot_btn"):
                with st.spinner("Generating pair plot..."):
                    pair_fig = plot_pair_plot(st.session_state.df, numeric_cols)
                    if pair_fig:
                        st.pyplot(pair_fig, use_container_width=True)
                        st.caption("Pair plot showing relationships between all numeric features (diagonal: histograms, off-diagonal: scatter plots)")
        else:
            st.info("ℹ️ Pair plots require at least 2 numeric columns.")

def page_model_training():
    """Model training and comparison page."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🤖 Model Training</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[4]}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        empty_state("⚠️", "No Data Available", "Please upload or load a dataset first.")
        return
    
    # Model Configuration
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox("Target Column", st.session_state.df.columns, key="target_col")
    
    # Auto-detect mode based on target column
    target_data = st.session_state.df[target_column]
    is_numeric = pd.api.types.is_numeric_dtype(target_data)
    is_categorical = pd.api.types.is_categorical_dtype(target_data) or target_data.dtype == 'object'
    unique_values = target_data.nunique()
    
    # Recommend mode based on data
    if is_categorical or (is_numeric and unique_values <= 10):
        recommended_mode = "Classification"
    else:
        recommended_mode = "Regression"
    
    with col2:
        mode = st.selectbox("Mode", ["Classification", "Regression"], 
                           index=0 if recommended_mode == "Classification" else 1, 
                           key="mode",
                           help=f"Auto-detected: {recommended_mode} (based on target column)")
    
    # Validate mode selection
    if mode == "Classification" and is_numeric and unique_values > 10:
        st.warning(f"⚠️ Warning: Classification selected but target has {unique_values} unique continuous values. Consider using Regression instead.")
    elif mode == "Regression" and is_categorical:
        st.warning("⚠️ Warning: Regression selected but target is categorical. Consider using Classification instead.")
    
    # Feature Selection
    st.subheader("🎯 Feature Selection")
    available_features = [col for col in st.session_state.df.columns if col != target_column]
    selected_features = st.multiselect("Select Features", available_features, default=available_features[:min(3, len(available_features))], key="features")
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature.")
        return
    
    # Model Selection
    st.subheader("📦 Model Selection")
    models = config.CLASSIFICATION_MODELS if mode == "Classification" else config.REGRESSION_MODELS
    selected_models = st.multiselect("Select Models", list(models.keys()), default=[list(models.keys())[0]], key="models")
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model.")
        return
    
    # Train Button
    if st.button("🚀 Train Models", use_container_width=True, key="train_btn"):
        # Prepare data
        X = st.session_state.df[selected_features]
        y = st.session_state.df[target_column]
        
        # Validate data
        is_valid, message = validate_data_for_modeling(X, y)
        if not is_valid:
            st.error(message)
            return
        
        # Validate mode-target compatibility
        unique_y = y.nunique()
        if mode == "Classification" and unique_y > 10:
            st.error(f"❌ Error: Classification mode requires discrete target values, but found {unique_y} unique continuous values. Please select Regression mode instead.")
            return
        
        if mode == "Regression" and pd.api.types.is_categorical_dtype(y):
            st.error("❌ Error: Regression mode requires numeric target values, but found categorical data. Please select Classification mode instead.")
            return
        
        # Train models
        st.subheader("📈 Training Results")
        results = []
        model_instances = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Training models..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TRAIN_TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE
            )
            
            for idx, model_name in enumerate(selected_models):
                status_text.text(f"Training {model_name}...")
                progress_bar.progress((idx + 1) / len(selected_models))
                
                # Create model instance based on mode and type
                if mode == "Classification":
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
                    else:
                        model = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
                else:
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
                    else:
                        model = DecisionTreeRegressor(random_state=config.RANDOM_STATE)
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Get probability predictions for classification (for ROC curve)
                y_pred_proba = None
                if mode == "Classification" and hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                if mode == "Classification":
                    test_metric = accuracy_score(y_test, y_pred)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=config.CROSS_VAL_FOLDS, scoring='accuracy')
                else:
                    test_metric = r2_score(y_test, y_pred)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=config.CROSS_VAL_FOLDS, scoring='r2')
                
                results.append({
                    "Model": model_name,
                    "Test Metric": test_metric,
                    "CV Mean": cv_scores.mean(),
                    "CV Std": cv_scores.std()
                })
                
                model_instances[model_name] = {
                    "model": model,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "y_pred_proba": y_pred_proba,
                    "cv_scores": cv_scores,
                    "mode": mode
                }
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results
        st.session_state.trained_models = model_instances
        
        # Display results
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Highlight best model
        best_idx = results_df["Test Metric"].idxmax()
        best_model = results_df.loc[best_idx, "Model"]
        best_score = results_df.loc[best_idx, "Test Metric"]
        
        st.success(f"🏆 Best Model: **{best_model}** (Score: {best_score:.4f})")
        
        # Model Diagnostics
        st.subheader("🔍 Model Diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if mode == "Classification":
                st.write("**Confusion Matrix (Best Model)**")
                best_model_data = model_instances[best_model]
                cm = confusion_matrix(best_model_data["y_test"], best_model_data["y_pred"])
                
                # Create normalized confusion matrix for percentages
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#1A1F2E')
                ax.set_facecolor('#252D3D')
                
                # Create heatmap
                heatmap = sns.heatmap(cm, annot=False, cmap='Blues', ax=ax, cbar_kws={'label': 'Count'},
                           linewidths=0.5, linecolor='#1A1F2E')
                
                # Add custom annotations with counts and percentages
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        count = cm[i, j]
                        pct = cm_normalized[i, j] * 100
                        text = ax.text(j + 0.5, i + 0.5, f'{int(count)}\n({pct:.1f}%)',
                                      ha="center", va="center", color='#E8EAED' if count < cm.max()/2 else '#0F1419',
                                      fontweight='600', fontsize=11)
                
                ax.set_xlabel('Predicted', color='#E8EAED', fontweight='600', fontsize=12)
                ax.set_ylabel('Actual', color='#E8EAED', fontweight='600', fontsize=12)
                ax.set_title('Confusion Matrix', color='#E8EAED', fontweight='600', fontsize=13, pad=15)
                ax.tick_params(colors='#E8EAED')
                cbar = heatmap.collections[0].colorbar
                if cbar:
                    cbar.set_label('Count', color='#E8EAED')
                    cbar.ax.tick_params(colors='#E8EAED')
                st.pyplot(fig, use_container_width=True)
        
        with col2:
            if mode == "Regression":
                st.write("**Residuals Plot (Best Model)**")
                best_model_data = model_instances[best_model]
                residuals = best_model_data["y_test"] - best_model_data["y_pred"]
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#1A1F2E')
                ax.set_facecolor('#252D3D')
                ax.scatter(best_model_data["y_pred"], residuals, alpha=0.6, color='#5B7FFF', s=50, edgecolors='#2D3748')
                ax.axhline(y=0, color='#FF6B6B', linestyle='--', linewidth=2)
                ax.set_xlabel('Predicted Values', color='#E8EAED', fontweight='600')
                ax.set_ylabel('Residuals', color='#E8EAED', fontweight='600')
                ax.set_title('Residuals Plot', color='#E8EAED', fontweight='600', pad=15)
                ax.tick_params(colors='#E8EAED')
                for spine in ax.spines.values():
                    spine.set_color('#2D3748')
                st.pyplot(fig, use_container_width=True)
            elif mode == "Classification" and model_instances[best_model]["y_pred_proba"] is not None:
                st.write("**ROC Curve (Best Model)**")
                best_model_data = model_instances[best_model]
                roc_fig, roc_auc = plot_roc_curve(best_model_data["y_test"], best_model_data["y_pred_proba"], best_model)
                st.pyplot(roc_fig, use_container_width=True)
        
        # Sample Predictions
        st.subheader("📋 Sample Predictions")
        best_model_data = model_instances[best_model]
        sample_predictions = pd.DataFrame({
            "Actual": best_model_data["y_test"].values[:10],
            "Predicted": best_model_data["y_pred"][:10]
        })
        st.dataframe(sample_predictions, use_container_width=True)
        
        # Engineered Features Notice
        engineered_cols = [col for col in st.session_state.df.columns 
                          if any(x in col for x in ['_poly', '_x_', '_binned'])]
        if engineered_cols:
            st.success(f"✅ Using {len(engineered_cols)} engineered features from Clean Data page")
            with st.expander("📋 View engineered features"):
                st.write(engineered_cols)
        
        # Feature Importance (if available)
        st.subheader("🎯 Feature Importance Analysis")
        best_model_obj = model_instances[best_model]["model"]
        importance_df = get_feature_importance(best_model_obj, selected_features)
        
        if importance_df is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                importance_fig = plot_feature_importance(importance_df, f"Feature Importance ({best_model})")
                st.pyplot(importance_fig, use_container_width=True)
            
            with col2:
                st.write("**Importance Scores:**")
                st.dataframe(importance_df, use_container_width=True)
        else:
            st.info("ℹ️ Feature importance not available for this model type (available for tree-based models only).")
        
        # Improvement Suggestions
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2A3A4A 0%, #1A2A3A 100%); padding: 1.5rem; 
                    border-radius: 10px; border-left: 4px solid #FFD700; margin: 2rem 0;">
            <div style="color: #FFD700; font-weight: 600; margin-bottom: 0.5rem;">💡 Want Better Performance?</div>
            <p style="color: #E8EAED; margin: 0; font-size: 0.95rem; line-height: 1.6;">
                👇 <strong>Scroll down</strong> to optimize your model with hyperparameter tuning and SHAP explainability!
            </p>
        </div>
        """)
        
        # Advanced Features: Hyperparameter Tuning
        st.subheader("⚡ Hyperparameter Optimization")
        
        tune_col1, tune_col2 = st.columns([1, 1])
        
        with tune_col1:
            tune_model = st.selectbox("Select Model to Tune", selected_models, key="tune_model_select")
            if st.button("🔧 Start Hyperparameter Tuning", use_container_width=True, key="tune_btn"):
                with st.spinner("⏳ Tuning hyperparameters with GridSearchCV (5-fold CV)...\n\nThis may take 20-60 seconds depending on dataset size."):
                    best_tuned_model, best_params, comparison = tune_hyperparameters(
                        X_train, y_train, X_test, y_test, tune_model, mode
                    )
                    
                    if best_tuned_model is not None:
                        st.success("✅ Tuning Complete!")
                        
                        # Display comparison with visual indicators
                        st.markdown("**Performance Comparison:**")
                        comparison_df = pd.DataFrame([comparison])
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Show improvement clearly
                        improvement = comparison.get('Improvement', 0)
                        improvement_pct = comparison.get('Improvement %', 0)
                        if improvement > 0:
                            st.success(f"✅ **Model improved by {improvement:.4f} ({improvement_pct:.2f}%)**")
                        elif improvement == 0:
                            st.info("ℹ️ Model already had optimal parameters")
                        
                        # Display best parameters
                        st.markdown("**Best Parameters Found:**")
                        params_display = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
                        st.dataframe(params_display, use_container_width=True)
                        
                        # Store tuned model
                        st.session_state.tuned_models = {tune_model: best_tuned_model}
                        
                        # Option to use tuned model with clear explanation
                        st.markdown("**What to do next:**")
                        st.markdown("""
                        - **View tuned model in feature importance** (refresh page)
                        - **Compare with original** using the main results table
                        - **Export the tuned model** using the Export Model section below
                        """)
                        
                        if st.button("💾 Save Tuned Model", use_container_width=True, key="save_tuned"):
                            model_instances[f"{tune_model} (Tuned)"] = {
                                "model": best_tuned_model,
                                "X_test": X_test,
                                "y_test": y_test,
                                "y_pred": best_tuned_model.predict(X_test),
                                "y_pred_proba": best_tuned_model.predict_proba(X_test)[:, 1] if mode == "Classification" and hasattr(best_tuned_model, 'predict_proba') else None,
                                "cv_scores": cross_val_score(best_tuned_model, X_train, y_train, cv=config.CROSS_VAL_FOLDS, scoring='accuracy' if mode == "Classification" else 'r2'),
                                "mode": mode
                            }
                            st.success("✓ Tuned model saved! You can now export it.")
                    else:
                        st.error("⚠️ Could not tune this model type. Try a different model.")
        
        with tune_col2:
            st.info("ℹ️ Hyperparameter tuning uses GridSearchCV with 5-fold cross-validation to find optimal parameters that maximize model performance on your data.")
        
        # Advanced Features: SHAP Explainability
        st.subheader("🔬 Model Explainability (SHAP)")
        
        if not HAS_SHAP:
            st.warning("""
            ⚠️ **SHAP is not installed** - but don't worry, the app works fine without it!
            
            **To enable SHAP model explanations:**
            ```
            pip install shap
            ```
            Then restart Streamlit with: `streamlit run app.py`
            
            **What does SHAP do?**
            - Shows which features most influence each prediction
            - Creates visual explanations (why did the model predict this?)
            - Helps debug and understand model behavior
            
            **Optional:** You can continue using all other features without SHAP.
            """)
        else:
            st.markdown("""
            **What is SHAP?**
            - SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain predictions
            - Shows how much each feature contributes to the final prediction (positive or negative)
            - Different from feature importance: shows per-prediction explanations, not just overall importance
            
            **Summary Plot** 👈 Shows feature importance across all predictions
            - Horizontal axis: Impact on model prediction
            - Each dot: One data sample
            - Red dots going right: Feature increases prediction
            - Blue dots going left: Feature decreases prediction
            
            **Individual Explanation** 👈 Shows why one specific prediction was made
            - Select a sample and see its feature values
            - Shows SHAP value for each feature (contribution to prediction)
            """)
            
            shap_col1, shap_col2 = st.columns([1, 1])
            
            with shap_col1:
                st.write("**Feature Importance via SHAP:**")
                if best_model in model_instances:
                    if st.button("📊 Generate SHAP Summary", use_container_width=True, key="shap_summary_btn"):
                        try:
                            with st.spinner("🔄 Generating SHAP summary plot (this may take 10-30 seconds)..."):
                                best_model_obj = model_instances[best_model]["model"]
                                model_type = 'tree' if 'Forest' in best_model or 'Tree' in best_model else 'linear'
                                
                                # Sample data if too large
                                X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42)
                                
                                shap_fig = plot_shap_summary(best_model_obj, X_train.sample(min(50, len(X_train)), random_state=42), 
                                                            X_test_sample, model_type)
                                
                                if shap_fig is not None:
                                    st.pyplot(shap_fig, use_container_width=True)
                                    st.success("✓ SHAP summary plot generated!")
                        except Exception as e:
                            st.error(f"⚠️ SHAP visualization error: {str(e)}")
            
            with shap_col2:
                st.write("**Individual Prediction Explanation:**")
                if best_model in model_instances:
                    # Show sample statistics to help selection
                    sample_stats_col1, sample_stats_col2, sample_stats_col3 = st.columns(3)
                    with sample_stats_col1:
                        st.metric("📊 Total Samples", len(X_test))
                    with sample_stats_col2:
                        correct = (model_instances[best_model]["y_pred"] == model_instances[best_model]["y_test"]).sum()
                        st.metric("✅ Correct", correct)
                    with sample_stats_col3:
                        incorrect = (model_instances[best_model]["y_pred"] != model_instances[best_model]["y_test"]).sum()
                        st.metric("❌ Incorrect", incorrect)
                    
                    pred_idx = st.number_input("Select sample index (0 to {}):".format(len(X_test)-1), 
                                              min_value=0, max_value=len(X_test)-1, value=0, step=1, key="shap_idx")
                    
                    # Preview sample
                    st.info(f"👀 Sample {pred_idx} values: {dict(zip(X_test.columns[:3], X_test.iloc[pred_idx].values[:3]))}...")
                    
                    if st.button("📈 Explain Prediction", use_container_width=True, key="shap_force_btn"):
                        try:
                            with st.spinner("Generating SHAP force plot..."):
                                best_model_obj = model_instances[best_model]["model"]
                                model_type = 'tree' if 'Forest' in best_model or 'Tree' in best_model else 'linear'
                                
                                shap_explanation = plot_shap_force(best_model_obj, 
                                                                  X_train.sample(min(50, len(X_train)), random_state=42), 
                                                                  X_test, pred_idx, model_type)
                                
                                if shap_explanation is not None:
                                    st.success("✓ Explanation Generated!")
                                    
                                    sample = X_test.iloc[pred_idx]
                                    st.markdown(f"**Sample #{pred_idx} - Feature Values:**")
                                    st.dataframe(pd.DataFrame([sample]), use_container_width=True)
                                    
                                    st.markdown(f"**SHAP Values (Feature Contributions):**")
                                    shap_contrib = pd.DataFrame({
                                        'Feature': X_test.columns,
                                        'Value': sample.values,
                                        'SHAP Impact': shap_explanation['shap_values']
                                    })
                                    st.dataframe(shap_contrib.sort_values('SHAP Impact', key=abs, ascending=False), use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ SHAP explanation error: {str(e)}")
        
        # Report Export
        st.subheader("📄 Export Report")
        profile, col_profile = generate_data_profile(st.session_state.df)
        stat_summary = get_statistical_summary(st.session_state.df)
        
        html_report = generate_html_report(st.session_state.df, profile, stat_summary, 
                                           trained_models=model_instances, mode=mode)
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html_report,
            file_name=f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
        
        # Model Export
        st.subheader("💾 Export Model")
        model_bytes = pickle.dumps(model_instances[best_model]["model"])
        st.download_button(
            label="📥 Download Best Model (Pickle)",
            data=model_bytes,
            file_name=f"{best_model.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

# ============================================================================
# MAIN APP
# ============================================================================

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=config.STEP_NAMES,
        icons=["🏠", "📤", "🧹", "📊", "🤖"],
        default_index=st.session_state.current_step,
        orientation="vertical",
        key="main_menu",
        styles={
            "container": {"padding": "0.5rem!important", "background-color": "#1A1F2E"},
            "icon": {"color": "#5B7FFF", "font-size": "22px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0.5rem 0",
                "--hover-color": "#252D3D",
                "color": "#E8EAED",
                "border-radius": "8px",
                "padding": "0.75rem 1rem",
                "font-weight": "500"
            },
            "nav-link-selected": {
                "background-color": "#5B7FFF",
                "color": "white",
                "border-radius": "8px",
                "font-weight": "600"
            }
        }
    )
    
    # Update current_step when selection changes
    for i, name in enumerate(config.STEP_NAMES):
        if selected == name:
            st.session_state.current_step = i
            break
    
    st.divider()
    st.markdown(f"""
    <div class="help-box">
        <strong>Current Step:</strong> {config.STEP_NAMES[st.session_state.current_step]}<br><br>
        {config.HELP_TEXTS[st.session_state.current_step]}
    </div>
    """, unsafe_allow_html=True)

# Route to current page
if st.session_state.current_step == 0:
    landing_page()
elif st.session_state.current_step == 1:
    upload_and_schema()
elif st.session_state.current_step == 2:
    clean_data()
elif st.session_state.current_step == 3:
    visualize_data()
elif st.session_state.current_step == 4:
    page_model_training()
