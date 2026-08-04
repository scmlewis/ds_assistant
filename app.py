import streamlit as st

# Page Configuration - MUST be first Streamlit command
st.set_page_config(page_title="AI Data Science Assistant", layout="wide", page_icon=":bar_chart:")

# Load Outfit font from Google Fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Now import other modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, roc_curve, auc
from scipy import stats
import pickle
from streamlit_option_menu import option_menu
import config
import core
import modeling

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
if "ml" not in st.session_state:
    st.session_state.ml = {}


# Custom CSS Styling
def apply_custom_styling():
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #111318;
        --bg-secondary: #181b22;
        --bg-surface: #1e2129;
        --bg-elevated: #252830;
        --accent: #6b8aed;
        --accent-muted: #4e6bc2;
        --text-primary: #dfe2e8;
        --text-secondary: #8b9099;
        --text-tertiary: #5c6068;
        --border: #282c34;
        --border-subtle: #1f2229;
        --success: #4ade80;
        --error: #f87171;
        --warning: #facc15;
    }

    * {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Content container — max width */
    [data-testid="stAppViewContainer"] > section {
        padding: 2rem 2rem;
        max-width: 100%;
    }

    .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding-top: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--bg-secondary) !important;
    }

    /* Page header — flat, no gradient */
    .app-header {
        padding: 1.75rem 2rem;
        border-radius: 10px;
        margin: 0 0 2rem 0;
        border: 1px solid var(--border-subtle);
        background-color: var(--bg-secondary);
    }

    .app-title {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.5px;
    }

    /* Headings */
    h1 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.75rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.4px;
    }

    h2 {
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 2rem 0 1rem 0;
    }

    h3 {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 1.05rem;
        margin: 1.25rem 0 0.5rem 0;
    }

    /* Help/Info Box — minimal */
    .help-box {
        background-color: var(--bg-surface);
        border-left: 3px solid var(--accent-muted);
        padding: 1rem 1.25rem;
        border-radius: 0 6px 6px 0;
        margin: 0 0 1.5rem 0;
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* Stat Card — flat, left-aligned */
    .stat-card {
        background-color: var(--bg-surface);
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
        text-align: left;
        margin: 0.25rem 0;
    }

    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0.25rem 0;
        font-variant-numeric: tabular-nums;
    }

    .stat-label {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    /* Empty State — clean, no dashed border */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
    }

    .empty-state-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
        opacity: 0.6;
    }

    .empty-state-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-primary);
        margin: 0.75rem 0 0.5rem 0;
    }

    .empty-state-message {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* Buttons — flat, no gradient */
    .stButton > button {
        background-color: var(--accent);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.25rem;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 0.01em;
        transition: background-color 0.2s ease, transform 0.15s ease;
    }

    .stButton > button:hover {
        background-color: var(--accent-muted);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    .stButton > button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }

    /* Input Fields */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] > div > div,
    [data-testid="stTextInput"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background-color: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text-primary);
    }

    [data-testid="stSelectbox"] > div > div:focus-within,
    [data-testid="stNumberInput"] > div > div:focus-within,
    [data-testid="stTextInput"] > div > div:focus-within,
    [data-testid="stMultiSelect"] > div > div:focus-within {
        border-color: var(--accent-muted);
        box-shadow: 0 0 0 2px rgba(107, 138, 237, 0.15);
        background-color: var(--bg-elevated);
    }

    /* Checkbox and Radio */
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: var(--text-primary);
    }

    [data-testid="stCheckbox"] {
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
    }

    [data-testid="stCheckbox"]:hover {
        background-color: var(--bg-surface);
    }

    /* Messages */
    .stSuccess {
        background-color: rgba(74, 222, 128, 0.08) !important;
        border-left: 3px solid var(--success) !important;
        color: #bbf7d0 !important;
        border-radius: 0 6px 6px 0 !important;
        padding: 0.85rem 1rem !important;
    }

    .stError {
        background-color: rgba(248, 113, 113, 0.08) !important;
        border-left: 3px solid var(--error) !important;
        color: #fecaca !important;
        border-radius: 0 6px 6px 0 !important;
        padding: 0.85rem 1rem !important;
    }

    .stWarning {
        background-color: rgba(250, 204, 21, 0.08) !important;
        border-left: 3px solid var(--warning) !important;
        color: #fef08a !important;
        border-radius: 0 6px 6px 0 !important;
        padding: 0.85rem 1rem !important;
    }

    .stInfo {
        background-color: rgba(107, 138, 237, 0.08) !important;
        border-left: 3px solid var(--accent-muted) !important;
        color: #c7d2fe !important;
        border-radius: 0 6px 6px 0 !important;
        padding: 0.85rem 1rem !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-subtle);
    }

    /* Expander */
    [data-testid="stExpander"] > div > button {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        padding: 0.85rem 1rem !important;
    }

    [data-testid="stExpander"] > div > button:hover {
        background-color: var(--bg-elevated) !important;
        border-color: var(--border) !important;
    }

    [data-testid="stExpander"] > div > button:focus {
        outline: none;
        border-color: var(--accent-muted) !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.5rem 0;
    }

    /* Links */
    a {
        color: var(--accent);
        text-decoration: none;
    }

    a:hover {
        color: var(--accent-muted);
        text-decoration: underline;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div {
        background-color: var(--bg-elevated);
    }

    [data-testid="stProgress"] > div > div {
        background-color: var(--accent);
    }

    /* Metric */
    [data-testid="stMetric"] {
        background-color: var(--bg-surface);
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
    }

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0;
        background-color: var(--bg-surface);
        border-radius: 8px;
        padding: 4px;
        border: 1px solid var(--border-subtle);
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 500;
        color: var(--text-secondary);
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background-color: var(--bg-elevated);
        color: var(--text-primary);
    }

    /* Landing — capability cards */
    .cap-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 0.75rem;
        min-height: 170px;
        transition: border-color 0.2s ease, background-color 0.2s ease, transform 0.2s ease;
    }

    .cap-card:hover {
        border-color: var(--border);
        background-color: var(--bg-elevated);
        transform: translateY(-2px);
    }

    .cap-index {
        color: var(--accent-muted);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        margin-bottom: 0.5rem;
    }

    .cap-title {
        color: var(--text-primary);
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: -0.2px;
    }

    .cap-card p {
        color: var(--text-secondary);
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Landing — export CTA band */
    .cta-band {
        background-color: rgba(107, 138, 237, 0.07);
        border: 1px solid rgba(107, 138, 237, 0.25);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 1.75rem 2rem;
        margin: 0.25rem 0 2rem 0;
    }

    .cta-kicker {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.35rem;
    }

    .cta-title {
        color: var(--text-primary);
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        letter-spacing: -0.2px;
    }

    .cta-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Responsive */
    @media (max-width: 800px) {
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
        }
        .app-title {
            font-size: 1.5rem;
        }
        h1 {
            font-size: 1.35rem;
        }
        .block-container {
            padding: 1rem;
        }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_styling()

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

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
        st.dataframe(missing_table[missing_table["Missing Count"] > 0], width="stretch")

def empty_state(title, message):
    """Display user-friendly empty state UI."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-message">{message}</div>
    </div>
    """, unsafe_allow_html=True)

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
    fig.patch.set_facecolor('#181b22')
    ax.set_facecolor('#1e2129')
    
    ax.plot(fpr, tpr, color='#6b8aed', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#f87171', lw=2, linestyle='--', label='Random classifier')
    
    ax.set_xlabel('False positive rate', color='#dfe2e8', fontweight='500')
    ax.set_ylabel('True positive rate', color='#dfe2e8', fontweight='500')
    ax.set_title(f'ROC curve — {model_name}', color='#dfe2e8', fontweight='500', pad=15)
    ax.tick_params(colors='#dfe2e8')
    ax.legend(loc='lower right', facecolor='#1e2129', edgecolor='#282c34', labelcolor='#dfe2e8')
    
    for spine in ax.spines.values():
        spine.set_color('#282c34')
    
    ax.grid(True, alpha=0.1, color='#282c34')
    plt.tight_layout()
    return fig, roc_auc

def get_missing_value_heatmap(df):
    """Create missing value heatmap."""
    missing_matrix = df.isna().astype(int)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#181b22')
    ax.set_facecolor('#1e2129')
    
    heatmap = sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar=True, ax=ax, 
                         cbar_kws={'label': 'Missing (1) vs Present (0)'},
                         linewidths=0.2, linecolor='#181b22')
    
    ax.set_xlabel('Row index', color='#dfe2e8', fontweight='500')
    ax.set_ylabel('Columns', color='#dfe2e8', fontweight='500')
    ax.set_title('Missing data pattern', color='#dfe2e8', fontweight='500', pad=15)
    ax.tick_params(colors='#dfe2e8')
    
    cbar = heatmap.collections[0].colorbar
    if cbar:
        cbar.set_label('Missing (1) vs Present (0)', color='#dfe2e8')
        cbar.ax.tick_params(colors='#dfe2e8')
    
    plt.tight_layout()
    return fig

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
                font-family: 'Outfit', 'Segoe UI', sans-serif;
                background: #111318;
                color: #dfe2e8;
                line-height: 1.6;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: #181b22;
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid #282c34;
            }}
            header {{
                background: #1e2129;
                padding: 40px 20px;
                text-align: center;
                border-bottom: 1px solid #282c34;
            }}
            header h1 {{
                font-size: 2.2em;
                margin-bottom: 8px;
                color: #dfe2e8;
                font-weight: 600;
            }}
            header p {{
                color: #8b9099;
                font-size: 1em;
            }}
            .content {{
                padding: 40px;
            }}
            section {{
                margin-bottom: 40px;
                border-bottom: 1px solid #282c34;
                padding-bottom: 30px;
            }}
            section:last-child {{
                border-bottom: none;
            }}
            h2 {{
                color: #6b8aed;
                margin-bottom: 20px;
                font-size: 1.6em;
                border-left: 3px solid #6b8aed;
                padding-left: 15px;
                font-weight: 500;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: #1e2129;
                border-radius: 8px;
                overflow: hidden;
            }}
            th {{
                background: #252830;
                color: #dfe2e8;
                padding: 12px;
                text-align: left;
                font-weight: 500;
                border-bottom: 1px solid #282c34;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #282c34;
            }}
            tr:hover {{
                background: #252830;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #1e2129;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #282c34;
            }}
            .metric-label {{
                color: #5c6068;
                font-size: 0.85em;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .metric-value {{
                color: #6b8aed;
                font-size: 1.8em;
                font-weight: 700;
            }}
            footer {{
                background: #111318;
                padding: 20px;
                text-align: center;
                color: #5c6068;
                font-size: 0.85em;
                border-top: 1px solid #282c34;
            }}
            .info-box {{
                background: #1e2129;
                border-left: 3px solid #6b8aed;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 6px 6px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Data Science Analysis Report</h1>
                <p>Comprehensive analysis and insights from your dataset</p>
            </header>
            
            <div class="content">
                <section>
                    <h2>Dataset overview</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-label">Total rows</div>
                            <div class="metric-value">{profile['Total Rows']:,}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Total columns</div>
                            <div class="metric-value">{profile['Total Columns']}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Memory usage</div>
                            <div class="metric-value">{profile['Memory Usage (MB)']:.2f} MB</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Completeness</div>
                            <div class="metric-value">{profile['Completeness %']:.1f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Duplicate rows</div>
                            <div class="metric-value">{profile['Duplicate Rows']}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Complete rows</div>
                            <div class="metric-value">{profile['Complete Rows']:,}</div>
                        </div>
                    </div>
                </section>
                
                {f'''
                <section>
                    <h2>Statistical summary</h2>
                    {stat_summary.to_html() if stat_summary is not None else '<p>No numeric columns found.</p>'}
                </section>
                ''' if stat_summary is not None else ''}
                
                {f'''
                <section>
                    <h2>Model training results</h2>
                    <div class="info-box">
                        <strong>Mode:</strong> {mode}
                    </div>
                </section>
                ''' if trained_models else ''}
                
            </div>
            
            <footer>
                <p>Generated by AI Data Science Assistant &middot; Report created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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
    
    if len(df) > sample_size:
        df_sample = df[numeric_cols].sample(n=sample_size, random_state=42)
    else:
        df_sample = df[numeric_cols]
    
    fig = plt.figure(figsize=(max(10, len(numeric_cols) * 2), max(10, len(numeric_cols) * 2)))
    fig.patch.set_facecolor('#181b22')
    
    n_cols = len(numeric_cols)
    for i, col_x in enumerate(numeric_cols):
        for j, col_y in enumerate(numeric_cols):
            ax = plt.subplot(n_cols, n_cols, i * n_cols + j + 1)
            ax.set_facecolor('#1e2129')
            
            if i == j:
                ax.hist(df_sample[col_x], bins=20, color='#6b8aed', alpha=0.7, edgecolor='#282c34')
            else:
                ax.scatter(df_sample[col_x], df_sample[col_y], alpha=0.5, color='#6b8aed', s=20, edgecolors='#282c34')
            
            ax.tick_params(colors='#dfe2e8', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#282c34')
            
            if i == n_cols - 1:
                ax.set_xlabel(col_x, color='#dfe2e8', fontsize=9, fontweight='500')
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(col_y, color='#dfe2e8', fontsize=9, fontweight='500')
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
    
    fig.patch.set_facecolor('#181b22')
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.set_facecolor('#1e2129')
        
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#6b8aed', alpha=0.7),
                        whiskerprops=dict(color='#dfe2e8'),
                        capprops=dict(color='#dfe2e8'),
                        medianprops=dict(color='#f87171', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='#f87171', markersize=6, alpha=0.5))
        
        ax.set_title(col, color='#dfe2e8', fontweight='500', fontsize=11)
        ax.set_ylabel('Value', color='#dfe2e8', fontweight='500')
        ax.tick_params(colors='#dfe2e8')
        for spine in ax.spines.values():
            spine.set_color('#282c34')
        ax.grid(True, alpha=0.1, color='#282c34', axis='y')
    
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
    
    fig.patch.set_facecolor('#181b22')
    
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        ax.set_facecolor('#1e2129')
        
        value_counts = df[col].value_counts().head(max_categories)
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='#6b8aed', 
                      edgecolor='#282c34', alpha=0.8)
        
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels([str(v)[:15] for v in value_counts.index], rotation=45, ha='right', 
                           color='#dfe2e8', fontsize=9)
        ax.set_ylabel('Count', color='#dfe2e8', fontweight='500')
        ax.set_title(f'{col} distribution', color='#dfe2e8', fontweight='500', fontsize=11)
        ax.tick_params(colors='#dfe2e8')
        for spine in ax.spines.values():
            spine.set_color('#282c34')
        ax.grid(True, alpha=0.1, color='#282c34', axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', color='#dfe2e8', fontsize=8)
    
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# ============================================================================
# ADVANCED FEATURES - HYPERPARAMETER TUNING
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

def plot_correlation_with_significance(df, numeric_cols):
    """
    Plot correlation matrix with significance annotations.
    """
    corr_df, pval_df = core.calculate_correlation_significance(df, numeric_cols)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.set_facecolor('#111318')
    ax.set_facecolor('#181b22')
    
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, cbar=True, vmin=-1, vmax=1)
    
    ax.set_title('Correlation matrix with Pearson r', color='#dfe2e8', fontweight='500', fontsize=12, pad=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color='#dfe2e8')
    plt.setp(ax.get_yticklabels(), rotation=0, color='#dfe2e8')
    
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            if i != j and pval_df.iloc[i, j] < 0.05:
                ax.text(j+0.5, i+0.7, '*', ha='center', va='center', 
                       color='#facc15', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig, corr_df, pval_df

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
                    'Significant': 'Yes' if pval < 0.05 else 'No',
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
                'Significant': 'Yes' if pval < 0.05 else 'No',
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
    # Hero — value-proposition headline
    st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <div style="color: var(--accent); font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.9rem;">AI Data Science Assistant</div>
        <h1 style="font-size: 2.5rem; font-weight: 700; color: var(--text-primary); margin: 0 0 1rem 0; letter-spacing: -0.5px; line-height: 1.15; max-width: 760px;">
            From raw CSV to a trained model — without writing code.
        </h1>
        <div style="width: 48px; height: 3px; background: var(--accent); border-radius: 2px; margin-bottom: 1.25rem;"></div>
        <p style="color: var(--text-secondary); font-size: 1.05rem; margin: 0; font-weight: 400; max-width: 620px;">
            Upload a file, clean it, explore it, and compare models in a few clicks. No Python required.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start analyzing", key="hero_cta"):
        st.session_state.current_step = 1
        st.rerun()

    st.markdown("""
    <div style="height: 2rem;"></div>
    """, unsafe_allow_html=True)

    # Capabilities — balanced 2x2 grid, consistent card system
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="cap-card">
            <div class="cap-index">01</div>
            <div class="cap-title">Data profiling</div>
            <p>Column types, missing values, summary statistics, and a correlation matrix with p-values — instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="cap-card">
            <div class="cap-index">02</div>
            <div class="cap-title">Cleaning</div>
            <p>Deduplication, imputation, encoding, scaling, and outlier removal — with a live before/after preview.</p>
        </div>
        """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
        <div class="cap-card">
            <div class="cap-index">03</div>
            <div class="cap-title">Visualization</div>
            <p>Histograms, boxplots, scatter, KDE, violin, and pair plots, all in a consistent dark theme.</p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="cap-card">
            <div class="cap-index">04</div>
            <div class="cap-title">ML Lab</div>
            <p>Auto-detects classification vs regression, trains models with cross-validation, ROC curves, and feature importance. Now with hyperparameter tuning, interpretability, and diagnostics.</p>
        </div>
        """, unsafe_allow_html=True)

    # Export — accent CTA band
    st.markdown("""
    <div class="cta-band">
        <div class="cta-kicker">Export</div>
        <div class="cta-title">Take your results with you</div>
        <p class="cta-text">Download the cleaned data as CSV, the trained models as pickle files, and a full HTML analysis report.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick tips — subtle bullet list
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 2rem;">
        <div style="display: flex; gap: 0.75rem; align-items: flex-start;">
            <div style="width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex-shrink: 0; margin-top: 0.5rem;"></div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5;">Start with the built-in Iris or Diabetes datasets — no upload needed.</div>
        </div>
        <div style="display: flex; gap: 0.75rem; align-items: flex-start;">
            <div style="width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex-shrink: 0; margin-top: 0.5rem;"></div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5;">Live previews show exactly what changes before you commit a cleaning step.</div>
        </div>
        <div style="display: flex; gap: 0.75rem; align-items: flex-start;">
            <div style="width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex-shrink: 0; margin-top: 0.5rem;"></div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5;">Your data persists across pages — navigate freely without losing progress.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def upload_and_schema():
    """Upload data and schema inspection page."""
    st.markdown(f"""
    <div class="app-header">
        <div style="color: var(--text-tertiary); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">Step 1</div>
        <h1 class="app-title">Upload Data</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[1]}
    </div>
    """, unsafe_allow_html=True)
    
    # Sample Datasets Section
    st.subheader("Sample datasets")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Iris dataset", width="stretch", key="iris_btn"):
            df = load_sample_dataset("iris")
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.pending_clean_options = {}
            st.success("Iris dataset loaded.")
            st.rerun()
    
    with col2:
        if st.button("Load Diabetes dataset", width="stretch", key="diabetes_btn"):
            df = load_sample_dataset("diabetes")
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.pending_clean_options = {}
            st.success("Diabetes dataset loaded.")
            st.rerun()
    
    # CSV File Upload
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.pending_clean_options = {}
            st.success("CSV file uploaded.")
            st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Show data if loaded
    if st.session_state.df is not None:
        st.subheader("Data quality report")
        data_quality_report(st.session_state.df)
        
        # Data Profiling Dashboard
        st.subheader("Data profiling")
        profile, col_profile = generate_data_profile(st.session_state.df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Memory usage", f"{profile['Memory Usage (MB)']:.2f} MB")
        with col2:
            st.metric("Completeness", f"{profile['Completeness %']:.1f}%")
        with col3:
            st.metric("Complete rows", f"{profile['Complete Rows']:,}")
        
        st.write("**Column profile:**")
        st.dataframe(col_profile, width="stretch")
        
        st.subheader("Data preview")
        st.dataframe(st.session_state.df.head(config.DATA_PREVIEW_ROWS), width="stretch")
        
        st.subheader("Summary statistics")
        st.dataframe(st.session_state.df.describe(), width="stretch")
        
        st.subheader("Schema")
        schema_info = pd.DataFrame({
            "Column": st.session_state.df.columns,
            "Type": st.session_state.df.dtypes.astype(str),
            "Missing": st.session_state.df.isna().sum(),
            "Unique": st.session_state.df.nunique()
        })
        st.dataframe(schema_info, width="stretch")
        
        # Missing Value Heatmap
        st.subheader("Missing data pattern")
        if st.session_state.df.isna().sum().sum() > 0:
            missing_fig = get_missing_value_heatmap(st.session_state.df)
            st.pyplot(missing_fig, width="stretch")
        else:
            st.info("No missing values detected.")
        
        # Statistical Summary
        st.subheader("Statistical summary")
        stat_summary = core.get_statistical_summary(st.session_state.df)
        if stat_summary is not None:
            st.dataframe(stat_summary, width="stretch")
        else:
            st.info("No numeric columns found for statistical summary.")
    else:
        empty_state("No data yet", "Upload a CSV file or load a sample dataset to get started.")

def clean_data():
    """Data cleaning page."""
    st.markdown(f"""
    <div class="app-header">
        <div style="color: var(--text-tertiary); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">Step 2</div>
        <h1 class="app-title">Clean Data</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[2]}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        empty_state("No data available", "Please upload or load a dataset first.")
        return
    
    # Status banner
    rows = len(st.session_state.df)
    cols = len(st.session_state.df.columns)
    size_kb = st.session_state.df.memory_usage(deep=True).sum() / 1024
    
    st.markdown(f"""
    <div style="background: var(--bg-surface); padding: 1.25rem 1.5rem; border-radius: 8px;
                border: 1px solid var(--border-subtle); margin-bottom: 1.5rem;">
        <div style="color: var(--text-tertiary); font-weight: 500; font-size: 0.75rem; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.35rem;">Current data</div>
        <div style="color: var(--text-primary); font-size: 0.95rem;">
            {rows:,} rows, {cols} columns, {size_kb:.2f} KB
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Original Data Preview
    with st.expander("Original data preview", expanded=False):
        st.dataframe(st.session_state.original_df.head(config.DATA_PREVIEW_ROWS), width="stretch")
    
    st.subheader("Cleaning options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic operations**")
        
        st.session_state.pending_clean_options["standardize_columns"] = st.checkbox(
            "Standardize column names",
            value=st.session_state.pending_clean_options.get("standardize_columns", False),
            help="Convert to lowercase with underscores"
        )
        
        st.session_state.pending_clean_options["remove_duplicates"] = st.checkbox(
            "Remove duplicates",
            value=st.session_state.pending_clean_options.get("remove_duplicates", False)
        )
        
        st.session_state.pending_clean_options["drop_missing"] = st.checkbox(
            "Drop rows with missing values",
            value=st.session_state.pending_clean_options.get("drop_missing", False)
        )
        
        fill_missing = st.checkbox(
            "Fill missing values",
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
        st.write("**Advanced operations**")
        
        st.session_state.pending_clean_options["remove_outliers"] = st.checkbox(
            "Remove outliers (IQR method)",
            value=st.session_state.pending_clean_options.get("remove_outliers", False)
        )
        
        st.session_state.pending_clean_options["encode_categorical"] = st.checkbox(
            "Encode categorical columns",
            value=st.session_state.pending_clean_options.get("encode_categorical", False)
        )
        
        scale_features = st.checkbox(
            "Scale features",
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
    st.subheader("Live preview")
    
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
        st.success(f"Original: {original_shape[0]}x{original_shape[1]} -> Cleaned: {preview_shape[0]}x{preview_shape[1]}")
        
        st.dataframe(preview_df.head(config.DATA_PREVIEW_ROWS), width="stretch")
    
    except Exception as e:
        st.error(f"Error during cleaning: {str(e)}")
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Apply", width="stretch", key="apply_clean"):
            if st.session_state.pending_df is None:
                st.error("No preview available. Adjust cleaning options first.")
            else:
                st.session_state.df = st.session_state.pending_df.copy()
                st.success("Cleaning applied.")
                st.rerun()
    
    with col2:
        if st.button("Revert", width="stretch", key="revert_clean"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.pending_clean_options = {}
            st.info("Data reverted to original state.")
            st.rerun()
    
    with col3:
        if st.button("Download", width="stretch", key="download_clean"):
            if st.session_state.pending_df is None:
                st.error("No preview available. Adjust cleaning options first.")
            else:
                csv = st.session_state.pending_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    width="stretch"
                )
    
    # Suggested Workflow Banner
    st.markdown("""
    <div style="background: var(--bg-surface); padding: 1.25rem 1.5rem; border-radius: 8px;
                border-left: 3px solid var(--accent-muted); margin-bottom: 1.5rem;">
        <div style="color: var(--text-tertiary); font-weight: 500; font-size: 0.75rem; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.35rem;">Suggested next steps</div>
        <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem; line-height: 1.6;">
            Clean data (current) -> Feature engineering -> Visualize correlations -> Train models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Feature Engineering
    st.subheader("Feature engineering")
    
    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        feat_col1, feat_col2 = st.columns([1, 1])
        
        with feat_col1:
            feature_type = st.selectbox(
                "Feature engineering type",
                ["Polynomial Features", "Interaction Terms", "Binning"],
                key="feat_type"
            )
            
            if feature_type == "Polynomial Features":
                degree = st.slider("Polynomial degree", 2, 5, 2, key="poly_degree")
                selected_numeric = st.multiselect(
                    "Select columns for polynomial features",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))],
                    key="poly_cols"
                )
                
                if st.button("Generate polynomial features", width="stretch", key="gen_poly"):
                    df_engineered, new_features = engineer_features(
                        st.session_state.df, selected_numeric, "polynomial", degree
                    )
                    st.session_state.df = df_engineered
                    st.success(f"Created {len(new_features)} polynomial features.")
                    st.write(f"**New features:** {', '.join(new_features[:5])}{'...' if len(new_features) > 5 else ''}")
            
            elif feature_type == "Interaction Terms":
                interaction_cols = st.multiselect(
                    "Select columns for interactions",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))],
                    key="interact_cols"
                )
                
                if st.button("Generate interactions", width="stretch", key="gen_interact"):
                    if len(interaction_cols) >= 2:
                        df_engineered, new_features = engineer_features(
                            st.session_state.df, None, "interaction", interaction_cols=interaction_cols
                        )
                        st.session_state.df = df_engineered
                        st.success(f"Created {len(new_features)} interaction features.")
                        st.write(f"**New features:** {', '.join(new_features)}")
                    else:
                        st.warning("Select at least 2 columns for interactions.")
            
            else:  # Binning
                binning_cols = st.multiselect(
                    "Select columns for binning",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))],
                    key="bin_cols"
                )
                
                if st.button("Generate binned features", width="stretch", key="gen_bin"):
                    df_engineered, new_features = engineer_features(
                        st.session_state.df, binning_cols, "binning"
                    )
                    st.session_state.df = df_engineered
                    st.success(f"Created {len(new_features)} binned features.")
        
        with feat_col2:
            st.info("**Feature engineering tips:**\n\n"
                   "- **Polynomial**: Creates new columns (e.g., x^2, y^2, x*y). Original features are kept.\n"
                   "- **Interactions**: Combines features (e.g., height * weight). Shows joint effects on target.\n"
                   "- **Binning**: Divides values into 5 equal groups (quintiles). Creates ordinal categories.\n\n"
                   "**When to use:**\n"
                   "- Polynomial: When relationships are non-linear\n"
                   "- Interactions: When features influence each other\n"
                   "- Binning: For tree models or when you want to discretize continuous values")
    else:
        st.info("Feature engineering requires numeric columns.")

def visualize_data():
    """Data visualization page."""
    st.markdown(f"""
    <div class="app-header">
        <div style="color: var(--text-tertiary); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">Step 3</div>
        <h1 class="app-title">Visualize Data</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[3]}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        empty_state("No data available", "Please upload or load a dataset first.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Correlations", "Charts", "Distributions", "Pair Plot"])
    
    # TAB 1: CORRELATIONS & STATISTICAL TESTS
    with tab1:
        st.subheader("Correlation matrix with statistical tests")
        
        numeric_df = st.session_state.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) >= 2:
            corr_fig, corr_df, pval_df = plot_correlation_with_significance(st.session_state.df, numeric_df.columns.tolist())
            st.pyplot(corr_fig, width="stretch")
            
            st.markdown("**Legend:** Gold stars (*) = statistically significant correlations (p < 0.05)")
            st.info("Expand sections below to view detailed statistics and run hypothesis tests.")
            
            with st.expander("Detailed correlation statistics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Correlation coefficients (r):**")
                    st.dataframe(corr_df, width="stretch")
                with col2:
                    st.write("**P-values:**")
                    st.dataframe(pval_df, width="stretch")
            
            with st.expander("Hypothesis testing", expanded=False):
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
                    test_type = st.selectbox("Test type", ["pearson", "spearman", "ttest"], key="test_type")
                
                if st.button("Run test", width="stretch", key="run_test"):
                    if var1 != var2:
                        with st.spinner("Running statistical test..."):
                            result = core.perform_hypothesis_test(st.session_state.df, var1, var2, test_type)
                        st.dataframe(result.to_frame().T, width="stretch")
                        
                        if result.get('P-value', 1) < 0.05:
                            st.success("Statistically significant relationship found (p < 0.05). This means the relationship is unlikely due to random chance.")
                        else:
                            st.info("No statistically significant relationship (p >= 0.05). Could be due to random variation in the data.")
                    else:
                        st.warning("Please select different variables.")
        else:
            st.info("Correlation matrix requires at least 2 numeric columns.")
    
    # TAB 2: CUSTOM CHARTS
    with tab2:
        st.subheader("Custom chart creation")
        
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox("Chart type", list(config.CHART_TYPES.keys()), key="chart_type")
        with col2:
            selected_columns = st.multiselect("Select columns", st.session_state.df.columns, key="chart_cols")
        
        if selected_columns and chart_type:
            try:
                fig, ax = plt.subplots(figsize=config.CHART_SIZE)
                fig.patch.set_facecolor('#181b22')
                ax.set_facecolor('#1e2129')
                
                if chart_type == "Histogram":
                    st.session_state.df[selected_columns[0]].hist(bins=30, ax=ax, color='#6b8aed', edgecolor='#282c34', alpha=0.8)
                    ax.set_title(f'Histogram: {selected_columns[0]}', fontweight='500', color='#dfe2e8', pad=15)
                    ax.set_xlabel(selected_columns[0], color='#dfe2e8')
                    ax.set_ylabel('Frequency', color='#dfe2e8')
                
                elif chart_type == "Boxplot":
                    st.session_state.df[selected_columns].boxplot(ax=ax, patch_artist=True)
                    for patch in ax.artists:
                        patch.set_facecolor('#6b8aed')
                        patch.set_edgecolor('#4e6bc2')
                    ax.set_title('Boxplot', fontweight='500', color='#dfe2e8', pad=15)
                    ax.set_ylabel('Value', color='#dfe2e8')
                
                elif chart_type == "Scatter":
                    ax.scatter(st.session_state.df[selected_columns[0]], st.session_state.df[selected_columns[1]], 
                              alpha=0.6, color='#6b8aed', s=50, edgecolors='#282c34', linewidth=0.5)
                    ax.set_xlabel(selected_columns[0], color='#dfe2e8')
                    ax.set_ylabel(selected_columns[1], color='#dfe2e8')
                    ax.set_title(f'Scatter: {selected_columns[0]} vs {selected_columns[1]}', fontweight='500', color='#dfe2e8', pad=15)
                
                elif chart_type == "Bar":
                    st.session_state.df[selected_columns[0]].value_counts().plot(kind='bar', ax=ax, color='#6b8aed', edgecolor='#282c34')
                    ax.set_title(f'Bar chart: {selected_columns[0]}', fontweight='500', color='#dfe2e8', pad=15)
                    ax.set_ylabel('Count', color='#dfe2e8')
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, color='#dfe2e8')
                
                elif chart_type == "Column":
                    st.session_state.df[selected_columns[0]].value_counts().plot(kind='barh', ax=ax, color='#6b8aed', edgecolor='#282c34')
                    ax.set_title(f'Column chart: {selected_columns[0]}', fontweight='500', color='#dfe2e8', pad=15)
                    ax.set_xlabel('Count', color='#dfe2e8')
                
                elif chart_type == "Pie":
                    colors = ['#6b8aed', '#4e6bc2', '#8ba5f0', '#3d5bc4', '#5a7ae6']
                    ax.pie(st.session_state.df[selected_columns[0]].value_counts(), 
                          labels=st.session_state.df[selected_columns[0]].value_counts().index,
                          autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': '#dfe2e8'})
                    ax.set_title(f'Pie chart: {selected_columns[0]}', fontweight='500', color='#dfe2e8', pad=15)
                
                ax.tick_params(colors='#dfe2e8')
                for spine in ax.spines.values():
                    spine.set_color('#282c34')
                    spine.set_linewidth(0.5)
                
                plt.tight_layout()
                st.pyplot(fig, width="stretch")
            
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    # TAB 3: DISTRIBUTION ANALYSIS
    with tab3:
        st.subheader("Distribution analysis")
        
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            dist_col = st.selectbox("Select column for distribution analysis", numeric_cols, key="dist_col")
            dist_type = st.radio("Distribution plot type", ["Histogram with KDE", "KDE Plot", "Violin Plot"], 
                                horizontal=True, key="dist_type")
            
            if dist_col:
                fig, ax = plt.subplots(figsize=config.CHART_SIZE)
                fig.patch.set_facecolor('#181b22')
                ax.set_facecolor('#1e2129')
                
                if dist_type == "Histogram with KDE":
                    st.session_state.df[dist_col].hist(bins=30, ax=ax, color='#6b8aed', 
                                                        alpha=0.6, edgecolor='#282c34', density=True)
                    st.session_state.df[dist_col].plot(kind='kde', ax=ax, color='#f87171', linewidth=2.5)
                    ax.set_title(f'Distribution: {dist_col} (Histogram + KDE)', fontweight='500', color='#dfe2e8', pad=15)
                
                elif dist_type == "KDE Plot":
                    st.session_state.df[dist_col].plot(kind='kde', ax=ax, color='#6b8aed', linewidth=3)
                    ax.fill_between(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata(), 
                                   alpha=0.3, color='#6b8aed')
                    ax.set_title(f'KDE plot: {dist_col}', fontweight='500', color='#dfe2e8', pad=15)
                
                elif dist_type == "Violin Plot":
                    parts = ax.violinplot([st.session_state.df[dist_col].dropna()], 
                                         positions=[0], showmeans=True, showmedians=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor('#6b8aed')
                        pc.set_alpha(0.7)
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                        if partname in parts:
                            vp = parts[partname]
                            vp.set_edgecolor('#dfe2e8')
                            vp.set_linewidth(2)
                    ax.set_xticks([0])
                    ax.set_xticklabels([dist_col])
                    ax.set_title(f'Violin plot: {dist_col}', fontweight='500', color='#dfe2e8', pad=15)
                
                ax.set_ylabel('Density' if 'KDE' in dist_type or 'Histogram' in dist_type else 'Frequency', 
                             color='#dfe2e8', fontweight='500')
                ax.set_xlabel(dist_col, color='#dfe2e8', fontweight='500')
                ax.tick_params(colors='#dfe2e8')
                
                for spine in ax.spines.values():
                    spine.set_color('#282c34')
                    spine.set_linewidth(0.5)
                
                plt.tight_layout()
                st.pyplot(fig, width="stretch")
        else:
            st.info("Distribution analysis requires at least one numeric column.")
    
    # TAB 4: PAIR PLOT
    with tab4:
        st.subheader("Pair plot analysis")
        
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            if st.button("Generate pair plot", help="Shows pairwise relationships between numeric features", 
                        width="stretch", key="pair_plot_btn"):
                with st.spinner("Generating pair plot..."):
                    pair_fig = plot_pair_plot(st.session_state.df, numeric_cols)
                    if pair_fig:
                        st.pyplot(pair_fig, width="stretch")
                        st.caption("Pair plot showing relationships between all numeric features (diagonal: histograms, off-diagonal: scatter plots)")
        else:
            st.info("Pair plots require at least 2 numeric columns.")

def _tab_interpret():
    if not st.session_state.trained_models:
        st.info("Train a model in the **Configure & Train** tab first.")
        return

    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Model to interpret", model_names, key="interp_model")

    md = st.session_state.trained_models[selected_model]
    mode = md["mode"]
    feature_names = list(md["X_test"].columns)

    st.subheader("Feature importance")

    source = st.radio("Importance source", ["Permutation", "Model coefficients", "Built-in"],
                       horizontal=True, key="imp_source")

    if st.button("Compute importance", key="comp_imp_btn"):
        with st.spinner("Computing..."):
            if source == "Permutation":
                importance_df = modeling.permutation_feature_importance(
                    md["model"], md["X_test"], md["y_test"], feature_names
                )
            elif source == "Model coefficients":
                importance_df = modeling.get_model_coefficients(md["model"], feature_names)
            else:
                if hasattr(md["model"], "feature_importances_"):
                    raw = md["model"].feature_importances_
                elif hasattr(md["model"], "named_steps"):
                    final = md["model"].named_steps.get("model")
                    if final is not None and hasattr(final, "feature_importances_"):
                        raw = final.feature_importances_
                    else:
                        raw = None
                else:
                    raw = None

                if raw is None:
                    importance_df = None
                else:
                    importance_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": raw,
                    }).sort_values("importance", ascending=False)

        if importance_df is None or importance_df.empty:
            st.warning("Cannot extract importance for this model/source combination.")
        else:
            fig, ax = plt.subplots(figsize=(8, max(3, len(importance_df) * 0.4)))
            fig.patch.set_facecolor('#181b22')
            ax.set_facecolor('#1e2129')

            importance_col = "importance" if "importance" in importance_df.columns else "coefficient"
            plot_df = importance_df.sort_values(importance_col)

            ax.barh(plot_df["feature"], plot_df[importance_col], color="#6b8aed", edgecolor="#282c34")
            ax.set_xlabel("Importance" if source != "Model coefficients" else "Coefficient",
                          color="#dfe2e8", fontweight="500")
            ax.set_title(f"{source} importance", color="#dfe2e8", fontweight="500", pad=15)
            ax.tick_params(colors="#dfe2e8")
            for spine in ax.spines.values():
                spine.set_color("#282c34")
            plt.tight_layout()
            st.pyplot(fig, width="stretch")

            st.dataframe(importance_df, width="stretch")

    st.divider()
    st.subheader("Partial dependence")

    pdp_features = st.multiselect("Features (1-2)", feature_names,
                                   default=[feature_names[0]] if feature_names else [],
                                   max_selections=2, key="pdp_features")

    if pdp_features and st.button("Compute partial dependence", key="pdp_btn"):
        with st.spinner("Computing..."):
            pdp = modeling.partial_dependence_data(md["model"], md["X_test"], pdp_features)

        if len(pdp_features) == 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#181b22')
            ax.set_facecolor('#1e2129')
            ax.plot(pdp["grid_values"], pdp["pdp_values"], color="#6b8aed", linewidth=2.5)
            ax.fill_between(pdp["grid_values"], pdp["pdp_values"], alpha=0.15, color="#6b8aed")
            ax.set_xlabel(pdp_features[0], color="#dfe2e8", fontweight="500")
            ax.set_ylabel("Partial dependence", color="#dfe2e8", fontweight="500")
            ax.set_title(f"PDP — {pdp_features[0]}", color="#dfe2e8", fontweight="500", pad=15)
            ax.tick_params(colors="#dfe2e8")
            for spine in ax.spines.values():
                spine.set_color("#282c34")
            ax.grid(True, alpha=0.1, color="#282c34")
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
        else:
            if isinstance(pdp.get("pdp_values"), np.ndarray) and pdp["pdp_values"].ndim == 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#181b22')
                ax.set_facecolor('#1e2129')
                im = ax.pcolormesh(pdp["grid_values"][0], pdp["grid_values"][1],
                                   pdp["pdp_values"], cmap="RdBu_r", shading="auto")
                ax.set_xlabel(pdp_features[0], color="#dfe2e8", fontweight="500")
                ax.set_ylabel(pdp_features[1], color="#dfe2e8", fontweight="500")
                ax.set_title(f"PDP — {pdp_features[0]} x {pdp_features[1]}",
                             color="#dfe2e8", fontweight="500", pad=15)
                ax.tick_params(colors="#dfe2e8")
                fig.colorbar(im, ax=ax)
                plt.tight_layout()
                st.pyplot(fig, width="stretch")
            else:
                st.dataframe(pdp, width="stretch")

    st.divider()
    st.subheader("Explain a single prediction")

    row_idx = st.slider("Row index", 0, len(md["X_test"]) - 1, 0, key="explain_row")

    if st.button("Explain prediction", key="explain_btn"):
        with st.spinner("Computing..."):
            explanation = modeling.explain_prediction(
                md["model"], md["X_test"], md["y_test"], row_idx,
                feature_names=feature_names
            )

        st.dataframe(explanation, width="stretch")

        fig, ax = plt.subplots(figsize=(8, max(3, len(explanation) * 0.5)))
        fig.patch.set_facecolor('#181b22')
        ax.set_facecolor('#1e2129')

        colors = ["#4ade80" if v >= 0 else "#f87171" for v in explanation["contribution"]]
        ax.barh(explanation["feature"], explanation["contribution"], color=colors, edgecolor="#282c34")
        ax.set_xlabel("Contribution", color="#dfe2e8", fontweight="500")
        ax.set_title(f"Prediction explanation (row {row_idx})", color="#dfe2e8", fontweight="500", pad=15)
        ax.tick_params(colors="#dfe2e8")
        ax.axvline(x=0, color="#dfe2e8", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("#282c34")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")


def _tab_tune():
    if not st.session_state.trained_models:
        st.info("Train a model in the **Configure & Train** tab first.")
        return

    st.subheader("Hyperparameter tuning")

    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Model to tune", model_names, key="tune_model")

    search_type = st.radio("Search type", ["Grid Search", "Randomized Search"],
                           horizontal=True, key="search_type")

    pipe = st.session_state.trained_models[selected_model]["pipeline"]
    param_grid = modeling.get_param_grid(selected_model)

    if not param_grid:
        st.info(f"**{selected_model}** has no tunable hyperparameters in the grid. Try a different model.")
        return

    with st.expander("Parameter grid", expanded=False):
        st.json(param_grid)

    n_iter = 10
    if search_type == "Randomized Search":
        n_iter = st.slider("n_iter (random search iterations)", 5, 100, 10, key="n_iter")

    cv_folds = st.slider("CV folds", 2, 10, config.CROSS_VAL_FOLDS, key="tune_cv")

    if st.button("Run tuning", width="stretch", key="run_tuning_btn"):
        X = None
        y = None
        for name in model_names:
            if name == selected_model:
                md = st.session_state.trained_models[name]
                X = pd.concat([md["X_test"], md["X_test"]])
                y = pd.concat([md["y_test"], md["y_test"]])
                break

        if X is None:
            st.error("Could not retrieve training data.")
            return

        with st.spinner("Running hyperparameter search..."):
            mode = st.session_state.trained_models[selected_model]["mode"]
            scoring = "accuracy" if mode == "Classification" else "r2"

            if search_type == "Grid Search":
                result = modeling.run_grid_search(pipe, X, y, param_grid, cv=cv_folds, scoring=scoring)
            else:
                result = modeling.run_random_search(pipe, X, y, param_grid, n_iter=n_iter, cv=cv_folds, scoring=scoring)

        st.success("Tuning complete!")

        st.subheader("Best parameters")
        st.json(result.best_params_)

        st.subheader("CV results")
        cv_results = pd.DataFrame(result.cv_results_)
        display_cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        existing_cols = [c for c in display_cols if c in cv_results.columns]
        st.dataframe(cv_results[existing_cols].head(20), width="stretch")

        st.session_state.trained_models[selected_model]["tuned_model"] = result.best_estimator_
        st.session_state.trained_models[selected_model]["best_params"] = result.best_params_

    st.divider()
    st.subheader("Complexity curves")

    if st.session_state.trained_models[selected_model].get("tuned_model"):
        tuned = st.session_state.trained_models[selected_model]["tuned_model"]
    else:
        tuned = pipe

    mode = st.session_state.trained_models[selected_model]["mode"]
    X_for_curves = st.session_state.trained_models[selected_model]["X_test"]
    y_for_curves = st.session_state.trained_models[selected_model]["y_test"]
    scoring = "accuracy" if mode == "Classification" else "r2"

    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        st.write("**Learning curve**")
        if st.button("Compute learning curve", key="lc_btn"):
            with st.spinner("Computing..."):
                lc = modeling.compute_learning_curve(tuned, X_for_curves, y_for_curves, cv=cv_folds, scoring=scoring)

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#181b22')
            ax.set_facecolor('#1e2129')

            ax.plot(lc["train_sizes"], lc["train_mean"], "o-", color="#6b8aed", label="Training score")
            ax.fill_between(lc["train_sizes"], lc["train_mean"] - lc["train_std"],
                            lc["train_mean"] + lc["train_std"], alpha=0.15, color="#6b8aed")
            ax.plot(lc["train_sizes"], lc["cv_mean"], "o-", color="#f87171", label="CV score")
            ax.fill_between(lc["train_sizes"], lc["cv_mean"] - lc["cv_std"],
                            lc["cv_mean"] + lc["cv_std"], alpha=0.15, color="#f87171")

            ax.set_xlabel("Training set size", color="#dfe2e8", fontweight="500")
            ax.set_ylabel("Score", color="#dfe2e8", fontweight="500")
            ax.set_title("Learning curve", color="#dfe2e8", fontweight="500", pad=15)
            ax.tick_params(colors="#dfe2e8")
            ax.legend(facecolor="#1e2129", edgecolor="#282c34", labelcolor="#dfe2e8")
            for spine in ax.spines.values():
                spine.set_color("#282c34")
            ax.grid(True, alpha=0.1, color="#282c34")
            plt.tight_layout()
            st.pyplot(fig, width="stretch")

            gap = lc["train_mean"][-1] - lc["cv_mean"][-1]
            if gap > 0.1:
                st.warning(f"**Overfitting detected.** Gap between train ({lc['train_mean'][-1]:.3f}) and CV ({lc['cv_mean'][-1]:.3f}) is {gap:.3f}. Try more data or simpler model.")
            elif lc["cv_mean"][-1] < 0.6:
                st.info("**Underfitting.** Both scores are low. Try a more complex model or more features.")
            else:
                st.success(f"**Good fit.** Train ({lc['train_mean'][-1]:.3f}) and CV ({lc['cv_mean'][-1]:.3f}) are close.")

    with curve_col2:
        st.write("**Validation curve**")
        param_name = st.selectbox("Hyperparameter", list(param_grid.keys()), key="vc_param")
        param_range = param_grid[param_name]

        if st.button("Compute validation curve", key="vc_btn"):
            with st.spinner("Computing..."):
                vc = modeling.compute_validation_curve(tuned, X_for_curves, y_for_curves,
                                                       param_name=param_name, param_range=param_range,
                                                       cv=cv_folds, scoring=scoring)

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#181b22')
            ax.set_facecolor('#1e2129')

            ax.plot(param_range, vc["train_mean"], "o-", color="#6b8aed", label="Training score")
            ax.fill_between(param_range, vc["train_mean"] - vc["train_std"],
                            vc["train_mean"] + vc["train_std"], alpha=0.15, color="#6b8aed")
            ax.plot(param_range, vc["cv_mean"], "o-", color="#f87171", label="CV score")
            ax.fill_between(param_range, vc["cv_mean"] - vc["cv_std"],
                            vc["cv_mean"] + vc["cv_std"], alpha=0.15, color="#f87171")

            best_idx = np.argmax(vc["cv_mean"])
            ax.axvline(x=param_range[best_idx], color="#4ade80", linestyle="--", linewidth=1.5, label=f"Best: {param_range[best_idx]}")

            ax.set_xlabel(param_name, color="#dfe2e8", fontweight="500")
            ax.set_ylabel("Score", color="#dfe2e8", fontweight="500")
            ax.set_title(f"Validation curve — {param_name}", color="#dfe2e8", fontweight="500", pad=15)
            ax.tick_params(colors="#dfe2e8")
            ax.legend(facecolor="#1e2129", edgecolor="#282c34", labelcolor="#dfe2e8")
            for spine in ax.spines.values():
                spine.set_color("#282c34")
            ax.grid(True, alpha=0.1, color="#282c34")
            plt.tight_layout()
            st.pyplot(fig, width="stretch")

            st.info(f"**Best value:** `{param_range[best_idx]}` (score: {vc['cv_mean'][best_idx]:.3f})")


def _tab_configure_and_train():
    st.subheader("Model configuration")

    col1, col2 = st.columns(2)

    with col1:
        target_column = st.selectbox("Target column", st.session_state.df.columns, key="target_col")

    target_data = st.session_state.df[target_column]
    is_numeric = pd.api.types.is_numeric_dtype(target_data)
    is_categorical = pd.api.types.is_categorical_dtype(target_data) or target_data.dtype == 'object'
    unique_values = target_data.nunique()

    if is_categorical or (is_numeric and unique_values <= 10):
        recommended_mode = "Classification"
    else:
        recommended_mode = "Regression"

    with col2:
        mode = st.selectbox("Mode", ["Classification", "Regression"],
                           index=0 if recommended_mode == "Classification" else 1,
                           key="mode",
                           help=f"Auto-detected: {recommended_mode} (based on target column)")

    if mode == "Classification" and is_numeric and unique_values > 10:
        st.warning(f"Classification selected but target has {unique_values} unique continuous values. Consider using Regression instead.")
    elif mode == "Regression" and is_categorical:
        st.warning("Regression selected but target is categorical. Consider using Classification instead.")

    st.subheader("Feature selection")
    available_features = [col for col in st.session_state.df.columns if col != target_column]
    selected_features = st.multiselect("Select features", available_features,
                                       default=available_features[:min(3, len(available_features))], key="features")

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    st.subheader("Model selection")
    models = config.CLASSIFICATION_MODELS if mode == "Classification" else config.REGRESSION_MODELS
    selected_models = st.multiselect("Select models", list(models.keys()),
                                     default=[list(models.keys())[0]], key="models")

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    if st.button("Train models", width="stretch", key="train_btn"):
        X = st.session_state.df[selected_features]
        y = st.session_state.df[target_column]

        is_valid, message = core.validate_data_for_modeling(X, y)
        if not is_valid:
            st.error(message)
            return

        unique_y = y.nunique()
        if mode == "Classification" and unique_y > 10:
            st.error(f"Classification mode requires discrete target values, but found {unique_y} unique continuous values. Please select Regression mode instead.")
            return

        if mode == "Regression" and pd.api.types.is_categorical_dtype(y):
            st.error("Regression mode requires numeric target values, but found categorical data. Please select Classification mode instead.")
            return

        scoring = "accuracy" if mode == "Classification" else "r2"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TRAIN_TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE
        )

        st.subheader("Training results")
        results_rows = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((idx + 1) / len(selected_models))

            pipe = modeling.build_pipeline(model_name)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            train_score = pipe.score(X_train, y_train)
            test_score = pipe.score(X_test, y_test)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=config.CROSS_VAL_FOLDS, scoring=scoring)

            results_rows.append({
                "Model": model_name,
                "Train Score": train_score,
                "Test Score": test_score,
                "CV Mean": cv_scores.mean(),
                "CV Std": cv_scores.std(),
            })

            is_binary = mode == "Classification" and unique_y == 2
            y_pred_proba = None
            if is_binary and hasattr(pipe, "predict_proba"):
                y_pred_proba = pipe.predict_proba(X_test)[:, 1]

            st.session_state.trained_models[model_name] = {
                "pipeline": pipe,
                "model": pipe,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "cv_scores": cv_scores,
                "mode": mode,
                "is_binary": is_binary,
            }

        progress_bar.empty()
        status_text.empty()

        results_df = pd.DataFrame(results_rows)
        st.dataframe(results_df, width="stretch")

        best_idx = results_df["Test Score"].idxmax()
        best_model = results_df.loc[best_idx, "Model"]
        best_score = results_df.loc[best_idx, "Test Score"]
        st.success(f"Best model: **{best_model}** (Score: {best_score:.4f})")

        best = st.session_state.trained_models[best_model]

        st.subheader("Model diagnostics")

        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            if mode == "Classification":
                st.write("**Confusion matrix (best model)**")
                cm = confusion_matrix(best["y_test"], best["y_pred"])
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#181b22')
                ax.set_facecolor('#1e2129')

                heatmap = sns.heatmap(cm, annot=False, cmap='Blues', ax=ax,
                                      cbar_kws={'label': 'Count'},
                                      linewidths=0.5, linecolor='#181b22')

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        count = cm[i, j]
                        pct = cm_normalized[i, j] * 100
                        ax.text(j + 0.5, i + 0.5, f'{int(count)}\n({pct:.1f}%)',
                                ha="center", va="center",
                                color='#dfe2e8' if count < cm.max()/2 else '#111318',
                                fontweight='600', fontsize=11)

                ax.set_xlabel('Predicted', color='#dfe2e8', fontweight='500', fontsize=12)
                ax.set_ylabel('Actual', color='#dfe2e8', fontweight='500', fontsize=12)
                ax.set_title('Confusion matrix', color='#dfe2e8', fontweight='500', fontsize=13, pad=15)
                ax.tick_params(colors='#dfe2e8')
                cbar = heatmap.collections[0].colorbar
                if cbar:
                    cbar.set_label('Count', color='#dfe2e8')
                    cbar.ax.tick_params(colors='#dfe2e8')
                st.pyplot(fig, width="stretch")

        with diag_col2:
            if mode == "Regression":
                st.write("**Residuals plot (best model)**")
                residuals = best["y_test"] - best["y_pred"]
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#181b22')
                ax.set_facecolor('#1e2129')
                ax.scatter(best["y_pred"], residuals, alpha=0.6, color='#6b8aed', s=50, edgecolors='#282c34')
                ax.axhline(y=0, color='#f87171', linestyle='--', linewidth=2)
                ax.set_xlabel('Predicted values', color='#dfe2e8', fontweight='500')
                ax.set_ylabel('Residuals', color='#dfe2e8', fontweight='500')
                ax.set_title('Residuals plot', color='#dfe2e8', fontweight='500', pad=15)
                ax.tick_params(colors='#dfe2e8')
                for spine in ax.spines.values():
                    spine.set_color('#282c34')
                st.pyplot(fig, width="stretch")
            elif mode == "Classification":
                if best.get("is_binary") and best["y_pred_proba"] is not None:
                    st.write("**ROC curve (best model)**")
                    roc_fig, roc_auc = plot_roc_curve(best["y_test"], best["y_pred_proba"], best_model)
                    st.pyplot(roc_fig, width="stretch")
                else:
                    st.info("ROC curve is available only for binary classification with probability outputs.")

        st.subheader("Sample predictions")
        sample_predictions = pd.DataFrame({
            "Actual": best["y_test"].values[:10],
            "Predicted": best["y_pred"][:10]
        })
        st.dataframe(sample_predictions, width="stretch")

        st.subheader("Export report")
        profile, col_profile = generate_data_profile(st.session_state.df)
        stat_summary = core.get_statistical_summary(st.session_state.df)

        html_report = generate_html_report(st.session_state.df, profile, stat_summary,
                                           trained_models=st.session_state.trained_models, mode=mode)

        st.download_button(
            label="Download HTML report",
            data=html_report,
            file_name=f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            width="stretch"
        )

        st.subheader("Export model")
        model_bytes = pickle.dumps(best["model"])
        st.download_button(
            label="Download best model (pickle)",
            data=model_bytes,
            file_name=f"{best_model.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream",
            width="stretch"
        )


def page_model_training():
    st.markdown(f"""
    <div class="app-header">
        <div style="color: var(--text-tertiary); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">Step 4</div>
        <h1 class="app-title">ML Lab</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="help-box">
        {config.HELP_TEXTS[4]}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        empty_state("No data available", "Please upload or load a dataset first.")
        return

    tab_train, tab_tune, tab_interpret, tab_diagnose = st.tabs(
        ["Configure & Train", "Tune", "Interpret", "Diagnose"]
    )

    with tab_train:
        _tab_configure_and_train()

    with tab_tune:
        _tab_tune()

    with tab_interpret:
        _tab_interpret()

# ============================================================================
# MAIN APP
# ============================================================================

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=config.STEP_NAMES,
        icons=None,
        default_index=st.session_state.current_step,
        orientation="vertical",
        key="main_menu",
        styles={
            "container": {
                "padding": "0 !important",
                "background-color": "#181b22",
                "border-radius": "0px"
            },
            "icon": {
                "color": "#6b8aed",
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "0.85rem",
                "text-align": "left",
                "margin": "0.2rem 0",
                "color": "#8b9099",
                "border-radius": "6px",
                "padding": "0.55rem 1rem",
                "font-weight": "400",
                "border": "1px solid transparent"
            },
            "nav-link-selected": {
                "background": "#252830 !important",
                "color": "#dfe2e8 !important",
                "border-radius": "6px",
                "font-weight": "500",
                "border": "1px solid #282c34"
            }
        }
    )
    
    for i, name in enumerate(config.STEP_NAMES):
        if selected == name:
            st.session_state.current_step = i
            break

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
