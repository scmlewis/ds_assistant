# Technical Implementation Details: Advanced Features (Tier 3)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Clean Data   │ Visualize    │ Model Train  │   Reports      │
│ Page         │ Data Page    │  Page        │   Export       │
│              │              │              │                │
│ • Feature    │ • Corr +     │ • Hyper      │ • HTML Report  │
│   Eng        │   P-values   │   Tuning     │ • Model Pickle │
│              │ • Hypo Test  │ • SHAP Exp   │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│            ADVANCED ML FUNCTIONS (NEW)                      │
├─────────────────────────────────────────────────────────────┤
│ • tune_hyperparameters()      [GridSearchCV wrapper]        │
│ • engineer_features()          [Poly, Interaction, Binning] │
│ • plot_shap_summary()          [SHAP TreeExplainer]         │
│ • plot_shap_force()            [SHAP Force for instance]    │
│ • calculate_correlation_significance() [Pearson p-vals]    │
│ • perform_hypothesis_test()    [Pearson, Spearman, T-test] │
│ • plot_correlation_with_significance() [Heatmap + stars]   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         ML LIBRARIES & DEPENDENCIES                         │
├─────────────────────────────────────────────────────────────┤
│ • scikit-learn: GridSearchCV, PolynomialFeatures            │
│ • scipy.stats: pearsonr, spearmanr, ttest_ind              │
│ • shap: TreeExplainer, KernelExplainer (optional)           │
│ • numpy, pandas: Data manipulation                          │
│ • matplotlib, seaborn: Visualization                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Hyperparameter Tuning Implementation

### Function Signature
```python
def tune_hyperparameters(X_train, y_train, X_test, y_test, 
                         model_name, mode):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Returns:
        - best_model: Trained model with optimal parameters
        - best_params: Dict of parameter names → values
        - comparison: Dict with metrics (baseline, optimized, improvement)
    """
```

### Parameter Grids

#### Logistic Regression
```python
'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Regularization strength
'solver': ['lbfgs', 'liblinear']      # Optimization algorithm
'max_iter': [200, 500, 1000]          # Iteration limit
```
**Total combinations**: 6 × 2 × 3 = 36 models tested per fold

#### Random Forest (Classification & Regression)
```python
'n_estimators': [50, 100, 200]        # Number of trees
'max_depth': [5, 10, 15, None]        # Tree depth
'min_samples_split': [2, 5, 10]       # Min samples to split
'min_samples_leaf': [1, 2, 4]         # Min samples per leaf
```
**Total combinations**: 3 × 4 × 3 × 3 = 108 models per fold

#### Decision Tree
```python
'max_depth': [5, 10, 15, 20, None]    # Tree depth
'min_samples_split': [2, 5, 10]       # Min samples to split
'min_samples_leaf': [1, 2, 4]         # Min samples per leaf
'criterion': ['gini', 'entropy']      # Split criterion (classification)
```
**Total combinations**: 5 × 3 × 3 × 2 = 90 models per fold

### GridSearchCV Configuration
```python
GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,                              # 5-fold cross-validation
    scoring=metric,                    # 'accuracy' or 'r2'
    n_jobs=-1,                         # Use all CPU cores (parallel)
    verbose=0                          # Silent execution
)
```

### Performance Comparison Metrics
```python
comparison = {
    'Baseline': accuracy_score(y_test, baseline_predictions),
    'Optimized': accuracy_score(y_test, tuned_predictions),
    'Improvement': (optimized - baseline),
    'Improvement %': ((optimized - baseline) / baseline) * 100,
    'Metric': 'Accuracy' or 'R² Score'
}
```

### Integration in UI
```python
# User selects model → tuning starts → results displayed
tune_model = st.selectbox("Select Model to Tune", selected_models)
if st.button("🔧 Start Hyperparameter Tuning"):
    best_tuned_model, best_params, comparison = tune_hyperparameters(
        X_train, y_train, X_test, y_test, tune_model, mode
    )
    
    # Display comparison
    comparison_df = pd.DataFrame([comparison])
    st.dataframe(comparison_df)
    
    # Display best parameters
    params_display = pd.DataFrame(list(best_params.items()), 
                                 columns=['Parameter', 'Value'])
    st.dataframe(params_display)
```

---

## Feature 2: Feature Engineering Implementation

### Function Signature
```python
def engineer_features(df, numeric_cols, feature_type='polynomial', 
                      degree=2, interaction_cols=None):
    """
    Create engineered features from numeric columns.
    
    Args:
        feature_type: 'polynomial', 'interaction', or 'binning'
        degree: Degree for polynomial features (2-5)
        interaction_cols: Columns to interact
    
    Returns:
        - df_engineered: DataFrame with new features
        - new_features: List of created feature names
    """
```

### Implementation Details

#### A. Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=degree, include_bias=False)
X = df[numeric_cols]
X_poly = poly.fit_transform(X)

# Get feature names: ['x1', 'x2', 'x1^2', 'x1*x2', 'x2^2'] for degree=2
feature_names = poly.get_feature_names_out(numeric_cols)

# Create DataFrame with polynomial features
df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)

# Add new features (exclude originals)
new_features = [col for col in df_poly.columns if col not in numeric_cols]
for col in new_features:
    df_engineered[col] = df_poly[col]
```

**Degree 2 Example** (features: age, income):
- New: `age^2`, `income^2`, `age × income`
- Total: 3 new features

**Degree 3 Example** (same features):
- New: `age^2`, `age^3`, `income^2`, `income^3`, `age × income`, `age^2 × income`, `age × income^2`
- Total: 7 new features

#### B. Interaction Terms
```python
# Create all pairwise interactions
new_features = []
for i, col1 in enumerate(interaction_cols):
    for col2 in interaction_cols[i+1:]:
        new_col = f'{col1}_x_{col2}'
        df_engineered[new_col] = df[col1] * df[col2]
        new_features.append(new_col)
```

**Example** (features: age, income, education_years):
- New: `age_x_income`, `age_x_education_years`, `income_x_education_years`
- Total: 3 new features (C(n,2) combinations)

#### C. Binning/Discretization
```python
# Convert continuous to categorical quintiles
for col in numeric_cols:
    new_col = f'{col}_binned'
    df_engineered[new_col] = pd.qcut(df[col], q=5, labels=False, 
                                     duplicates='drop')
```

**Example** (feature: income):
- Values: 0, 1, 2, 3, 4 (quintile bins)
- Represents: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
- Captures: Income thresholds/brackets

---

## Feature 3: SHAP Explainability Implementation

### Architecture

```
Model + Data
    ↓
[TreeExplainer/KernelExplainer]
    ↓
SHAP Values Matrix [samples × features]
    ↓
Visualization/Analysis
```

### Function Signatures

#### A. Summary Plot
```python
def plot_shap_summary(model, X_train, X_test, model_type='tree'):
    """Generate SHAP summary plot (feature importance).
    
    For tree models:
        TreeExplainer(model) → SHAP values → matplotlib figure
    
    For others:
        KernelExplainer(model.predict, X_train_sample) → figure
    """
```

#### B. Force Plot
```python
def plot_shap_force(model, X_train, X_test, instance_idx=0, 
                    model_type='tree'):
    """Generate SHAP values for single prediction.
    
    Returns dict with:
        - shap_values: Feature contributions [array]
        - features: Feature values [Series]
        - base_value: Model's average prediction [float]
    """
```

### SHAP Explainer Selection

```python
if model_type == 'tree':
    # Fast, exact SHAP values for tree models
    explainer = shap.TreeExplainer(model)
else:
    # General explainer, slower but works for any model
    explainer = shap.KernelExplainer(
        model.predict, 
        X_train.sample(min(50, len(X_train)))  # Background data
    )
```

### Handling Multi-Class Predictions

```python
shap_values = explainer.shap_values(X_test)

# Tree models return list for multi-class:
# shap_values = [class_0_shap, class_1_shap, class_2_shap, ...]

if isinstance(shap_values, list):
    shap_values = shap_values[0]  # Use first class for display
```

### UI Integration

```python
# Summary Plot
if st.button("📊 Generate SHAP Summary"):
    X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    shap_fig = plot_shap_summary(best_model_obj, X_train_sample, 
                                X_test_sample, model_type)
    st.pyplot(shap_fig)

# Force Plot
pred_idx = st.number_input("Select sample index", 0, len(X_test)-1)
if st.button("📈 Explain Prediction"):
    shap_explanation = plot_shap_force(best_model_obj, X_train_sample, 
                                       X_test, pred_idx, model_type)
    
    # Display results
    st.dataframe(pd.DataFrame({
        'Feature': X_test.columns,
        'Value': X_test.iloc[pred_idx].values,
        'SHAP Impact': shap_explanation['shap_values']
    }).sort_values('SHAP Impact', key=abs, ascending=False))
```

### Error Handling with Graceful Degradation

```python
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# In UI:
if not HAS_SHAP:
    st.warning("⚠️ SHAP is not installed. Install with:\npip install shap")
else:
    # Show SHAP features
    ...
```

---

## Feature 4: Statistical Significance Testing Implementation

### Function Signatures

#### A. Correlation with P-values
```python
def calculate_correlation_significance(df, numeric_cols):
    """Calculate Pearson r and p-values for all pairs.
    
    Returns:
        - corr_df: Correlation matrix [n_cols × n_cols]
        - pval_df: P-value matrix [n_cols × n_cols]
    """
```

#### B. Hypothesis Testing
```python
def perform_hypothesis_test(df, col1, col2, test_type='pearson'):
    """Run statistical test on two variables.
    
    test_type:
        - 'pearson': Pearson correlation r
        - 'spearman': Spearman rank correlation
        - 'ttest': Independent samples t-test
    
    Returns:
        pd.Series with: Test, Statistic, P-value, Significant, Sample Size
    """
```

#### C. Visualization
```python
def plot_correlation_with_significance(df, numeric_cols):
    """Create correlation heatmap with significance markers.
    
    Gold stars (*) indicate p < 0.05
    """
```

### Statistical Tests Implemented

#### Pearson Correlation
```python
from scipy.stats import pearsonr

corr, pval = pearsonr(df[col1].dropna(), df[col2].dropna())
# corr: r-value (-1 to 1)
# pval: probability under null hypothesis
```

**Interpretation**:
- r = 0: No relationship
- r = 0.5: Moderate positive relationship
- r = 1: Perfect positive relationship
- p < 0.05: Statistically significant

#### Spearman Correlation
```python
from scipy.stats import spearmanr

corr, pval = spearmanr(df[col1].dropna(), df[col2].dropna())
# Same structure as Pearson
# Better for: monotonic, ordinal, or non-normal data
```

#### Independent Samples T-Test
```python
from scipy.stats import ttest_ind

stat, pval = ttest_ind(df[col1].dropna(), df[col2].dropna())
# stat: t-statistic
# pval: probability under null hypothesis (means are equal)
```

**Interpretation**:
- t > 0: col1 mean > col2 mean
- t < 0: col1 mean < col2 mean
- p < 0.05: Significant difference

### Visualization Details

```python
# Correlation heatmap
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=0.5, ax=ax)

# Significance markers (gold stars)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        if pval_df.iloc[i, j] < 0.05:
            ax.text(j+0.5, i+0.7, '*',  # Gold star
                   ha='center', va='center',
                   color='#FFD700', fontsize=16, fontweight='bold')
```

### Integration in UI

```python
# View correlations with p-values
fig, corr_df, pval_df = plot_correlation_with_significance(
    st.session_state.df, numeric_cols
)
st.pyplot(fig)
st.markdown("Legend: Gold stars (*) = statistically significant (p < 0.05)")

# Detailed statistics
with st.expander("📊 Detailed Correlation Statistics"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("Correlation Coefficients (r):")
        st.dataframe(corr_df)
    with col2:
        st.write("P-Values:")
        st.dataframe(pval_df)

# Hypothesis testing
with st.expander("🔬 Hypothesis Testing"):
    var1 = st.selectbox("Variable 1", numeric_cols)
    var2 = st.selectbox("Variable 2", numeric_cols)
    test_type = st.selectbox("Test Type", ["pearson", "spearman", "ttest"])
    
    if st.button("🔍 Run Test"):
        result = perform_hypothesis_test(df, var1, var2, test_type)
        st.dataframe(result.to_frame().T)
        
        if result.get('P-value', 1) < 0.05:
            st.success("✓ Statistically significant (p < 0.05)")
        else:
            st.info("ℹ️ Not statistically significant (p ≥ 0.05)")
```

---

## Performance Considerations

### Hyperparameter Tuning
- **Time**: ~5-60 seconds depending on model/data size
- **Optimization**: GridSearchCV with `n_jobs=-1` (parallel execution)
- **Bottleneck**: Model fitting (especially for large datasets)

### Feature Engineering
- **Polynomial**: O(n × m^d) where n=samples, m=features, d=degree
  - Degree 2: 3 features → 9 new features
  - Degree 3: 3 features → 28 new features
- **Interactions**: O(C(m,2)) = m*(m-1)/2
  - 5 features → 10 interactions
- **Binning**: O(n) - fast, memory efficient

### SHAP Calculation
- **TreeExplainer**: O(n × T) where T=number of trees (fast)
- **KernelExplainer**: O(n × k × m) where k=background samples (slow)
- **Optimization**: Sampling to max 100 test samples for speed

### Statistical Testing
- **Correlation**: O(n) for all pairs → O(n × m²)
- **Hypothesis Tests**: O(n) per test
- **Total time**: < 100ms for typical datasets

---

## Error Handling & Robustness

### Graceful SHAP Degradation
```python
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Checks before SHAP usage:
if not HAS_SHAP:
    st.warning("Install SHAP: pip install shap")
    return  # Don't crash app
```

### Data Validation
```python
# Hyperparameter tuning
if best_model is None:
    st.error("Could not tune this model type")
    return

# Feature engineering
if len(interaction_cols) < 2:
    st.warning("Select at least 2 columns for interactions")
    return

# SHAP
try:
    shap_values = explainer.shap_values(X_test)
except Exception as e:
    st.error(f"SHAP error: {str(e)}")
    return

# Statistical tests
valid_data_1 = df[col1].dropna()
valid_data_2 = df[col2].dropna()
if len(valid_data_1) < 2 or len(valid_data_2) < 2:
    st.warning("Insufficient data for test")
    return
```

### Sample Size Handling
```python
# SHAP: Limit to 100 samples for speed
X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42)

# Statistics: Use all available data
valid_data = df[col].dropna()
n = len(valid_data)

# Feature engineering: Handle all data
df_engineered = df.copy()
```

---

## Testing Checklist

- [ ] **Hyperparameter Tuning**: Model improves after tuning
- [ ] **Feature Engineering**: New columns created and visible
- [ ] **SHAP**: Summary plot and force plot generate (if installed)
- [ ] **Statistics**: P-values < 0.05 marked with stars
- [ ] **Error Handling**: App doesn't crash on edge cases
- [ ] **Performance**: Each feature completes in < 1 minute
- [ ] **Integration**: Features accessible from correct pages

---

## Dependencies & Imports

```python
# Core ML
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

# Statistics  
from scipy import stats  # pearsonr, spearmanr, ttest_ind

# Explainability (optional)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Data & visualization (existing)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Configuration & Tuning

### GridSearchCV Parameters (Can Be Modified)
```python
# In tune_hyperparameters():
GridSearchCV(...
    cv=5,           # Change to 3 for faster, 10 for thorough
    n_jobs=-1,      # -1 = all cores, 1 = single core
    verbose=0       # 1 = show progress
)
```

### SHAP Sampling
```python
# In plot_shap_summary():
X_test_sample = X_test.sample(min(100, len(X_test)))  # Max 100
X_train_sample = X_train.sample(min(50, len(X_train)))  # Max 50

# Increase for more accuracy, decrease for speed
```

### Polynomial Degree Range
```python
# In engineer_features UI:
degree = st.slider("Polynomial Degree", 2, 5, 2)  # Min=2, Max=5, Default=2

# Can increase max to 6+, but risk overfitting & huge feature space
```

