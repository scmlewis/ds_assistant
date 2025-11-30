# Testing Guide - Advanced Features

## Overview
Complete testing checklist and guide for all 4 advanced features.

---

## Test Environment Setup

### Prerequisites
```bash
# Install dependencies
python -m pip install streamlit pandas numpy scikit-learn scipy matplotlib seaborn

# Optional (for full SHAP support)
python -m pip install shap
```

### Start the App
```bash
python -m streamlit run app.py
```

### Test Data Available
- Iris dataset (built-in, for classification)
- Diabetes dataset (built-in, for regression)
- Any CSV upload

---

## Feature 1: Hyperparameter Tuning

### Test Case 1.1: Basic Hyperparameter Tuning
**Data**: Iris dataset (classification)

**Steps**:
1. Navigate to Model Training page
2. Click "Load sample dataset" → Select "Iris"
3. Set target column to "target"
4. Select "Classification" mode
5. Select "Logistic Regression" and "Random Forest"
6. Click "🚀 Train Models"
7. Wait for training to complete
8. Scroll to "⚡ Hyperparameter Optimization"
9. Select "Logistic Regression" in dropdown
10. Click "🔧 Start Hyperparameter Tuning"
11. Wait for tuning to complete

**Expected Results**:
- ✅ Baseline accuracy displayed
- ✅ Optimized accuracy displayed
- ✅ Improvement percentage shown (positive if tuning helped)
- ✅ "Best Parameters Found" table shows parameter values
- ✅ "💾 Save Tuned Model" button available

**Verification**:
- Optimized score ≥ Baseline score (sometimes equal if already optimal)
- Best parameters are valid (C between 0.001-100, solver is valid, etc)
- No errors in console

### Test Case 1.2: Hyperparameter Tuning with Regression
**Data**: Diabetes dataset (regression)

**Steps**:
1. Load diabetes dataset
2. Set target column to "target"
3. Select "Regression" mode
4. Select "Random Forest" model
5. Train models
6. Scroll to "⚡ Hyperparameter Optimization"
7. Select "Random Forest"
8. Click "🔧 Start Hyperparameter Tuning"

**Expected Results**:
- ✅ Baseline R² score displayed
- ✅ Optimized R² score displayed
- ✅ Best parameters for Random Forest displayed
- ✅ Performance improvement shown

### Test Case 1.3: Save Tuned Model
**Steps**:
1. Complete tuning (Test Case 1.1 or 1.2)
2. Click "💾 Save Tuned Model"

**Expected Results**:
- ✅ Success message: "✓ Tuned model saved!"
- ✅ Tuned model available in feature importance section
- ✅ Can download tuned model as pickle

---

## Feature 2: Feature Engineering

### Test Case 2.1: Polynomial Features
**Data**: Iris or custom numeric data

**Steps**:
1. Load dataset with numeric data
2. Go to Clean Data page
3. Scroll to "⚡ Feature Engineering"
4. Select "Polynomial Features"
5. Set degree to 2
6. Select "sepal length" and "sepal width" columns
7. Click "✨ Generate Polynomial Features"

**Expected Results**:
- ✅ Success message: "✅ Created X polynomial features!"
- ✅ Message shows new feature names (x1^2, x2^2, x1*x2, etc)
- ✅ Dataset now includes new columns
- ✅ Can proceed to model training with engineered features

**Verification**:
- For degree 2 with 2 features: Should create 3 new features
- Values should be numeric (squares, products)
- Original features still present

### Test Case 2.2: Interaction Terms
**Steps**:
1. Go to Clean Data page
2. Scroll to "⚡ Feature Engineering"
3. Select "Interaction Terms"
4. Select 3+ numeric columns (e.g., sepal length, sepal width, petal length)
5. Click "✨ Generate Interactions"

**Expected Results**:
- ✅ Created C(n,2) new features where n = number of selected columns
- ✅ Feature names show interactions (e.g., "sepal_length_x_sepal_width")
- ✅ Values are products of original columns
- ✅ Dataset updated with new columns

**Verification**:
- For 3 columns: Should create 3 interactions (C(3,2) = 3)
- For 5 columns: Should create 10 interactions (C(5,2) = 10)
- Values equal to product of original features

### Test Case 2.3: Binning
**Steps**:
1. Go to Clean Data page
2. Scroll to "⚡ Feature Engineering"
3. Select "Binning"
4. Select numeric columns
5. Click "✨ Generate Binned Features"

**Expected Results**:
- ✅ Created 1 binned feature per selected column
- ✅ Feature names show "_binned" suffix
- ✅ Values are 0-4 (quintile bins)
- ✅ Dataset updated

**Verification**:
- Values should be integers: 0, 1, 2, 3, 4
- Each value should appear roughly equally (quintiles)
- Original columns still present

### Test Case 2.4: Compare Model Performance
**Steps**:
1. Train model on original data
2. Note baseline accuracy
3. Apply feature engineering (polynomial or interactions)
4. Train model on engineered features
5. Compare accuracy

**Expected Results**:
- ✅ Model with engineered features performs same or better
- ✅ Performance improvement on non-linear data
- ✅ No errors during training

---

## Feature 3: SHAP Model Explainability

### Test Case 3.1: SHAP Summary Plot (if installed)
**Data**: Any classification dataset

**Steps**:
1. Train models on classification data
2. Scroll to "🔬 Model Explainability (SHAP)"
3. Click "📊 Generate SHAP Summary"

**Expected Results**:
- ✅ SHAP summary plot appears (bar chart)
- ✅ Features ranked by importance
- ✅ Success message: "✓ SHAP summary plot generated!"

**Verification**:
- Plot shows feature importance scores
- Top features should align with domain knowledge
- No errors in console

### Test Case 3.2: Individual Prediction Explanation
**Steps**:
1. Train models
2. Scroll to "🔬 Model Explainability (SHAP)"
3. Select sample index (e.g., 0)
4. Click "📈 Explain Prediction"

**Expected Results**:
- ✅ Success message: "✓ Explanation Generated!"
- ✅ Sample feature values shown in table
- ✅ SHAP impact scores shown (Feature → Value → SHAP Impact)
- ✅ Features sorted by impact magnitude

**Verification**:
- Feature values match actual data
- SHAP values are numeric (positive = increases prediction, negative = decreases)
- No NaN values in output

### Test Case 3.3: SHAP Not Installed (Graceful Degradation)
**Steps**:
1. If SHAP not installed, go to Model Training page
2. Scroll to "🔬 Model Explainability (SHAP)"

**Expected Results**:
- ✅ Warning message displayed: "⚠️ SHAP is not installed"
- ✅ Install instructions shown: "pip install shap"
- ✅ App continues to work without SHAP
- ✅ No errors or crashes

---

## Feature 4: Statistical Significance Testing

### Test Case 4.1: View Correlation with Significance
**Data**: Any dataset with numeric columns

**Steps**:
1. Go to Visualize Data page
2. View "🔗 Correlation Matrix with Statistical Tests"

**Expected Results**:
- ✅ Correlation heatmap displayed
- ✅ Gold stars (*) visible on significant correlations
- ✅ Message: "Legend: Gold stars (*) = statistically significant (p < 0.05)"

**Verification**:
- Heatmap shows values from -1 to +1
- Gold stars appear only where correlations are significant
- Diagonal (self-correlations) are all 1.0 with no stars

### Test Case 4.2: View Detailed Statistics
**Steps**:
1. Go to Visualize Data page
2. View correlation section
3. Expand "📊 Detailed Correlation Statistics"

**Expected Results**:
- ✅ Two tables appear: Correlation Coefficients and P-Values
- ✅ Correlation values between -1 and 1
- ✅ P-values between 0 and 1
- ✅ Tables match heatmap visualization

**Verification**:
- Correlation matrix is symmetric
- P-values < 0.05 correspond to gold stars on heatmap
- Diagonal correlations are all 1.0

### Test Case 4.3: Run Hypothesis Test
**Steps**:
1. Go to Visualize Data page
2. Expand "🔬 Hypothesis Testing"
3. Select Variable 1 (e.g., "sepal length")
4. Select Variable 2 (e.g., "sepal width")
5. Select Test Type: "pearson"
6. Click "🔍 Run Test"

**Expected Results**:
- ✅ Test results table displayed
- ✅ Contains: Correlation, P-value, Significant (Yes/No)
- ✅ Interpretation message:
  - If p < 0.05: "✓ Statistically significant"
  - If p ≥ 0.05: "ℹ️ Not statistically significant"

**Verification**:
- P-value matches correlation matrix
- Significance matches gold star status
- Correlation value between -1 and 1

### Test Case 4.4: Spearman Test
**Steps**:
1. Go to Visualize Data page
2. Expand "🔬 Hypothesis Testing"
3. Select 2 variables
4. Select Test Type: "spearman"
5. Click "🔍 Run Test"

**Expected Results**:
- ✅ Spearman rank correlation displayed
- ✅ P-value and significance shown
- ✅ Results may differ from Pearson (more robust to outliers)

### Test Case 4.5: T-Test
**Steps**:
1. Go to Visualize Data page
2. Expand "🔬 Hypothesis Testing"
3. Select 2 different variables
4. Select Test Type: "ttest"
5. Click "🔍 Run Test"

**Expected Results**:
- ✅ T-statistic and p-value displayed
- ✅ Sample sizes for both variables shown
- ✅ Significance interpretation shown

**Verification**:
- T-statistic is numeric
- P-value between 0 and 1
- Sample sizes match data

---

## Integration Tests

### Test 5.1: Full Workflow - Classification
**Scenario**: Complete classification pipeline with new features

**Steps**:
1. Load Iris dataset
2. Clean data (apply feature engineering - polynomials)
3. Visualize (check correlations & p-values)
4. Train models
5. Explain with SHAP (if installed)
6. Optimize with hyperparameter tuning
7. Export report

**Expected Results**:
- ✅ All steps complete without errors
- ✅ Model performance improves with engineered features
- ✅ SHAP explanations make sense
- ✅ Hyperparameter tuning shows improvement
- ✅ Statistical tests show which features are significant

### Test 5.2: Full Workflow - Regression
**Scenario**: Complete regression pipeline

**Steps**:
1. Load Diabetes dataset
2. Feature engineering (interactions)
3. Visualize correlations
4. Train regression models
5. Optimize hyperparameters
6. Export model

**Expected Results**:
- ✅ All steps work for regression
- ✅ R² scores improve
- ✅ No classification-specific errors (ROC curve, etc)

### Test 5.3: Mixed Feature Engineering
**Steps**:
1. Load dataset
2. Create polynomial features
3. Create interaction terms
4. Create binned features
5. Train model with all engineered features

**Expected Results**:
- ✅ All features coexist without conflicts
- ✅ Model trains successfully
- ✅ Performance improves from combined features

---

## Performance Tests

### Test 6.1: Hyperparameter Tuning Speed
**Data**: Iris (150 samples)

**Steps**:
1. Load Iris
2. Train Logistic Regression
3. Time hyperparameter tuning

**Expected Results**:
- ✅ Completes in < 30 seconds
- ✅ Baseline vs optimized comparison shown

### Test 6.2: SHAP Performance
**Data**: Iris (150 samples)

**Steps**:
1. Load Iris
2. Train Random Forest
3. Time SHAP summary generation
4. Time SHAP force plot

**Expected Results**:
- ✅ Summary completes in < 15 seconds
- ✅ Force plot completes in < 5 seconds

### Test 6.3: Feature Engineering Speed
**Data**: Iris (150 samples, 4 features)

**Steps**:
1. Load Iris
2. Time polynomial feature generation (degree 2)
3. Time interaction generation
4. Time binning

**Expected Results**:
- ✅ All complete in < 1 second
- ✅ No lag in UI

### Test 6.4: Statistical Testing Speed
**Data**: Iris (150 samples, 5 numeric columns)

**Steps**:
1. Load Iris
2. Time correlation + p-value calculation
3. Time single hypothesis test

**Expected Results**:
- ✅ Correlation matrix < 1 second
- ✅ Single test < 100ms

---

## Error Handling Tests

### Test 7.1: Missing Data
**Steps**:
1. Upload CSV with missing values
2. Try feature engineering
3. Try statistical tests

**Expected Results**:
- ✅ Missing values handled gracefully
- ✅ dropna() applied for statistics
- ✅ No crashes

### Test 7.2: Small Dataset
**Steps**:
1. Create CSV with 5-10 samples
2. Try all features

**Expected Results**:
- ✅ Feature engineering works
- ✅ Statistics warnings for small sample size
- ✅ No crashes

### Test 7.3: Invalid Feature Combinations
**Steps**:
1. Clean Data → Feature Engineering
2. Interaction Terms → Select only 1 column
3. Click Generate

**Expected Results**:
- ✅ Warning: "Please select at least 2 columns"
- ✅ Feature not generated
- ✅ App continues working

---

## Compatibility Tests

### Test 8.1: Existing Features Still Work
**Steps**:
1. Verify all 5 pages load without error
2. Test data upload
3. Test data cleaning
4. Test basic visualizations
5. Test model training without new features

**Expected Results**:
- ✅ All existing functionality preserved
- ✅ No regressions
- ✅ Dark theme consistent

### Test 8.2: Session State Preserved
**Steps**:
1. Upload data
2. Clean data
3. Use feature engineering
4. Go back to visualize
5. Go forward to training

**Expected Results**:
- ✅ Data preserved across pages
- ✅ Engineered features available in training
- ✅ Session state consistent

---

## Checklist for Deployment

- [ ] All 4 features tested with sample data
- [ ] Error handling tested (missing data, invalid inputs)
- [ ] Performance acceptable (< 1 minute per feature)
- [ ] SHAP graceful fallback confirmed
- [ ] Documentation complete
- [ ] No console errors
- [ ] Dark theme consistent
- [ ] Integration with existing code verified
- [ ] Export functionality working
- [ ] Models can be saved and loaded

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SHAP module not found | pip install shap (optional) |
| Hyperparameter tuning very slow | Normal, may take 1-2 minutes on large data |
| Feature engineering creates too many columns | Use lower polynomial degree |
| P-value doesn't match expected | Check sample selection, handle missing values |
| Models crash with > 1000 samples | Try reducing sample size for SHAP visualization |

---

## Success Criteria

✅ **Hyperparameter Tuning**: 
- GridSearchCV runs successfully
- Best parameters displayed
- Performance comparison shown
- Improvement is realistic (0-20%)

✅ **Feature Engineering**:
- New features created without errors
- Feature names are clear
- Can train models with new features
- Performance improves or stays same

✅ **SHAP Explainability**:
- Summary plots generate (if installed)
- Individual explanations work
- Graceful fallback if not installed
- Values are interpretable

✅ **Statistical Testing**:
- P-values calculated correctly
- Significant correlations marked
- Hypothesis tests match calculations
- Interpretations are accurate

---

## Final Sign-Off

After completing all tests:
- [ ] Feature 1: Hyperparameter Tuning - PASS ✓
- [ ] Feature 2: Feature Engineering - PASS ✓
- [ ] Feature 3: SHAP Explainability - PASS ✓
- [ ] Feature 4: Statistical Testing - PASS ✓
- [ ] Integration Tests - PASS ✓
- [ ] Performance Tests - PASS ✓
- [ ] Error Handling - PASS ✓
- [ ] Compatibility - PASS ✓

**Status**: Ready for Production ✅

