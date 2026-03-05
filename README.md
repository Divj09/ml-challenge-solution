# IEEE ML Challenge 2026 - Qualifier Submission

## 🏆 Competition Results
- **Event:** alrIEEEna'26 ML Challenge Qualifiers
- **Model:** XGBoost Classifier
- **Cross-Validation F1 Score:** 0.9821 ± 0.0009
- **Submission Date:** March 2026

---

## 📋 Submission Contents

| File | Description | Status |
|------|-------------|--------|
| `FINAL.csv` | Competition predictions (10,944 rows) | ✅ Ready |
| `solution_notebook.ipynb` | Complete solution with detailed explanations | ✅ Ready |
| `train_model.py` | Model training script | ✅ Included |
| `predict.py` | Prediction generation script | ✅ Included |
| `exploratory_analysis.py` | Data exploration | ✅ Included |
| `requirements.txt` | Python dependencies | ✅ Included |

---

## 📊 Dataset Information

- **Training Samples:** 43,776
- **Test Samples:** 10,944
- **Features:** 47 numerical features
- **Target:** Binary classification (Class 0: 60.46%, Class 1: 39.54%)
- **Evaluation Metric:** Weighted F1 Score

---

## 🔍 Solution Approach

### **See `solution_notebook.ipynb` for complete details**

The notebook includes:
- ✅ **Problem Understanding & Intuition**
- ✅ **Exploratory Data Analysis**
- ✅ **Statistical Tests** (Pearson Correlation Analysis)
- ✅ **Feature Selection Rationale** (Why all 47 features were used)
- ✅ **Model Comparison** (XGBoost, LightGBM, Random Forest, Gradient Boosting)
- ✅ **Hyperparameter Justification** (Why n_estimators=300, learning_rate=0.05, etc.)
- ✅ **Cross-Validation Results** (5-Fold Stratified CV)
- ✅ **Feature Importance Analysis**
- ✅ **Final Predictions**

---

## 🎯 Model Performance

### Cross-Validation Results (5-Fold Stratified)

| Model | F1 Score | Std Dev | Status |
|-------|----------|---------|--------|
| **XGBoost** | **0.9821** | **0.0009** | ✅ **SELECTED** |
| LightGBM | 0.9757 | 0.0017 | - |
| Random Forest | 0.9555 | 0.0036 | - |
| Gradient Boosting | 0.9524 | 0.0031 | - |

**Winner:** XGBoost achieved highest F1 score with lowest variance

---

## 🛠️ Technical Details

### Preprocessing Pipeline
1. Remove ID column
2. Feature scaling (StandardScaler)
3. No missing values (verified)
4. All features numerical (no encoding needed)

### XGBoost Hyperparameters
```python
n_estimators = 300          # Number of boosting rounds
learning_rate = 0.05        # Lower for better generalization
max_depth = 7               # Controls tree complexity
subsample = 0.8             # Row sampling for regularization
colsample_bytree = 0.8      # Feature sampling
random_state = 42           # Reproducibility
