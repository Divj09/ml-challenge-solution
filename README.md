# ML Challenge Competition Solution

## 🎯 Competition Results
- **Model**: XGBoost Classifier
- **F1 Score**: 0.9821
- **Cross-validation**: 5-fold Stratified

## 📊 Dataset
- Training samples: 43,776
- Test samples: 10,944
- Features: 47
- Target: Binary Classification (Class 0/1)

## 🛠️ Technology Stack
- Python 3.x
- XGBoost 3.2.0
- LightGBM 4.6.0
- scikit-learn
- pandas, numpy
- imbalanced-learn (SMOTE)

📈 Model Performance
Model	F1 Score	Std Dev
XGBoost	0.9821	0.0009
LightGBM	0.9757	0.0017
RandomForest	0.9555	0.0036
GradientBoosting	0.9524	0.0031

## 📁 Project Structure
ml-challenge-solution/
├── train_model.py # Model training script
├── predict.py # Prediction generation
├── exploratory_analysis.py # Data exploration
├── requirements.txt # Dependencies
├── FINAL.csv # Competition predictions
└── README.md # Documentation

🎯 Features
Automatic target detection
Missing value imputation
Categorical variable encoding
Feature scaling
SMOTE for class balancing (when needed)
5-fold stratified cross-validation
Automated best model selection

📊 Class Distribution
Class 0: 60.4%
Class 1: 39.6%
🏆 Solution Highlights
Clean, well-documented code
Modular design with separate training and prediction scripts
Comprehensive data exploration
Multiple model comparison
Production-ready pipeline

📝 Requirements
See requirements.txt for full list of dependencies.


## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python train_model.py
python train_model.py

📈 Model Performance
Model	F1 Score	Std Dev
XGBoost	0.9821	0.0009
LightGBM	0.9757	0.0017
RandomForest	0.9555	0.0036
GradientBoosting	0.9524	0.0031

