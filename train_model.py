import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from imblearn.over_sampling import SMOTE
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class MLChallengeSolution:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        
    def identify_target(self, train_df, test_df):
        """Identify target column"""
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        diff_cols = train_cols - test_cols
        
        # Remove ID columns
        potential_targets = [col for col in diff_cols 
                           if col.lower() not in ['id', 'index', 'unnamed']]
        
        if len(potential_targets) >= 1:
            self.target_column = potential_targets[0]
            print(f"🎯 Target column: {self.target_column}")
            return self.target_column
        else:
            raise ValueError("Cannot identify target column automatically")
    
    def preprocess_data(self, df, is_train=True):
        """Preprocess the data"""
        df = df.copy()
        
        # Identify ID column
        id_cols = [col for col in df.columns 
                  if col.lower() in ['id', 'index'] or 'id' in col.lower()]
        
        # Separate features
        if is_train:
            if self.target_column in df.columns:
                X = df.drop(columns=[self.target_column] + id_cols, errors='ignore')
                y = df[self.target_column]
            else:
                raise ValueError(f"Target column '{self.target_column}' not found")
        else:
            X = df.drop(columns=id_cols, errors='ignore')
            y = None
        
        print(f"   Features: {X.shape[1]} columns")
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype == 'object':
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing', inplace=True)
                else:
                    X[col].fillna(X[col].median(), inplace=True)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            print(f"   Encoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X[col] = X[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    X[col] = le.transform(X[col])
        
        # Scale features
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def train_ensemble(self, X, y):
        """Train ensemble of models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Check class distribution
        print("\n📊 Class distribution:")
        print(pd.Series(y).value_counts())
        print("\nClass percentages:")
        print(pd.Series(y).value_counts(normalize=True) * 100)
        
        # Handle imbalanced data with SMOTE
        unique_counts = pd.Series(y).value_counts()
        if len(unique_counts) > 1 and unique_counts.min() / unique_counts.max() < 0.5:
            print("\n⚖️ Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            print(f"   New shape: {X.shape}")
            print(f"   New class distribution:\n{pd.Series(y).value_counts()}")
        
        # Define models
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                random_state=42,
                verbose=-1
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        }
        
        # Cross-validation
        best_score = 0
        best_model_name = None
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("\n🔄 Training and evaluating models...")
        print("-" * 60)
        
        for name, model in models.items():
            print(f"\n📊 {name}:")
            print("   Training with 5-fold cross-validation...")
            
            scores = cross_val_score(
                model, X, y, 
                cv=kfold, 
                scoring='f1_weighted', 
                n_jobs=-1
            )
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            print(f"   ✅ F1 Score: {mean_score:.4f} (+/- {std_score:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                self.model = model
        
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"🏆 F1 SCORE: {best_score:.4f}")
        print("="*60)
        
        # Train best model on full data
        print(f"\n🚀 Training {best_model_name} on full dataset...")
        self.model.fit(X, y)
        print("   ✅ Training complete!")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def save_model(self, filepath='model/trained_model.pkl'):
        """Save trained model"""
        os.makedirs('model', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved to: {filepath}")
    
    def load_model(self, filepath='model/trained_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        
        print(f"✅ Model loaded from: {filepath}")

def main():
    print("\n" + "="*60)
    print("ML CHALLENGE - MODEL TRAINING")
    print("="*60)
    
    try:
        # Load data
        print("\n📂 Loading data...")
        train_df = pd.read_csv('TRAIN.csv')
        test_df = pd.read_csv('test.csv')
        
        print(f"   ✅ Training data: {train_df.shape}")
        print(f"   ✅ Test data: {test_df.shape}")
        
        # Initialize solution
        solution = MLChallengeSolution()
        
        # Identify target
        print("\n🔍 Identifying target column...")
        target_col = solution.identify_target(train_df, test_df)
        
        # Preprocess
        print("\n🔧 Preprocessing training data...")
        X_train, y_train = solution.preprocess_data(train_df, is_train=True)
        print(f"   ✅ Preprocessed shape: {X_train.shape}")
        
        # Train model
        solution.train_ensemble(X_train, y_train)
        
        # Save model
        solution.save_model()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print("\nNext step: Run predict.py to generate FINAL.csv")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()