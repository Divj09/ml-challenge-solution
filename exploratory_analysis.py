import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load training and test data"""
    print("="*60)
    print("LOADING DATASET")
    print("="*60)
    
    train_df = pd.read_csv('TRAIN.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"\n✅ Training data shape: {train_df.shape}")
    print(f"✅ Test data shape: {test_df.shape}")
    
    print("\n" + "="*60)
    print("COLUMNS IN TRAINING DATA")
    print("="*60)
    for i, col in enumerate(train_df.columns, 1):
        print(f"{i}. {col} ({train_df[col].dtype})")
    
    print("\n" + "="*60)
    print("COLUMNS IN TEST DATA")
    print("="*60)
    for i, col in enumerate(test_df.columns, 1):
        print(f"{i}. {col} ({test_df[col].dtype})")
    
    # Identify target
    target_col = identify_target_column(train_df, test_df)
    
    print("\n" + "="*60)
    print("MISSING VALUES IN TRAINING DATA")
    print("="*60)
    missing = train_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("✅ No missing values!")
    
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(train_df.describe())
    
    if target_col:
        print("\n" + "="*60)
        print(f"TARGET COLUMN: {target_col}")
        print("="*60)
        print("\nValue counts:")
        print(train_df[target_col].value_counts())
        print("\nPercentage distribution:")
        print(train_df[target_col].value_counts(normalize=True) * 100)
    
    return train_df, test_df, target_col

def identify_target_column(train_df, test_df):
    """Identify the target column"""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    diff_cols = train_cols - test_cols
    
    # Remove ID columns
    potential_targets = [col for col in diff_cols 
                        if col.lower() not in ['id', 'index', 'unnamed']]
    
    if len(potential_targets) == 1:
        target = potential_targets[0]
        print(f"\n🎯 Target column identified: {target}")
        return target
    elif len(potential_targets) > 1:
        print(f"\n⚠️ Multiple potential targets found: {potential_targets}")
        print(f"Using first one: {potential_targets[0]}")
        return potential_targets[0]
    else:
        print("\n⚠️ Could not identify target column")
        return None

def visualize_data(train_df, target_col):
    """Create visualizations"""
    if target_col is None:
        print("\n⚠️ Skipping visualization - no target column")
        return
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    Path('visualizations').mkdir(exist_ok=True)
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    train_df[target_col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {target_col}', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/target_distribution.png', dpi=100)
    print("✅ Saved: visualizations/target_distribution.png")
    plt.close()
    
    # Correlation heatmap
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(14, 10))
        correlation = train_df[numeric_cols].corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0, 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=100)
        print("✅ Saved: visualizations/correlation_heatmap.png")
        plt.close()
    
    print("\n✅ All visualizations saved in 'visualizations/' folder")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60 + "\n")
    
    try:
        train_df, test_df, target_col = load_data()
        visualize_data(train_df, target_col)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()