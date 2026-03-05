import pandas as pd
import numpy as np
from train_model import MLChallengeSolution

def generate_predictions():
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    try:
        # Load test data
        print("\n📂 Loading test data...")
        test_df = pd.read_csv('test.csv')
        print(f"   ✅ Test data shape: {test_df.shape}")
        
        # Identify ID column
        id_cols = [col for col in test_df.columns 
                  if col.lower() in ['id', 'index'] or 'id' in col.lower()]
        
        if id_cols:
            id_col = id_cols[0]
            test_ids = test_df[id_col]
            print(f"   ✅ ID column: {id_col}")
        else:
            print("   ⚠️ No ID column found, using index")
            test_ids = test_df.index
            id_col = 'id'
        
        # Load model
        print("\n🔄 Loading trained model...")
        solution = MLChallengeSolution()
        solution.load_model('model/trained_model.pkl')
        
        # Preprocess test data
        print("\n🔧 Preprocessing test data...")
        X_test, _ = solution.preprocess_data(test_df, is_train=False)
        print(f"   ✅ Preprocessed shape: {X_test.shape}")
        
        # Make predictions
        print("\n🎯 Generating predictions...")
        predictions = solution.predict(X_test)
        print(f"   ✅ Generated {len(predictions)} predictions")
        
        # Create submission file
        print("\n📝 Creating FINAL.csv...")
        
        target_name = solution.target_column if solution.target_column else 'target'
        
        final_df = pd.DataFrame({
            id_col: test_ids,
            target_name: predictions
        })
        
        final_df.to_csv('FINAL.csv', index=False)
        
        print(f"   ✅ FINAL.csv created successfully!")
        print(f"   Shape: {final_df.shape}")
        
        print("\n📊 Prediction distribution:")
        print(final_df[target_name].value_counts())
        print("\nPercentage distribution:")
        print(final_df[target_name].value_counts(normalize=True) * 100)
        
        print("\n" + "="*60)
        print("✅ PREDICTION COMPLETE!")
        print("="*60)
        print("\n📁 FINAL.csv is ready for submission!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_predictions()