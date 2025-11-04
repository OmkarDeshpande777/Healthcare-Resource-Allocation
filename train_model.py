import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Define model path
MODEL_PATH = r"C:\Document Local\Projects\DAA Project\trained_model.pkl"
DATASET_PATH = r"C:\Document Local\Projects\DAA Project\healthcare_dataset.csv"

def train_model():
    """Train a RandomForest model from the healthcare dataset and save it."""
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return False
    
    print("Loading healthcare dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Define features based on the app's input requirements
    # Note: Using available columns from dataset that match closest to app requirements
    feature_columns = ['Age', 'Gender', 'Blood Type', 'Admission Type']
    target_column = 'Medical Condition'
    
    # Check if required columns exist
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    # Prepare features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"\nTarget variable unique values: {y.unique()}")
    print(f"Target variable value counts:\n{y.value_counts()}")
    
    # Encode categorical features
    label_encoders = {}
    for col in ['Gender', 'Blood Type', 'Admission Type']:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    print("\nTraining RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Save model and encoders together
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model trained and saved successfully to: {MODEL_PATH}")
    print(f"Model accuracy on training data: {model.score(X, y):.4f}")
    
    return True

if __name__ == "__main__":
    success = train_model()
    if not success:
        print("\nFailed to train model. Please check the errors above.")
        exit(1)
