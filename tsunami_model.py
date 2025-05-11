# Tsunami Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import joblib

# File paths - update these to match your local file locations
TRAINING_DATA_PATH = 'earthquake_1995-2023.csv'  # Main dataset
TEST_DATA_PATH = 'testing_data1.csv'             # Test dataset

print("Loading earthquake data...")
earthquake_data = pd.read_csv(TRAINING_DATA_PATH)

print(f"Dataset shape: {earthquake_data.shape}")

# Select only features available from USGS API in real-time
# These are the features we can reliably expect from the API
features_for_production = [
    'magnitude',    # Earthquake magnitude
    'depth',        # Depth of the earthquake
    'latitude',     # Latitude coordinate
    'longitude',    # Longitude coordinate
    'sig',          # Significance number
    'gap',          # Azimuthal gap
    'dmin',         # Minimum distance
    'magType',      # Magnitude type
    'mmi'           # Modified Mercalli Intensity
]

# Check if all selected features exist in the dataset
missing_features = [f for f in features_for_production if f not in earthquake_data.columns]
if missing_features:
    print(f"Warning: These features are missing from the dataset: {missing_features}")
    # Remove missing features from our selection
    features_for_production = [f for f in features_for_production if f not in missing_features]

print(f"Training model with these features: {features_for_production}")

# Handle categorical features
# Convert magType to numerical using one-hot encoding
if 'magType' in features_for_production:
    earthquake_data = pd.get_dummies(earthquake_data, columns=['magType'], drop_first=True)
    # Update features list to include the one-hot encoded columns
    mag_type_columns = [col for col in earthquake_data.columns if col.startswith('magType_')]
    features_for_production = [f for f in features_for_production if f != 'magType'] + mag_type_columns

# Prepare features and target
X = earthquake_data[features_for_production].copy()
y = earthquake_data['tsunami']  # Target variable

# Handle missing values
X = X.fillna({
    'depth': X['depth'].median(),
    'dmin': X['dmin'].median(),
    'gap': X['gap'].median(),
    'sig': X['sig'].median(),
    'mmi': X['mmi'].median() if 'mmi' in X.columns else 0
})

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

print("Training Random Forest model...")
# Create a pipeline with scaling and the RandomForest model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
val_score = pipeline.score(X_val, y_val)
print(f"Validation accuracy: {val_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

# Make predictions on the validation set
y_pred = pipeline.predict(X_val)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Feature importance
if hasattr(pipeline['model'], 'feature_importances_'):
    importance = pipeline['model'].feature_importances_
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature Importance:")
    print(feature_importance)

# Save the trained model and feature list
print("Saving model and feature list...")
joblib.dump(pipeline, 'tsunami_prediction_model.pkl')

# Save the list of features used for later reference
with open('model_features.txt', 'w') as f:
    f.write('\n'.join(X.columns))

print("Model training complete!")

# Test on separate test dataset if available
try:
    test_data = pd.read_csv(TEST_DATA_PATH)
    print(f"\nTesting on separate test dataset: {TEST_DATA_PATH}")
    
    # Prepare test features
    X_test = test_data[features_for_production].copy()
    
    # Handle missing values
    X_test = X_test.fillna({
        'depth': X['depth'].median(),
        'dmin': X['dmin'].median(),
        'gap': X['gap'].median(),
        'sig': X['sig'].median(),
        'mmi': X['mmi'].median() if 'mmi' in X.columns else 0
    })
    
    # If we used one-hot encoding for categorical features, ensure test set has same columns
    if 'magType' not in features_for_production and any(col.startswith('magType_') for col in X.columns):
        # Get magType categorical feature from test data
        test_mag_type = pd.get_dummies(test_data, columns=['magType'], drop_first=True)
        
        # Add missing columns with zeros
        for col in X.columns:
            if col.startswith('magType_') and col not in test_mag_type.columns:
                test_mag_type[col] = 0
        
        # Keep only the columns used in training
        X_test = test_mag_type[X.columns].copy()
    
    y_test = test_data['tsunami']
    
    # Evaluate on test set
    test_score = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f}")
    
    # Make predictions
    y_test_pred = pipeline.predict(X_test)
    
    # Print classification report
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
except Exception as e:
    print(f"Error testing model on test dataset: {e}")
