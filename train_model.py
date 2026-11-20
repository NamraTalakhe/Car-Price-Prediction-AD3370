import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Paths relative to this script's directory
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'Cleaned_Car_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'LinearRegressionModel.pkl')
BACKUP_MODEL_PATH = os.path.join(BASE_DIR, 'LinearRegressionModel_old.pkl')

print('Loading dataset...')
car = pd.read_csv(DATA_PATH)

# Define features and target
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Preprocess categorical and numeric columns
categorical_features = ['name', 'company', 'fuel_type']
numeric_features = ['year', 'kms_driven']
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('regressor', LinearRegression())
])

# Train‑test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print('Training model...')
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print('R2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))

# Backup existing model if present
if os.path.exists(MODEL_PATH):
    print('Backing up existing model to LinearRegressionModel_old.pkl')
    os.replace(MODEL_PATH, BACKUP_MODEL_PATH)

# Save new pipeline
joblib.dump(pipeline, MODEL_PATH)
print('Saved new model to', MODEL_PATH)
