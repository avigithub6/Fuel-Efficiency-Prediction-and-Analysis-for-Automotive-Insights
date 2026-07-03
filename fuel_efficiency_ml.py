#!/usr/bin/env python
"""
Fuel Efficiency Prediction Machine Learning Pipeline

This script implements a complete machine learning pipeline for predicting fuel efficiency
(MPG - Miles Per Gallon) of automobiles based on various attributes.

Features:
- Data loading and preprocessing
- Model training and evaluation
- Feature importance analysis
- Model persistence
- Prediction functionality
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def load_data(file_path='autos_mpg.csv'):
    """
    Load and prepare the Auto MPG dataset
    
    Args:
        file_path: Path to the dataset CSV file
        
    Returns:
        DataFrame containing the cleaned dataset
    """
    print(f"Loading data from {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Display basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {', '.join(df.columns)}")
        
        # Replace '?' with NaN (if any)
        df = df.replace('?', np.nan)
        
        # Convert data types
        numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    
    Args:
        df: DataFrame with the raw data
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop(['mpg', 'car_name'], axis=1, errors='ignore')
    y = df['mpg']
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit the preprocessor
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"Training set shape: {X_train_transformed.shape}")
    print(f"Test set shape: {X_test_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple regression models and evaluate their performance
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary of trained models and DataFrame with results
    """
    print("Training models...")
    
    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Results storage
    results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        
        print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv('output/model_results.csv', index=False)
    
    return models, results

def find_best_model(models, results):
    """
    Identify the best performing model based on R² score
    
    Args:
        models: Dictionary of trained models
        results: List of model results
        
    Returns:
        Best model name and model object
    """
    # Convert results to DataFrame if it's a list
    if isinstance(results, list):
        results_df = pd.DataFrame(results)
    else:
        results_df = results
    
    # Find the model with the highest R² score
    best_model_idx = results_df['R²'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_model = models[best_model_name]
    
    # Save the best model
    joblib.dump(best_model, 'output/best_model.pkl')
    print(f"Best model ({best_model_name}) saved to 'output/best_model.pkl'")
    
    # Save feature importances if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': [f"feature_{i}" for i in range(len(best_model.feature_importances_))],
            'Importance': best_model.feature_importances_
        })
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        feature_importances.to_csv('output/feature_importances.csv', index=False)
        print("Feature importances saved to 'output/feature_importances.csv'")
    
    return best_model_name, best_model

def save_prediction_script(preprocessor):
    """
    Generate a standalone script for making predictions
    
    Args:
        preprocessor: Fitted preprocessor for transforming input data
    """
    # Save preprocessor
    joblib.dump(preprocessor, 'output/preprocessor.pkl')
    
    # Create prediction script
    with open('output/predict.py', 'w') as f:
        f.write("""#!/usr/bin/env python
\"\"\"
Fuel Efficiency Prediction Script

This script loads the trained model and preprocessor, and provides a function
to predict MPG based on car features.
\"\"\"

import joblib
import pandas as pd
import numpy as np

# Load model and preprocessor
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

def predict_mpg(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    \"\"\"
    Predict MPG for a car with the given features
    
    Args:
        cylinders: Number of cylinders (integer)
        displacement: Engine displacement (float)
        horsepower: Engine horsepower (float)
        weight: Vehicle weight (float)
        acceleration: Time to accelerate from 0 to 60 mph (float)
        model_year: Model year (integer)
        origin: Origin of car (integer: 1=USA, 2=Europe, 3=Japan)
        
    Returns:
        Predicted MPG value
    \"\"\"
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'model_year': [model_year],
        'origin': [origin]
    })
    
    # Transform the data using the preprocessor
    X = preprocessor.transform(data)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction

if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Predict fuel efficiency (MPG) for a car')
    parser.add_argument('--cylinders', type=int, required=True, help='Number of cylinders')
    parser.add_argument('--displacement', type=float, required=True, help='Engine displacement')
    parser.add_argument('--horsepower', type=float, required=True, help='Horsepower')
    parser.add_argument('--weight', type=float, required=True, help='Vehicle weight')
    parser.add_argument('--acceleration', type=float, required=True, help='Time to accelerate 0-60 mph')
    parser.add_argument('--model_year', type=int, required=True, help='Model year (70-82)')
    parser.add_argument('--origin', type=int, required=True, help='Origin (1=USA, 2=Europe, 3=Japan)')
    
    args = parser.parse_args()
    
    # Make prediction
    predicted_mpg = predict_mpg(
        args.cylinders, 
        args.displacement,
        args.horsepower,
        args.weight,
        args.acceleration,
        args.model_year,
        args.origin
    )
    
    print(f"Predicted MPG: {predicted_mpg:.2f}")
""")
    
    print("Prediction script saved to 'output/predict.py'")

def predict_mpg(model, preprocessor, car_features):
    """
    Make a prediction for a single car
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        car_features: Dictionary of car features
        
    Returns:
        Predicted MPG value
    """
    # Convert to DataFrame
    df = pd.DataFrame([car_features])
    
    # Transform features
    X = preprocessor.transform(df)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction

def main():
    """
    Main function to run the complete ML pipeline
    """
    print("Starting Fuel Efficiency Prediction ML Pipeline")
    print("=" * 50)
    start_time = datetime.now()
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)
    
    # Train models
    models, results = train_models(X_train, y_train, X_test, y_test)
    
    # Find best model
    best_model_name, best_model = find_best_model(models, results)
    
    # Save prediction script
    save_prediction_script(preprocessor)
    
    # Example prediction
    example_car = {
        'cylinders': 4,
        'displacement': 120.0,
        'horsepower': 95.0,
        'weight': 2500.0,
        'acceleration': 15.0,
        'model_year': 80,
        'origin': 3
    }
    
    prediction = predict_mpg(best_model, preprocessor, example_car)
    print(f"\nExample prediction: {prediction:.2f} MPG for a car with:")
    for feature, value in example_car.items():
        print(f"  {feature}: {value}")
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print("\nPipeline completed successfully!")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fuel Efficiency Prediction ML Pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode")
    
    main() 