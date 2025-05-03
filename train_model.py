import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

class WeatherModelTrainer:
    def __init__(self):
        self.processed_data_dir = "data"
        self.models_dir = "models"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_processed_data(self):
        """
        Load the processed weather data
        """
        try:
            filepath = os.path.join(self.processed_data_dir, "processed_data.csv")
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Prepare features for model training based on processed data structure
        """
        if df is None:
            return None, None
        
        try:
            # First, create a copy of the dataframe to avoid modifying the original
            df_processed = df.copy()
            
            # Convert boolean columns to numeric
            df_processed['is_raining'] = df_processed['is_raining'].astype(int)
            df_processed['is_cloudy'] = df_processed['is_cloudy'].astype(int)
            
            # Select only numeric features for prediction
            feature_columns = [
                'humidity_normalized',
                'wind_kph_normalized',
                'pressure_mb_normalized',
                'precip_mm_normalized',
                'cloud_normalized',
                'is_raining',
                'is_cloudy'
            ]
            
            # Verify all required columns exist
            missing_columns = [col for col in feature_columns if col not in df_processed.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in processed data: {missing_columns}")
            
            # Select features and target
            X = df_processed[feature_columns]
            y = df_processed['temperature_c']
            
            # Verify all features are numeric
            non_numeric_cols = X.select_dtypes(include=['object']).columns
            if len(non_numeric_cols) > 0:
                raise ValueError(f"Non-numeric columns found: {non_numeric_cols}")
            
            return X, y
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None, None
    
    def train_model(self, X, y):
        """
        Train a Random Forest model for weather prediction
        """
        if X is None or y is None:
            return None
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Print model performance
            print("\nModel Performance:")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Root Mean Squared Error: {rmse:.2f}")
            print(f"R² Score: {r2:.2f}")
            
            # Print feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance)
            
            # Save metrics
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2)
            }
            
            metrics_path = os.path.join(self.models_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            return model
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def save_model(self, model):
        """
        Save the trained model
        """
        if model is None:
            return False
        
        try:
            filepath = os.path.join(self.models_dir, "model.pkl")
            joblib.dump(model, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

if __name__ == "__main__":
    # Initialize trainer
    trainer = WeatherModelTrainer()
    
    # Load and prepare data
    processed_data = trainer.load_processed_data()
    if processed_data is not None:
        X, y = trainer.prepare_features(processed_data)
        if X is not None and y is not None:
            # Train and save model
            model = trainer.train_model(X, y)
            if model is not None:
                trainer.save_model(model) 