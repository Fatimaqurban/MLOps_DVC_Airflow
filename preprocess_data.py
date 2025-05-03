import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np

class WeatherDataProcessor:
    def __init__(self):
        self.raw_data_dir = "data"
        self.processed_data_dir = "data"
        self.scaler = StandardScaler()
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_latest_data(self):
        """
        Load the raw weather data
        """
        try:
            filepath = os.path.join(self.raw_data_dir, "raw_data.csv")
            if not os.path.exists(filepath):
                print("No raw data file found")
                return None
            
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def process_data(self, df):
        """
        Process the weather data with normalization and missing value handling
        """
        if df is None:
            return None
        
        try:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Handle missing values
            numeric_columns = ['temperature_c', 'temperature_f', 'humidity', 
                             'wind_kph', 'pressure_mb', 'precip_mm', 'cloud',
                             'feelslike_c', 'feelslike_f']
            
            # Fill missing numeric values with median
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            # Fill missing categorical values with mode
            categorical_columns = ['condition']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])
            
            # Normalize numeric features
            numeric_features = ['temperature_c', 'humidity', 'wind_kph', 
                             'pressure_mb', 'precip_mm', 'cloud']
            
            for feature in numeric_features:
                if feature in df.columns:
                    df[f'{feature}_normalized'] = self.scaler.fit_transform(df[[feature]])
            
            # Add derived features
            df['is_raining'] = df['precip_mm'] > 0
            df['is_cloudy'] = df['cloud'] > 50
            df['wind_direction'] = df['wind_degree'].apply(self._get_wind_direction)
            
            # Select and reorder columns
            processed_df = df[[
                'timestamp', 'city', 'country',
                'temperature_c', 'temperature_c_normalized',
                'temperature_f', 'humidity', 'humidity_normalized',
                'wind_kph', 'wind_kph_normalized', 'wind_direction',
                'pressure_mb', 'pressure_mb_normalized',
                'precip_mm', 'precip_mm_normalized',
                'cloud', 'cloud_normalized',
                'feelslike_c', 'feelslike_f',
                'condition', 'is_raining', 'is_cloudy'
            ]]
            
            return processed_df
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
    
    def _get_wind_direction(self, degree):
        """
        Convert wind degree to cardinal direction
        """
        if pd.isna(degree):
            return 'Unknown'
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round(degree / (360. / len(directions))) % len(directions)
        return directions[index]
    
    def save_processed_data(self, df, filename=None):
        """
        Save the processed data to a CSV file
        """
        if df is None:
            return False
        
        if filename is None:
            filename = "processed_data.csv"
        
        filepath = os.path.join(self.processed_data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
        return True

if __name__ == "__main__":
    # Example usage
    processor = WeatherDataProcessor()
    raw_data = processor.load_latest_data()
    if raw_data is not None:
        processed_data = processor.process_data(raw_data)
        if processed_data is not None:
            processor.save_processed_data(processed_data) 