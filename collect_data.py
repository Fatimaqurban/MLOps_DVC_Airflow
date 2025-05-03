import os
import requests
import pandas as pd
from datetime import datetime
import yaml

class WeatherDataCollector:
    def __init__(self):
        # Load API key from params.yaml
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        self.api_key = params['WEATHER_API_KEY']
        self.base_url = "http://api.weatherapi.com/v1"
        self.output_dir = "data"
        
        # List of cities to collect data from
        self.cities = [
            "London", "New York", "Tokyo", "Paris", "Sydney",
            "Dubai", "Singapore", "Mumbai", "Toronto", "Berlin"
        ]
        
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY not found in params.yaml")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_weather_data(self, city):
        """
        Fetch current weather data for a given city
        """
        endpoint = f"{self.base_url}/current.json"
        params = {
            "key": self.api_key,
            "q": city,
            "aqi": "no"
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant data
            current_data = data['current']
            location_data = data['location']
            
            # Create a DataFrame
            weather_data = {
                'timestamp': [datetime.now().isoformat()],
                'city': [location_data['name']],
                'country': [location_data['country']],
                'temperature_c': [current_data['temp_c']],
                'temperature_f': [current_data['temp_f']],
                'humidity': [current_data['humidity']],
                'wind_kph': [current_data['wind_kph']],
                'wind_degree': [current_data['wind_degree']],
                'pressure_mb': [current_data['pressure_mb']],
                'precip_mm': [current_data['precip_mm']],
                'cloud': [current_data['cloud']],
                'feelslike_c': [current_data['feelslike_c']],
                'feelslike_f': [current_data['feelslike_f']],
                'condition': [current_data['condition']['text']]
            }
            
            return pd.DataFrame(weather_data)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data for {city}: {e}")
            return None
    
    def collect_all_cities(self):
        """
        Collect weather data for all cities
        """
        all_data = []
        for city in self.cities:
            print(f"Collecting data for {city}...")
            df = self.fetch_weather_data(city)
            if df is not None:
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None
    
    def save_data(self, df, filename=None):
        """
        Save the weather data to a CSV file
        """
        if df is None:
            return False
        
        if filename is None:
            filename = "raw_data.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return True

if __name__ == "__main__":
    # Example usage
    collector = WeatherDataCollector()
    weather_data = collector.collect_all_cities()
    if weather_data is not None:
        collector.save_data(weather_data)