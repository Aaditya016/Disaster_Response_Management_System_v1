# tsunami_monitor.py
import requests
import pandas as pd
import numpy as np
import joblib
import time
import datetime
import os
import json

# Load the trained model
print("Loading tsunami prediction model...")
model = joblib.load('tsunami_prediction_model.pkl')

# Load the list of features used in the model
features = []
try:
    with open('model_features.txt', 'r') as f:
        features = f.read().splitlines()
    print(f"Loaded {len(features)} features: {features}")
except Exception as e:
    print(f"Error loading feature list: {e}")
    print("Using default feature list")
    features = ['magnitude', 'depth', 'latitude', 'longitude', 'sig', 'gap', 'dmin', 'mmi']

# USGS Earthquake API URL
USGS_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"

def fetch_recent_earthquakes():
    """Fetch recent significant earthquakes from USGS"""
    try:
        response = requests.get(USGS_API_URL)
        if response.status_code == 200:
            data = response.json()
            return data['features']
        else:
            print(f"Error fetching earthquake data: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching earthquake data: {e}")
        return []

def process_earthquake(earthquake):
    """Process earthquake data and prepare features for prediction"""
    try:
        properties = earthquake['properties']
        geometry = earthquake['geometry']
        
        # Create a dictionary with default values
        earthquake_features = {
            'magnitude': properties.get('mag', 0),
            'depth': geometry['coordinates'][2] if len(geometry['coordinates']) > 2 else 0,
            'latitude': geometry['coordinates'][1] if len(geometry['coordinates']) > 1 else 0,
            'longitude': geometry['coordinates'][0] if len(geometry['coordinates']) > 0 else 0,
            'sig': properties.get('sig', 0),
            'gap': properties.get('gap', 0),
            'dmin': properties.get('dmin', 0),
            'mmi': properties.get('mmi', 0)
        }
        
        # Handle magType if it's in our model features
        if any(col.startswith('magType_') for col in features):
            mag_type = properties.get('magType', '')
            # Initialize all magType columns to 0
            for col in features:
                if col.startswith('magType_'):
                    earthquake_features[col] = 0
            
            # Set the matching magType column to 1 if it exists
            mag_type_col = f'magType_{mag_type}'
            if mag_type_col in features:
                earthquake_features[mag_type_col] = 1
        
        # Create a dataframe with only the features used by our model
        df = pd.DataFrame([earthquake_features])
        
        # Ensure all model features are in the dataframe
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the columns needed by the model, in the right order
        prediction_df = df[features]
        
        # Make prediction
        prediction = model.predict(prediction_df)[0]
        probability = model.predict_proba(prediction_df)[0][1]
        
        result = {
            'id': earthquake['id'],
            'time': datetime.datetime.fromtimestamp(properties['time']/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'title': properties.get('title', 'Unknown Earthquake'),
            'magnitude': properties.get('mag', 0),
            'depth': geometry['coordinates'][2] if len(geometry['coordinates']) > 2 else 0,
            'latitude': geometry['coordinates'][1] if len(geometry['coordinates']) > 1 else 0,
            'longitude': geometry['coordinates'][0] if len(geometry['coordinates']) > 0 else 0,
            'tsunami_predicted': bool(prediction),
            'tsunami_probability': float(probability),
            'features_used': earthquake_features
        }
        
        return result
    except Exception as e:
        print(f"Error processing earthquake: {e}")
        return None

def monitor_earthquakes():
    """Main monitoring function to check for tsunami risks"""
    print(f"Starting earthquake monitoring at {datetime.datetime.now()}")
    
    while True:
        print("\nFetching recent earthquakes...")
        earthquakes = fetch_recent_earthquakes()
        
        if not earthquakes:
            print("No earthquakes found or error fetching data")
            time.sleep(300)  # Wait 5 minutes before checking again
            continue
            
        print(f"Processing {len(earthquakes)} earthquakes...")
        
        for earthquake in earthquakes:
            result = process_earthquake(earthquake)
            
            if result and result['tsunami_predicted']:
                # High risk of tsunami detected
                print("\n" + "!" * 80)
                print(f"TSUNAMI RISK DETECTED: {result['title']}")
                print(f"Time: {result['time']}")
                print(f"Magnitude: {result['magnitude']}")
                print(f"Location: ({result['latitude']}, {result['longitude']})")
                print(f"Depth: {result['depth']} km")
                print(f"Tsunami Probability: {result['tsunami_probability']*100:.2f}%")
                print("!" * 80 + "\n")
                
                # Here you could trigger the email alert system or other notifications
            elif result:
                print(f"Processed: {result['title']} - No tsunami risk detected (probability: {result['tsunami_probability']*100:.2f}%)")
        
        print(f"Completed processing at {datetime.datetime.now()}")
        print("Waiting for next check cycle...")
        time.sleep(300)  # Wait 5 minutes before checking again

if __name__ == "__main__":
    try:
        monitor_earthquakes()
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except Exception as e:
        print(f"Error in monitoring: {e}")
