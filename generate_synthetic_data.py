import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
data = {
    'latitude': np.random.uniform(30, 50, n_samples),  # Random latitudes
    'longitude': np.random.uniform(-130, -70, n_samples),  # Random longitudes
    'temperature': np.random.normal(20, 5, n_samples),  # Random temperatures
    'humidity': np.random.normal(50, 10, n_samples),  # Random humidity
    'wind_speed': np.random.normal(5, 2, n_samples),  # Random wind speeds
    'NO2': np.random.normal(30, 10, n_samples)  # Random NO2 levels
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/synthetic_air_quality_data.csv', index=False)
