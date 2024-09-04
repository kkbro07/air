import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load and preprocess data
data = pd.read_csv('data/synthetic_air_quality_data.csv')  # Path to the synthetic data
X = data.drop('NO2', axis=1)
y = data['NO2']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Print results using .format() method
print('MAE: {}, RMSE: {}'.format(mae, rmse))

# Save model
joblib.dump(model, 'model/air_quality_model.pkl')
