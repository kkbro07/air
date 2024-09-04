import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
data = pd.read_csv('data.csv')  # Path to the synthetic data

# Preprocess the data
# For simplicity, let's assume the data is already clean and ready for use

# Split the data into features and target
X = data.drop('no2', axis=1)
y = data['no2']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f'MAE: {mae}, RMSE: {rmse}')
