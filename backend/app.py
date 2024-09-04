from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/air_quality_model.pkl')

@app.route('/api/air_quality', methods=['GET'])
def get_air_quality():
    # Example data (replace with actual data fetching logic)
    data = [
        {'lat': 51.505, 'lon': -0.09, 'no2': 20},
        {'lat': 51.51, 'lon': -0.1, 'no2': 22},
        # Add more data points as needed
    ]
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def predict():
    input_data = request.json
    features = np.array(input_data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'no2': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
