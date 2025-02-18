# app2.py

from flask import Flask, request, jsonify
from joblib import load
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load the model
model = load('random_forest_fraud_model2.joblib')
logging.info('Model loaded successfully.')

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log incoming request
        logging.info('Received prediction request')
        
        # Get data from the request
        data = request.get_json(force=True)
        features = data['features']
        
        # Make prediction
        prediction = model.predict([features])
        prediction_proba = model.predict_proba([features])[:, 1]
        
        # Log prediction
        logging.info(f'Prediction: {prediction[0]}, Probability: {prediction_proba[0]}')
        
        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0])
        })
    except Exception as e:
        # Log errors
        logging.error(f'Error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)