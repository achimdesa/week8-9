from flask import Flask, jsonify
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
fraud_data = pd.read_csv('../data/Fraud_Data.csv')

# Convert purchase_time to datetime
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

# API endpoint for summary statistics
@app.route('/summary', methods=['GET'])
def summary():
    total_transactions = len(fraud_data)
    fraud_cases = fraud_data['class'].sum()
    fraud_percentage = (fraud_cases / total_transactions) * 100

    return jsonify({
        'total_transactions': total_transactions,
        'fraud_cases': fraud_cases,
        'fraud_percentage': fraud_percentage
    })

# API endpoint for fraud trends over time
@app.route('/fraud_trends', methods=['GET'])
def fraud_trends():
    fraud_data['date'] = fraud_data['purchase_time'].dt.date
    fraud_trends = fraud_data.groupby('date')['class'].sum().reset_index()

    return jsonify(fraud_trends.to_dict(orient='records'))

# API endpoint for geographical fraud distribution
@app.route('/geographical_fraud', methods=['GET'])
def geographical_fraud():
    geo_fraud = fraud_data.groupby('country')['class'].sum().reset_index()

    return jsonify(geo_fraud.to_dict(orient='records'))

# API endpoint for fraud by device and browser
@app.route('/fraud_by_device_browser', methods=['GET'])
def fraud_by_device_browser():
    device_fraud = fraud_data.groupby('device_id')['class'].sum().reset_index()
    browser_fraud = fraud_data.groupby('browser')['class'].sum().reset_index()

    return jsonify({
        'device_fraud': device_fraud.to_dict(orient='records'),
        'browser_fraud': browser_fraud.to_dict(orient='records')
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)