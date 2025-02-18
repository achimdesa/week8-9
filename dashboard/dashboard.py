from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import requests

# Initialize Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    html.Div(id='summary-boxes'),
    dcc.Graph(id='fraud-trends'),
    dcc.Graph(id='geographical-fraud'),
    dcc.Graph(id='device-fraud'),
    dcc.Graph(id='browser-fraud'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # Auto-refresh every 5 seconds
])

# Callback for summary boxes
@app.callback(
    Output('summary-boxes', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_summary(n):
    response = requests.get('http://127.0.0.1:5000/summary')
    data = response.json()

    return [
        html.Div([
            html.H3("Total Transactions"),
            html.P(data['total_transactions'])
        ], style={'display': 'inline-block', 'margin': '10px'}),
        html.Div([
            html.H3("Fraud Cases"),
            html.P(data['fraud_cases'])
        ], style={'display': 'inline-block', 'margin': '10px'}),
        html.Div([
            html.H3("Fraud Percentage"),
            html.P(f"{data['fraud_percentage']:.2f}%")
        ], style={'display': 'inline-block', 'margin': '10px'})
    ]

# Callback for fraud trends over time
@app.callback(
    Output('fraud-trends', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_fraud_trends(n):
    response = requests.get('http://127.0.0.1:5000/fraud_trends')
    data = pd.DataFrame(response.json())

    fig = px.line(data, x='date', y='class', title='Fraud Cases Over Time')
    return fig

# Callback for geographical fraud distribution
@app.callback(
    Output('geographical-fraud', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_geographical_fraud(n):
    response = requests.get('http://127.0.0.1:5000/geographical_fraud')
    data = pd.DataFrame(response.json())

    fig = px.bar(data, x='country', y='class', title='Geographical Fraud Distribution')
    return fig

# Callback for fraud by device
@app.callback(
    Output('device-fraud', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_device_fraud(n):
    response = requests.get('http://127.0.0.1:5000/fraud_by_device_browser')
    data = pd.DataFrame(response.json()['device_fraud'])

    fig = px.bar(data, x='device_id', y='class', title='Fraud Cases by Device')
    return fig

# Callback for fraud by browser
@app.callback(
    Output('browser-fraud', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_browser_fraud(n):
    response = requests.get('http://127.0.0.1:5000/fraud_by_device_browser')
    data = pd.DataFrame(response.json()['browser_fraud'])

    fig = px.bar(data, x='browser', y='class', title='Fraud Cases by Browser')
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)