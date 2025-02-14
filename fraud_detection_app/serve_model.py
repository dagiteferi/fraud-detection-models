import os
import joblib
import logging
import pandas as pd
from flask import Flask, request, jsonify
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Initialize Flask app
app = Flask(__name__)

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Define model paths (adjusted for Docker container)
FRAUD_MODEL_PATH = os.path.join("models", "random_forest_model_fraud.pkl")
CREDIT_CARD_MODEL_PATH = os.path.join("models", "randomforestfor_credit_card_data.pkl")

# Load models
fraud_model = joblib.load(FRAUD_MODEL_PATH)
credit_card_model = joblib.load(CREDIT_CARD_MODEL_PATH)

# Configure logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# Load fraud data globally
fraud_data = pd.read_csv('data/processed/processed_fraud_data.csv')

# Filter and preprocess data globally
fraud_data = fraud_data[fraud_data['purchase_value'] > 0]
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data.set_index('purchase_time', inplace=True)

# Home route for Flask
@app.route("/")
def home():
    app.logger.info("Home route accessed.")
    return jsonify({"message": "Fraud Detection API is running!"})

# Prediction route for Flask
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get input data from request
        app.logger.info(f"Received prediction request: {data}")

        model_type = data.get("model_type")

        # Validate model type
        if model_type not in ["fraud", "credit_card"]:
            app.logger.error("Invalid model type received.")
            return jsonify({"error": "Invalid model type"}), 400

        # Extract features from the request (remove model_type)
        features = {k: v for k, v in data.items() if k != "model_type"}

        # Convert the features to a DataFrame
        df = pd.DataFrame([features])

        # Make prediction based on model type
        if model_type == "fraud":
            prediction = fraud_model.predict(df)
            proba = fraud_model.predict_proba(df)[:, 1]  # Probability of fraud
            app.logger.info(f"Fraud model prediction: {prediction[0]} with probability: {round(proba[0], 4)}")
        elif model_type == "credit_card":
            prediction = credit_card_model.predict(df)
            proba = credit_card_model.predict_proba(df)[:, 1]  # Probability of fraud for credit card
            app.logger.info(f"Credit card model prediction: {prediction[0]} with probability: {round(proba[0], 4)}")

        return jsonify({
            "model_used": model_type,
            "fraud_prediction": int(prediction[0]),
            "fraud_probability": round(proba[0], 4)
        })

    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Dashboard (using Dash)
dash_app.layout = html.Div([  # Dash layout
    html.Header([  # Header section
        html.H1("Fraud Detection Dashboard", className="title"),
        html.P("Analyze and track fraud cases in real time.", className="subtitle")
    ], className="header"),

    # Filters Section
    html.Div([
        html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=fraud_data.index.min().date(),
                end_date=fraud_data.index.max().date(),
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Fraud Type:"),
            dcc.Dropdown(
                id='fraud-type-dropdown',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Fraud', 'value': 1},
                    {'label': 'Non-Fraud', 'value': 0},
                ],
                value='all',
                style={'width': '100%'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Top N Fraud Cases:"),
            dcc.Slider(
                id='top-n-slider',
                min=1,
                max=20,
                step=1,
                value=10,
                marks={i: str(i) for i in range(1, 21)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], className="filters"),

    # Summary Boxes
    html.Div([  # Summary statistics section
        html.Div([  
            html.H4("Total Transactions"),
            html.P(id="total-transactions", children="Loading..."),
        ], className="box"),
        html.Div([  
            html.H4("Fraud Cases"),
            html.P(id="fraud-cases", children="Loading..."),
        ], className="box"),
        html.Div([  
            html.H4("Fraud Percentage"),
            html.P(id="fraud-percentage", children="Loading..."),
        ], className="box"),
    ], className="summary-boxes"),

    # Line Chart for Fraud Trends
    dcc.Graph(id="fraud-trends", className="graph"),

    # Geographic Analysis
    dcc.Graph(id="geographic-fraud", className="graph"),

    # Device Analysis
    dcc.Graph(id="device-fraud", className="graph"),

    # Browser Analysis (new addition)
    dcc.Graph(id="browser-fraud", className="graph"),
])

# Callbacks to update the dashboard
@dash_app.callback(
    [Output("total-transactions", "children"),
     Output("fraud-cases", "children"),
     Output("fraud-percentage", "children"),
     Output("fraud-trends", "figure"),
     Output("geographic-fraud", "figure"),
     Output("device-fraud", "figure"),
     Output("browser-fraud", "figure")],
    [Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date"),
     Input("fraud-type-dropdown", "value"),
     Input("top-n-slider", "value")]
)
def update_dashboard(start_date, end_date, fraud_type, top_n):
    # Filter data based on the selected date range
    filtered_data = fraud_data.loc[start_date:end_date]

    # Filter data based on fraud type (Fraud=1, Non-Fraud=0)
    if fraud_type != 'all':
        filtered_data = filtered_data[filtered_data['class'] == int(fraud_type)]

    # Get total transactions and fraud cases
    total_transactions = len(filtered_data)
    fraud_cases = filtered_data[filtered_data['class'] == 1].shape[0]
    fraud_percentage = (fraud_cases / total_transactions) * 100

    # Create fraud trends data (time series of fraud cases)
    fraud_trends = filtered_data.resample('D').apply(lambda x: (x['class'] == 1).sum())  # Daily fraud count
    fraud_trends_dates = fraud_trends.index.astype(str)

    # Highlight top N fraud cases based on purchase value (or frequency)
    top_fraud_cases = filtered_data.nlargest(top_n, 'purchase_value')

    # Example geographic data (assuming 'ip_address' is the location)
    fraud_by_location = filtered_data.groupby('ip_address')['class'].sum().sort_values(ascending=False).head(10)
    locations = fraud_by_location.index
    fraud_counts_by_location = fraud_by_location.values

    # Example device data (assuming 'device_id' represents the device)
    fraud_by_device = filtered_data.groupby('device_id')['class'].sum().sort_values(ascending=False).head(10)
    devices = fraud_by_device.index
    fraud_counts_by_device = fraud_by_device.values

    # Browser fraud analysis (assuming binary flags for each browser)
    browsers = ['browser_FireFox', 'browser_IE', 'browser_Opera', 'browser_Safari']
    fraud_by_browser = {browser: filtered_data[browser].sum() for browser in browsers}
    browser_names = list(fraud_by_browser.keys())
    fraud_counts_by_browser = list(fraud_by_browser.values())

    # Create figures using Plotly for the graphs
    fraud_trends_fig = {
        "data": [go.Scatter(x=fraud_trends_dates, y=fraud_trends, mode='lines')],
        "layout": go.Layout(title="Fraud Cases Over Time", xaxis={"title": "Date"}, yaxis={"title": "Fraud Cases"})
    }

    geo_fraud_fig = {
        "data": [go.Bar(x=locations, y=fraud_counts_by_location)],
        "layout": go.Layout(title="Geographic Fraud Analysis", xaxis={"title": "IP Address"}, yaxis={"title": "Fraud Cases"})
    }

    device_fraud_fig = {
        "data": [go.Bar(x=devices, y=fraud_counts_by_device)],
        "layout": go.Layout(title="Fraud Cases by Device", xaxis={"title": "Device ID"}, yaxis={"title": "Fraud Cases"})
    }

    # Browser fraud bar chart
    browser_fraud_fig = {
        "data": [go.Bar(x=browser_names, y=fraud_counts_by_browser)],
        "layout": go.Layout(title="Fraud Cases by Browser", xaxis={"title": "Browser"}, yaxis={"title": "Fraud Cases"})
    }

    return total_transactions, fraud_cases, round(fraud_percentage, 2), fraud_trends_fig, geo_fraud_fig, device_fraud_fig, browser_fraud_fig

if __name__ == "__main__":
    # Ensure logs directory exists for logging
    if not os.path.exists("logs"):
        os.makedirs("logs")

    app.run(host="0.0.0.0", port=5000)
