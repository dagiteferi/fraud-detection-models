# serve_model.py

import os
import joblib
import logging
import pandas as pd
from flask import Flask, request, jsonify
import dash
from dash import dcc, html
from callbacks import register_callbacks  # Import the callback function

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
logging.basicConfig(filename="logs/api.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Load data once before the first request and return fraud_data
def load_data_once():
    global fraud_data
    fraud_data = pd.read_csv('data/processed/processed_fraud_data.csv')
    fraud_data = fraud_data[fraud_data['purchase_value'] > 0]
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data.set_index('purchase_time', inplace=True)
    return fraud_data

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

# Load data
fraud_data = load_data_once()  # Load data before callbacks

# Register callbacks
register_callbacks(dash_app, fraud_data)

# Dashboard layout
dash_app.layout = html.Div([
    html.Header([html.H1("Fraud Detection Dashboard", className="title")], className="header"),
    html.Div([  # Summary statistics section
        html.Div([html.H4("Total Transactions"), html.P(id="total-transactions", children="")], className="box"),
        html.Div([html.H4("Fraud Cases"), html.P(id="fraud-cases", children="")], className="box"),
        html.Div([html.H4("Fraud Percentage"), html.P(id="fraud-percentage", children="")], className="box"),
    ], className="summary-boxes"),
    dcc.Graph(id="fraud-trends", className="graph"),
    dcc.Graph(id="geographic-fraud", className="graph"),
    dcc.Graph(id="device-fraud", className="graph"),
    dcc.Graph(id="browser-fraud", className="graph"),
])

if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
    app.run(host="0.0.0.0", port=5000)
