import os
import joblib
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Define model paths (adjusted for Docker container)
FRAUD_MODEL_PATH = os.path.join("models", "random_forest_model_fraud.pkl")
CREDIT_CARD_MODEL_PATH = os.path.join("models", "randomforestfor_credit_card_data.pkl")

# Load the models
fraud_model = joblib.load(FRAUD_MODEL_PATH)
credit_card_model = joblib.load(CREDIT_CARD_MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# Home route
@app.route("/")
def home():
    app.logger.info("Home route accessed.")
    return jsonify({"message": "Fraud Detection API is running!"})

# Prediction route
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

        # Convert the features to a DataFrame, assuming the order matches X_train columns
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

# Dashboard data endpoints
@app.route("/dashboard/summary", methods=["GET"])
def get_summary():
    # Example summary data, replace with actual logic
    data = {
        "total_transactions": 100000,  # Replace with actual value
        "fraud_cases": 1200,  # Replace with actual value
        "fraud_percentage": 1.2  # Replace with actual value
    }
    return jsonify(data)

@app.route("/dashboard/trends", methods=["GET"])
def get_fraud_trends():
    # Example trends data, replace with actual logic
    trends = {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],  # Replace with actual dates
        "fraud_cases": [100, 200, 150]  # Replace with actual fraud cases data
    }
    return jsonify(trends)

@app.route("/dashboard/geography", methods=["GET"])
def get_geographic_data():
    # Example geographic fraud analysis data, replace with actual data
    geo_data = {
        "locations": ["New York", "Los Angeles", "Chicago"],  # Replace with actual locations
        "fraud_cases": [50, 30, 40]  # Replace with actual fraud cases by location
    }
    return jsonify(geo_data)

@app.route("/dashboard/devices", methods=["GET"])
def get_device_data():
    # Example device fraud analysis data, replace with actual data
    device_data = {
        "devices": ["Desktop", "Mobile", "Tablet"],  # Replace with actual device types
        "fraud_cases": [800, 300, 100]  # Replace with actual fraud cases by device
    }
    return jsonify(device_data)

if __name__ == "__main__":
    # Ensure logs directory exists for logging
    if not os.path.exists("logs"):
        os.makedirs("logs")

    app.run(host="0.0.0.0", port=5000)
