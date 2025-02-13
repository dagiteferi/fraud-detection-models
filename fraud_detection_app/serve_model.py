import os
import joblib
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Define model paths
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

if __name__ == "__main__":
    # Ensure logs directory exists for logging
    if not os.path.exists("logs"):
        os.makedirs("logs")

    app.run(host="0.0.0.0", port=5000)
