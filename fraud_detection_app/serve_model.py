import pickle
import logging
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Define model paths
FRAUD_MODEL_PATH = os.path.join("models", "random_forest_model_fraud.pkl")
CREDIT_CARD_MODEL_PATH = os.path.join("models", "randomforestfor_credit_card_data.pkl")

# Load the models
with open(FRAUD_MODEL_PATH, "rb") as f:
    fraud_model = pickle.load(f)

with open(CREDIT_CARD_MODEL_PATH, "rb") as f:
    credit_card_model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO, 
                    format="%(asctime)s %(levelname)s: %(message)s")

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Check if model type is provided
        model_type = data.get("model_type")
        if model_type not in ["fraud", "credit_card"]:
            return jsonify({"error": "Invalid model_type. Use 'fraud' or 'credit_card'"}), 400
        
        # Extract features (remove model_type from input)
        features = {k: v for k, v in data.items() if k != "model_type"}

        # Convert JSON to DataFrame
        df = pd.DataFrame([features])

        # Check for missing values
        if df.isnull().values.any():
            return jsonify({"error": "Missing values in input"}), 400

        # Select the appropriate model
        model = fraud_model if model_type == "fraud" else credit_card_model

        # Make prediction
        prediction = model.predict(df)
        proba = model.predict_proba(df)[:, 1]  # Probability of fraud

        # Log the request
        logging.info(f"Model: {model_type}, Input: {features}, Prediction: {prediction[0]}, Probability: {proba[0]:.4f}")

        return jsonify({"model_used": model_type, "fraud_prediction": int(prediction[0]), "fraud_probability": round(proba[0], 4)})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
