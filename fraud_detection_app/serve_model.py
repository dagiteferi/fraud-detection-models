import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='logs/api.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')

@app.route('/')
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to Pandas DataFrame
        df = pd.DataFrame([data])

        # Ensure proper feature format
        if df.isnull().values.any():
            return jsonify({"error": "Missing values in input"}), 400

        # Make prediction
        prediction = model.predict(df)
        proba = model.predict_proba(df)[:, 1]  # Probability of fraud

        # Log the request
        logging.info(f"Input: {data}, Prediction: {prediction[0]}, Probability: {proba[0]:.4f}")

        return jsonify({"fraud_prediction": int(prediction[0]), "fraud_probability": round(proba[0], 4)})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
