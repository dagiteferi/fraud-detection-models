import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import logging
import os
from joblib import load  # To load the saved model

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'model_explainability.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(model_path):
    """
    Load the trained model from a saved file.
    Args:
        model_path: Path to the saved model file
    Returns:
        Loaded model
    """
    try:
        model = load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def explain_with_lime(random_forest_model, X_train, y_train, X_test):
    """
    Generate LIME explanations for the Random Forest model
    Args:
        random_forest_model: Trained Random Forest model
        X_train: Training data features
        y_train: Training data labels
        X_test: Test data features
    """
    try:
        # Create a LIME explainer
        explainer_lime = LimeTabularExplainer(
            X_train.values, 
            training_labels=y_train.values, 
            mode='classification', 
            feature_names=X_train.columns,
            class_names=["Non-Fraud", "Fraud"],  # Class names for fraud detection
            discretize_continuous=True
        )
        
        # Explain a single instance (e.g., the first test instance)
        explanation = explainer_lime.explain_instance(X_test.iloc[0].values, random_forest_model.predict_proba)

        # Show explanation in the notebook
        logging.info("Generating LIME explanation for the first instance.")
        explanation.show_in_notebook()

        # Feature importance plot
        explanation_list = explanation.as_list()
        logging.info(f"LIME feature importance for the first instance: {explanation_list}")
        print(f"LIME feature importance for the first instance: {explanation_list}")

        logging.info("LIME explanation completed successfully.")
        print("LIME explanation completed successfully.")
    except Exception as e:
        logging.error(f"Error in LIME explanation: {e}")
        print(f"Error in LIME explanation: {e}")

# Example usage:

# Path to your saved model (replace with actual path)
model_path = 'path_to_saved_model.joblib'

# Load the model
random_forest_model = load_model(model_path)

# Assuming you already have X_train, y_train, and X_test from your dataset
# X_train, y_train, X_test should be pandas DataFrames (or numpy arrays) containing your data

# Generate LIME explanations
explain_with_lime(random_forest_model, X_train, y_train, X_test)
