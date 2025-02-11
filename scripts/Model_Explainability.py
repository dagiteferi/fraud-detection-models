import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import logging
import os
import pandas as pd
from joblib import load
import numpy as np

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
        model_path: Path to the saved model file.
    Returns:
        Loaded model.
    """
    try:
        model = load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def explain_with_lime(model, X_train, y_train, X_test, instance_index=0):
    """
    Generate LIME explanations for the Random Forest model.
    Args:
        model: Trained model.
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Test data features.
        instance_index: Index of the test instance to explain.
    """
    try:
        # Ensure data is in pandas DataFrame/Series format
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)

        # Create a LIME explainer
        explainer = LimeTabularExplainer(
            X_train.values, 
            feature_names=X_train.columns,
            class_names=["Non-Fraud", "Fraud"],
            training_labels=y_train.values,
            discretize_continuous=True
        )
        
        # Explain a single instance
        explanation = explainer.explain_instance(X_test.iloc[instance_index].values, model.predict_proba)

        # Show explanation in the notebook
        logging.info("Generating LIME explanation for the instance.")
        explanation.show_in_notebook(show_all=False)

        # Feature importance plot
        explanation_list = explanation.as_list()
        logging.info(f"LIME feature importance for the instance: {explanation_list}")
        print(f"LIME feature importance for the instance: {explanation_list}")

        logging.info("LIME explanation completed successfully.")
        print("LIME explanation completed successfully.")
    except Exception as e:
        logging.error(f"Error in LIME explanation: {e}")
        print(f"Error in LIME explanation: {e}")

def generate_lime_explanation(model_path, X_train, y_train, X_test, instance_index=0):
    """
    Wrapper function to load a model and generate LIME explanations.
    Args:
        model_path: Path to the saved model.
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Test data features.
        instance_index: Index of the test instance to explain.
    """
    # Load the model
    model = load_model(model_path)

    # Generate LIME explanations
    explain_with_lime(model, X_train, y_train, X_test, instance_index)
