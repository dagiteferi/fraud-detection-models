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

def lime_explainability(model, X_train, X_test, instance_index=0):
    """
    Generate LIME explanations for a model.
    Args:
        model: Trained model.
        X_train: Training data features.
        X_test: Test data features.
        instance_index: Index of the test instance to explain.
    """
    try:
        # Convert X_train to a pandas DataFrame
        X_train_df = pd.DataFrame(X_train)
        
        # Create a LIME explainer
        lime_explainer = LimeTabularExplainer(
            X_train_df.values,
            feature_names=X_train_df.columns,
            class_names=["Not Fraud", "Fraud"],
            discretize_continuous=True
        )
        
        # Get LIME explanations for the specified test instance
        lime_explanation = lime_explainer.explain_instance(X_test[instance_index], model.predict_proba, num_features=5)
        
        # Display LIME feature importance plot
        lime_explanation.as_pyplot_figure()
        plt.show()
        
        # Log and print the LIME explanation details
        explanation_list = lime_explanation.as_list()
        logging.info(f"LIME feature importance for the instance: {explanation_list}")
        print(f"LIME feature importance for the instance: {explanation_list}")
        
        logging.info("LIME explanation completed successfully.")
        print("LIME explanation completed successfully.")
    except Exception as e:
        logging.error(f"Error in LIME explanation: {e}")
        print(f"Error in LIME explanation: {e}")

def generate_lime_explanation(model_path, X_train, X_test, instance_index=0):
    """
    Wrapper function to load a model and generate LIME explanations.
    Args:
        model_path: Path to the saved model.
        X_train: Training data features.
        X_test: Test data features.
        instance_index: Index of the test instance to explain.
    """
    # Load the model
    model = load_model(model_path)

    # Generate LIME explanations
    lime_explainability(model, X_train, X_test, instance_index)
