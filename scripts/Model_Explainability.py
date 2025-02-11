import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import logging
import os

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'model_explainability.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
