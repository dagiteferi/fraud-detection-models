# Model_Explainability.py

import shap
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

def explain_with_shap(random_forest_model, X_test):
    """
    Generate SHAP explanations for the Random Forest model
    Args:
        random_forest_model: Trained Random Forest model
        X_test: Test data features
    """
    try:
        # Create a SHAP explainer object
        explainer = shap.TreeExplainer(random_forest_model)
        
        # Calculate SHAP values for the test set
        shap_values = explainer.shap_values(X_test)

        # Summary Plot
        logging.info("Generating SHAP summary plot.")
        shap.summary_plot(shap_values[1], X_test)
        plt.show()

        # Force Plot for a single prediction (e.g., the first instance)
        logging.info("Generating SHAP force plot for the first instance.")
        shap.force_plot(shap_values[1][0], X_test.iloc[0])
        plt.show()

        # Dependence Plot for a specific feature (e.g., 'Amount')
        logging.info("Generating SHAP dependence plot for 'Amount'.")
        shap.dependence_plot('Amount', shap_values[1], X_test)
        plt.show()

        logging.info("SHAP explanation completed successfully.")
        print("SHAP explanation completed successfully.")
    except Exception as e:
        logging.error(f"Error in SHAP explanation: {e}")
        print(f"Error in SHAP explanation: {e}")

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
        explainer_lime = LimeTabularExplainer(X_train.values, training_labels=y_train, mode='classification', training_mode='regression')
        
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
