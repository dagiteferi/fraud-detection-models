import numpy as np
import pandas as pd
import lime
from lime.lime_tabular import LimeTabularExplainer
import logging
import os
from joblib import load
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

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

def lime_explainability(model, X_train, X_test):
    """
    Generate LIME explanations for a model.
    Args:
        model: Trained model.
        X_train: Training data features.
        X_test: Test data features.
    """
    try:
        # Convert X_train to a pandas DataFrame if it's not already
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        # Create a LIME explainer
        lime_explainer = LimeTabularExplainer(
            X_train.values, 
            feature_names=X_train.columns, 
            class_names=["Not Fraud", "Fraud"], 
            discretize_continuous=True
        )
        
        # Explain a single instance (e.g., the first test instance)
        lime_explanation = lime_explainer.explain_instance(X_test[0], model.predict_proba, num_features=5)
        
        # Display LIME feature importance plot
        lime_explanation.as_pyplot_figure()
        plt.show()

        logging.info("LIME explanation generated successfully.")
    except Exception as e:
        logging.error(f"Error in LIME explanation: {e}")
        print(f"Error in LIME explanation: {e}")

def global_feature_importance(model, X_train):
    """
    Plot global feature importance.
    Args:
        model: Trained model.
        X_train: Training data features (DataFrame).
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title("Global Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [X_train.columns[i] for i in indices], rotation=90)
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        plt.show()

        logging.info("Global feature importance plotted successfully.")
    except Exception as e:
        logging.error(f"Error in plotting global feature importance: {e}")
        print(f"Error in plotting global feature importance: {e}")

def pdp_ice_plots(model, X_train, features_to_plot):
    """
    Plot Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots.
    Args:
        model: Trained model.
        X_train: Training data features (DataFrame).
        features_to_plot: List of feature indices or names to plot.
    """
    try:
        # Plot Partial Dependence (PDP) and Individual Conditional Expectation (ICE)
        fig, ax = plt.subplots(figsize=(10, 6))
        display = PartialDependenceDisplay.from_estimator(model, X_train, features=features_to_plot, ax=ax)
        
        # Adding title dynamically based on the features being plotted
        feature_names = [X_train.columns[i] if isinstance(i, int) else i for i in features_to_plot]
        plt.title(f"PDP and ICE for Features: {', '.join(feature_names)}")

        display.plot(ax=ax)

        logging.info("PDP and ICE plots generated successfully.")
    except Exception as e:
        logging.error(f"Error in generating PDP and ICE plots: {e}")
        print(f"Error in generating PDP and ICE plots: {e}")

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_auc(model, X_test, y_test):
    """
    Plot ROC curve and calculate AUC for the uploaded model.
    Args:
        model: Loaded classification model.
        X_test: Features of the test set.
        y_test: True labels of the test set.
    """
    try:
        # Get predicted probabilities for the positive class (fraud)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1 (fraud)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        logging.info(f"ROC Curve plotted successfully with AUC = {roc_auc:.2f}.")
    except Exception as e:
        logging.error(f"Error in plotting ROC curve: {e}")
        print(f"Error in plotting ROC curve: {e}")

# Example usage:
# Assuming `model` is your loaded model, `X_test` is the test features, and `y_test` is the true labels.

# plot_roc_auc(model, X_test, y_test)

