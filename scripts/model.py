import os
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Setup logger
logger = logging.getLogger('fraud_detection_logger')
logger.setLevel(logging.DEBUG)

# Ensure logs directory exists
os.makedirs("../logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/model.log"),  # Save logs to file outside the notebook directory
        logging.StreamHandler()  # Print logs in Jupyter notebook
    ]
)

# Prepare data function
def prepare_data(df, target_column):
    logger.info(f"Preparing data by separating features and target column: {target_column}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# Split data function
def split_data(X, y, test_size=0.2, random_state=42):
    logger.info("Splitting data into train and test sets")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    
    
    # Feature and Target Separation for creditcard.csv
    X_credit, y_credit = prepare_data(credit_df, 'Class')
    
    # Train-Test Split for creditcard.csv
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = split_data(X_credit, y_credit)

    # Feature and Target Separation for Fraud_Data.csv
    X_fraud, y_fraud = prepare_data(fraud_df, 'class')
    
    # Train-Test Split for Fraud_Data.csv
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = split_data(X_fraud, y_fraud)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP": MLPClassifier()
    }

    # Step 1: Logistic Regression
    if "Logistic Regression" in models:
        model_name = "Logistic Regression"
        model = models[model_name]
        
        with mlflow.start_run(run_name=f"{model_name} - Credit Card Data"):
            logger.info(f"Training {model_name} for credit card data")
            model.fit(X_train_credit, y_train_credit)
            y_pred_credit = model.predict(X_test_credit)
            report_credit = classification_report(y_test_credit, y_pred_credit, output_dict=True)
            accuracy_credit = report_credit['accuracy']

            # Log parameters and metrics
            mlflow.log_param("model", model_name)
            mlflow.log_metric("accuracy", accuracy_credit)

            # Log the model
            mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_credit")

            logger.info(f"{model_name} - Credit Card Data:\n{classification_report(y_test_credit, y_pred_credit)}")

        with mlflow.start_run(run_name=f"{model_name} - Fraud Data"):
            logger.info(f"Training {model_name} for fraud data")
            model.fit(X_train_fraud, y_train_fraud)
            y_pred_fraud = model.predict(X_test_fraud)
            report_fraud = classification_report(y_test_fraud, y_pred_fraud, output_dict=True)
            accuracy_fraud = report_fraud['accuracy']

            # Log parameters and metrics
            mlflow.log_param("model", model_name)
            mlflow.log_metric("accuracy", accuracy_fraud)

            # Log the model
            mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_fraud")

            logger.info(f"{model_name} - Fraud Data:\n{classification_report(y_test_fraud, y_pred_fraud)}")
