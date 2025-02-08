import os
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

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
    # Load data using the notebook's data loader
    from src import data_loading as dl
    fraud_df = dl.load_data("processed/processed_fraud_data.csv")
    credit_df = dl.load_data("creditcard.csv")

    # Feature and Target Separation for creditcard.csv
    X_credit, y_credit = prepare_data(credit_df, 'Class')

    # Train-Test Split for creditcard.csv
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = split_data(X_credit, y_credit)

    # Feature and Target Separation for Fraud_Data.csv
    X_fraud, y_fraud = prepare_data(fraud_df, 'class')

    # Train-Test Split for Fraud_Data.csv
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = split_data(X_fraud, y_fraud)

    # Train Logistic Regression Model for Fraud Data
    with mlflow.start_run(run_name="Logistic Regression - Fraud Data"):
        logger.info("Training Logistic Regression for fraud data")
        logistic_model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')

        # Ensure feature data is in float64 to avoid MLflow warnings
        X_train_fraud = X_train_fraud.astype('float64')
        X_test_fraud = X_test_fraud.astype('float64')

        logistic_model.fit(X_train_fraud, y_train_fraud)
        y_pred_fraud = logistic_model.predict(X_test_fraud)
        report_fraud = classification_report(y_test_fraud, y_pred_fraud, output_dict=True)
        accuracy_fraud = report_fraud['accuracy']

        # Log parameters and metrics
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_metric("accuracy", accuracy_fraud)

        # Log the model
        mlflow.sklearn.log_model(logistic_model, "logistic_model_fraud", input_example=X_test_fraud[:5])

        logger.info(f"Logistic Regression - Fraud Data:\n{classification_report(y_test_fraud, y_pred_fraud)}")

    # Train Decision Tree Model for Fraud Data
    with mlflow.start_run(run_name="Decision Tree - Fraud Data"):
        logger.info("Training Decision Tree for fraud data")
        decision_tree_model = DecisionTreeClassifier()

        # Ensure feature data is in float64 to avoid MLflow warnings
        X_train_fraud = X_train_fraud.astype('float64')
        X_test_fraud = X_test_fraud.astype('float64')

        decision_tree_model.fit(X_train_fraud, y_train_fraud)
        y_pred_fraud = decision_tree_model.predict(X_test_fraud)
        report_fraud = classification_report(y_test_fraud, y_pred_fraud, output_dict=True)
        accuracy_fraud = report_fraud['accuracy']

        # Log parameters and metrics
        mlflow.log_param("model", "Decision Tree")
        mlflow.log_metric("accuracy", accuracy_fraud)

        # Log the model
        mlflow.sklearn.log_model(decision_tree_model, "decision_tree_model_fraud", input_example=X_test_fraud[:5])

        logger.info(f"Decision Tree - Fraud Data:\n{classification_report(y_test_fraud, y_pred_fraud)}")

    # Train Random Forest Model for Fraud Data
    with mlflow.start_run(run_name="Random Forest - Fraud Data"):
        logger.info("Training Random Forest for fraud data")
        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Ensure feature data is in float64 to avoid MLflow warnings
        X_train_fraud = X_train_fraud.astype('float64')
        X_test_fraud = X_test_fraud.astype('float64')

        random_forest_model.fit(X_train_fraud, y_train_fraud)
        y_pred_fraud = random_forest_model.predict(X_test_fraud)
        report_fraud = classification_report(y_test_fraud, y_pred_fraud, output_dict=True)
        accuracy_fraud = report_fraud['accuracy']

        # Log parameters and metrics
        mlflow.log_param("model", "Random Forest")
        mlflow.log_metric("accuracy", accuracy_fraud)

        # Log the model
        mlflow.sklearn.log_model(random_forest_model, "random_forest_model_fraud", input_example=X_test_fraud[:5])

        logger.info(f"Random Forest - Fraud Data:\n{classification_report(y_test_fraud, y_pred_fraud)}")

    # Train and evaluate Gradient Boosting model for Fraud_Data.csv
    with mlflow.start_run(run_name="Gradient Boosting - Fraud Data"):
        gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        # Ensure feature data is in float64 to avoid MLflow warnings
        X_train_fraud = X_train_fraud.astype('float64')
        X_test_fraud = X_test_fraud.astype('float64')

        gradient_boosting_model.fit(X_train_fraud, y_train_fraud)
        y_pred_fraud = gradient_boosting_model.predict(X_test_fraud)

        # Generate classification report
        report_fraud = classification_report(y_test_fraud, y_pred_fraud, output_dict=True)
        accuracy_fraud = report_fraud['accuracy']

        # Log parameters, metrics, and model
        mlflow.log_param("model", "Gradient Boosting")
        mlflow.log_metric("accuracy", accuracy_fraud)
        mlflow.sklearn.log_model(gradient_boosting_model, "gradient_boosting_model_fraud", input_example=X_test_fraud[:5])

        # Print classification report
        print("Gradient Boosting - Fraud Data:\n", classification_report(y_test_fraud, y_pred_fraud))


# Train and evaluate MLP model for Fraud_Data.csv
with mlflow.start_run(run_name="MLP - Fraud Data"):
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, solver='adam', random_state=42)
    
    # Ensure feature data is in float64 to avoid MLflow warnings
    X_train_fraud = X_train_fraud.astype('float64')
    X_test_fraud = X_test_fraud.astype('float64')

    mlp_model.fit(X_train_fraud, y_train_fraud)
    y_pred_fraud = mlp_model.predict(X_test_fraud)

    # Generate classification report
    report_fraud = classification_report(y_test_fraud, y_pred_fraud, output_dict=True)
    accuracy_fraud = report_fraud['accuracy']

    # Log parameters, metrics, and model
    mlflow.log_param("model", "MLP")
    mlflow.log_metric("accuracy", accuracy_fraud)
    mlflow.sklearn.log_model(mlp_model, "mlp_model_fraud", input_example=X_test_fraud[:5])

    # Print classification report
    print("MLP - Fraud Data:\n", classification_report(y_test_fraud, y_pred_fraud))
