import pandas as pd
from sklearn.preprocessing import StandardScaler
from scripts.logger import logger  # Import logger

class FeatureEngineering:
    def __init__(self, df):
        self.df = df
        self.processed_df = None
        self.logging = logger
        self.scaler = StandardScaler()

    def preprocess_datetime(self):
        """Extracts hour and day features from datetime columns."""
        self.logging.info("Extracting time-based features...")
        try:
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
            self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
            self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
            self.logging.info("Time-based features extracted successfully.")
        except Exception as e:
            self.logging.error("Error in extracting time-based features: %s", e)
            raise

    def calculate_transaction_frequency(self):
        """Calculates the transaction frequency and velocity for each user and device."""
        self.logging.info("Calculating transaction frequency and velocity...")
        try:
            # Ensure 'purchase_time' is in datetime format
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])

            # Sort the DataFrame by user_id and purchase_time
            self.df = self.df.sort_values(by=['user_id', 'purchase_time'])

            # Calculate the time difference between consecutive transactions for each user
            self.df['purchase_delay'] = self.df.groupby('user_id')['purchase_time'].diff().dt.total_seconds().fillna(0)

            # Transaction frequency per user
            user_freq = self.df.groupby('user_id').size()
            self.df['user_transaction_frequency'] = self.df['user_id'].map(user_freq)

            # Transaction frequency per device
            device_freq = self.df.groupby('device_id').size()
            self.df['device_transaction_frequency'] = self.df['device_id'].map(device_freq)

            # Transaction velocity: transactions per hour for each user
            self.df['user_transaction_velocity'] = self.df['user_transaction_frequency'] / (self.df['purchase_delay'] / 3600).replace(0, 1)  # Avoid division by zero
            self.logging.info("Transaction frequency and velocity calculated successfully.")
        except Exception as e:
            self.logging.error("Error in calculating transaction frequency and velocity: %s", e)
            raise

    def normalize_and_scale(self):
        """Normalizes and scales numerical features using StandardScaler."""
        self.logging.info("Normalizing and scaling numerical features...")
        try:
            numerical_features = [
                'purchase_value', 'user_transaction_frequency', 'device_transaction_frequency', 
                'user_transaction_velocity', 'hour_of_day', 'day_of_week', 'purchase_delay', 'age'
            ]
            self.df[numerical_features] = self.scaler.fit_transform(self.df[numerical_features])
            self.logging.info("Numerical features normalized and scaled successfully.")
        except Exception as e:
            self.logging.error("Error in normalizing and scaling numerical features: %s", e)
            raise

    def encode_categorical_features(self):
        """Encodes categorical features such as 'source', 'browser', and 'sex' using one-hot encoding."""
        self.logging.info("Encoding categorical features...")
        try:
            categorical_features = ['source', 'browser', 'sex']
            self.df = pd.get_dummies(self.df, columns=categorical_features, drop_first=True)
            self.logging.info("Categorical features encoded successfully.")
        except Exception as e:
            self.logging.error("Error in encoding categorical features: %s", e)
            raise

    def pipeline(self):
        """Executes the full feature engineering pipeline."""
        self.logging.info("Starting the feature engineering pipeline...")
        try:
            self.preprocess_datetime()
            self.calculate_transaction_frequency()
            self.normalize_and_scale()
            self.encode_categorical_features()
            self.processed_df = self.df
            self.logging.info("Feature engineering pipeline executed successfully.")
        except Exception as e:
            self.logging.error("Error in the feature engineering pipeline: %s", e)
            raise

    def get_processed_data(self) -> pd.DataFrame:
        """Returns the processed DataFrame with all the engineered features."""
        self.logging.info("Retrieving processed data...")
        if self.processed_df is None:
            self.logging.error("Data has not been processed. Run the pipeline() method first.")
            raise ValueError("Data has not been processed. Run the pipeline() method first.")
        self.logging.info("Processed data retrieved successfully.")
        return self.processed_df
