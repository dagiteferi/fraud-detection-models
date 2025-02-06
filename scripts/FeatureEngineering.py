import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class LoadData:
    """
    A class for preprocessing a dataset, including loading, cleaning, and handling missing values.

    Attributes:
    ----------
    filepath : str
        The file path of the dataset.
    logger : logging.Logger
        The logger instance for logging actions and errors.
    data : pd.DataFrame, optional
        The dataset loaded from the file path.
    """

    def __init__(self, filepath, logger):
        """
        Initializes the DataPreprocessor with a dataset filepath and logger.

        Parameters:
        ----------
        filepath : str
            The path to the dataset file (CSV format).
        logger : logging.Logger
            A logger instance for logging information and errors.
        """
        self.filepath = filepath
        self.logger = logger
        self.data = None
    
    def load_dataset(self):
        """
        Loads the dataset from the specified filepath.

        Returns:
        -------
        pd.DataFrame
            The loaded dataset as a DataFrame.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.logger.info("Dataset loaded successfully.")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None  # Return None if there's an error loading the dataset
        

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame, logging):
        """
        Initializes the FeatureEngineering class with the transaction data DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the transaction data.
            logging (logging.Logger): Logger instance for logging actions and errors.
        """
        self.df = df.copy()
        self.processed_df = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.logging = logging
        self.logging.info("FeatureEngineering class initialized with the provided DataFrame.")

    def preprocess_datetime(self):
        """
        Converts 'signup_time' and 'purchase_time' columns to datetime format and creates time-based features.
        """
        self.logging.info("Preprocessing datetime features...")
        try:
            # Convert 'signup_time' and 'purchase_time' to datetime
            self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])

            # Create time-based features
            self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
            self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
            self.df['purchase_delay'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds() / 3600  # Time difference in hours
            self.logging.info("Datetime features successfully created.")
        except Exception as e:
            self.logging.error(f"Error in preprocessing datetime features: {e}")
            raise

    def calculate_transaction_frequency(self):
        """
        Calculates the transaction frequency and velocity for each user and device.
        """
        self.logging.info("Calculating transaction frequency and velocity...")
        try:
            # Transaction frequency per user
            user_freq = self.df.groupby('user_id').size()
            self.df['user_transaction_frequency'] = self.df['user_id'].map(user_freq)

            # Transaction frequency per device
            device_freq = self.df.groupby('device_id').size()
            self.df['device_transaction_frequency'] = self.df['device_id'].map(device_freq)

            # Transaction velocity: transactions per hour for each user
            self.df['user_transaction_velocity'] = self.df['user_transaction_frequency'] / self.df['purchase_delay']
            self.logging.info("Transaction frequency and velocity calculated successfully.")
        except Exception as e:
            self.logging.error(f"Error in calculating transaction frequency and velocity: {e}")
            raise

    def normalize_and_scale(self):
        """
        Normalizes and scales numerical features using StandardScaler.
        Applies scaling to selected columns and stores the transformed DataFrame.
        """
        self.logging.info("Normalizing and scaling numerical features...")
        try:
            numerical_features = ['purchase_value', 'user_transaction_frequency', 'device_transaction_frequency', 
                                  'user_transaction_velocity', 'hour_of_day', 'day_of_week', 'purchase_delay', 'age']
            self.df[numerical_features] = self.scaler.fit_transform(self.df[numerical_features])
            self.logging.info("Numerical features normalized and scaled successfully.")
        except Exception as e:
            self.logging.error(f"Error in normalizing and scaling numerical features: {e}")
            raise

    def encode_categorical_features(self):
        """
        Encodes categorical features such as 'source', 'browser', and 'sex' using one-hot encoding.
        """
        self.logging.info("Encoding categorical features...")
        try:
            categorical_features = ['source', 'browser', 'sex']
            self.df = pd.get_dummies(self.df, columns=categorical_features, drop_first=True)
            self.logging.info("Categorical features encoded successfully.")
        except Exception as e:
            self.logging.error(f"Error in encoding categorical features: {e}")
            raise

    def pipeline(self):
        """
        Executes the full feature engineering pipeline, including time-based feature extraction, 
        transaction frequency/velocity calculation, normalization, scaling, and encoding categorical features.
        """
        self.logging.info("Starting the feature engineering pipeline...")
        try:
            self.preprocess_datetime()
            self.calculate_transaction_frequency()
            self.normalize_and_scale()
            self.encode_categorical_features()
            self.processed_df = self.df
            self.logging.info("Feature engineering pipeline executed successfully.")
        except Exception as e:
            self.logging.error(f"Error in the feature engineering pipeline: {e}")
            raise
    def get_processed_data(self) -> pd.DataFrame:
        """
        Returns the processed DataFrame with all the engineered features.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        self.logging.info("Retrieving processed data...")
        if self.processed_df is None:
            self.logging.error("Data has not been processed. Run the pipeline() method first.")
            raise ValueError("Data has not been processed. Run the pipeline() method first.")
        self.logging.info("Processed data retrieved successfully.")
        return self.processed_df