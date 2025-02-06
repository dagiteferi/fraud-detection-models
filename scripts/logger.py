import logging
import os

# Ensure logs directory exists
os.makedirs("../logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/eda.log"),  # Save logs to file
        logging.StreamHandler()  # Print logs in Jupyter notebook
    ]
)

# Create a logger instance
logger = logging.getLogger("EDA Logger")
