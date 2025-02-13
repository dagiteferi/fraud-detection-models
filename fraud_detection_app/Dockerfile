# Use the official Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the application code (from the fraud_detection_app directory)
COPY . .  
# Now copies files from fraud_detection_app/ to /app

# Copy the models directory (from the parent directory of the build context)
COPY ../models/ ./models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python", "serve_model.py"]
