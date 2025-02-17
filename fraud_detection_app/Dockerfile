FROM python:3.13.1-slim


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install all dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (if applicable)
EXPOSE 8050

# Define the command to run the application
CMD ["python", "fraud_detection_app/serve_model.py"]