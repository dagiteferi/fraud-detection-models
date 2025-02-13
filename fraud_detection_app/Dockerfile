FROM python:3.8-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files from the project to /app
COPY . /app

# Set the entry point to start your model server
CMD ["python", "fraud_detection_app/serve_model.py"]
