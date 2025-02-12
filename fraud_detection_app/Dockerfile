# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents (including the models folder) into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 (Flask default port)
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "serve_model.py"]
