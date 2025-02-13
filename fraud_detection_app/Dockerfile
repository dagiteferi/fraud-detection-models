# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install the required versions of numpy and scikit-learn
RUN pip install numpy==1.24.0 scikit-learn==1.3.2

# Install any other needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run serve_model.py when the container launches
CMD ["python", "fraud_detection_app/serve_model.py"]
