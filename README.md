# 🚀 Fraud Detection System  

A machine learning-powered fraud detection system for e-commerce and banking transactions. This project includes data preprocessing, feature engineering, model training, explainability (SHAP & LIME), API development with Flask, deployment with Docker, and visualization with Dash.  

## 📌 Project Overview  
Fraud detection is critical for securing online transactions and banking operations. This system detects fraudulent activities in e-commerce and bank credit transactions using advanced machine learning techniques and real-time monitoring.  

## 📑 Table of Contents  
- [Project Overview](#-project-overview)  
- [Key Features](#-key-features)  
- [Project Structure](#-project-structure)  
- [Tech Stack](#-tech-stack)  
- [Installation](#-installation)  
- [Running with Docker](#-running-with-docker)  
- [Dashboard Features](#-dashboard-features)  
- [Model Explainability](#-model-explainability)  


### 🔹 Key Features  
✔️ Data preprocessing and feature engineering  
✔️ Fraud detection model training with multiple algorithms  
✔️ Model explainability using SHAP and LIME  
✔️ REST API for real-time fraud detection (Flask)  
✔️ Deployment using Docker  
✔️ Interactive fraud analysis dashboard (Dash)  

## 📂 Project Structure  
 ```bash
   --------------------------------------
```

## 🛠 Tech Stack  
- **Programming Language:** Python (Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch)  
- **Machine Learning Models:** Logistic Regression, Random Forest, Gradient Boosting, LSTM, CNN  
- **API & Deployment:** Flask, Docker  
- **Explainability:** SHAP, LIME  
- **Visualization:** Dash, Matplotlib, Seaborn  

## 🔧 Installation  

### 1️⃣ Clone the repository  
```sh
git clone https://github.com/dagiteferi/fraud-detection-models.git
cd fraud-detection-models
```
2️⃣ Create a virtual environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
###  3️⃣ Install dependencies
```sh
pip install -r requirements.txt
```
### 4️⃣ Run the API
```sh
cd fraud_detection_app
python serve_model.py
```
The API will run at http://127.0.0.1:5000/.
### 5️⃣ Run the Dashboard
```sh
cd fraud_detection_app
python serve_model.py
```
The Dashboard will run at http://127.0.0.1:5000/.
## 🚀 Running with Docker
1️⃣ Build the Docker Image
```sh
docker build -t fraud-detection-model -f fraud_detection_app/Dockerfile .

```
2️⃣ Run the Docker Container
```sh
docker run -d -p 5000:5000 --name fraud-detection-container fraud-detection-model
```
The API will be accessible at http://127.0.0.1:5000/ inside the container.

## 📊 Dashboard Features
## Dashboard Features

1. **Fraud Detection Summary**:
   - **Total Transactions**: Displays the total number of transactions in the dataset.
   - **Fraud Cases**: Shows the total number of fraudulent transactions.
   - **Fraud Percentage**: Displays the percentage of fraudulent transactions out of the total.

2. **Fraud Trends Over Time**:
   - A time series graph visualizing the number of fraud cases over time. It helps track how fraud patterns evolve.

3. **Geographic Fraud Analysis**:
   - A bar chart representing the fraud cases grouped by geographic locations (IP addresses). This helps identify regions with high fraudulent activity.

4. **Device-based Fraud Analysis**:
   - A bar chart showing fraud cases broken down by device ID, allowing identification of devices commonly used for fraudulent transactions.

5. **Browser-based Fraud Analysis**:
   - A bar chart comparing fraud cases across different browsers, helping to identify any browser-specific fraud patterns.


## 📌 Model Explainability
SHAP Summary & Force Plots for feature importance
LIME explanations for individual predictions
