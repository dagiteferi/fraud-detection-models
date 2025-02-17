# callbacks.py

from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

def register_callbacks(dash_app, fraud_data):
    @dash_app.callback(
        [
            Output("total-transactions", "children"),
            Output("fraud-cases", "children"),
            Output("fraud-percentage", "children"),
            Output("fraud-trends", "figure"),
            Output("geographic-fraud", "figure"),
            Output("device-fraud", "figure"),
            Output("browser-fraud", "figure")
        ],
        Input("fraud-trends", "id")  # Trigger update when the page loads
    )
    def update_dashboard(_):
        # Get total transactions and fraud cases
        total_transactions = len(fraud_data)
        fraud_cases = fraud_data[fraud_data['class'] == 1].shape[0]
        fraud_percentage = (fraud_cases / total_transactions) * 100
        
        # Create fraud trends data (time series of fraud cases)
        fraud_trends = fraud_data.resample('D').apply(lambda x: (x['class'] == 1).sum())
        fraud_trends_dates = fraud_trends.index.astype(str)

        # Example geographic data (assuming 'ip_address' is the location)
        fraud_by_location = fraud_data.groupby('ip_address')['class'].sum().sort_values(ascending=False).head(10)
        locations = fraud_by_location.index
        fraud_counts_by_location = fraud_by_location.values

        # Example device data 
        fraud_by_device = fraud_data.groupby('device_id')['class'].sum().sort_values(ascending=False).head(10)
        devices = fraud_by_device.index
        fraud_counts_by_device = fraud_by_device.values

        # Browser fraud analysis 
        browsers = ['browser_FireFox', 'browser_IE', 'browser_Opera', 'browser_Safari']
        fraud_by_browser = {browser: fraud_data[browser].sum() for browser in browsers}
        browser_names = list(fraud_by_browser.keys())
        fraud_counts_by_browser = list(fraud_by_browser.values())

        # Create figures using Plotly for the graphs
        fraud_trends_fig = {
            "data": [go.Scatter(x=fraud_trends_dates, y=fraud_trends, mode='lines')],
            "layout": go.Layout(title="Fraud Cases Over Time", xaxis={"title": "Date"}, yaxis={"title": "Fraud Cases"})
        }

        geo_fraud_fig = {
            "data": [go.Bar(x=locations, y=fraud_counts_by_location)],
            "layout": go.Layout(title="Geographic Fraud Analysis", xaxis={"title": "IP Address"}, yaxis={"title": "Fraud Cases"})
        }

        device_fraud_fig = {
            "data": [go.Bar(x=devices, y=fraud_counts_by_device)],
            "layout": go.Layout(title="Fraud Cases by Device", xaxis={"title": "Device ID"}, yaxis={"title": "Fraud Cases"})
        }

        browser_fraud_fig = {
            "data": [go.Bar(x=browser_names, y=fraud_counts_by_browser)],
            "layout": go.Layout(title="Fraud Cases by Browser", xaxis={"title": "Browser"}, yaxis={"title": "Fraud Cases"})
        }

        return total_transactions, fraud_cases, round(fraud_percentage, 2), fraud_trends_fig, geo_fraud_fig, device_fraud_fig, browser_fraud_fig
