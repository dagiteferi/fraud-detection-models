dash_app.layout = html.Div([
    html.Header([
        html.H1("Fraud Detection Dashboard", className="title"),
        html.P("Analyze and track fraud cases in real time.", className="subtitle")
    ], className="header"),
    
    # Summary Boxes
    html.Div([
        html.Div([
            html.H4("Total Transactions"),
            html.P(id="total-transactions", children="Loading..."),
        ], className="box"),
        html.Div([
            html.H4("Fraud Cases"),
            html.P(id="fraud-cases", children="Loading..."),
        ], className="box"),
        html.Div([
            html.H4("Fraud Percentage"),
            html.P(id="fraud-percentage", children="Loading..."),
        ], className="box"),
    ], className="summary-boxes"),
    
    # Line Chart for Fraud Trends
    dcc.Graph(id="fraud-trends", className="graph"),

    # Geographic Analysis
    dcc.Graph(id="geographic-fraud", className="graph"),

    # Device Analysis
    dcc.Graph(id="device-fraud", className="graph"),
], className="dashboard")
