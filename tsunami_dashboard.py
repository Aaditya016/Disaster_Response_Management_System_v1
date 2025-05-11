# tsunami_dashboard.py
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import plotly.express as px
import os
import json
from flask import Flask

# Define an enhanced color scheme
colors = {
    'background': '#f5f7fa',
    'card': '#ffffff',
    'primary': '#4e73df',
    'secondary': '#858796',
    'success': '#1cc88a',
    'danger': '#e74a3b',
    'warning': '#f6c23e',
    'info': '#36b9cc',
    'light': '#f8f9fc',
    'dark': '#5a5c69',
    'text': '#3a3b45',
    'border': '#e3e6f0'
}

# Load the model
print("Loading tsunami prediction model...")
model = None
try:
    model = joblib.load('tsunami_prediction_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Dashboard will run in limited mode.")

# Load the list of features used in the model
features = []
try:
    with open('model_features.txt', 'r') as f:
        features = f.read().splitlines()
    print(f"Loaded {len(features)} features: {features}")
except Exception as e:
    print(f"Error loading feature list: {e}")
    print("Using default feature list")
    features = ['magnitude', 'depth', 'latitude', 'longitude', 'sig', 'gap', 'dmin', 'mmi', 'magType_mb', 'magType_md', 'magType_ml', 'magType_ms', 'magType_mw', 'magType_mwb', 'magType_mwc', 'magType_mwr']

# USGS Earthquake API URL
USGS_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"

# Create a Dash application with custom stylesheets
external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap'
]

app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                external_stylesheets=external_stylesheets,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# Custom CSS for enhanced styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>TsunamiWatch AI Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Nunito', sans-serif;
            }
            body {
                background-color: #f5f7fa;
                color: #3a3b45;
            }
            .dashboard-container {
                max-width: 1800px;
                margin: 0 auto;
                padding: 20px;
            }
            .header-card {
                background: linear-gradient(135deg, #4e73df 0%, #224abe 100%);
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                padding: 25px;
                margin-bottom: 25px;
                color: white;
            }
            .card {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                padding: 25px;
                margin-bottom: 25px;
                border: 1px solid #e3e6f0;
            }
            .card-title {
                font-size: 1.2rem;
                font-weight: 700;
                color: #4e73df;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .stat-card {
                text-align: center;
                padding: 20px;
                border-radius: 8px;
                color: white;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
            .stat-value {
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 5px;
            }
            .stat-label {
                font-size: 14px;
                opacity: 0.9;
            }
            .refresh-btn {
                background-color: #4e73df;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                cursor: pointer;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 10px rgba(78, 115, 223, 0.3);
            }
            .refresh-btn:hover {
                background-color: #3a5bc7;
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(78, 115, 223, 0.4);
            }
            .last-updated {
                font-size: 14px;
                color: rgba(255,255,255,0.8);
                margin-top: 10px;
            }
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
                margin-bottom: 25px;
            }
            .map-container {
                height: 600px;
                position: relative;
            }
            .map-legend {
                position: absolute;
                bottom: 30px;
                right: 30px;
                background: rgba(255,255,255,0.95);
                padding: 15px;
                border-radius: 8px;
                z-index: 1000;
                font-size: 13px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                border: 1px solid #e3e6f0;
            }
            .risk-badge {
                padding: 6px 15px;
                border-radius: 20px;
                color: white;
                font-weight: 600;
                display: inline-block;
                min-width: 100px;
                text-align: center;
                font-size: 14px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .custom-input {
                width: 100%;
                padding: 12px;
                border: 1px solid #e3e6f0;
                border-radius: 6px;
                margin-bottom: 15px;
                transition: all 0.3s ease;
                font-size: 14px;
            }
            .custom-input:focus {
                border-color: #4e73df;
                box-shadow: 0 0 0 0.2rem rgba(78, 115, 223, 0.25);
                outline: none;
            }
            .calculate-btn {
                background: linear-gradient(135deg, #4e73df 0%, #224abe 100%);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                cursor: pointer;
                font-weight: 600;
                width: 100%;
                transition: all 0.3s ease;
                box-shadow: 0 4px 10px rgba(78, 115, 223, 0.3);
                margin-top: 10px;
            }
            .calculate-btn:hover {
                background: linear-gradient(135deg, #3a5bc7 0%, #1a3bb0 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(78, 115, 223, 0.4);
            }
            .section-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #4e73df;
                margin-bottom: 15px;
                border-bottom: 2px solid #e3e6f0;
                padding-bottom: 8px;
            }
            .footer {
                text-align: center;
                padding: 15px;
                font-size: 13px;
                color: #858796;
                margin-top: 20px;
            }
            .table-container {
                overflow-x: auto;
                border-radius: 8px;
                border: 1px solid #e3e6f0;
            }
            @media (max-width: 768px) {
                .grid-container {
                    grid-template-columns: 1fr;
                }
                .card {
                    padding: 15px;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Function to fetch recent earthquakes from USGS API
def fetch_recent_earthquakes():
    try:
        response = requests.get(USGS_API_URL)
        data = response.json()
        
        earthquakes = []
        for feature in data['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            
            earthquake = {
                'id': feature['id'],
                'magnitude': props.get('mag', 0),
                'location': props.get('place', 'Unknown'),
                'time': datetime.datetime.fromtimestamp(props.get('time', 0)/1000).strftime('%Y-%m-%d %H:%M:%S'),
                'depth': coords[2] if len(coords) > 2 else 0,
                'latitude': coords[1] if len(coords) > 1 else 0,
                'longitude': coords[0] if len(coords) > 0 else 0,
                'sig': props.get('sig', 0),
                'magType': props.get('magType', 'unknown'),
                'mmi': props.get('mmi', 0),
                'gap': props.get('gap', 0),
                'dmin': props.get('dmin', 0),
                'tsunami_risk': 'N/A'  # Will be calculated later
            }
            earthquakes.append(earthquake)
        
        return pd.DataFrame(earthquakes)
    except Exception as e:
        print(f"Error fetching earthquake data: {e}")
        return pd.DataFrame()

# Function to prepare data for prediction
def prepare_data_for_prediction(df):
    # Create a copy to avoid modifying the original
    prediction_df = df.copy()
    
    # Handle missing values for required features
    for feature in features:
        if feature.startswith('magType_'):
            # Handle categorical features
            mag_type = feature.replace('magType_', '')
            if 'magType' in prediction_df.columns:
                prediction_df[feature] = (prediction_df['magType'] == mag_type).astype(int)
            else:
                prediction_df[feature] = 0
        elif feature not in prediction_df.columns:
            prediction_df[feature] = 0
    
    # Select only the needed features for prediction
    prediction_data = prediction_df[features].copy()
    
    # Fill missing values with appropriate defaults
    numeric_cols = prediction_data.select_dtypes(include=[np.number]).columns
    prediction_data[numeric_cols] = prediction_data[numeric_cols].fillna(0)
    
    return prediction_data

# Function to predict tsunami risk
def predict_tsunami_risk(df):
    if model is None:
        return ["N/A"] * len(df)
    
    try:
        # Prepare data for prediction
        prediction_data = prepare_data_for_prediction(df)
        
        # Make predictions
        predictions = model.predict(prediction_data)
        probabilities = model.predict_proba(prediction_data)[:, 1]  # Probability of class 1
        
        # Return risk levels based on probabilities
        risk_levels = []
        for prob in probabilities:
            if prob < 0.2:
                risk_levels.append("Very Low")
            elif prob < 0.4:
                risk_levels.append("Low")
            elif prob < 0.6:
                risk_levels.append("Moderate")
            elif prob < 0.8:
                risk_levels.append("High")
            else:
                risk_levels.append("Very High")
        
        return risk_levels
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["Error"] * len(df)

# Helper function to get risk color for a given risk level
def get_risk_color(risk_level):
    if risk_level == "Very Low":
        return colors['info']
    elif risk_level == "Low":
        return colors['success']
    elif risk_level == "Moderate":
        return colors['warning']
    elif risk_level == "High":
        return colors['danger']
    elif risk_level == "Very High":
        return "#8b0000"  # Dark red
    else:
        return colors['secondary']

# Define the dashboard layout with all components on one page
app.layout = html.Div(className='dashboard-container', children=[
    # Header Section
    html.Div(className='header-card', children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
            html.Div(children=[
                html.H1(children=[
                    html.I(className="fas fa-water", style={'marginRight': '10px'}),
                    "TsunamiWatch AI Dashboard"
                ], style={'fontSize': '28px', 'fontWeight': '700', 'marginBottom': '5px'}),
                html.P("Real-time earthquake monitoring and tsunami risk assessment", style={'opacity': '0.9'})
            ]),
            html.Div(children=[
                html.Button(id="refresh-button", className="refresh-btn", children=[
                    html.I(className="fas fa-sync-alt"), 
                    " Refresh Data"
                ]),
                html.Div(id='last-updated', className='last-updated')
            ])
        ])
    ]),
    
    # Statistics Cards
    html.Div(className='grid-container', children=[
        html.Div(className='stat-card', style={'backgroundColor': colors['primary']}, children=[
            html.Div(id='total-earthquakes', className='stat-value', children="0"),
            html.Div(className='stat-label', children="Recent Earthquakes")
        ]),
        html.Div(className='stat-card', style={'backgroundColor': colors['success']}, children=[
            html.Div(id='avg-magnitude', className='stat-value', children="0.0"),
            html.Div(className='stat-label', children="Average Magnitude")
        ]),
        html.Div(className='stat-card', style={'backgroundColor': colors['danger']}, children=[
            html.Div(id='high-risk-count', className='stat-value', children="0"),
            html.Div(className='stat-label', children="High Risk Events")
        ]),
        html.Div(className='stat-card', style={'backgroundColor': colors['info']}, children=[
            html.Div(id='last-detection', className='stat-value', children="None"),
            html.Div(className='stat-label', children="Last Detection")
        ])
    ]),
    
    # Map Section
    html.Div(className='card', children=[
        html.Div(className='card-title', children=[
            html.I(className="fas fa-globe-americas"),
            "Global Earthquake Map"
        ]),
        dcc.Graph(
            id='earthquake-map',
            config={'displayModeBar': True, 'scrollZoom': True},
            style={'height': '600px'}
        ),
        html.Div(className='map-legend', children=[
            html.Div("Tsunami Risk Levels:", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '5px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Div(style={'width': '15px', 'height': '15px', 'backgroundColor': colors['info'], 'borderRadius': '50%'}),
                    html.Span("Very Low")
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Div(style={'width': '15px', 'height': '15px', 'backgroundColor': colors['success'], 'borderRadius': '50%'}),
                    html.Span("Low")
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Div(style={'width': '15px', 'height': '15px', 'backgroundColor': colors['warning'], 'borderRadius': '50%'}),
                    html.Span("Moderate")
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Div(style={'width': '15px', 'height': '15px', 'backgroundColor': colors['danger'], 'borderRadius': '50%'}),
                    html.Span("High")
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Div(style={'width': '15px', 'height': '15px', 'backgroundColor': "#8b0000", 'borderRadius': '50%'}),
                    html.Span("Very High")
                ])
            ])
        ])
    ]),
    
    # Data Table and Custom Prediction in a 2-column layout
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '25px', 'marginBottom': '25px'}, children=[
        # Data Table Section
        html.Div(className='card', children=[
            html.Div(className='card-title', children=[
                html.I(className="fas fa-table"),
                "Recent Earthquake Data"
            ]),
            html.Div(className='table-container', children=[
                dash_table.DataTable(
                    id='earthquake-table',
                    columns=[
                        {"name": "Magnitude", "id": "magnitude"},
                        {"name": "Location", "id": "location"},
                        {"name": "Time (UTC)", "id": "time"},
                        {"name": "Depth (km)", "id": "depth"},
                        {"name": "Risk", "id": "tsunami_risk"}
                    ],
                    data=[],
                    sort_action='native',
                    filter_action='native',
                    page_size=10,
                    style_table={'overflowX': 'auto', 'minWidth': '100%'},
                    style_header={
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'fontSize': '14px'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '12px',
                        'fontFamily': 'Nunito, sans-serif',
                        'fontSize': '14px',
                        'border': f'1px solid {colors["border"]}'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'tsunami_risk', 'filter_query': '{tsunami_risk} = "Very High"'},
                            'backgroundColor': 'rgba(139, 0, 0, 0.1)',
                            'color': '#8b0000',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'column_id': 'tsunami_risk', 'filter_query': '{tsunami_risk} = "High"'},
                            'backgroundColor': 'rgba(231, 74, 59, 0.1)',
                            'color': colors['danger'],
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'column_id': 'tsunami_risk', 'filter_query': '{tsunami_risk} = "Moderate"'},
                            'backgroundColor': 'rgba(246, 194, 62, 0.1)',
                            'color': colors['warning'],
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'column_id': 'magnitude', 'filter_query': '{magnitude} >= 6'},
                            'fontWeight': 'bold',
                            'color': colors['danger']
                        },
                        {
                            'if': {'state': 'active'},
                            'backgroundColor': 'rgba(78, 115, 223, 0.1)',
                            'border': f'1px solid {colors["primary"]}'
                        }
                    ],
                    style_as_list_view=True
                )
            ]),
            html.Div(style={'marginTop': '15px', 'fontSize': '13px', 'color': colors['secondary']}, children=[
                html.I(className="fas fa-info-circle", style={'marginRight': '5px'}),
                "Note: Tsunami risk is calculated based on earthquake parameters using a machine learning model."
            ])
        ]),
        
        # Custom Prediction Section
        html.Div(className='card', children=[
            html.Div(className='card-title', children=[
                html.I(className="fas fa-calculator"),
                "Custom Tsunami Risk Prediction"
            ]),
            html.P("Enter earthquake parameters to calculate tsunami risk.", style={'color': colors['text'], 'marginBottom': '20px'}),
            
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'}, children=[
                html.Div(children=[
                    html.Label("Magnitude", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '5px', 'color': colors['dark']}),
                    dcc.Input(
                        id='magnitude-input',
                        type='number',
                        min=0,
                        max=10,
                        step=0.1,
                        value=6.5,
                        className='custom-input',
                        placeholder="Enter magnitude (e.g., 6.5)"
                    ),
                ]),
                html.Div(children=[
                    html.Label("Depth (km)", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '5px', 'color': colors['dark']}),
                    dcc.Input(
                        id='depth-input',
                        type='number',
                        min=0,
                        max=1000,
                        step=1,
                        value=10,
                        className='custom-input',
                        placeholder="Enter depth (e.g., 10)"
                    ),
                ]),
                html.Div(children=[
                    html.Label("Latitude", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '5px', 'color': colors['dark']}),
                    dcc.Input(
                        id='latitude-input',
                        type='number',
                        min=-90,
                        max=90,
                        step=0.1,
                        value=35.0,
                        className='custom-input',
                        placeholder="Enter latitude (e.g., 35.0)"
                    ),
                ]),
                html.Div(children=[
                    html.Label("Longitude", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '5px', 'color': colors['dark']}),
                    dcc.Input(
                        id='longitude-input',
                        type='number',
                        min=-180,
                        max=180,
                        step=0.1,
                        value=140.0,
                        className='custom-input',
                        placeholder="Enter longitude (e.g., 140.0)"
                    ),
                ]),
                html.Div(children=[
                    html.Label("Magnitude Type", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '5px', 'color': colors['dark']}),
                    dcc.Dropdown(
                        id='magtype-dropdown',
                        options=[
                            {'label': 'mb (Body wave)', 'value': 'mb'},
                            {'label': 'md (Duration)', 'value': 'md'},
                            {'label': 'ml (Local)', 'value': 'ml'},
                            {'label': 'ms (Surface wave)', 'value': 'ms'},
                            {'label': 'mw (Moment)', 'value': 'mw'},
                        ],
                        value='mw',
                        clearable=False,
                        className='custom-input',
                        style={'border': 'none', 'padding': '0'}
                    ),
                ]),
                html.Div(children=[
                    html.Label("Significance", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '5px', 'color': colors['dark']}),
                    dcc.Input(
                        id='sig-input',
                        type='number',
                        min=0,
                        max=1000,
                        step=1,
                        value=600,
                        className='custom-input',
                        placeholder="Enter significance"
                    ),
                ]),
            ]),
            
            html.Button("Calculate Tsunami Risk", id="calculate-button", className="calculate-btn"),
            
            html.Div(id='risk-output', style={
                'marginTop': '25px',
                'textAlign': 'center',
                'padding': '20px',
                'borderRadius': '8px',
                'backgroundColor': colors['light'],
                'border': f'1px solid {colors["border"]}'
            }),
            
            html.Div(style={'marginTop': '15px', 'fontSize': '13px', 'color': colors['secondary']}, children=[
                html.I(className="fas fa-exclamation-triangle", style={'marginRight': '5px'}),
                "Note: This prediction is based on machine learning analysis and should not be used as the sole basis for emergency decisions."
            ])
        ])
    ]),
    
    # Footer
    html.Div(className='footer', children=[
        html.P([
            "TsunamiWatch AI Dashboard © 2025 | Powered by machine learning | Data source: ",
            html.A("USGS Earthquake API", href="https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php", target="_blank", style={'color': colors['primary']})
        ])
    ]),
    
    # Store component to save the earthquake data
    dcc.Store(id='earthquake-data'),
    
    # Interval component for auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # Update every 5 minutes
        n_intervals=0
    )
])

# Callback to update earthquake data when refresh button is clicked
@app.callback(
    Output('earthquake-data', 'data'),
    [Input('interval-component', 'n_intervals'),
     Input('refresh-button', 'n_clicks')])
def update_data(n_intervals, n_clicks):
    """Update earthquake data periodically or on button click"""
    earthquakes = fetch_recent_earthquakes()
    return earthquakes.to_dict('records') if not earthquakes.empty else []

@app.callback(
    Output('last-updated', 'children'),
    [Input('earthquake-data', 'data')])
def update_last_updated(data):
    """Update the last updated timestamp"""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"Last updated: {now}"

# Callback to update all components when data changes
@app.callback(
    [Output('earthquake-map', 'figure'),
     Output('earthquake-table', 'data'),
     Output('total-earthquakes', 'children'),
     Output('avg-magnitude', 'children'),
     Output('high-risk-count', 'children'),
     Output('last-detection', 'children')],
    [Input('earthquake-data', 'data')]
)
def update_all_components(data):
    if not data:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor=colors['card'],
            paper_bgcolor=colors['card'],
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'No earthquake data available',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        return empty_fig, [], "0", "0.0", "0", "None"
    
    earthquake_df = pd.DataFrame(data)
    
    # Add tsunami risk predictions if model is available
    if model is not None:
        earthquake_df['tsunami_risk'] = predict_tsunami_risk(earthquake_df)
    
    # Create map figure
    map_fig = go.Figure()
    
    # Add earthquake markers
    for i, row in earthquake_df.iterrows():
        risk = row['tsunami_risk']
        color = get_risk_color(risk)
        size = min(25, max(10, row['magnitude'] * 4))  # Scale dot size based on magnitude
        
        map_fig.add_trace(go.Scattergeo(
            lon=[row['longitude']],
            lat=[row['latitude']],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=[f"M{row['magnitude']} - {row['location']}<br>Time: {row['time']}<br>Depth: {row['depth']}km<br>Tsunami Risk: {risk}"],
            hoverinfo='text',
            name=f"M{row['magnitude']}"
        ))
    
    # Configure the map layout
    map_fig.update_layout(
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            oceancolor='rgb(217, 240, 252)',
            showocean=True,
            showcoastlines=True,
            coastlinecolor='rgb(180, 180, 180)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        showlegend=False,
        plot_bgcolor=colors['card'],
        paper_bgcolor=colors['card'],
    )
    
    # Calculate statistics
    total_earthquakes = len(earthquake_df)
    avg_magnitude = earthquake_df['magnitude'].mean()
    high_risk_count = sum(earthquake_df['tsunami_risk'].isin(['High', 'Very High'])) if 'tsunami_risk' in earthquake_df.columns else 0
    last_detection = earthquake_df.iloc[0]['time'] if total_earthquakes > 0 else "None"
    
    # Prepare table data
    table_data = earthquake_df[['magnitude', 'location', 'time', 'depth', 'tsunami_risk']].to_dict('records')
    
    return map_fig, table_data, str(total_earthquakes), f"{avg_magnitude:.1f}", str(high_risk_count), last_detection

# Callback to calculate tsunami risk for manual inputs
@app.callback(
    Output('risk-output', 'children'),
    [Input('calculate-button', 'n_clicks')],
    [State('magnitude-input', 'value'),
     State('depth-input', 'value'),
     State('latitude-input', 'value'),
     State('longitude-input', 'value'),
     State('magtype-dropdown', 'value'),
     State('sig-input', 'value')]
)
def calculate_tsunami_risk(n_clicks, magnitude, depth, latitude, longitude, magtype, sig):
    if n_clicks is None or n_clicks == 0:
        return ""
    
    if None in [magnitude, depth, latitude, longitude, magtype, sig]:
        return html.Div([
            html.I(className="fas fa-exclamation-circle", style={'marginRight': '8px', 'color': colors['danger']}),
            "Please fill in all required fields"
        ], style={'color': colors['danger']})
    
    try:
        # Create a dataframe with the input values
        input_df = pd.DataFrame({
            'magnitude': [magnitude],
            'depth': [depth],
            'latitude': [latitude],
            'longitude': [longitude],
            'magType': [magtype],
            'sig': [sig],
            'gap': [50],  # Default values
            'dmin': [1],   # Default values
            'mmi': [5]     # Default values
        })
        
        # Predict tsunami risk
        risk = predict_tsunami_risk(input_df)[0]
        risk_color = get_risk_color(risk)
        
        return html.Div([
            html.Div("Prediction Result", style={
                'fontSize': '16px',
                'fontWeight': '600',
                'color': colors['dark'],
                'marginBottom': '10px'
            }),
            html.Div(risk, style={
                'fontSize': '32px',
                'fontWeight': '700',
                'color': risk_color,
                'margin': '10px 0'
            }),
            html.Div(f"Magnitude: {magnitude} | Depth: {depth} km", style={
                'fontSize': '14px',
                'color': colors['secondary'],
                'marginTop': '10px'
            }),
            html.Div(f"Location: {latitude}°N, {longitude}°E", style={
                'fontSize': '14px',
                'color': colors['secondary']
            })
        ], style={'textAlign': 'center'})
    except Exception as e:
        print(f"Error in prediction: {e}")
        return html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px', 'color': colors['danger']}),
            "Error calculating risk. Please check inputs."
        ], style={'color': colors['danger']})

# Add a new callback to show loading state when refreshing data
@app.callback(
    Output('refresh-button', 'children'),
    [Input('refresh-button', 'n_clicks')],
    [State('refresh-button', 'children')]
)


def show_loading_state(n_clicks, current_children):
    if n_clicks is None:
        return current_children
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_children
    
    return [
        html.I(className="fas fa-spinner fa-spin", style={'marginRight': '8px'}),
        "Loading..."
    ]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)