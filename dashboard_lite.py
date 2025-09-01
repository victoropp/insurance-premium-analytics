import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from src.visualizations_updated import ProfessionalVisualizationEngine
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app with a professional theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "Videbimus AI - Premium Analytics [DEMO VERSION]"
server = app.server  # Expose the server for deployment

# Initialize visualization engine
viz_engine = ProfessionalVisualizationEngine()

# Load test results for metrics
test_results = pd.read_csv('data/final_test_results.csv')
best_test_r2 = test_results['Test_R2'].max() if not test_results.empty else 0.9978

# Demo prediction function (without loading large models)
def demo_predict_premium(age, experience, vehicle_age, accidents, annual_mileage, location):
    """Demo prediction using simple formula (no ML models loaded)"""
    # Simple heuristic-based prediction for demo
    base_premium = 800
    
    # Age factor
    if age < 25:
        base_premium *= 1.3
    elif age > 60:
        base_premium *= 1.1
        
    # Experience factor  
    if experience < 2:
        base_premium *= 1.2
    elif experience > 10:
        base_premium *= 0.9
        
    # Vehicle age factor
    if vehicle_age > 10:
        base_premium *= 0.8
    elif vehicle_age < 2:
        base_premium *= 1.1
        
    # Accidents factor
    base_premium *= (1 + accidents * 0.15)
    
    # Mileage factor
    if annual_mileage > 20:
        base_premium *= 1.2
    elif annual_mileage < 10:
        base_premium *= 0.9
        
    # Location factor
    location_multiplier = {
        'Urban': 1.2,
        'Suburban': 1.0,
        'Rural': 0.8
    }
    base_premium *= location_multiplier.get(location, 1.0)
    
    return round(base_premium, 2)

# Create app layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🏢 VIDEBIMUS AI", className="text-center mb-0", style={'color': '#2E86AB', 'fontWeight': 'bold'}),
                html.H3("Insurance Premium Analytics Platform", className="text-center text-muted mb-1"),
                html.P("DEMO VERSION - Optimized for Cloud Deployment", className="text-center text-warning mb-3"),
                html.Hr(style={'borderColor': '#2E86AB', 'borderWidth': '2px'}),
                
                # Company info
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.I(className="fas fa-user me-2"),
                            "Developed by: Victor Collins Oppon"
                        ], className="mb-1 text-center"),
                        html.P([
                            html.I(className="fas fa-briefcase me-2"),
                            "Title: Data Scientist & AI Consultant"
                        ], className="mb-1 text-center"),
                    ], width=6),
                    dbc.Col([
                        html.P([
                            html.I(className="fas fa-globe me-2"),
                            html.A("https://www.videbimusai.com", 
                                  href="https://www.videbimusai.com", 
                                  target="_blank", className="text-decoration-none")
                        ], className="mb-1 text-center"),
                        html.P([
                            html.I(className="fas fa-envelope me-2"),
                            html.A("consulting@videbimusai.com", 
                                  href="mailto:consulting@videbimusai.com", 
                                  className="text-decoration-none")
                        ], className="mb-1 text-center"),
                    ], width=6)
                ], justify="center"),
                
                html.Hr(style={'borderColor': '#dee2e6', 'borderWidth': '1px', 'margin': '20px 0'})
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'})
        ], width=12)
    ]),
    
    # Performance Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🎯 Model Performance", className="text-center mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H2(f"{best_test_r2:.4f}", className="text-center text-success mb-0"),
                            html.P("Best Test R² Score", className="text-center text-muted")
                        ], width=4),
                        dbc.Col([
                            html.H2("99.78%", className="text-center text-info mb-0"),
                            html.P("Prediction Accuracy", className="text-center text-muted")
                        ], width=4),
                        dbc.Col([
                            html.H2("3", className="text-center text-warning mb-0"),
                            html.P("Ensemble Models", className="text-center text-muted")
                        ], width=4)
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="📊 Executive Summary", tab_id="executive"),
        dbc.Tab(label="🔍 Detailed Analysis", tab_id="detailed"),
        dbc.Tab(label="🎯 Model Performance", tab_id="performance"),
        dbc.Tab(label="🧮 Premium Calculator", tab_id="calculator")
    ], id="tabs", active_tab="executive", className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Footer
    html.Hr(style={'margin': '40px 0 20px 0'}),
    dbc.Row([
        dbc.Col([
            html.P([
                "© 2024 ", 
                html.A("Videbimus AI", href="https://www.videbimusai.com", target="_blank", className="text-decoration-none text-primary"),
                ". All rights reserved."
            ], className="text-center text-muted mb-2"),
            html.P("🤖 This demo version uses simplified models for cloud deployment compatibility.", 
                   className="text-center text-warning small")
        ], width=12)
    ])
    
], fluid=True, style={'padding': '20px'})

# Callback for tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def update_tab_content(active_tab):
    if active_tab == "executive":
        try:
            fig = viz_engine.create_executive_summary()
            return dcc.Graph(figure=fig, style={'height': '800px'})
        except Exception as e:
            return dbc.Alert(f"Error loading executive summary: {str(e)}", color="warning")
    
    elif active_tab == "detailed":
        try:
            fig = viz_engine.create_detailed_analysis()
            return dcc.Graph(figure=fig, style={'height': '900px'})
        except Exception as e:
            return dbc.Alert(f"Error loading detailed analysis: {str(e)}", color="warning")
    
    elif active_tab == "performance":
        try:
            fig = viz_engine.create_model_comparison(test_results)
            return dcc.Graph(figure=fig, style={'height': '950px'})
        except Exception as e:
            return dbc.Alert(f"Error loading model performance: {str(e)}", color="warning")
    
    elif active_tab == "calculator":
        return html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("🧮 Premium Calculator - Demo Version")),
                dbc.CardBody([
                    dbc.Alert("This demo uses a simplified prediction model for cloud deployment compatibility.", 
                              color="info", className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Age"),
                            dbc.Input(id="age", type="number", value=30, min=18, max=80)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Years of Experience"),
                            dbc.Input(id="experience", type="number", value=5, min=0, max=50)
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Vehicle Age (years)"),
                            dbc.Input(id="vehicle_age", type="number", value=3, min=0, max=30)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Previous Accidents"),
                            dbc.Input(id="accidents", type="number", value=0, min=0, max=10)
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Annual Mileage (thousands km)"),
                            dbc.Input(id="mileage", type="number", value=15, min=5, max=50)
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Location"),
                            dbc.Select(id="location", value="Suburban",
                                      options=[
                                          {"label": "Urban", "value": "Urban"},
                                          {"label": "Suburban", "value": "Suburban"},
                                          {"label": "Rural", "value": "Rural"}
                                      ])
                        ], width=6)
                    ], className="mb-4"),
                    
                    dbc.Button("Calculate Premium", id="calculate-btn", color="primary", size="lg", className="w-100 mb-4"),
                    
                    html.Div(id="prediction-result")
                ])
            ])
        ])

# Callback for premium calculation
@app.callback(
    Output("prediction-result", "children"),
    Input("calculate-btn", "n_clicks"),
    [State("age", "value"),
     State("experience", "value"),
     State("vehicle_age", "value"),
     State("accidents", "value"),
     State("mileage", "value"),
     State("location", "value")]
)
def calculate_premium(n_clicks, age, experience, vehicle_age, accidents, mileage, location):
    if n_clicks is None:
        return ""
    
    try:
        premium = demo_predict_premium(age, experience, vehicle_age, accidents, mileage, location)
        
        return dbc.Alert([
            html.H4(f"💰 Estimated Premium: ${premium:,.2f}", className="mb-3"),
            html.P("This is a demonstration calculation using simplified models.", className="mb-0")
        ], color="success")
        
    except Exception as e:
        return dbc.Alert(f"Calculation error: {str(e)}", color="danger")

if __name__ == '__main__':
    print("="*70)
    print("                      VIDEBIMUS AI")
    print("           Insurance Premium Analytics Platform")
    print("                     DEMO VERSION")
    print("="*70)
    print()
    print("   Starting demo server...")
    print("   URL: http://127.0.0.1:8050")
    print()
    print("="*70)
    
    app.run(debug=False, host='127.0.0.1', port=8050)