import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import os
from src.visualizations_updated import ProfessionalVisualizationEngine
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app with a professional theme
import time
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "Videbimus AI - Premium Analytics [UPDATED VERSION]"
server = app.server  # Expose the server for deployment

# Initialize visualization engine
viz_engine = ProfessionalVisualizationEngine()

# Load test results for metrics (memory optimized)
test_results = pd.read_csv('data/final_test_results.csv')
best_test_r2 = test_results['Test_R2'].max() if not test_results.empty else 0.9978

# Clear unnecessary data to save memory
import gc
gc.collect()

# Feature engineering function (matching the training pipeline)
def create_features(df):
    """Apply the same feature engineering as in training"""
    df_feat = df.copy()
    
    # Interaction features
    df_feat['Age_Experience_Ratio'] = df_feat['Driver Age'] / (df_feat['Driver Experience'] + 1)
    df_feat['Accidents_Per_Year_Driving'] = df_feat['Previous Accidents'] / (df_feat['Driver Experience'] + 1)
    df_feat['Mileage_Per_Year_Driving'] = df_feat['Annual Mileage (x1000 km)'] / (df_feat['Driver Experience'] + 1)
    df_feat['Car_Age_Driver_Age_Ratio'] = df_feat['Car Age'] / df_feat['Driver Age']
    df_feat['Experience_Rate'] = df_feat['Driver Experience'] / df_feat['Driver Age']
    
    # Polynomial features
    df_feat['Driver_Age_Squared'] = df_feat['Driver Age'] ** 2
    df_feat['Experience_Squared'] = df_feat['Driver Experience'] ** 2
    df_feat['Accidents_Squared'] = df_feat['Previous Accidents'] ** 2
    
    # Risk indicators
    df_feat['High_Risk_Driver'] = ((df_feat['Driver Age'] < 25) | (df_feat['Driver Age'] > 65)).astype(int)
    df_feat['New_Driver'] = (df_feat['Driver Experience'] < 2).astype(int)
    df_feat['Old_Car'] = (df_feat['Car Age'] > 10).astype(int)
    df_feat['High_Mileage'] = (df_feat['Annual Mileage (x1000 km)'] > 20).astype(int)
    
    # Composite risk score
    df_feat['Risk_Score'] = (
        df_feat['High_Risk_Driver'] * 2 + 
        df_feat['New_Driver'] * 3 + 
        df_feat['Previous Accidents'] * 4 + 
        df_feat['Old_Car'] * 1 +
        df_feat['High_Mileage'] * 1
    )
    
    return df_feat

# Load models and scaler
def load_models():
    models = {}
    # Only load the top 3 performing models based on test set results
    model_files = {
        'Stacking (Linear) - Best Performer': 'models/stacking_linear.pkl',
        'Stacking (Ridge)': 'models/stacking_ridge.pkl',
        'Voting Ensemble': 'models/voting_ensemble.pkl'
    }
    
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                models[name] = joblib.load(file)
            except:
                pass
    
    # Also load the scaler from training
    scaler = RobustScaler()
    # Fit scaler on training data
    df_train = pd.read_csv('data/insurance_tranining_dataset.csv')
    df_train_engineered = create_features(df_train)
    X_train = df_train_engineered.drop('Insurance Premium ($)', axis=1)
    scaler.fit(X_train)
    
    return models, scaler

available_models, scaler = load_models()

# Professional color scheme
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#73AB84',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6C91C2',
    'light': '#F5F5F5',
    'dark': '#2D3436'
}

# Navigation bar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.I(className="fas fa-chart-line fa-2x", style={'color': 'white', 'marginRight': '10px'}),
                html.Div([
                    dbc.NavbarBrand("Videbimus AI | Insurance Premium Analytics", className="ms-2", 
                                  style={'fontSize': '22px', 'fontWeight': 'bold'}),
                    html.Small("Developed by Victor Collins Oppon", 
                             style={'color': '#E0E0E0', 'fontSize': '11px', 'display': 'block', 'marginLeft': '45px'})
                ])
            ], width="auto"),
        ], align="center", className="g-0"),
        dbc.Row([
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Executive Summary", id="executive-link", href="#executive", style={'color': 'white', 'cursor': 'pointer'})),
                    dbc.NavItem(dbc.NavLink("Detailed Analysis", id="analysis-link", href="#analysis", style={'color': 'white', 'cursor': 'pointer'})),
                    dbc.NavItem(dbc.NavLink("Model Performance", id="models-link", href="#models", style={'color': 'white', 'cursor': 'pointer'})),
                    dbc.NavItem(dbc.NavLink("Premium Calculator", id="calculator-link", href="#calculator", style={'color': 'white', 'cursor': 'pointer'})),
                ], navbar=True)
            ], width="auto"),
        ], align="center"),
    ], fluid=True),
    color=colors['primary'],
    dark=True,
    className="mb-4",
    style={'boxShadow': '0 2px 4px rgba(0,0,0,.1)'}
)

# Key metrics cards
def create_metric_card(title, value, subtitle, icon, color):
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.I(className=f"fas {icon} fa-3x", style={'color': color, 'opacity': '0.3'})
                ], width=4),
                dbc.Col([
                    html.H6(title, className="text-muted mb-0", style={'fontSize': '14px'}),
                    html.H3(value, className="mb-0", style={'color': color, 'fontWeight': 'bold'}),
                    html.P(subtitle, className="text-muted mb-0", style={'fontSize': '12px'})
                ], width=8)
            ])
        ])
    ], className="mb-3", style={'border': 'none', 'boxShadow': '0 2px 4px rgba(0,0,0,.1)'})

metrics_row = dbc.Container([
    dbc.Row([
        dbc.Col(create_metric_card(
            "Total Records", 
            f"{len(viz_engine.df):,}", 
            "Dataset Size",
            "fa-database",
            colors['primary']
        ), width=3),
        dbc.Col(create_metric_card(
            "Average Premium", 
            f"${viz_engine.df['Insurance Premium ($)'].mean():.2f}", 
            "Mean Value",
            "fa-dollar-sign",
            colors['success']
        ), width=3),
        dbc.Col(create_metric_card(
            "Premium Range", 
            f"${viz_engine.df['Insurance Premium ($)'].max() - viz_engine.df['Insurance Premium ($)'].min():.2f}", 
            "Max - Min",
            "fa-chart-bar",
            colors['warning']
        ), width=3),
        dbc.Col(create_metric_card(
            "Best Model R²", 
            f"{best_test_r2:.4f}", 
            "Test Set Score",
            "fa-trophy",
            colors['info']
        ), width=3),
    ])
], fluid=False, className="px-0")

# Executive Summary Section
executive_section = dbc.Container([
    html.Div(id="executive"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-chart-pie me-2"),
                "Executive Summary"
            ], className="mb-3", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P([
                "This section provides a bird's-eye view of the insurance premium landscape. ",
                "The visualizations below show premium distributions, key risk factors, model performance metrics, ",
                "and demographic patterns that influence pricing decisions."
            ], className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    dbc.Card([
        dbc.CardBody([
            dcc.Graph(
                id='executive-summary-plot', 
                figure=viz_engine.create_executive_summary(),
                config={
                    'displayModeBar': 'hover',  # Show on hover
                    'displaylogo': False,
                    'modeBarButtonsToRemove': [],  # Keep all buttons
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'executive_summary',
                        'height': 800,
                        'width': 1200,
                        'scale': 1
                    }
                }
            )
        ])
    ], style={'border': 'none', 'boxShadow': '0 2px 8px rgba(0,0,0,.1)'})
], className="mb-5")

# Detailed Analysis Section
analysis_section = dbc.Container([
    html.Div(id="analysis"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-microscope me-2"),
                "Detailed Analysis"
            ], className="mb-3 mt-5", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P([
                "Explore the intricate relationships between various risk factors. ",
                "These detailed visualizations reveal how driver experience, vehicle age, accident history, ",
                "and annual mileage interact to determine insurance premiums. Use these insights to understand pricing patterns."
            ], className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    dbc.Card([
        dbc.CardBody([
            dcc.Graph(
                id='detailed-analysis-plot', 
                figure=viz_engine.create_detailed_analysis(),
                config={
                    'displayModeBar': 'hover',  # Show on hover
                    'displaylogo': False,
                    'modeBarButtonsToRemove': [],  # Keep all buttons
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'detailed_analysis',
                        'height': 800,
                        'width': 1200,
                        'scale': 1
                    }
                }
            )
        ])
    ], style={'border': 'none', 'boxShadow': '0 2px 8px rgba(0,0,0,.1)'})
], className="mb-5")

# Model Performance Section
models_section = dbc.Container([
    html.Div(id="models"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-robot me-2"),
                "Model Performance"
            ], className="mb-3 mt-5", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P([
                "Compare the performance of various machine learning models trained on insurance data. ",
                "The charts show accuracy metrics (R², RMSE, MAE) for different algorithms. ",
                "Our best model achieves 99.78% accuracy using advanced ensemble techniques."
            ], className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    dbc.Card([
        dbc.CardBody([
            dcc.Graph(
                id='model-comparison-plot', 
                figure=viz_engine.create_model_comparison(),
                config={
                    'displayModeBar': 'hover',  # Show on hover
                    'displaylogo': False,
                    'modeBarButtonsToRemove': [],  # Keep all buttons
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'model_comparison',
                        'height': 800,
                        'width': 1200,
                        'scale': 1
                    }
                }
            )
        ])
    ], style={'border': 'none', 'boxShadow': '0 2px 8px rgba(0,0,0,.1)'})
], className="mb-5")

# Premium Calculator Section with Explainability
calculator_section = dbc.Container([
    html.Div(id="calculator"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-calculator me-2"),
                "Interactive Premium Calculator"
            ], className="mb-3 mt-5", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            dbc.Alert([
                html.H6("How to Use the Premium Calculator:", className="alert-heading mb-2"),
                html.Ol([
                    html.Li("Enter driver information: age (18-80 years) and driving experience"),
                    html.Li("Specify accident history (0-10 previous accidents)"),
                    html.Li("Enter annual mileage in thousands of kilometers"),
                    html.Li("Input vehicle manufacturing year (vehicle age calculates automatically)"),
                    html.Li("Select a prediction model (Linear Stacking recommended)"),
                    html.Li("Click 'Calculate Premium' to get instant predictions with AI explanations")
                ], style={'fontSize': '13px', 'marginBottom': '10px'}),
                html.P([
                    html.I(className="fas fa-lightbulb me-2"),
                    html.Strong("Tip: "),
                    "The explainability charts show how each factor affects your premium and which features have the most impact."
                ], className="mb-0", style={'fontSize': '13px'})
            ], color="success", className="mb-4", 
               style={'backgroundColor': 'rgba(115, 171, 132, 0.1)', 'border': '1px solid #73AB84'})
        ])
    ]),
    dbc.Row([
        # Input Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-sliders-h me-2"),
                    html.H5("Input Parameters", className="mb-0")
                ], style={'backgroundColor': colors['primary'], 'color': 'white'}),
                dbc.CardBody([
                    # Driver Information
                    html.H6("Driver Information", className="text-muted mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Driver Age", className="fw-bold"),
                            dbc.Input(
                                id="input-age", 
                                type="number", 
                                value=35, 
                                min=18, 
                                max=80, 
                                step=1,
                                className="mb-3"
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Years of Experience", className="fw-bold"),
                            dbc.Input(
                                id="input-experience", 
                                type="number", 
                                value=10, 
                                min=0, 
                                max=60, 
                                step=1,
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Previous Accidents", className="fw-bold"),
                            dbc.Input(
                                id="input-accidents", 
                                type="number", 
                                value=0, 
                                min=0, 
                                max=10, 
                                step=1,
                                className="mb-3"
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Annual Mileage (x1000 km)", className="fw-bold"),
                            dbc.Input(
                                id="input-mileage", 
                                type="number", 
                                value=15, 
                                min=1, 
                                max=100, 
                                step=1,
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),
                    
                    html.Hr(),
                    
                    # Vehicle Information
                    html.H6("Vehicle Information", className="text-muted mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Manufacturing Year", className="fw-bold"),
                            dbc.Input(
                                id="input-car-year", 
                                type="number", 
                                value=2020, 
                                min=1980, 
                                max=2025, 
                                step=1,
                                className="mb-2"
                            ),
                            html.Small(id="vehicle-age-display", className="text-muted"),
                        ], width=12),
                    ]),
                    
                    html.Hr(),
                    
                    # Model Selection
                    dbc.Label("Select Model", className="fw-bold"),
                    dcc.Dropdown(
                        id="model-selector",
                        options=[
                            {"label": "Linear Stacking ⭐", "value": 'Stacking (Linear) - Best Performer'},
                            {"label": "Ridge Stacking", "value": 'Stacking (Ridge)'},
                            {"label": "Voting Ensemble", "value": 'Voting Ensemble'}
                        ] if available_models else [],
                        value='Stacking (Linear) - Best Performer' if 'Stacking (Linear) - Best Performer' in available_models else list(available_models.keys())[0] if available_models else None,
                        className="mb-3",
                        style={'fontWeight': '500'}
                    ),
                    
                    dbc.Button(
                        [html.I(className="fas fa-magic me-2"), "Calculate Premium"],
                        id="predict-button", 
                        color="primary", 
                        size="lg", 
                        className="w-100",
                        style={'fontWeight': 'bold'}
                    ),
                ])
            ], style={'border': 'none', 'boxShadow': '0 2px 8px rgba(0,0,0,.1)'})
        ], width=4),
        
        # Results Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-poll me-2"),
                    html.H5("Prediction Results", className="mb-0")
                ], style={'backgroundColor': colors['secondary'], 'color': 'white'}),
                dbc.CardBody([
                    html.Div(id="prediction-output", children=[
                        html.Div([
                            html.I(className="fas fa-hand-point-left fa-3x mb-3", 
                                  style={'color': colors['light'], 'opacity': '0.5'}),
                            html.H5("Ready to Calculate Your Premium", 
                                   className="text-muted"),
                            html.P("Enter your information in the left panel", 
                                  className="text-muted mb-3"),
                            html.Hr(style={'width': '50%', 'margin': '20px auto'}),
                            html.P([
                                html.Strong("What you'll receive:"), html.Br(),
                                "✓ Accurate premium prediction", html.Br(),
                                "✓ Risk level assessment", html.Br(),
                                "✓ Comparison to average premiums", html.Br(),
                                "✓ Detailed factor analysis", html.Br(),
                                "✓ Interactive explanations"
                            ], className="text-muted", style={'fontSize': '13px'})
                        ], className="text-center", style={'padding': '40px'})
                    ])
                ])
            ], style={'border': 'none', 'boxShadow': '0 2px 8px rgba(0,0,0,.1)', 'minHeight': '500px'})
        ], width=8),
    ]),
    
    # Explainability Section
    html.Hr(className="my-5"),
    dbc.Row([
        dbc.Col([
            html.H4([
                html.I(className="fas fa-lightbulb me-2"),
                "Model Explainability"
            ], className="mb-4", style={'color': colors['dark'], 'fontWeight': 'bold'})
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='feature-importance-plot', config={
                        'displayModeBar': 'hover',  # Show on hover
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [],  # Keep all buttons
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'feature_importance',
                            'height': 400,
                            'width': 600,
                            'scale': 1
                        }
                    })
                ])
            ], style={'border': 'none', 'boxShadow': '0 2px 8px rgba(0,0,0,.1)'})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='sensitivity-analysis-plot', config={
                        'displayModeBar': 'hover',  # Show on hover
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [],  # Keep all buttons
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'sensitivity_analysis',
                            'height': 400,
                            'width': 600,
                            'scale': 1
                        }
                    })
                ])
            ], style={'border': 'none', 'boxShadow': '0 2px 8px rgba(0,0,0,.1)'})
        ], width=6),
    ]),
], className="mb-5")

# Footer
footer = dbc.Container([
    html.Hr(style={'marginTop': '50px'}),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Victor Collins Oppon", className="text-center mb-1", 
                       style={'color': colors['primary'], 'fontWeight': 'bold'}),
                html.P("Data Scientist & AI Consultant", className="text-center mb-2", 
                      style={'color': colors['secondary'], 'fontSize': '14px', 'fontWeight': '500'}),
                html.P([
                    html.I(className="fas fa-building me-2"),
                    html.A("Videbimus AI", 
                          href="https://www.videbimusai.com",
                          target="_blank",
                          style={'color': 'inherit', 'textDecoration': 'none'}),
                    html.Span(" | ", className="mx-2"),
                    html.I(className="fas fa-envelope me-2"),
                    html.A("consulting@videbimusai.com",
                          href="mailto:consulting@videbimusai.com",
                          style={'color': 'inherit', 'textDecoration': 'none'}),
                    html.Span(" | ", className="mx-2"),
                    html.I(className="fas fa-globe me-2"),
                    html.A("https://www.videbimusai.com",
                          href="https://www.videbimusai.com",
                          target="_blank",
                          style={'color': 'inherit', 'textDecoration': 'none'}),
                    html.Span(" | ", className="mx-2"),
                    html.I(className="fab fa-whatsapp me-2", style={'color': '#25D366'}),
                    html.A("+233 277 390 051", 
                          href="https://wa.me/233277390051?text=Hello%20Victor,%20I'm%20interested%20in%20your%20AI%20consulting%20services",
                          target="_blank",
                          style={'color': '#25D366', 'textDecoration': 'none', 'fontWeight': '500'})
                ], className="text-center text-muted", style={'fontSize': '13px'}),
                html.Hr(style={'width': '50%', 'margin': '20px auto'}),
                html.P([
                    html.I(className="fas fa-copyright me-1"),
                    "2025 ",
                    html.A("Videbimus AI", 
                          href="https://www.videbimusai.com",
                          target="_blank",
                          style={'color': 'inherit', 'textDecoration': 'none'}),
                    ". All Rights Reserved | ",
                    html.Span("Powered by Advanced Machine Learning & AI", 
                            style={'fontStyle': 'italic'})
                ], className="text-center text-muted", style={'fontSize': '12px'})
            ])
        ])
    ])
], className="mt-5", style={'paddingBottom': '30px'})

# Welcome Section
welcome_section = dbc.Container([
    dbc.Alert([
        html.H4([html.I(className="fas fa-info-circle me-2"), "Welcome to the Insurance Premium Analytics Platform"], className="alert-heading"),
        html.Hr(),
        html.P([
            "This advanced analytics dashboard leverages machine learning to predict car insurance premiums with ",
            html.Strong("99.78% accuracy"),
            ". Developed by ", html.Strong("Victor Collins Oppon"), " at ", html.Strong("Videbimus AI"),
            ", this platform provides comprehensive insights into insurance risk factors and pricing."
        ], className="mb-3"),
        html.P([
            html.I(className="fas fa-chart-bar me-2"),
            html.Strong("Executive Summary: "), "High-level overview of premium distributions and key metrics",
            html.Br(),
            html.I(className="fas fa-search me-2"),
            html.Strong("Detailed Analysis: "), "Deep dive into specific risk factors and their relationships",
            html.Br(),
            html.I(className="fas fa-robot me-2"),
            html.Strong("Model Performance: "), "Comparison of machine learning models and their accuracy",
            html.Br(),
            html.I(className="fas fa-calculator me-2"),
            html.Strong("Premium Calculator: "), "Interactive tool to predict premiums with explainable AI"
        ], className="mb-0"),
    ], color="info", dismissable=True, fade=True, className="mb-4", is_open=True, 
       style={'backgroundColor': 'rgba(108, 145, 194, 0.1)', 'border': '1px solid #6C91C2'})
], className="mt-3")

# App Layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        welcome_section,
        metrics_row,
        html.Hr(className="my-4"),
        executive_section,
        analysis_section,
        models_section,
        calculator_section,
        footer,
        html.Div(id='dummy-output', style={'display': 'none'})  # Hidden div for navigation callback
    ], fluid=True, style={'backgroundColor': '#f8f9fa'})
])

# Callback to display calculated vehicle age
@app.callback(
    Output('vehicle-age-display', 'children'),
    [Input('input-car-year', 'value')]
)
def update_vehicle_age_display(car_year):
    if car_year is None:
        return "Vehicle age will be calculated automatically"
    
    from datetime import datetime
    current_year = datetime.now().year
    
    if car_year > current_year + 1:
        return f"⚠️ Invalid year (max: {current_year + 1})"
    elif car_year < 1980:
        return "⚠️ Invalid year (min: 1980)"
    else:
        calculated_age = max(0, current_year - car_year)
        if calculated_age == 0:
            return f"✓ Brand new vehicle (0 years old)"
        elif calculated_age == 1:
            return f"✓ Vehicle age: 1 year old"
        else:
            return f"✓ Vehicle age: {calculated_age} years old"

# Callback for predictions
@app.callback(
    [Output('prediction-output', 'children'),
     Output('feature-importance-plot', 'figure'),
     Output('sensitivity-analysis-plot', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [State('input-age', 'value'),
     State('input-experience', 'value'),
     State('input-accidents', 'value'),
     State('input-mileage', 'value'),
     State('input-car-year', 'value'),
     State('model-selector', 'value')]
)
def predict_premium(n_clicks, age, experience, accidents, mileage, car_year, model_name):
    if n_clicks is None or not model_name or model_name not in available_models:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Click 'Calculate Premium' to see analysis",
            height=300,
            template='plotly_white'
        )
        return [
            html.Div([
                html.I(className="fas fa-hand-point-left fa-3x mb-3", 
                      style={'color': colors['light'], 'opacity': '0.5'}),
                html.H5("Enter parameters and click 'Calculate Premium'", 
                       className="text-muted"),
                html.P("The system will provide detailed predictions and explanations", 
                      className="text-muted")
            ], className="text-center", style={'padding': '40px'})
        ], empty_fig, empty_fig
    
    # Validate inputs
    if age is None or experience is None or accidents is None or mileage is None or car_year is None:
        error_output = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Input Error: "),
            "Please fill in all fields with valid values"
        ], color="warning")
        
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Missing input values", height=300)
        return error_output, empty_fig, empty_fig
    
    # Calculate vehicle age automatically from manufacturing year
    from datetime import datetime
    current_year = datetime.now().year
    
    # Validate manufacturing year
    if car_year < 1980:
        car_year = 1980  # Set minimum year
    elif car_year > current_year + 1:  # Allow next year's model
        car_year = current_year + 1
    
    # Calculate car age automatically based on current year
    car_age = max(0, current_year - car_year)
    
    # Ensure experience doesn't exceed age - 16 (minimum driving age)
    if experience > age - 16:
        experience = max(0, age - 16)
    
    # Create input dataframe with original features
    input_data = pd.DataFrame({
        'Driver Age': [float(age)],
        'Driver Experience': [float(experience)],
        'Previous Accidents': [float(accidents)],
        'Annual Mileage (x1000 km)': [float(mileage)],
        'Car Manufacturing Year': [float(car_year)],
        'Car Age': [float(car_age)]
    })
    
    # Apply feature engineering
    input_engineered = create_features(input_data)
    
    # Scale features
    input_scaled = scaler.transform(input_engineered)
    
    try:
        # Make prediction
        model = available_models[model_name]
        prediction = model.predict(input_scaled)[0]
        
        # Calculate statistics
        avg_premium = viz_engine.df['Insurance Premium ($)'].mean()
        std_premium = viz_engine.df['Insurance Premium ($)'].std()
        percentile = (viz_engine.df['Insurance Premium ($)'] < prediction).mean() * 100
        
        # Determine risk level
        if prediction < avg_premium - std_premium:
            risk_level = "Low Risk"
            risk_color = colors['success']
            risk_icon = "fa-shield-alt"
        elif prediction < avg_premium:
            risk_level = "Below Average Risk"
            risk_color = colors['info']
            risk_icon = "fa-check-circle"
        elif prediction < avg_premium + std_premium:
            risk_level = "Above Average Risk"
            risk_color = colors['warning']
            risk_icon = "fa-exclamation-triangle"
        else:
            risk_level = "High Risk"
            risk_color = colors['danger']
            risk_icon = "fa-exclamation-circle"
        
        # Enhanced Risk Assessment Visualization
        # Create risk gauge figure
        risk_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={'reference': avg_premium, 'relative': False, 'valueformat': '.2f'},
            title={'text': '<b>Premium Assessment</b>', 'font': {'size': 14}},
            number={'prefix': "$", 'valueformat': '.2f', 'font': {'size': 24, 'color': colors['primary']}},
            gauge={
                'axis': {'range': [viz_engine.df['Insurance Premium ($)'].min(), 
                                  viz_engine.df['Insurance Premium ($)'].quantile(0.95)],
                        'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': risk_color, 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [viz_engine.df['Insurance Premium ($)'].min(), avg_premium - std_premium], 
                     'color': colors['success'], 'name': 'Low'},
                    {'range': [avg_premium - std_premium, avg_premium], 
                     'color': colors['info'], 'name': 'Below Avg'},
                    {'range': [avg_premium, avg_premium + std_premium], 
                     'color': colors['warning'], 'name': 'Above Avg'},
                    {'range': [avg_premium + std_premium, viz_engine.df['Insurance Premium ($)'].quantile(0.95)], 
                     'color': colors['danger'], 'name': 'High'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': avg_premium
                }
            }
        ))
        
        risk_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            template='plotly_white'
        )
        
        # Create risk factors breakdown
        risk_factors = {
            'Age Risk': 1 if age < 25 or age > 65 else 0,
            'Experience Risk': 1 if experience < 3 else 0,
            'Accident Risk': min(accidents / 2, 1),
            'Mileage Risk': 1 if mileage > 30 else 0,
            'Vehicle Risk': 1 if car_age > 10 else 0
        }
        
        risk_breakdown = go.Figure(go.Bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            marker=dict(
                color=[colors['danger'] if v > 0.7 else colors['warning'] if v > 0.3 else colors['success'] 
                      for v in risk_factors.values()],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{v*100:.0f}%' for v in risk_factors.values()],
            textposition='outside',
            textfont=dict(size=11, weight='bold')
        ))
        
        risk_breakdown.update_layout(
            title={'text': '<b>Risk Factor Analysis</b>', 'font': {'size': 14}},
            height=250,
            xaxis=dict(range=[0, 1.2], title='Risk Level', tickformat='.0%'),
            yaxis=dict(title=''),
            template='plotly_white',
            margin=dict(l=100, r=20, t=40, b=40)
        )
        
        # Create prediction output with enhanced layout
        prediction_output = html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=risk_gauge, config={
                        'displayModeBar': 'hover',  # Show on hover
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [],  # Keep all buttons
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'risk_gauge',
                            'height': 300,
                            'width': 400,
                            'scale': 1
                        }
                    })
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=risk_breakdown, config={
                        'displayModeBar': 'hover',  # Show on hover
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [],  # Keep all buttons
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'risk_breakdown',
                            'height': 300,
                            'width': 400,
                            'scale': 1
                        }
                    })
                ], width=6)
            ]),
            
            html.Hr(style={'margin': '20px 0'}),
            
            # Enhanced risk metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className=f"fas {risk_icon}", 
                                      style={'fontSize': '2.5rem', 'color': risk_color}),
                                html.H4(risk_level, 
                                       style={'color': risk_color, 'fontWeight': 'bold', 'marginTop': '10px'}),
                                html.Hr(style={'margin': '10px 0'}),
                                html.P([
                                    html.Strong("Percentile: "),
                                    f"{percentile:.1f}%"
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("Risk Category: "),
                                    f"{int((percentile-1)//20) + 1}/5"
                                ], className="mb-0")
                            ], className="text-center")
                        ])
                    ], style={'height': '100%', 'border': f'2px solid {risk_color}', 'borderRadius': '10px'})
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-dollar-sign", 
                                      style={'fontSize': '2.5rem', 'color': colors['primary']}),
                                html.H4(f"${prediction:.2f}", 
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                                html.Hr(style={'margin': '10px 0'}),
                                html.P([
                                    html.Strong("vs Average: "),
                                    f"${abs(prediction - avg_premium):.2f}"
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("Direction: "),
                                    f"{'Above' if prediction > avg_premium else 'Below'} Avg"
                                ], className="mb-0")
                            ], className="text-center")
                        ])
                    ], style={'height': '100%', 'border': f'2px solid {colors["primary"]}', 'borderRadius': '10px'})
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-tachometer-alt", 
                                      style={'fontSize': '2.5rem', 'color': colors['secondary']}),
                                html.H4(f"{int(input_engineered['Risk_Score'].iloc[0])}/20", 
                                       style={'fontWeight': 'bold', 'marginTop': '10px'}),
                                html.Hr(style={'margin': '10px 0'}),
                                html.P([
                                    html.Strong("Composite: "),
                                    f"{(input_engineered['Risk_Score'].iloc[0]/20)*100:.0f}%"
                                ], className="mb-1"),
                                html.P([
                                    html.Strong("Confidence: "),
                                    "High"
                                ], className="mb-0")
                            ], className="text-center")
                        ])
                    ], style={'height': '100%', 'border': f'2px solid {colors["secondary"]}', 'borderRadius': '10px'})
                ], width=4)
            ], className="mb-3"),
            
            # Detailed breakdown
            dbc.Card([
                dbc.CardHeader(html.H6("Detailed Profile Analysis", className="mb-0", style={'fontWeight': 'bold'})),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P([html.I(className="fas fa-user me-2"), 
                                   html.Strong("Driver: "), 
                                   f"{int(age)} yrs, {int(experience)} yrs exp"], className="mb-2"),
                            html.P([html.I(className="fas fa-car-crash me-2"), 
                                   html.Strong("Safety: "), 
                                   f"{int(accidents)} accident{'s' if accidents != 1 else ''}"], className="mb-2")
                        ], width=6),
                        dbc.Col([
                            html.P([html.I(className="fas fa-car me-2"), 
                                   html.Strong("Vehicle: "), 
                                   f"{int(car_year)}, {int(car_age)} yrs old"], className="mb-2"),
                            html.P([html.I(className="fas fa-road me-2"), 
                                   html.Strong("Usage: "), 
                                   f"{int(mileage)},000 km/year"], className="mb-2")
                        ], width=6)
                    ])
                ])
            ], style={'borderRadius': '10px'})
        ])
        
        # Simplified Feature Comparison
        feature_names = ['Age', 'Exp.', 'Accidents', 'Mileage', 'Car Year', 'Car Age']
        feature_values = [age, experience, accidents, mileage, car_year, car_age]
        avg_values = [viz_engine.df[col].mean() for col in input_data.columns]
        std_values = [viz_engine.df[col].std() for col in input_data.columns]
        
        # Calculate percentage differences instead of deviations
        pct_diff = [((feature_values[i] - avg_values[i]) / (avg_values[i] + 0.001)) * 100 for i in range(len(feature_values))]
        
        # Create single clean comparison chart
        from plotly.subplots import make_subplots
        importance_fig = go.Figure()
        
        # Add baseline (average) as a subtle line
        importance_fig.add_trace(go.Scatter(
            x=feature_names,
            y=avg_values,
            mode='lines+markers',
            name='Population Average',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=2, dash='dash'),
            marker=dict(size=8, color='rgba(150, 150, 150, 0.5)'),
            text=[f'{v:.0f}' for v in avg_values],
            textposition='top center',
            textfont=dict(size=9, color='gray')
        ))
        
        # Add user's values as prominent bars
        bar_colors = []
        for i, diff in enumerate(pct_diff):
            if abs(diff) > 50:
                bar_colors.append(colors['danger'] if diff > 0 else colors['success'])
            elif abs(diff) > 20:
                bar_colors.append(colors['warning'])
            else:
                bar_colors.append(colors['info'])
        
        importance_fig.add_trace(go.Bar(
            x=feature_names,
            y=feature_values,
            name='Your Profile',
            marker=dict(
                color=bar_colors,
                line=dict(color='rgba(0,0,0,0.2)', width=1),
                opacity=0.8
            ),
            text=[f'{v:.0f}<br>({d:+.0f}%)' for v, d in zip(feature_values, pct_diff)],
            textposition='outside',
            textfont=dict(size=10, weight='bold'),
            width=0.6
        ))
        
        importance_fig.update_layout(
            title={'text': '<b>Your Profile vs Population Average</b>', 'font': {'size': 14}},
            height=320,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),
            xaxis=dict(
                title='',
                tickfont=dict(size=11),
                showgrid=False
            ),
            yaxis=dict(
                title='Value',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                zeroline=False
            ),
            bargap=0.3,
            font=dict(size=11),
            margin=dict(t=50, b=60, l=60, r=20),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Enhanced Sensitivity Analysis with Tornado Chart
        sensitivity_fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            subplot_titles=('<b>Premium Sensitivity</b>', '<b>Feature Impact</b>'),
            horizontal_spacing=0.15
        )
        
        # Calculate sensitivity for each feature
        base_prediction = prediction
        impact_ranges = []
        feature_impacts = []
        
        # Define cleaner color palette for only the most important features
        line_colors = [colors['primary'], colors['secondary'], colors['warning']]
        
        # Full detailed sensitivity analysis for all features with optimization
        all_impacts = []
        
        # Pre-calculate common values for efficiency
        base_input_engineered = create_features(input_data)
        
        for i, col in enumerate(input_data.columns):
            test_values = []
            predictions = []
            
            # Get reasonable range for the feature
            feature_min = viz_engine.df[col].quantile(0.1)
            feature_max = viz_engine.df[col].quantile(0.9)
            current_value = input_data[col].iloc[0]
            
            # Use 15 points for smooth, accurate curves
            test_range = np.linspace(feature_min, feature_max, 15)
            
            # Batch create test inputs for efficiency
            test_inputs = []
            for test_val in test_range:
                test_input = input_data.copy()
                test_input[col] = test_val
                test_inputs.append(test_input)
                test_values.append(test_val)
            
            # Batch process all predictions for this feature
            # Process in smaller batches to optimize memory and speed
            for test_input in test_inputs:
                test_engineered = create_features(test_input)
                test_scaled = scaler.transform(test_engineered)
                test_pred = model.predict(test_scaled)[0]
                predictions.append(test_pred)
            
            # Calculate impact range
            min_pred = min(predictions)
            max_pred = max(predictions)
            impact_range = max_pred - min_pred
            
            all_impacts.append({
                'name': feature_names[i],
                'min': min_pred,
                'max': max_pred,
                'range': impact_range,
                'current': current_value,
                'current_pred': base_prediction,
                'test_values': test_values,
                'predictions': predictions,
                'index': i
            })
        
        # Sort by impact range and show only top 3 most impactful features in curves
        all_impacts_sorted = sorted(all_impacts, key=lambda x: x['range'], reverse=True)
        
        # Plot only top 3 most impactful features
        for idx, impact in enumerate(all_impacts_sorted[:3]):
            sensitivity_fig.add_trace(go.Scatter(
                x=impact['test_values'],
                y=impact['predictions'],
                mode='lines',
                name=impact['name'],
                line=dict(width=3, color=line_colors[idx]),
                opacity=0.8,
                hovertemplate='<b>%{fullData.name}</b><br>Value: %{x:.1f}<br>Premium: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # Add current position marker with larger size
            sensitivity_fig.add_trace(go.Scatter(
                x=[impact['current']],
                y=[base_prediction],
                mode='markers',
                marker=dict(size=12, color=line_colors[idx], symbol='diamond', 
                          line=dict(color='white', width=2)),
                showlegend=False,
                hovertemplate=f'<b>{impact["name"]}</b><br>Current: %{{x:.1f}}<br>Premium: $%{{y:.2f}}<extra></extra>'
            ), row=1, col=1)
        
        # Store all impacts for tornado chart
        impact_ranges = [imp['range'] for imp in all_impacts]
        feature_impacts = all_impacts
        
        # Create simplified tornado chart for all 6 features
        for i, impact in enumerate(all_impacts_sorted[:6]):
            # Single bar showing total range
            sensitivity_fig.add_trace(go.Bar(
                y=[impact['name']],
                x=[impact['range']],
                orientation='h',
                marker=dict(
                    color=colors['primary'] if i < 3 else colors['light'],
                    opacity=0.9 if i < 3 else 0.6,
                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                ),
                text=f"${impact['range']:.0f}",
                textposition='outside',
                textfont=dict(size=10, weight='bold' if i < 3 else 'normal'),
                showlegend=False,
                hovertemplate=f"<b>{impact['name']}</b><br>Range: ${impact['range']:.2f}<br>Min: ${impact['min']:.2f}<br>Max: ${impact['max']:.2f}<extra></extra>"
            ), row=1, col=2)
        
        sensitivity_fig.update_layout(
            height=380,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.3,
                font=dict(size=11),
                itemsizing='constant'
            ),
            font=dict(size=11),
            margin=dict(t=50, b=70, l=60, r=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white'
        )
        
        # Update axes with cleaner styling
        sensitivity_fig.update_xaxes(
            title_text="Feature Value", 
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False
        )
        sensitivity_fig.update_yaxes(
            title_text="Predicted Premium ($)", 
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False
        )
        sensitivity_fig.update_xaxes(
            title_text="Total Impact ($)", 
            row=1, col=2,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            range=[0, max(impact_ranges)*1.2]
        )
        sensitivity_fig.update_yaxes(
            showticklabels=True, 
            row=1, col=2,
            automargin=True
        )
        
        return prediction_output, importance_fig, sensitivity_fig
        
    except Exception as e:
        error_output = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong("Prediction Error: "),
            str(e)
        ], color="danger")
        
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Error in calculation", height=300)
        
        return error_output, empty_fig, empty_fig

# Add client-side callback for smooth scrolling navigation
app.clientside_callback(
    """
    function(n1, n2, n3, n4) {
        const triggered = window.dash_clientside.callback_context.triggered[0];
        if (triggered) {
            const href = triggered.prop_id.split('.')[0];
            const element = document.getElementById(href.replace('-link', ''));
            if (element) {
                element.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        }
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    [Input('executive-link', 'n_clicks'),
     Input('analysis-link', 'n_clicks'),
     Input('models-link', 'n_clicks'),
     Input('calculator-link', 'n_clicks')]
)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("                      VIDEBIMUS AI")
    print("           Insurance Premium Analytics Platform")
    print("="*70)
    print("\n   Developed by: Victor Collins Oppon")
    print("   Title: Data Scientist & AI Consultant")
    print("   Company: Videbimus AI")
    print("\n" + "-"*70)
    print("\nStarting dashboard server...")
    print("\nDashboard URL: http://127.0.0.1:8050")
    print("\nPress CTRL+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(debug=False, port=8050)