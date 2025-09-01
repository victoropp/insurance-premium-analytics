import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class ProfessionalVisualizationEngine:
    def __init__(self, data_path='data/insurance_tranining_dataset.csv', 
                 model_results_path='data/model_results.csv',
                 feature_importance_path='data/feature_importance.csv',
                 test_results_path='data/final_test_results.csv'):
        self.df = pd.read_csv(data_path)
        self.model_results = pd.read_csv(model_results_path)
        self.feature_importance = pd.read_csv(feature_importance_path) if feature_importance_path else None
        try:
            self.test_results = pd.read_csv(test_results_path) if test_results_path else None
        except:
            self.test_results = None
        
        # Professional color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#73AB84',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C91C2',
            'light': '#F5F5F5',
            'dark': '#2D3436'
        }
        
        self.template = 'plotly_white'
        self.font_family = 'Arial, sans-serif'
        
    def create_executive_summary(self):
        """Create a clean, executive-level summary dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                '<b>📊 Premium Distribution</b>',
                '<b>⚠️ Key Risk Factors</b>',
                '<b>🎯 Test Set Performance</b>',
                '<b>🔗 Feature Correlations</b>',
                '<b>👥 Age vs Premium Analysis</b>',
                '<b>🎲 Risk Segmentation</b>'
            ),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'heatmap'}, {'type': 'box'}, {'type': 'pie'}]],
            vertical_spacing=0.25,  # Increased from 0.15 to 0.25 for better spacing
            horizontal_spacing=0.15,  # Slightly increased horizontal spacing
            row_heights=[0.45, 0.45],  # Equal heights for both rows
            shared_xaxes=False,
            shared_yaxes=False  # Ensure each subplot is independent
        )
        
        # 1. Premium Distribution - Clean histogram
        fig.add_trace(
            go.Histogram(
                x=self.df['Insurance Premium ($)'],
                nbinsx=25,
                marker=dict(
                    color=self.colors['primary'],
                    line=dict(color='white', width=1)
                ),
                name='Premium Distribution',
                hovertemplate='<b>Premium Range:</b> $%{x}<br><b>Count:</b> %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Key Risk Factors - Horizontal bar chart
        risk_impact = pd.DataFrame({
            'Factor': ['Previous Accidents', 'Driver Age < 25', 'New Driver', 'High Mileage', 'Old Vehicle'],
            'Impact': [
                self.df.groupby('Previous Accidents')['Insurance Premium ($)'].mean().diff().mean(),
                self.df[self.df['Driver Age'] < 25]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean(),
                self.df[self.df['Driver Experience'] < 2]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean(),
                self.df[self.df['Annual Mileage (x1000 km)'] > 20]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean(),
                self.df[self.df['Car Age'] > 10]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean()
            ]
        }).sort_values('Impact')
        
        fig.add_trace(
            go.Bar(
                x=risk_impact['Impact'],
                y=risk_impact['Factor'],
                orientation='h',
                marker=dict(
                    color=risk_impact['Impact'],
                    colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=['${:.2f}'.format(x) for x in risk_impact['Impact']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Premium Impact: $%{x:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Model Performance - Clean bar chart showing test results
        if self.test_results is not None and not self.test_results.empty:
            test_results = self.test_results
            # Show test set performance for ensemble models with better spacing
            fig.add_trace(
                go.Bar(
                    x=['Linear<br>Stack', 'Ridge<br>Stack', 'Voting<br>Ens.'],  # Shorter names with line breaks
                    y=test_results['Test_R2'].values,
                    marker=dict(
                        color=[self.colors['success'], self.colors['info'], self.colors['secondary']],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.4f}' for v in test_results['Test_R2'].values],
                    textposition='outside',
                    textfont=dict(size=10, color=self.colors['dark']),  # Smaller font
                    width=0.6,  # Narrower bars for better spacing
                    hovertemplate='<b>%{customdata[0]}</b><br>Test R²: %{y:.4f}<br>RMSE: %{customdata[1]:.3f}<extra></extra>',
                    customdata=[[name, rmse] for name, rmse in zip(
                        ['Stacking (Linear)', 'Stacking (Ridge)', 'Voting Ensemble'],
                        test_results['Test_RMSE'].values
                    )]
                ),
                row=1, col=3
            )
        else:
            # Fallback to validation results if test results not available
            top3_models = self.model_results.nlargest(3, 'Val_R2')
            fig.add_trace(
                go.Bar(
                    x=[m[:12] for m in top3_models['Model']],
                    y=top3_models['Val_R2'].values,
                    marker=dict(
                        color=[self.colors['success'], self.colors['info'], self.colors['secondary']],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.4f}' for v in top3_models['Val_R2'].values],
                    textposition='outside',
                    textfont=dict(size=11, color=self.colors['dark']),
                    hovertemplate='<b>%{x}</b><br>Val R²: %{y:.4f}<extra></extra>'
                ),
                row=1, col=3
            )
        
        # 4. Feature Correlations - Improved correlation matrix
        # Get full correlation matrix
        corr_full = self.df.select_dtypes(include=[np.number]).corr()
        
        # Create a more informative correlation heatmap
        # Check which features exist in the dataframe
        possible_features = ['Driver Age', 'Driver Experience', 'Previous Accidents', 
                           'Annual Mileage (x1000 km)', 'Car Manufacturing Year', 
                           'Car Age', 'Insurance Premium ($)']
        
        # Use only features that exist in the correlation matrix
        features = [f for f in possible_features if f in corr_full.columns]
        
        # If we have all expected features, remove Car Manufacturing Year to avoid redundancy with Car Age
        if 'Car Manufacturing Year' in features and 'Car Age' in features:
            features.remove('Car Manufacturing Year')
        
        corr_subset = corr_full.loc[features, features]
        
        # Create custom text annotations
        text_annotations = []
        for i in range(len(features)):
            row_text = []
            for j in range(len(features)):
                val = corr_subset.iloc[i, j]
                if i == j:
                    row_text.append('1.00')
                else:
                    row_text.append(f'{val:.2f}')
            text_annotations.append(row_text)
        
        # Create shortened labels for better display
        label_map = {
            'Driver Age': 'Driver<br>Age',
            'Driver Experience': 'Driver<br>Exp.',
            'Previous Accidents': 'Previous<br>Accidents',
            'Annual Mileage (x1000 km)': 'Annual<br>Mileage',
            'Car Age': 'Car<br>Age',
            'Insurance Premium ($)': 'Premium<br>($)'
        }
        
        x_labels = [label_map.get(f, f) for f in features]
        y_labels = x_labels.copy()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_subset.values,
                x=x_labels,
                y=y_labels,
                colorscale=[
                    [0, self.colors['danger']],
                    [0.25, '#FF9999'],
                    [0.5, 'white'],
                    [0.75, '#99CC99'],
                    [1, self.colors['success']]
                ],
                zmid=0,
                text=text_annotations,
                texttemplate='%{text}',
                textfont={"size": 9, "color": "black"},
                showscale=True,
                colorbar=dict(
                    title="Correlation",
                    thickness=10,
                    len=0.5,  # Increased to match other colorbars
                    x=0.33,  # Moved left to stay in column 1
                    xanchor='right',  # Anchor to right side of position
                    y=0.22,  # Middle of bottom row
                    yanchor='middle',
                    tickmode='linear',
                    tick0=-1,
                    dtick=0.5,
                    tickfont=dict(size=9)
                ),
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 5. Age vs Premium - Box plots by age group
        age_groups = pd.cut(self.df['Driver Age'], bins=[0, 25, 35, 50, 65, 100], 
                           labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        for i, group in enumerate(['18-25', '26-35', '36-50', '51-65', '65+']):
            mask = age_groups == group
            fig.add_trace(
                go.Box(
                    y=self.df.loc[mask, 'Insurance Premium ($)'],
                    name=group,
                    marker_color=self.colors['primary'] if i % 2 == 0 else self.colors['secondary'],
                    boxmean='sd',
                    hovertemplate='<b>Age Group: %{x}</b><br>Premium: $%{y:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 6. Risk Segmentation - Professional pie chart
        risk_categories = pd.cut(self.df['Insurance Premium ($)'],
                                bins=[0, 490, 500, 510, float('inf')],
                                labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
        risk_counts = risk_categories.value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker=dict(
                    colors=[self.colors['success'], self.colors['info'], 
                           self.colors['warning'], self.colors['danger']],
                    line=dict(color='white', width=2)
                ),
                text=[f'{v}' for v in risk_counts.values],  # Simplified text to avoid overlap
                textposition='inside',  # Changed to inside to prevent overlap
                textfont=dict(size=11, color='white'),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                domain=dict(x=[0, 1], y=[0, 0.95])  # Slightly reduce vertical domain
            ),
            row=2, col=3
        )
        
        # Update layout for professional appearance
        fig.update_layout(
            height=800,  # Increased height to accommodate better spacing
            showlegend=False,
            title={
                'text': '',  # Remove duplicate title
                'font': {'size': 24, 'family': self.font_family}
            },
            template=self.template,
            font=dict(family=self.font_family, size=12),
            margin=dict(t=120, b=60, l=60, r=60),  # Increased margins
            hovermode='closest'  # Enable hover for all subplots
        )
        
        # Update axes for better readability
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        
        # Format the test performance chart axes
        fig.update_yaxes(range=[0.994, 1.001], row=1, col=3, title_text="Test R² Score")  # Extended range for text
        fig.update_xaxes(title_text="Model", row=1, col=3, tickangle=0, tickfont=dict(size=10))  # Smaller tick font
        
        return fig
    
    def create_detailed_analysis(self):
        """Create detailed analysis dashboard with clean visualizations"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '<b>Driver Experience Impact</b>',
                '<b>Vehicle Age Analysis</b>',
                '<b>Accident History Effect</b>',
                '<b>Annual Mileage Distribution</b>',
                '<b>Premium Percentiles</b>',
                '<b>Feature Importance Ranking</b>'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'violin'}],
                   [{'type': 'box'}, {'type': 'bar'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.15,
            row_heights=[0.33, 0.33, 0.34],
            shared_xaxes=False,
            shared_yaxes=False  # Ensure each subplot is independent
        )
        
        # 1. Driver Experience Impact
        exp_grouped = self.df.groupby('Driver Experience')['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        exp_grouped = exp_grouped[exp_grouped['count'] >= 5]  # Filter for statistical significance
        
        fig.add_trace(
            go.Scatter(
                x=exp_grouped.index,
                y=exp_grouped['mean'],
                mode='lines+markers',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=8, color=self.colors['primary']),
                error_y=dict(
                    type='data',
                    array=exp_grouped['std'],
                    visible=True,
                    color=self.colors['primary'],
                    thickness=1.5,
                    width=4
                ),
                hovertemplate='<b>Experience:</b> %{x} years<br><b>Avg Premium:</b> $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Vehicle Age Analysis - Scatter plot with trend line for clarity
        # Calculate average premium for each car age with sufficient data
        car_age_stats = self.df.groupby('Car Age')['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        car_age_stats = car_age_stats[car_age_stats['count'] >= 3]  # Only include ages with enough data
        
        # Add scatter plot with size representing sample size
        fig.add_trace(
            go.Scatter(
                x=car_age_stats.index,
                y=car_age_stats['mean'],
                mode='markers',
                marker=dict(
                    size=np.minimum(car_age_stats['count'] * 2, 30),  # Size based on sample, capped at 30
                    color=car_age_stats['mean'],
                    colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                    showscale=True,
                    colorbar=dict(
                        title="Avg<br>Premium",
                        thickness=10,
                        len=0.5,
                        y=0.75,
                        yanchor='middle'
                    ),
                    line=dict(color='white', width=1)
                ),
                text=[f'Age: {age} yrs<br>${mean:.2f}<br>n={int(count)}' 
                      for age, mean, count in zip(car_age_stats.index, car_age_stats['mean'], car_age_stats['count'])],
                hovertemplate='<b>Vehicle Age:</b> %{x} years<br><b>Avg Premium:</b> $%{y:.2f}<br><b>Sample Size:</b> %{text}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add polynomial trend line to show pattern
        if len(car_age_stats) > 3:
            z = np.polyfit(car_age_stats.index, car_age_stats['mean'], 2)  # Quadratic fit
            p = np.poly1d(z)
            x_trend = np.linspace(car_age_stats.index.min(), car_age_stats.index.max(), 50)
            
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    line=dict(color=self.colors['primary'], width=2, dash='dash'),
                    name='Trend',
                    showlegend=False,
                    hovertemplate='Trend: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
        # Add annotation for insight
        fig.add_annotation(
            text="Bubble size = sample size",
            xref="x2", yref="y2",
            x=car_age_stats.index.max() * 0.7,
            y=car_age_stats['mean'].min() + (car_age_stats['mean'].max() - car_age_stats['mean'].min()) * 0.15,
            showarrow=False,
            font=dict(size=9, color='gray'),
            row=1, col=2
        )
        
        # 3. Accident History Effect - Line chart with confidence intervals
        accident_grouped = self.df.groupby('Previous Accidents')['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        accident_grouped = accident_grouped[accident_grouped['count'] >= 3]  # Filter for reliability
        
        # Calculate confidence intervals
        confidence_interval = 1.96 * (accident_grouped['std'] / np.sqrt(accident_grouped['count']))
        
        fig.add_trace(
            go.Scatter(
                x=accident_grouped.index,
                y=accident_grouped['mean'],
                mode='lines+markers',
                name='Average Premium',
                line=dict(color=self.colors['danger'], width=3),
                marker=dict(size=10, color=self.colors['danger']),
                error_y=dict(
                    type='data',
                    array=confidence_interval,
                    visible=True,
                    color=self.colors['danger'],
                    thickness=1.5,
                    width=4
                ),
                text=[f'${m:.2f}<br>{int(n)} drivers' for m, n in zip(accident_grouped['mean'], accident_grouped['count'])],
                hovertemplate='<b>%{x} Accidents</b><br>Avg Premium: $%{y:.2f}<br>±$%{error_y.array:.2f} (95% CI)<br>Sample: %{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add trend line
        if len(accident_grouped) > 1:
            z = np.polyfit(accident_grouped.index, accident_grouped['mean'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=accident_grouped.index,
                    y=p(accident_grouped.index),
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=1
            )
        
        # 4. Annual Mileage Distribution - Scatter plot with trend
        # Create mileage bins for better visualization
        mileage_bins = pd.cut(self.df['Annual Mileage (x1000 km)'], 
                              bins=[0, 10, 20, 30, 40, float('inf')],
                              labels=['0-10k', '11-20k', '21-30k', '31-40k', '40k+'])
        mileage_grouped = self.df.groupby(mileage_bins)['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        mileage_grouped = mileage_grouped[mileage_grouped['count'] >= 5]
        
        # Create box plots for each mileage group
        for mileage_range in mileage_grouped.index:
            mask = mileage_bins == mileage_range
            fig.add_trace(
                go.Box(
                    y=self.df.loc[mask, 'Insurance Premium ($)'],
                    name=str(mileage_range),
                    marker=dict(color=self.colors['info']),
                    boxmean='sd',
                    width=0.6,
                    hovertemplate='<b>Mileage: %{x}</b><br>Premium: $%{y:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 5. Premium Percentiles
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [np.percentile(self.df['Insurance Premium ($)'], p) for p in percentiles]
        
        fig.add_trace(
            go.Box(
                y=self.df['Insurance Premium ($)'],
                name='Distribution',
                marker=dict(color=self.colors['primary']),
                boxmean='sd',
                notched=True,
                hovertemplate='Premium: $%{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Feature Importance
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            importance_df = self.feature_importance.nlargest(10, 'Importance')
            
            fig.add_trace(
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=importance_df['Importance'],
                        colorscale=[[0, self.colors['light']], [1, self.colors['primary']]],
                        showscale=False,
                        line=dict(color='white', width=1)
                    ),
                    text=['{:.3f}'.format(x) for x in importance_df['Importance']],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            height=900,
            showlegend=False,
            title={
                'text': '',  # Remove duplicate title
                'font': {'size': 24, 'family': self.font_family}
            },
            template=self.template,
            font=dict(family=self.font_family, size=11),
            margin=dict(t=100, b=50, l=60, r=60),
            hovermode='closest'  # Enable hover for all subplots
        )
        
        # Professional axes formatting
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)', title_font_size=12)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)', title_font_size=12)
        
        # Add axis labels
        fig.update_xaxes(title_text="Years of Experience", row=1, col=1)
        fig.update_xaxes(title_text="Vehicle Age (Years)", row=1, col=2)
        fig.update_xaxes(title_text="Number of Previous Accidents", row=2, col=1)
        fig.update_xaxes(title_text="Annual Mileage Range (km)", row=2, col=2)
        fig.update_yaxes(title_text="Insurance Premium ($)", row=1, col=1)
        fig.update_yaxes(title_text="Average Premium ($)", row=1, col=2)
        fig.update_yaxes(title_text="Average Premium ($)", row=2, col=1)
        fig.update_yaxes(title_text="Insurance Premium ($)", row=2, col=2)
        fig.update_yaxes(title_text="Insurance Premium ($)", row=3, col=1)
        fig.update_xaxes(title_text="Feature Importance Score", row=3, col=2)
        
        return fig
    
    def create_model_comparison(self):
        """Create professional model comparison dashboard"""
        # Use test results from initialization
        test_results = self.test_results
            
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '<b>🏆 Top 10 Models - Validation</b>',
                '<b>🤖 Ensemble Models - Test Set</b>',
                '<b>📈 Overfitting Analysis</b>',
                '<b>🥇 Model Rankings - Test</b>',
                '<b>📊 Performance Metrics</b>',
                ' '  # Space to maintain structure but avoid default positioning
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]],
            vertical_spacing=0.18,  # Increased vertical spacing to push titles up
            horizontal_spacing=0.15,
            row_heights=[0.32, 0.32, 0.36],  # Adjusted row heights for better spacing
            shared_xaxes=False,
            shared_yaxes=False  # Ensure each subplot is independent
        )
        
        # 1. Top 10 Models - Validation Performance (Horizontal bar for better readability)
        top10_val = self.model_results.nlargest(10, 'Val_R2').sort_values('Val_R2')
        
        # Shorten model names for display
        short_names = []
        for name in top10_val['Model']:
            if 'Regression' in name:
                short_names.append(name.replace(' Regression', ''))
            elif 'Random Forest' in name:
                short_names.append('Random Forest')
            elif 'Gradient Boosting' in name:
                short_names.append('Gradient Boost')
            elif 'Extra Trees' in name:
                short_names.append('Extra Trees')
            else:
                short_names.append(name[:12])
        
        fig.add_trace(
            go.Bar(
                y=short_names,
                x=top10_val['Val_R2'],
                orientation='h',
                marker=dict(
                    color=top10_val['Val_R2'],
                    colorscale=[[0, '#F0F0F0'], [1, self.colors['primary']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=[f'{v:.4f}' for v in top10_val['Val_R2']],
                textposition='outside',
                textfont=dict(size=9),
                hovertemplate='<b>%{customdata}</b><br>Val R²: %{x:.4f}<extra></extra>',
                customdata=top10_val['Model']
            ),
            row=1, col=1
        )
        
        # 2. Ensemble Models - Test Set Performance
        if test_results is not None:
            fig.add_trace(
                go.Bar(
                    x=['Stacking<br>(Linear)', 'Stacking<br>(Ridge)', 'Voting<br>Ensemble'],
                    y=test_results['Test_R2'],
                    marker=dict(
                        color=[self.colors['success'], self.colors['info'], self.colors['warning']],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.5f}' for v in test_results['Test_R2']],
                    textposition='outside',
                    textfont=dict(size=10),
                    width=0.5,  # Narrower bars
                    hovertemplate='<b>%{customdata[0]}</b><br>Test R²: %{y:.5f}<br>Test RMSE: %{customdata[1]:.4f}<extra></extra>',
                    customdata=[[name, rmse] for name, rmse in zip(test_results['Model'], test_results['Test_RMSE'])]
                ),
                row=1, col=2
            )
        
        # 3. Overfitting Analysis - Cleaner visualization
        # Select diverse models to show different overfitting patterns
        models_to_show = ['Linear Regression', 'Ridge Regression', 'Random Forest', 
                         'XGBoost', 'Neural Network']
        selected_models = self.model_results[self.model_results['Model'].isin(models_to_show)]
        
        # Calculate overfitting degree
        selected_models['Overfit_Degree'] = selected_models['Train_R2'] - selected_models['Val_R2']
        
        fig.add_trace(
            go.Scatter(
                x=selected_models['Train_R2'],
                y=selected_models['Val_R2'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=selected_models['Overfit_Degree'],
                    colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                    showscale=True,
                    colorbar=dict(
                        title="Overfit<br>Level",
                        thickness=10,
                        len=0.5,
                        x=0.45,
                        y=0.5
                    ),
                    line=dict(color='white', width=1)
                ),
                text=[m.replace(' Regression', '').replace('Random Forest', 'RF')[:10] 
                      for m in selected_models['Model']],
                hovertemplate='<b>%{text}</b><br>Train R²: %{x:.4f}<br>Val R²: %{y:.4f}<br>Overfit: %{marker.color:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add ideal line (no overfitting)
        min_val = min(selected_models['Train_R2'].min(), selected_models['Val_R2'].min()) * 0.95
        max_val = 1.0
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                showlegend=False,
                hovertemplate='Perfect fit line<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add annotation
        fig.add_annotation(
            text="Points below line = overfitting",
            xref="x3", yref="y3",
            x=0.98, y=0.95,
            showarrow=False,
            font=dict(size=9, color='gray'),
            row=2, col=1
        )
        
        # 4. Model Rankings - Test Set Results
        if test_results is not None and len(test_results) > 0:
            # Show test set rankings
            test_results_sorted = test_results.sort_values('Test_R2')
            
            fig.add_trace(
                go.Bar(
                    x=test_results_sorted['Test_R2'],
                    y=['Voting<br>Ensemble', 'Stacking<br>(Ridge)', 'Stacking<br>(Linear)'],
                    orientation='h',
                    marker=dict(
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'],  # Three distinct colors: red, teal, blue
                        showscale=False,
                        line=dict(color='white', width=1)
                    ),
                    text=[f'{x:.5f}' for x in test_results_sorted['Test_R2']],
                    textposition='outside',  # Changed to outside for visibility
                    textfont=dict(size=10, color=self.colors['dark']),  # Dark color for visibility
                    hovertemplate='<b>%{customdata[0]}</b><br>Test R²: %{x:.5f}<br>Test RMSE: %{customdata[1]:.4f}<extra></extra>',
                    customdata=[[name, rmse] for name, rmse in zip(test_results_sorted['Model'], test_results_sorted['Test_RMSE'])]
                ),
                row=2, col=2
            )
        else:
            # Fallback to validation rankings if test results not available
            top3_models = self.model_results.nlargest(3, 'Val_R2').sort_values('Val_R2')
            
            fig.add_trace(
                go.Bar(
                    x=top3_models['Val_R2'],
                    y=[m[:15] for m in top3_models['Model']],
                    orientation='h',
                    marker=dict(
                        color=top3_models['Val_R2'],
                        colorscale=[[0.99, '#F0F0F0'], [1, self.colors['primary']]],
                        showscale=False,
                        line=dict(color='white', width=1)
                    ),
                    text=[f'{x:.5f}' for x in top3_models['Val_R2']],
                    textposition='outside',
                    textfont=dict(size=10, color=self.colors['dark']),
                    hovertemplate='<b>%{y}</b><br>Val R²: %{x:.5f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 5. Performance Metrics Comparison - Scatter
        top10_models = self.model_results.nlargest(10, 'Val_R2')
        
        fig.add_trace(
            go.Scatter(
                x=top10_models['Val_RMSE'],
                y=top10_models['Val_R2'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.colors['secondary'],
                    line=dict(color='white', width=1)
                ),
                text=top10_models['Model'],
                hovertemplate='<b>%{text}</b><br>RMSE: %{x:.3f}<br>R²: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Best Test Set Model Indicator
        if test_results is not None and len(test_results) > 0:
            best_test_model = test_results.loc[test_results['Test_R2'].idxmax()]
            
            fig.add_trace(
                go.Indicator(
                    mode="number+gauge",
                    value=best_test_model['Test_R2'],
                    title={'text': "",  # Remove RMSE text completely
                           'font': {'size': 11}},
                    number={'font': {'size': 26}, 'valueformat': '.5f'},
                    gauge={
                        'axis': {'range': [0.994, 0.999], 'tickwidth': 1, 'tickmode': 'linear', 'tick0': 0.994, 'dtick': 0.001},
                        'bar': {'color': self.colors['success'], 'thickness': 0.8},
                        'steps': [
                            {'range': [0.994, 0.996], 'color': '#F5F5F5'},
                            {'range': [0.996, 0.997], 'color': '#E0E0E0'},
                            {'range': [0.997, 0.998], 'color': '#D0D0D0'},
                            {'range': [0.998, 0.999], 'color': '#C0C0C0'}
                        ],
                        'threshold': {
                            'line': {'color': self.colors['success'], 'width': 3},
                            'thickness': 0.8,
                            'value': best_test_model['Test_R2']
                        }
                    },
                    domain={'x': [0.15, 0.85], 'y': [0.2, 0.8]}  # Standard positioning
                ),
                row=3, col=2
            )
        else:
            # Fallback to validation best model if test results not available
            best_model = self.model_results.loc[self.model_results['Val_R2'].idxmax()]
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta+gauge",
                    value=best_model['Val_R2'],
                    delta={'reference': 0.99, 'position': "top"},
                    title={'text': f"<b>{best_model['Model']}</b><br><span style='font-size:11px'>RMSE: {best_model['Val_RMSE']:.3f}</span>"},
                    gauge={
                        'axis': {'range': [0.98, 1.0], 'tickwidth': 1},
                        'bar': {'color': self.colors['success']},
                        'steps': [
                            {'range': [0.98, 0.99], 'color': '#E8E8E8'},
                            {'range': [0.99, 0.995], 'color': '#D0D0D0'},
                            {'range': [0.995, 1.0], 'color': '#B8B8B8'}
                        ]
                    }
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            height=950,
            showlegend=False,
            title={
                'text': '',  # Remove duplicate title
                'font': {'size': 24, 'family': self.font_family}
            },
            template=self.template,
            font=dict(family=self.font_family, size=11),
            margin=dict(t=100, b=60, l=70, r=70),  # Standard margins
            hovermode='closest'  # Enable hover for all subplots
        )
        
        # Add custom annotation for Best Model Score without overriding others
        fig.add_annotation(
            text='<b>⭐ Best Model Score</b>',
            xref='paper',
            yref='paper',
            x=0.77,  # Position for right column
            y=0.28,  # Balanced position
            showarrow=False,
            font=dict(size=14, family=self.font_family),
            xanchor='center'
        )
        
        # Add axis labels and formatting
        fig.update_xaxes(title_text="Validation R² Score", row=1, col=1, range=[0.99, 1.001])
        fig.update_yaxes(title_text="Model", row=1, col=1, tickfont=dict(size=9))
        fig.update_xaxes(title_text="Model", row=1, col=2, tickfont=dict(size=10))
        fig.update_yaxes(title_text="Test R² Score", row=1, col=2, range=[0.994, 1.0])
        fig.update_xaxes(title_text="Training R²", row=2, col=1)
        fig.update_yaxes(title_text="Validation R²", row=2, col=1)
        fig.update_xaxes(title_text="Test R² Score", row=2, col=2, range=[0.994, 0.9995])
        fig.update_xaxes(title_text="Validation RMSE", row=3, col=1)
        fig.update_yaxes(title_text="Validation R²", row=3, col=1)
        
        # Professional grid
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        
        return fig