"""
Utility functions for creating visualizations in the W.A.R.P dashboard.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_temperature_over_time(df, city=None, start_year=None, end_year=None):
    """
    Create a line chart of temperature over time.
    
    Args:
        df: DataFrame with temperature data
        city: City to filter by (if None, show all cities)
        start_year: Start year for filtering
        end_year: End year for filtering
        
    Returns:
        Plotly figure object
    """
    # Filter data if needed
    if city:
        filtered_df = df[df['City'] == city]
    else:
        filtered_df = df
        
    if start_year:
        filtered_df = filtered_df[filtered_df['Year'] >= start_year]
        
    if end_year:
        filtered_df = filtered_df[filtered_df['Year'] <= end_year]
    
    # Create plot
    fig = px.line(filtered_df, x='Year', y='AvgYearlyTemp', color='City',
                  title='Average Yearly Temperature Over Time',
                  labels={'AvgYearlyTemp': 'Temperature (°C)', 'Year': 'Year'},
                  hover_data=['Country'])
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Temperature (°C)',
        legend_title='City',
        hovermode='closest'
    )
    
    return fig

def plot_temperature_distribution(df, by_country=True):
    """
    Create a box plot of temperature distribution.
    
    Args:
        df: DataFrame with temperature data
        by_country: Whether to group by country or city
        
    Returns:
        Plotly figure object
    """
    # Group by country or city
    if by_country:
        group_col = 'Country'
    else:
        group_col = 'City'
        # Limit to top 15 cities by data points to avoid overcrowding
        city_counts = df['City'].value_counts().nlargest(15).index
        df = df[df['City'].isin(city_counts)]
    
    # Create plot
    fig = px.box(df, x=group_col, y='AvgYearlyTemp',
                 title=f'Temperature Distribution by {group_col}',
                 labels={'AvgYearlyTemp': 'Temperature (°C)'})
    
    # Customize layout
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title='Temperature (°C)',
        hovermode='closest'
    )
    
    return fig

def plot_correlation_heatmap(corr_df, title="Feature Correlation Heatmap"):
    """
    Create a heatmap of feature correlations.
    
    Args:
        corr_df: DataFrame with correlation values
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create heatmap
    fig = px.imshow(
        corr_df,
        labels=dict(color="Correlation"),
        x=corr_df.columns,
        y=corr_df.columns,
        color_continuous_scale='RdBu_r',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=700
    )
    
    return fig

def plot_feature_importance(importance_df, title="Feature Importance"):
    """
    Create a bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with feature importance values
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if importance_df is None or importance_df.empty:
        return None
    
    # Sort by importance
    df = importance_df.sort_values('Importance', ascending=True).tail(20)
    
    # Create plot
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature',
        title=title,
        orientation='h'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=600
    )
    
    return fig

def plot_weather_comparison(comparison_data, metric_name):
    """
    Create a bar chart comparing weather metrics across stations.
    
    Args:
        comparison_data: Dictionary with comparison data
        metric_name: Name of the metric to plot
        
    Returns:
        Plotly figure object
    """
    if metric_name not in comparison_data:
        return None
    
    metric_data = comparison_data[metric_name]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Station': metric_data['stations'],
        'Value': metric_data['values']
    })
    
    # Create plot
    fig = px.bar(
        df,
        x='Station',
        y='Value',
        title=f'Comparison of {metric_name} across Weather Stations',
        color='Station'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Weather Station",
        yaxis_title=metric_name,
        height=500
    )
    
    return fig

def plot_predictions_vs_actual(predictions_df, actual_col, prediction_col, title="Predictions vs Actual"):
    """
    Create a scatter plot of predictions vs actual values.
    
    Args:
        predictions_df: DataFrame with predictions and actual values
        actual_col: Column name for actual values
        prediction_col: Column name for predicted values
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create plot
    fig = px.scatter(
        predictions_df,
        x=actual_col,
        y=prediction_col,
        title=title,
        labels={actual_col: "Actual", prediction_col: "Predicted"}
    )
    
    # Add 45-degree reference line
    min_val = min(predictions_df[actual_col].min(), predictions_df[prediction_col].min())
    max_val = max(predictions_df[actual_col].max(), predictions_df[prediction_col].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=500,
        width=700
    )
    
    return fig

def plot_model_comparison(comparison_df, metric_col, higher_is_better=True):
    """
    Create a bar chart comparing model performance.
    
    Args:
        comparison_df: DataFrame with model comparison data
        metric_col: Column name for the metric to compare
        higher_is_better: Whether higher values of the metric are better
        
    Returns:
        Plotly figure object
    """
    if comparison_df is None or comparison_df.empty or metric_col not in comparison_df.columns:
        return None
    
    # Sort by metric
    if higher_is_better:
        df = comparison_df.sort_values(metric_col, ascending=False)
    else:
        df = comparison_df.sort_values(metric_col, ascending=True)
    
    # Create plot
    fig = px.bar(
        df,
        x='Model',
        y=metric_col,
        title=f'Model Comparison by {metric_col}',
        color='Model'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric_col,
        height=500
    )
    
    return fig

def plot_confusion_matrix(confusion_matrix, title="Confusion Matrix"):
    """
    Create a heatmap visualization of a confusion matrix.
    
    Args:
        confusion_matrix: Dictionary or DataFrame with confusion matrix data
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if isinstance(confusion_matrix, dict):
        # Convert dictionary to DataFrame
        labels = sorted(confusion_matrix.keys())
        matrix = []
        
        for true_label in labels:
            row = []
            for pred_label in labels:
                row.append(confusion_matrix[true_label].get(pred_label, 0))
            matrix.append(row)
        
        df = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        df = confusion_matrix
    
    # Create heatmap
    fig = px.imshow(
        df,
        labels=dict(color="Count"),
        x=df.columns,
        y=df.index,
        color_continuous_scale='Blues',
        title=title,
        text_auto=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        width=500
    )
    
    return fig

def plot_residuals(predictions_df, actual_col, prediction_col, title="Residual Plot"):
    """
    Create a residual plot for regression model evaluation.
    
    Args:
        predictions_df: DataFrame with predictions and actual values
        actual_col: Column name for actual values
        prediction_col: Column name for predicted values
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Calculate residuals
    df = predictions_df.copy()
    df['residual'] = df[actual_col] - df[prediction_col]
    
    # Create plot
    fig = px.scatter(
        df,
        x=prediction_col,
        y='residual',
        title=title,
        labels={prediction_col: "Predicted Value", 'residual': "Residual"}
    )
    
    # Add horizontal reference line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    # Update layout
    fig.update_layout(
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Actual - Predicted)",
        height=500,
        width=700
    )
    
    return fig

def plot_cluster_visualization(predictions, feature1, feature2, title="Cluster Visualization"):
    """
    Create a scatter plot colored by cluster assignment.
    
    Args:
        predictions: DataFrame with cluster predictions
        feature1: First feature for x-axis
        feature2: Second feature for y-axis
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create plot
    fig = px.scatter(
        predictions,
        x=feature1,
        y=feature2,
        color='cluster',
        title=title,
        labels={feature1: feature1, feature2: feature2, 'cluster': "Cluster"}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=feature1,
        yaxis_title=feature2,
        height=600,
        width=800,
        legend_title="Cluster"
    )
    
    return fig

def plot_time_series(df, date_col, value_col, title="Time Series Plot"):
    """
    Create a time series line plot.
    
    Args:
        df: DataFrame with time series data
        date_col: Column name for date/time
        value_col: Column name for values
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create plot
    fig = px.line(
        df,
        x=date_col,
        y=value_col,
        title=title
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=value_col,
        height=500,
        width=800
    )
    
    return fig

def plot_seasonal_decomposition(decomposition_result, title="Seasonal Decomposition"):
    """
    Create a plot of seasonal decomposition components.
    
    Args:
        decomposition_result: Dictionary with trend, seasonal, and residual components
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(rows=4, cols=1, 
                         subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
                         vertical_spacing=0.1)
    
    # Add traces for each component
    fig.add_trace(
        go.Scatter(x=decomposition_result['original'].index, y=decomposition_result['original'], name="Original"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition_result['trend'].index, y=decomposition_result['trend'], name="Trend"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition_result['seasonal'].index, y=decomposition_result['seasonal'], name="Seasonal"),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition_result['residual'].index, y=decomposition_result['residual'], name="Residual"),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=800,
        title=title,
        showlegend=False
    )
    
    return fig

def plot_weather_map(df, lat_col, lon_col, value_col, title="Weather Map"):
    """
    Create a map visualization of weather data.
    
    Args:
        df: DataFrame with location and weather data
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        value_col: Column name for values to display
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create map
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=value_col,
        size_max=15,
        zoom=3,
        title=title,
        mapbox_style="carto-positron"
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

def plot_association_rules(rules_df, title="Association Rules"):
    """
    Create a scatter plot of association rules.
    
    Args:
        rules_df: DataFrame with association rules
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if rules_df is None or rules_df.empty:
        return None
    
    # Create plot
    fig = px.scatter(
        rules_df,
        x='confidence',
        y='lift',
        size='support' if 'support' in rules_df.columns else None,
        hover_data=['antecedent', 'consequent'],
        title=title
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Lift",
        height=600,
        width=800
    )
    
    # Add reference line at lift=1
    fig.add_hline(y=1, line_dash="dash", line_color="red")
    
    return fig
