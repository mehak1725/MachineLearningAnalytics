"""
W.A.R.P - Weather Analytics and Research Pipeline
Main application file for the Streamlit dashboard
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WARP")

# Import Spark utilities
from utils.spark_utils import create_spark_session, stop_spark_session
from utils.data_loader import load_weather_data, load_temperature_data, get_cities, get_weather_stations

# Import visualization modules
from visualization.dashboard import (
    render_home_page,
    render_data_exploration,
    render_feature_engineering,
    render_classification_models,
    render_regression_models,
    render_clustering_models,
    render_time_series,
    render_advanced_analytics
)

# Import data processing modules
from modules.data_ingestion import ingest_and_profile_data, prepare_delta_tables, create_silver_tables, get_delta_tables_info
from modules.eda import calculate_basic_stats, calculate_correlations, station_comparison, yearly_temperature_trends
from modules.data_cleaning import clean_dataset, standardize_features, detect_anomalies
from modules.feature_engineering import engineer_weather_features, engineer_temperature_features, create_prediction_datasets

# Create necessary folders to ensure Delta tables can be written
def ensure_dirs_exist():
    """Create necessary directories if they don't exist."""
    os.makedirs("delta_lake/bronze", exist_ok=True)
    os.makedirs("delta_lake/silver", exist_ok=True)
    os.makedirs("delta_lake/gold", exist_ok=True)
    os.makedirs("models", exist_ok=True)

# Application initialization
def init_app():
    """Initialize the application state."""
    if "spark" not in st.session_state:
        # Create a Spark session
        st.session_state.spark = create_spark_session()
        logger.info("Spark session created")
    
    if "app_initialized" not in st.session_state:
        # Set up application state
        st.session_state.app_initialized = True
        st.session_state.data_loaded = False
        st.session_state.delta_tables_prepared = False
        st.session_state.features_engineered = False
        st.session_state.current_page = "home"
        st.session_state.models_trained = {}
        st.session_state.data_cleaned = False

# Navigation
def navigation():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.title("W.A.R.P Navigation")
        
        # Main navigation
        st.sidebar.header("Main Sections")
        
        if st.sidebar.button("🏠 Home", key="nav_home"):
            st.session_state.current_page = "home"
            st.rerun()
            
        if st.sidebar.button("📊 Data Exploration", key="nav_eda"):
            st.session_state.current_page = "data_exploration"
            st.rerun()
            
        if st.sidebar.button("🧪 Feature Engineering", key="nav_features"):
            st.session_state.current_page = "feature_engineering"
            st.rerun()
            
        # Model sections
        st.sidebar.header("Models")
        
        if st.sidebar.button("🔍 Classification", key="nav_classification"):
            st.session_state.current_page = "classification_models"
            st.rerun()
            
        if st.sidebar.button("📈 Regression", key="nav_regression"):
            st.session_state.current_page = "regression_models"
            st.rerun()
            
        if st.sidebar.button("👥 Clustering", key="nav_clustering"):
            st.session_state.current_page = "clustering_models"
            st.rerun()
            
        if st.sidebar.button("⏰ Time Series", key="nav_time_series"):
            st.session_state.current_page = "time_series"
            st.rerun()
            
        if st.sidebar.button("🧠 Advanced Analytics", key="nav_advanced"):
            st.session_state.current_page = "advanced_analytics"
            st.rerun()
        
        # Delta Lake Status
        st.sidebar.header("Data Status")
        
        # Data loading status
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data loaded")
        else:
            st.sidebar.warning("❗ Data not loaded")
            
        # Delta tables status
        if st.session_state.delta_tables_prepared:
            st.sidebar.success("✅ Delta tables prepared")
        else:
            st.sidebar.warning("❗ Delta tables not prepared")
            
        # Feature engineering status
        if st.session_state.features_engineered:
            st.sidebar.success("✅ Features engineered")
        else:
            st.sidebar.warning("❗ Features not engineered")
            
        # Data cleaning status
        if st.session_state.data_cleaned:
            st.sidebar.success("✅ Data cleaned")
        else:
            st.sidebar.warning("❗ Data not cleaned")
        
        # App info
        st.sidebar.header("About")
        st.sidebar.info(
            "W.A.R.P - Weather Analytics & Research Pipeline\n\n"
            "A PySpark MLlib-based platform for weather analytics and prediction."
        )

# Data loading and processing
def load_and_process_data():
    """Load data and prepare Delta tables if not already done."""
    if not st.session_state.data_loaded:
        with st.spinner("Loading datasets..."):
            # Load weather and temperature data
            weather_df = load_weather_data(st.session_state.spark)
            temp_df = load_temperature_data(st.session_state.spark)
            
            if weather_df is not None and temp_df is not None:
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
                
                # Profile the loaded data
                weather_profile = ingest_and_profile_data(st.session_state.spark, weather_df, "weather")
                temp_profile = ingest_and_profile_data(st.session_state.spark, temp_df, "temperature")
                
                st.session_state.weather_profile = weather_profile
                st.session_state.temp_profile = temp_profile
                
                logger.info("Data loaded and profiled")
            else:
                st.error("Failed to load data. Check logs for details.")
    
    if st.session_state.data_loaded and not st.session_state.delta_tables_prepared:
        with st.spinner("Preparing Delta Lake tables..."):
            # Prepare Delta tables
            delta_info = prepare_delta_tables(st.session_state.spark)
            
            # Create Silver tables with cleaned data
            silver_success = create_silver_tables(st.session_state.spark)
            
            if silver_success:
                st.session_state.delta_tables_prepared = True
                st.success("Delta tables prepared successfully!")
                
                # Get information about Delta tables
                delta_tables_info = get_delta_tables_info(st.session_state.spark)
                st.session_state.delta_tables_info = delta_tables_info
                
                logger.info("Delta tables prepared")
            else:
                st.error("Failed to prepare Delta tables. Check logs for details.")

def engineer_features():
    """Engineer features if data is loaded and Delta tables are prepared."""
    if st.session_state.data_loaded and st.session_state.delta_tables_prepared and not st.session_state.features_engineered:
        with st.spinner("Engineering features..."):
            # Engineer weather features
            weather_features = engineer_weather_features(st.session_state.spark)
            
            # Engineer temperature features
            temp_features = engineer_temperature_features(st.session_state.spark)
            
            # Create prediction datasets
            prediction_datasets = create_prediction_datasets(st.session_state.spark)
            
            if weather_features is not None and temp_features is not None:
                st.session_state.features_engineered = True
                st.success("Features engineered successfully!")
                logger.info("Features engineered")
            else:
                st.error("Failed to engineer features. Check logs for details.")

def clean_data():
    """Clean data if it's loaded and not already cleaned."""
    if st.session_state.data_loaded and not st.session_state.data_cleaned:
        with st.spinner("Cleaning data..."):
            # Clean weather data
            clean_weather = clean_dataset(st.session_state.spark, "weather", "mean", "cap")
            
            # Clean temperature data
            clean_temp = clean_dataset(st.session_state.spark, "temperature", "mean", "cap")
            
            if clean_weather is not None and clean_temp is not None:
                st.session_state.data_cleaned = True
                st.success("Data cleaned successfully!")
                logger.info("Data cleaned")
            else:
                st.error("Failed to clean data. Check logs for details.")

# Main app
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="W.A.R.P - Weather Analytics",
        page_icon="🌦️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Initialize app
    init_app()
    
    # Render navigation
    navigation()
    
    # Load and process data
    load_and_process_data()
    
    # Clean data
    if st.session_state.delta_tables_prepared and not st.session_state.data_cleaned:
        clean_data()
    
    # Engineer features
    if st.session_state.delta_tables_prepared and st.session_state.data_cleaned and not st.session_state.features_engineered:
        engineer_features()
    
    # Render the current page
    if st.session_state.current_page == "home":
        render_home_page()
    
    elif st.session_state.current_page == "data_exploration":
        render_data_exploration(st.session_state.spark)
    
    elif st.session_state.current_page == "feature_engineering":
        render_feature_engineering(st.session_state.spark)
    
    elif st.session_state.current_page == "classification_models":
        render_classification_models(st.session_state.spark)
    
    elif st.session_state.current_page == "regression_models":
        render_regression_models(st.session_state.spark)
    
    elif st.session_state.current_page == "clustering_models":
        render_clustering_models(st.session_state.spark)
    
    elif st.session_state.current_page == "time_series":
        render_time_series(st.session_state.spark)
    
    elif st.session_state.current_page == "advanced_analytics":
        render_advanced_analytics(st.session_state.spark)

# Run the app
if __name__ == "__main__":
    main()
