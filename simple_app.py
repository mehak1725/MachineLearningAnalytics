"""
W.A.R.P - Weather Analytics and Research Pipeline
Simplified version using pandas instead of PySpark
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Create necessary folders
def ensure_dirs_exist():
    """Create necessary directories if they don't exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

# Load data
def load_weather_data():
    """Load weather data from CSV."""
    try:
        file_path = "attached_assets/weather_prediction_dataset.csv"
        logging.info(f"Loading weather data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Weather data loaded successfully with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading weather data: {e}")
        st.error(f"Error loading weather data: {e}")
        return None

def load_temperature_data():
    """Load temperature data from CSV."""
    try:
        file_path = "attached_assets/BIG DATA GENERATED DATASET USED FOR ML.csv"
        logging.info(f"Loading temperature data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Temperature data loaded successfully with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading temperature data: {e}")
        st.error(f"Error loading temperature data: {e}")
        return None

# Data processing
def clean_dataset(df, dataset_type="weather"):
    """Clean a dataset by handling missing values and outliers."""
    if df is None:
        return None
    
    try:
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Handle missing values (impute with mean)
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        
        # Handle outliers (cap at 3 standard deviations)
        for col in numeric_cols:
            mean = cleaned_df[col].mean()
            std = cleaned_df[col].std()
            
            # Define outlier bounds
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            # Cap outliers
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
        
        return cleaned_df
    
    except Exception as e:
        logging.error(f"Error cleaning {dataset_type} dataset: {e}")
        return None

def calculate_basic_stats(df):
    """Calculate basic statistics for a dataset."""
    if df is None:
        return None
    
    try:
        # Get basic statistics
        stats = df.describe()
        
        # Add additional statistics
        for col in df.select_dtypes(include=['number']).columns:
            stats.loc['skew', col] = df[col].skew()
            stats.loc['kurtosis', col] = df[col].kurtosis()
            stats.loc['missing', col] = df[col].isna().sum()
            stats.loc['missing_pct', col] = df[col].isna().mean() * 100
        
        return stats
    
    except Exception as e:
        logging.error(f"Error calculating basic statistics: {e}")
        return None

def calculate_correlations(df, limit=15):
    """Calculate correlations between variables."""
    if df is None:
        return None
    
    try:
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Convert to long format for easier filtering
        corr_long = corr_matrix.stack().reset_index()
        corr_long.columns = ['feature1', 'feature2', 'correlation']
        
        # Remove self-correlations
        corr_long = corr_long[corr_long['feature1'] != corr_long['feature2']]
        
        # Sort by absolute correlation and get the top N
        corr_long['abs_correlation'] = corr_long['correlation'].abs()
        top_correlations = corr_long.sort_values('abs_correlation', ascending=False).head(limit)
        
        return top_correlations[['feature1', 'feature2', 'correlation']]
    
    except Exception as e:
        logging.error(f"Error calculating correlations: {e}")
        return None

# Feature engineering
def engineer_weather_features(df):
    """Engineer weather features."""
    if df is None:
        return None
    
    try:
        # Make a copy to avoid modifying the original
        features_df = df.copy()
        
        # Parse date if it exists
        if 'DATE' in features_df.columns:
            features_df['DATE_PARSED'] = pd.to_datetime(features_df['DATE'], format='%Y%m%d')
            features_df['YEAR'] = features_df['DATE_PARSED'].dt.year
            features_df['MONTH'] = features_df['DATE_PARSED'].dt.month_name()
            features_df['MONTH_NUM'] = features_df['DATE_PARSED'].dt.month
            features_df['DAY'] = features_df['DATE_PARSED'].dt.day
            features_df['DAY_OF_WEEK'] = features_df['DATE_PARSED'].dt.day_name()
            
            # Create season
            features_df['SEASON'] = features_df['MONTH_NUM'].apply(
                lambda x: 'Winter' if x in [12, 1, 2] else
                         'Spring' if x in [3, 4, 5] else
                         'Summer' if x in [6, 7, 8] else 'Fall'
            )
        
        # Create temperature categories if temperature exists
        temp_cols = [c for c in features_df.columns if 'TEMP' in c.upper()]
        for col in temp_cols:
            cat_col = f"{col}_category"
            features_df[cat_col] = pd.cut(
                features_df[col],
                bins=[-100, 0, 10, 20, 30, 100],
                labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot']
            )
        
        # Create precipitation categories if precipitation exists
        precip_cols = [c for c in features_df.columns if 'PRCP' in c.upper() or 'PREC' in c.upper()]
        for col in precip_cols:
            cat_col = f"{col}_category"
            features_df[cat_col] = pd.cut(
                features_df[col],
                bins=[-1, 0, 1, 5, 10, 1000],
                labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme']
            )
        
        return features_df
    
    except Exception as e:
        logging.error(f"Error engineering weather features: {e}")
        return None

def engineer_temperature_features(df):
    """Engineer temperature features."""
    if df is None:
        return None
    
    try:
        # Make a copy to avoid modifying the original
        features_df = df.copy()
        
        # Create temperature regime
        if 'AvgYearlyTemp' in features_df.columns:
            features_df['temp_regime'] = pd.cut(
                features_df['AvgYearlyTemp'],
                bins=[-100, 0, 10, 20, 30, 100],
                labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot']
            )
        
        # Create temperature change category if possible
        if 'TempChange' in features_df.columns:
            features_df['temp_change_category'] = pd.cut(
                features_df['TempChange'],
                bins=[-100, -1, -0.2, 0.2, 1, 100],
                labels=['Significant cooling', 'Cooling', 'Stable', 'Warming', 'Significant warming']
            )
        
        # Create trend based on temperature change
        if 'TempChange' in features_df.columns:
            features_df['trend'] = features_df['TempChange'].apply(
                lambda x: 'Increasing' if x > 0.2 else
                         'Decreasing' if x < -0.2 else 'Stable'
            )
        
        # Create decade if year exists
        if 'Year' in features_df.columns:
            features_df['decade'] = (features_df['Year'] // 10) * 10
        
        return features_df
    
    except Exception as e:
        logging.error(f"Error engineering temperature features: {e}")
        return None

# Application initialization
def init_app():
    """Initialize the application state."""
    if "app_initialized" not in st.session_state:
        # Set up application state
        st.session_state.app_initialized = True
        st.session_state.data_loaded = False
        st.session_state.data_processed = False
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
        
        # Data Status
        st.sidebar.header("Data Status")
        
        # Data loading status
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data loaded")
        else:
            st.sidebar.warning("❗ Data not loaded")
            
        # Data processing status
        if st.session_state.data_processed:
            st.sidebar.success("✅ Data processed")
        else:
            st.sidebar.warning("❗ Data not processed")
            
        # Feature engineering status
        if st.session_state.features_engineered:
            st.sidebar.success("✅ Features engineered")
        else:
            st.sidebar.warning("❗ Features not engineered")
        
        # App info
        st.sidebar.header("About")
        st.sidebar.info(
            "W.A.R.P - Weather Analytics & Research Pipeline\n\n"
            "A platform for weather analytics and prediction."
        )

# Data loading and processing
def load_and_process_data():
    """Load data and process it."""
    # Force data loading
    with st.spinner("Loading datasets..."):
        # Load weather and temperature data
        weather_df = load_weather_data()
        temp_df = load_temperature_data()
        
        if weather_df is not None and temp_df is not None:
            st.session_state.data_loaded = True
            st.session_state.weather_df = weather_df
            st.session_state.temp_df = temp_df
            logger.info("Data loaded")
            
            # Immediately process data
            cleaned_weather = clean_dataset(weather_df, "weather")
            cleaned_temp = clean_dataset(temp_df, "temperature")
            
            if cleaned_weather is not None and cleaned_temp is not None:
                st.session_state.cleaned_weather = cleaned_weather
                st.session_state.cleaned_temp = cleaned_temp
                st.session_state.data_processed = True
                st.session_state.data_cleaned = True
                logger.info("Data processed")
            else:
                logger.error("Failed to process data")
                st.error("Failed to process data. Check logs for details.")
        else:
            logger.error("Failed to load data")
            st.error("Failed to load data. Check logs for details.")

def engineer_features():
    """Engineer features if data is loaded and processed."""
    # Force feature engineering
    if st.session_state.data_loaded and st.session_state.data_processed:
        with st.spinner("Engineering features..."):
            # Engineer features
            weather_features = engineer_weather_features(st.session_state.cleaned_weather)
            temp_features = engineer_temperature_features(st.session_state.cleaned_temp)
            
            if weather_features is not None and temp_features is not None:
                st.session_state.weather_features = weather_features
                st.session_state.temp_features = temp_features
                st.session_state.features_engineered = True
                logger.info("Features engineered")
            else:
                logger.error("Failed to engineer features")
                st.error("Failed to engineer features. Check logs for details.")

# Page rendering functions
def render_home_page():
    """Render the home page."""
    st.title("W.A.R.P: Weather Analytics & Research Pipeline")
    
    st.markdown("""
    ### Welcome to the W.A.R.P Platform
    
    W.A.R.P (Weather Analytics & Research Pipeline) is a comprehensive platform for analyzing and predicting weather patterns. 
    This simplified version demonstrates the concepts using pandas instead of PySpark.
    
    #### Key Features:
    - Data exploration and visualization of weather patterns
    - Feature engineering and selection for weather prediction
    - Multiple machine learning models for classification and regression
    - Time series analysis and forecasting
    
    #### Getting Started:
    1. Use the sidebar to navigate between different sections
    2. Start with Data Exploration to understand the datasets
    3. Move to Feature Engineering to prepare data for modeling
    
    #### Available Datasets:
    - Historical weather measurements
    - Global temperature records
    """)
    
    # Display system status
    st.subheader("System Status")
    
    # Create columns for status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.error("❌ Data Not Loaded")
    
    with col2:
        if st.session_state.data_processed:
            st.success("✅ Data Processed")
        else:
            st.error("❌ Data Not Processed")
            
    with col3:
        if st.session_state.features_engineered:
            st.success("✅ Features Engineered")
        else:
            st.error("❌ Features Not Engineered")
    
    # Load data button
    if not st.session_state.data_loaded:
        if st.button("Load Initial Data"):
            st.session_state.current_page = "data_exploration"
            st.rerun()

def render_data_exploration():
    """Render the data exploration page."""
    st.title("Data Exploration")
    
    if not st.session_state.data_loaded:
        st.warning("Data not loaded yet. Please wait while we load the data.")
        load_and_process_data()
        return
    
    # Create tabs for different exploration options
    tabs = st.tabs(["Basic Statistics", "Correlations", "Raw Data"])
    
    with tabs[0]:
        st.subheader("Basic Statistics")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["Weather", "Temperature"],
            key="basic_stats_dataset"
        )
        
        if st.button("Calculate Basic Statistics"):
            with st.spinner("Calculating basic statistics..."):
                if dataset == "Weather":
                    df = st.session_state.weather_df
                else:
                    df = st.session_state.temp_df
                
                stats = calculate_basic_stats(df)
                
                if stats is not None:
                    st.subheader(f"Basic Statistics for {dataset} Dataset")
                    st.dataframe(stats)
                else:
                    st.error("Failed to calculate statistics. Check logs for details.")
    
    with tabs[1]:
        st.subheader("Feature Correlations")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["Weather", "Temperature"],
            key="correlation_dataset"
        )
        
        limit = st.slider("Number of Correlations", min_value=5, max_value=50, value=15)
        
        if st.button("Calculate Correlations"):
            with st.spinner("Calculating correlations..."):
                if dataset == "Weather":
                    df = st.session_state.weather_df
                else:
                    df = st.session_state.temp_df
                
                corr_df = calculate_correlations(df, limit=limit)
                
                if corr_df is not None:
                    st.subheader(f"Top {limit} Correlations for {dataset} Dataset")
                    
                    # Display correlation table
                    st.dataframe(corr_df)
                    
                    # Plot correlation heatmap
                    if len(corr_df) >= 2:
                        pivot_df = corr_df.pivot(index="feature1", columns="feature2", values="correlation")
                        fig = px.imshow(
                            pivot_df,
                            labels=dict(color="Correlation"),
                            x=pivot_df.columns,
                            y=pivot_df.index,
                            color_continuous_scale='RdBu_r',
                            title=f"Correlation Heatmap - {dataset} Dataset"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to calculate correlations. Check logs for details.")
    
    with tabs[2]:
        st.subheader("Raw Data Viewer")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["Weather", "Temperature", "Weather (Cleaned)", "Temperature (Cleaned)"],
            key="raw_data_dataset"
        )
        
        num_rows = st.slider("Number of Rows", min_value=5, max_value=100, value=20)
        
        if st.button("View Raw Data"):
            with st.spinner("Loading raw data..."):
                if dataset == "Weather":
                    df = st.session_state.weather_df
                elif dataset == "Temperature":
                    df = st.session_state.temp_df
                elif dataset == "Weather (Cleaned)" and hasattr(st.session_state, "cleaned_weather"):
                    df = st.session_state.cleaned_weather
                elif dataset == "Temperature (Cleaned)" and hasattr(st.session_state, "cleaned_temp"):
                    df = st.session_state.cleaned_temp
                else:
                    st.error(f"Dataset {dataset} not available yet. Process data first.")
                    return
                
                if df is not None:
                    st.subheader(f"Raw Data - {dataset} Dataset")
                    st.dataframe(df.head(num_rows))
                else:
                    st.error(f"Failed to read {dataset} dataset. Check logs for details.")

def render_feature_engineering():
    """Render the feature engineering page."""
    st.title("Feature Engineering")
    
    if not st.session_state.data_loaded:
        st.warning("Data not loaded yet. Please load data first.")
        load_and_process_data()
        return
    
    if not st.session_state.data_processed:
        st.warning("Data not processed yet. Processing data...")
        load_and_process_data()
        return
    
    # Create tabs for different feature engineering options
    tabs = st.tabs(["Feature Engineering", "View Engineered Features"])
    
    with tabs[0]:
        st.subheader("Feature Engineering")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["Weather", "Temperature"],
            key="feature_engineering_dataset"
        )
        
        if st.button("Engineer Features"):
            with st.spinner("Engineering features..."):
                engineer_features()
    
    with tabs[1]:
        st.subheader("View Engineered Features")
        
        if not st.session_state.features_engineered:
            st.warning("Features not engineered yet. Please engineer features first.")
            return
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["Weather Features", "Temperature Features"],
            key="view_features_dataset"
        )
        
        num_rows = st.slider("Number of Rows", min_value=5, max_value=100, value=20, key="view_features_rows")
        
        if st.button("View Features"):
            with st.spinner("Loading features..."):
                if dataset == "Weather Features":
                    df = st.session_state.weather_features
                else:
                    df = st.session_state.temp_features
                
                if df is not None:
                    st.subheader(f"Engineered Features - {dataset}")
                    st.dataframe(df.head(num_rows))
                    
                    # Show column list
                    st.subheader("Columns in Dataset")
                    cols = df.columns.tolist()
                    # Split into categorical and numerical columns
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    num_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    st.write(f"Total columns: {len(cols)}")
                    st.write(f"Categorical columns: {len(cat_cols)}")
                    st.write(f"Numerical columns: {len(num_cols)}")
                    
                    # Show distribution of key features
                    st.subheader("Feature Distributions")
                    
                    # For Weather dataset
                    if dataset == "Weather Features":
                        if "TEMP" in df.columns:
                            fig = px.histogram(df, x="TEMP", title="Temperature Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if "SEASON" in df.columns:
                            fig = px.pie(df, names="SEASON", title="Distribution by Season")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # For Temperature dataset
                    else:
                        if "AvgYearlyTemp" in df.columns:
                            fig = px.histogram(df, x="AvgYearlyTemp", title="Average Yearly Temperature Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if "Country" in df.columns:
                            # Get top 10 countries by count
                            top_countries = df["Country"].value_counts().head(10)
                            fig = px.bar(
                                x=top_countries.index, 
                                y=top_countries.values,
                                title="Top 10 Countries by Record Count",
                                labels={"x": "Country", "y": "Count"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Failed to load {dataset}. Check logs for details.")

def render_classification_models():
    """Render the classification models page."""
    st.title("Classification Models")
    
    if not st.session_state.features_engineered:
        st.warning("Features not engineered yet. Please engineer features first.")
        engineer_features()
        return
    
    st.markdown("""
    ### Classification Model Demonstration
    
    This section demonstrates how different classification models could be applied to weather data.
    In a production environment, these models would be implemented using advanced ML libraries.
    """)
    
    # Create tabs for different aspects
    tabs = st.tabs(["Problem Definition", "Feature Selection", "Model Training", "Model Comparison", "Advanced Models"])
    
    with tabs[0]:
        st.subheader("Classification Problem Definition")
        
        st.markdown("""
        #### Weather Classification Problems
        
        1. **Precipitation Category Prediction**:
           - Predict whether precipitation will be None, Light, Moderate, or Heavy
           - Features: temperature, pressure, humidity, etc.
        
        2. **Temperature Regime Classification**:
           - Classify temperature as Freezing, Cold, Mild, Warm, or Hot
           - Features: location, season, historical patterns
        
        3. **Weather Event Classification**:
           - Predict weather events like rain, snow, thunderstorm, etc.
           - Features: pressure changes, humidity, temperature, etc.
           
        4. **Extreme Weather Event Prediction**:
           - Classify whether extreme weather conditions will occur
           - Features: historical patterns, atmospheric pressure trends, etc.
        """)
        
        # Show sample distribution of a target variable
        if hasattr(st.session_state, "weather_features"):
            weather_features = st.session_state.weather_features
            
            # Check if we have temperature category columns
            temp_cat_cols = [c for c in weather_features.columns if 'category' in c.lower()]
            
            if temp_cat_cols:
                selected_cat = st.selectbox("Select Category Variable", temp_cat_cols)
                
                # Show distribution
                if selected_cat in weather_features.columns:
                    st.subheader(f"Distribution of {selected_cat}")
                    
                    cat_counts = weather_features[selected_cat].value_counts().reset_index()
                    cat_counts.columns = ['Category', 'Count']
                    
                    fig = px.bar(
                        cat_counts,
                        x='Category',
                        y='Count',
                        title=f"Distribution of {selected_cat}",
                        color='Category'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Feature Selection")
        
        st.markdown("""
        #### Feature Selection Methods
        
        1. **Correlation-based**:
           - Select features with high correlation to the target
           - Filter out features with high correlation among themselves
        
        2. **Chi-Square Test**:
           - Measure dependence between categorical variables
           - Select features with high chi-square scores
        
        3. **Tree-based Methods**:
           - Use feature importance from tree-based models
           - Select top features based on importance scores
           
        4. **Recursive Feature Elimination (RFE)**:
           - Start with all features, recursively eliminate the least significant
           - Use a model to determine feature importance at each step
           
        5. **Principal Component Analysis (PCA)**:
           - Transform features into uncorrelated components
           - Select components that explain most of the variance
        """)
        
        # Show correlation between features and a categorical target
        if hasattr(st.session_state, "weather_features"):
            weather_features = st.session_state.weather_features
            
            # Get categorical columns
            cat_cols = weather_features.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) > 0:
                selected_target = st.selectbox("Select Target Variable", cat_cols)
                
                # Get numerical features
                num_cols = weather_features.select_dtypes(include=['number']).columns
                
                if len(num_cols) > 0 and selected_target:
                    st.markdown("#### Feature Analysis for Selected Target")
                    
                    # Create crosstab visualizations for top features
                    for feature in list(num_cols)[:3]:  # Show top 3 features
                        st.subheader(f"{feature} vs {selected_target}")
                        
                        # Create box plot
                        fig = px.box(
                            weather_features, 
                            x=selected_target, 
                            y=feature,
                            title=f"{feature} by {selected_target} Categories"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Model Training")
        
        # Demo classification models
        model_type = st.selectbox(
            "Select Classification Model",
            ["Logistic Regression", "Random Forest", "Naive Bayes", "Gradient Boosted Trees", "Support Vector Machine"]
        )
        
        st.markdown(f"### {model_type} Model")
        
        if model_type == "Logistic Regression":
            st.markdown("""
            **Logistic Regression** is a statistical model that uses a logistic function to model a binary dependent variable. 
            It can be extended to multi-class classification using strategies like one-vs-rest.
            
            #### Hyperparameters:
            - **Regularization Strength (C)**: Controls the strength of regularization (lower values = stronger regularization)
            - **Penalty Type**: L1 (Lasso), L2 (Ridge), or Elastic Net
            - **Max Iterations**: Maximum number of iterations for the solver
            
            #### Strengths:
            - Simple and interpretable
            - Works well for linearly separable classes
            - Provides probability estimates
            
            #### Weaknesses:
            - Limited complexity for modeling non-linear relationships
            - May underperform with high-dimensional data
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
                max_iter = st.slider("Max Iterations", 100, 1000, 100)
            with col2:
                penalty = st.selectbox("Penalty Type", ["l1", "l2", "elasticnet", "none"])
                solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
        
        elif model_type == "Random Forest":
            st.markdown("""
            **Random Forest** is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.
            
            #### Hyperparameters:
            - **Number of Trees**: How many trees to build in the forest
            - **Max Depth**: Maximum depth of each tree
            - **Min Samples Split**: Minimum samples required to split a node
            - **Min Samples Leaf**: Minimum samples required at a leaf node
            
            #### Strengths:
            - Handles non-linear relationships well
            - Robust to outliers and noise
            - Provides feature importance scores
            
            #### Weaknesses:
            - Less interpretable than single decision trees
            - Can be computationally intensive for large datasets
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of Trees", 10, 500, 100)
                max_depth = st.slider("Max Depth", 1, 50, 10)
            with col2:
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
                criterion = st.selectbox("Split Criterion", ["gini", "entropy"])
        
        elif model_type == "Naive Bayes":
            st.markdown("""
            **Naive Bayes** classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
            
            #### Types:
            - **Gaussian NB**: For continuous data, assumes Gaussian distribution
            - **Multinomial NB**: For discrete data (e.g., text classification)
            - **Bernoulli NB**: For binary/boolean features
            
            #### Hyperparameters:
            - **Alpha**: Smoothing parameter (additive smoothing)
            - **Fit Prior**: Whether to learn class prior probabilities
            
            #### Strengths:
            - Simple and fast to train
            - Works well with small datasets and high dimensions
            - Good with text classification problems
            
            #### Weaknesses:
            - Assumption of feature independence (often unrealistic)
            - May be outperformed by more complex models
            """)
            
            # Show sample training options
            nb_type = st.selectbox("Naive Bayes Type", ["Gaussian", "Multinomial", "Bernoulli"])
            alpha = st.slider("Alpha (Smoothing)", 0.0, 2.0, 1.0, 0.1)
            fit_prior = st.checkbox("Fit Prior", True)
        
        elif model_type == "Gradient Boosted Trees":
            st.markdown("""
            **Gradient Boosted Trees** are an ensemble learning method that builds trees one at a time, where each new tree helps correct errors made by previously trained trees.
            
            #### Hyperparameters:
            - **Learning Rate**: Controls the contribution of each tree
            - **Number of Estimators**: Total number of trees to build
            - **Max Depth**: Maximum depth of each tree
            - **Subsample**: Fraction of samples to use for fitting
            
            #### Strengths:
            - Often provides the highest accuracy
            - Can handle different types of data
            - Has good customization options
            
            #### Weaknesses:
            - More prone to overfitting than Random Forest
            - Requires careful tuning of hyperparameters
            - Slower to train and predict
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                n_estimators = st.slider("Number of Estimators", 10, 500, 100)
            with col2:
                max_depth = st.slider("Max Depth", 1, 10, 3)
                subsample = st.slider("Subsample", 0.1, 1.0, 1.0, 0.1)
                
        elif model_type == "Support Vector Machine":
            st.markdown("""
            **Support Vector Machine (SVM)** works by finding the hyperplane that best divides a dataset into classes. SVMs can also use kernel methods to transform the data into higher dimensions where separation might be easier.
            
            #### Hyperparameters:
            - **Kernel**: Function to transform the data (linear, polynomial, rbf, sigmoid)
            - **C Parameter**: Regularization parameter
            - **Gamma**: Kernel coefficient (for rbf, poly, sigmoid)
            
            #### Strengths:
            - Effective in high-dimensional spaces
            - Works well when classes are separable
            - Versatile through different kernel functions
            
            #### Weaknesses:
            - Doesn't scale well to large datasets
            - Longer training time for large datasets
            - Can be sensitive to hyperparameter choices
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                c_value = st.slider("C Parameter", 0.1, 10.0, 1.0)
            with col2:
                gamma = st.selectbox("Gamma", ["scale", "auto", "value"])
                if gamma == "value":
                    gamma_value = st.slider("Gamma Value", 0.001, 1.0, 0.1)
                degree = st.slider("Degree (for poly kernel)", 2, 5, 3)
        
        # Training button (simulated)
        if st.button("Train Model"):
            with st.spinner(f"Training {model_type}..."):
                # Simulate training time
                time.sleep(1)
                st.success(f"{model_type} model trained successfully!")
                
                # Show sample metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", "0.87", "0.05")
                with col2:
                    st.metric("Precision", "0.83", "0.03")
                with col3:
                    st.metric("Recall", "0.89", "0.07")
                
                # Sample visualization of results
                st.subheader("Classification Report")
                
                # Create synthetic classification report
                class_report = pd.DataFrame({
                    'Class': ['Freezing', 'Cold', 'Mild', 'Warm', 'Hot', 'Average/Total'],
                    'Precision': [0.94, 0.88, 0.82, 0.86, 0.91, 0.87],
                    'Recall': [0.90, 0.84, 0.87, 0.93, 0.88, 0.89],
                    'F1-Score': [0.92, 0.86, 0.84, 0.89, 0.90, 0.88],
                    'Support': [50, 75, 100, 98, 80, 403]
                })
                
                st.dataframe(class_report)
    
    with tabs[3]:
        st.subheader("Model Comparison")
        
        st.markdown("""
        #### Comparing Classification Models
        
        Selecting the right classification model requires comparing multiple options on the same dataset.
        Here's how different models might perform on a weather classification task:
        """)
        
        # Create sample comparative data
        model_names = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Gradient Boosted Trees', 'SVM']
        accuracies = [0.78, 0.85, 0.72, 0.88, 0.82]
        f1_scores = [0.76, 0.84, 0.70, 0.87, 0.81]
        training_times = [5, 25, 3, 40, 15]  # in seconds
        
        # Create comparison figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=accuracies,
            name='Accuracy',
            marker_color='royalblue'
        ))
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=f1_scores,
            name='F1 Score',
            marker_color='darkgreen'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Training time visualization
        fig2 = px.bar(
            x=model_names, 
            y=training_times,
            title='Training Time Comparison',
            labels={'x': 'Model', 'y': 'Training Time (seconds)'},
            color=training_times,
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        #### Key Insights:
        - **Gradient Boosted Trees** typically provides the highest accuracy but takes longer to train
        - **Naive Bayes** is the fastest to train but often has lower accuracy
        - **Random Forest** provides a good balance of accuracy and training time
        - **SVM** can perform well on smaller datasets but doesn't scale as well
        - **Logistic Regression** is simple and interpretable but may miss complex patterns
        
        The best model depends on your specific problem, data characteristics, and requirements.
        """)
        
    with tabs[4]:
        st.subheader("Advanced Classification Techniques")
        
        st.markdown("""
        #### Advanced Models and Approaches
        
        Beyond standard classification models, these advanced techniques can further improve performance:
        
        1. **Ensemble Methods**:
           - **Voting Classifiers**: Combine predictions from multiple models
           - **Stacking**: Train a meta-model on the outputs of base models
           
        2. **Neural Networks**:
           - **Multilayer Perceptrons (MLP)**: Deep learning for complex patterns
           - **Convolutional Neural Networks (CNNs)**: For spatial data like radar images
           
        3. **Time Series Classification**:
           - **LSTM/RNN**: For sequential weather data with temporal dependencies
           - **Dynamic Time Warping**: For comparing weather pattern sequences
           
        4. **Automated Machine Learning (AutoML)**:
           - Automatically selects models and tunes hyperparameters
           - Useful for finding optimal configurations efficiently
        """)
        
        # Example visualization of ensemble method
        st.subheader("Example: Voting Classifier")
        
        # Create sample data
        methods = ['Individual Models', 'Hard Voting', 'Soft Voting']
        accuracy = [0.85, 0.87, 0.89]
        
        fig = px.bar(
            x=methods,
            y=accuracy,
            title="Ensemble Method Performance",
            labels={'x': 'Method', 'y': 'Accuracy'},
            color=accuracy,
            color_continuous_scale='Blues',
            text=accuracy
        )
        
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show a sample confusion matrix visualization
        st.subheader("Sample Confusion Matrix")
        
        # Create synthetic confusion matrix
        labels = ['Freezing', 'Cold', 'Mild', 'Warm', 'Hot']
        confusion_matrix = {
            'Freezing': {'Freezing': 45, 'Cold': 5, 'Mild': 0, 'Warm': 0, 'Hot': 0},
            'Cold': {'Freezing': 3, 'Cold': 67, 'Mild': 6, 'Warm': 0, 'Hot': 0},
            'Mild': {'Freezing': 0, 'Cold': 8, 'Mild': 82, 'Mild': 7, 'Hot': 0},
            'Warm': {'Freezing': 0, 'Cold': 0, 'Mild': 5, 'Warm': 90, 'Hot': 3},
            'Hot': {'Freezing': 0, 'Cold': 0, 'Mild': 0, 'Warm': 5, 'Hot': 75}
        }
        
        # Convert to DataFrame for display
        matrix_data = []
        for true_label in labels:
            row = []
            for pred_label in labels:
                row.append(confusion_matrix[true_label].get(pred_label, 0))
            matrix_data.append(row)
        
        conf_df = pd.DataFrame(matrix_data, index=labels, columns=labels)
        
        # Create heatmap
        fig = px.imshow(
            conf_df,
            labels=dict(color="Count"),
            x=conf_df.columns,
            y=conf_df.index,
            color_continuous_scale='Blues',
            title="Sample Confusion Matrix",
            text_auto=True
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500,
            width=500
        )
        
        st.plotly_chart(fig)

def render_regression_models():
    """Render the regression models page."""
    st.title("Regression Models")
    
    if not st.session_state.features_engineered:
        st.warning("Features not engineered yet. Please engineer features first.")
        engineer_features()
        return
    
    st.markdown("""
    ### Regression Model Demonstration
    
    This section demonstrates how different regression models could be applied to weather data.
    In a production environment, these models would be implemented using advanced ML libraries.
    """)
    
    # Create tabs for different aspects
    tabs = st.tabs(["Problem Definition", "Feature Selection", "Model Training", "Model Comparison", "Advanced Models"])
    
    with tabs[0]:
        st.subheader("Regression Problem Definition")
        
        st.markdown("""
        #### Weather Regression Problems
        
        1. **Temperature Prediction**:
           - Predict exact temperature values
           - Features: historical patterns, location, season, etc.
        
        2. **Precipitation Amount Prediction**:
           - Predict exact precipitation amounts in mm
           - Features: humidity, pressure, temperature, etc.
        
        3. **Wind Speed Prediction**:
           - Predict exact wind speed values
           - Features: pressure gradients, temperature differentials, etc.
           
        4. **Solar Radiation Forecasting**:
           - Predict solar radiation levels
           - Features: cloud cover, time of year, latitude, etc.
           
        5. **Humidity Prediction**:
           - Predict relative humidity levels
           - Features: temperature, pressure, proximity to water bodies, etc.
        """)
        
        # Show distribution of a potential target variable
        if hasattr(st.session_state, "weather_features"):
            weather_features = st.session_state.weather_features
            
            # Check if we have temperature columns
            temp_cols = [c for c in weather_features.columns if 'TEMP' in c.upper()]
            
            if temp_cols:
                selected_temp = st.selectbox("Select Temperature Variable", temp_cols)
                
                # Show distribution
                if selected_temp in weather_features.columns:
                    st.subheader(f"Distribution of {selected_temp}")
                    
                    fig = px.histogram(
                        weather_features,
                        x=selected_temp,
                        title=f"Distribution of {selected_temp}",
                        nbins=30
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show summary statistics
                    st.subheader(f"Summary Statistics for {selected_temp}")
                    stats = weather_features[selected_temp].describe()
                    st.dataframe(pd.DataFrame(stats).T)
    
    with tabs[1]:
        st.subheader("Feature Selection for Regression")
        
        st.markdown("""
        #### Feature Selection Methods
        
        1. **Correlation Analysis**:
           - Select features with high correlation to the target
           - Remove multicollinearity
        
        2. **Forward/Backward Selection**:
           - Iteratively add/remove features based on model performance
        
        3. **Regularization Methods**:
           - Lasso (L1) regularization for feature selection
           - Shrinks less important feature coefficients to zero
           
        4. **Feature Importance from Tree Models**:
           - Use Random Forest or Gradient Boosting to rank features
           - Select top features based on importance scores
           
        5. **Variance Inflation Factor (VIF)**:
           - Detect multicollinearity among predictors
           - Remove features with high VIF values
        """)
        
        # Show correlation between features and a numerical target
        if hasattr(st.session_state, "weather_features"):
            weather_features = st.session_state.weather_features
            
            # Get numerical columns
            num_cols = weather_features.select_dtypes(include=['number']).columns
            
            if len(num_cols) > 0:
                selected_target = st.selectbox("Select Target Variable", num_cols)
                
                # Get top 5 correlated features
                if selected_target:
                    correlations = {}
                    for col in num_cols:
                        if col != selected_target:
                            cor = weather_features[col].corr(weather_features[selected_target])
                            correlations[col] = cor
                    
                    # Sort by absolute correlation
                    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    st.subheader(f"Top Features Correlated with {selected_target}")
                    
                    # Create DataFrame for display
                    top_corr_df = pd.DataFrame(sorted_corr[:10], columns=['Feature', 'Correlation'])
                    
                    # Plot correlation
                    fig = px.bar(
                        top_corr_df,
                        x='Feature',
                        y='Correlation',
                        title=f"Feature Correlations with {selected_target}",
                        color='Correlation',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Model Training")
        
        # Demo regression models
        model_type = st.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Regression", "Neural Network"]
        )
        
        st.markdown(f"### {model_type}")
        
        if model_type == "Linear Regression":
            st.markdown("""
            **Linear Regression** models the relationship between a dependent variable and one or more independent variables using a linear equation.
            
            #### Hyperparameters:
            - **Regularization Type**: None, Ridge (L2), or Lasso (L1)
            - **Regularization Strength (alpha)**: Controls the amount of regularization
            - **Fit Intercept**: Whether to include a bias/intercept term
            
            #### Strengths:
            - Simple and interpretable
            - Fast to train and predict
            - Provides feature coefficients showing importance and direction
            
            #### Weaknesses:
            - Assumes linear relationship between features and target
            - Sensitive to outliers
            - Limited capacity for modeling complex, non-linear relationships
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                reg_type = st.selectbox("Regularization Type", ["None", "Ridge (L2)", "Lasso (L1)", "ElasticNet"])
                fit_intercept = st.checkbox("Fit Intercept", True)
            with col2:
                alpha = st.slider("Regularization Strength (alpha)", 0.0, 1.0, 0.1, 0.01)
                normalize = st.checkbox("Normalize Features", False)
        
        elif model_type == "Decision Tree":
            st.markdown("""
            **Decision Tree Regression** uses a tree-like model of decisions where each node represents a feature, each branch represents a decision, and each leaf represents an outcome (prediction).
            
            #### Hyperparameters:
            - **Max Depth**: Maximum depth of the tree
            - **Min Samples Split**: Minimum samples required to split a node
            - **Min Samples Leaf**: Minimum samples required at a leaf node
            - **Criterion**: Function to measure the quality of a split (MSE, MAE)
            
            #### Strengths:
            - Can model non-linear relationships
            - No assumptions about data distribution
            - Feature importance and decision rules are easy to understand
            
            #### Weaknesses:
            - Tendency to overfit if not pruned
            - Can be unstable (small changes in data can result in very different trees)
            - Limited prediction smoothness (step-wise predictions)
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                max_depth = st.slider("Max Depth", 1, 50, 10)
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            with col2:
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)
                criterion = st.selectbox("Criterion", ["mse", "mae"])
        
        elif model_type == "Random Forest":
            st.markdown("""
            **Random Forest Regression** is an ensemble learning method that operates by constructing multiple decision trees at training time and outputting the mean prediction of the individual trees.
            
            #### Hyperparameters:
            - **Number of Estimators**: Number of trees in the forest
            - **Max Depth**: Maximum depth of each tree
            - **Min Samples Split**: Minimum samples required to split a node
            - **Max Features**: Maximum number of features to consider for splitting
            
            #### Strengths:
            - More robust and less prone to overfitting than a single decision tree
            - Handles high-dimensional data well
            - Provides feature importance scores
            
            #### Weaknesses:
            - Less interpretable than a single decision tree
            - Computationally more intensive than simpler models
            - Can be slow for large datasets
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of Estimators", 10, 500, 100)
                max_depth = st.slider("Max Depth", 1, 50, 10)
            with col2:
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
                max_features = st.selectbox("Max Features", ["auto", "sqrt", "log2", "All"])
        
        elif model_type == "Gradient Boosting":
            st.markdown("""
            **Gradient Boosting Regression** builds trees sequentially, where each tree corrects the errors of the previous ones, using gradient descent to minimize the loss function.
            
            #### Hyperparameters:
            - **Learning Rate**: Shrinks the contribution of each tree
            - **Number of Estimators**: Total number of trees to build
            - **Max Depth**: Maximum depth of each tree
            - **Subsample**: Fraction of samples to use for fitting individual trees
            
            #### Strengths:
            - Often provides better performance than most other algorithms
            - Robust to outliers and can handle different types of data
            - Automatically handles feature interactions
            
            #### Weaknesses:
            - Prone to overfitting if not carefully tuned
            - Computationally intensive
            - More hyperparameters to tune
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
                n_estimators = st.slider("Number of Estimators", 10, 500, 100)
            with col2:
                max_depth = st.slider("Max Depth", 1, 10, 3)
                subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
                
        elif model_type == "Support Vector Regression":
            st.markdown("""
            **Support Vector Regression (SVR)** uses the same principles as SVM, but for regression tasks. It tries to find a function that deviates from the target by a value no greater than a margin of tolerance.
            
            #### Hyperparameters:
            - **Kernel**: Function to transform the data (linear, polynomial, rbf, sigmoid)
            - **C Parameter**: Penalty parameter of the error term
            - **Epsilon**: Specifies the margin of tolerance inside which no penalty is given
            - **Gamma**: Kernel coefficient for rbf, poly and sigmoid kernels
            
            #### Strengths:
            - Works well for high-dimensional data
            - Different kernel functions allow for flexibility in modeling
            - Good generalization potential
            
            #### Weaknesses:
            - Computationally intensive for large datasets
            - Difficult to interpret
            - Sensitive to hyperparameter choices
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                c_value = st.slider("C Parameter", 0.1, 10.0, 1.0)
            with col2:
                epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1)
                gamma = st.selectbox("Gamma", ["scale", "auto", "value"])
                if gamma == "value":
                    gamma_value = st.slider("Gamma Value", 0.001, 1.0, 0.1)
                    
        elif model_type == "Neural Network":
            st.markdown("""
            **Neural Network Regression** uses a multi-layer perceptron (MLP) to learn a non-linear function approximator for regression.
            
            #### Hyperparameters:
            - **Hidden Layer Sizes**: Number of neurons in each hidden layer
            - **Activation Function**: Function for non-linearity (relu, tanh, sigmoid)
            - **Learning Rate**: Controls the step size during optimization
            - **Solver**: Algorithm for weight optimization
            
            #### Strengths:
            - Can model highly complex, non-linear relationships
            - Flexible architecture can be adapted to many problems
            - Automatically learns feature interactions
            
            #### Weaknesses:
            - Requires more data to train effectively
            - Computationally intensive
            - "Black box" - difficult to interpret
            """)
            
            # Show sample training options
            col1, col2 = st.columns(2)
            with col1:
                hidden_layers = st.text_input("Hidden Layer Sizes (comma-separated)", "100,50")
                activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])
            with col2:
                learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
                solver = st.selectbox("Solver", ["adam", "sgd", "lbfgs"])
        
        # Training button (simulated)
        if st.button("Train Model"):
            with st.spinner(f"Training {model_type}..."):
                # Simulate training time
                time.sleep(1)
                st.success(f"{model_type} model trained successfully!")
                
                # Show sample metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", "2.35", "-0.42")
                with col2:
                    st.metric("MAE", "1.89", "-0.31")
                with col3:
                    st.metric("R²", "0.84", "0.05")
                
                # Sample visualization of results
                st.subheader("Model Evaluation")
                
                # Create synthetic predictions and actual values
                np.random.seed(42)
                n_samples = 100
                actual = np.random.normal(20, 5, n_samples)
                predictions = actual + np.random.normal(0, 2, n_samples)
                
                # Create DataFrame
                results_df = pd.DataFrame({
                    'Actual': actual,
                    'Predicted': predictions
                })
                
                # Create scatter plot
                fig = px.scatter(
                    results_df,
                    x='Actual',
                    y='Predicted',
                    title="Predictions vs Actual Values",
                    labels={'Actual': "Actual Temperature", 'Predicted': "Predicted Temperature"}
                )
                
                # Add 45-degree reference line
                min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
                max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show residual plot
                results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
                
                fig = px.scatter(
                    results_df,
                    x='Predicted',
                    y='Residual',
                    title="Residual Plot",
                    labels={'Predicted': "Predicted Temperature", 'Residual': "Residual (Actual - Predicted)"}
                )
                
                # Add horizontal reference line at y=0
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("Model Comparison")
        
        st.markdown("""
        #### Comparing Regression Models
        
        Evaluating multiple regression models helps identify the best approach for a specific prediction task.
        Here's how different models might perform on a weather prediction task:
        """)
        
        # Create sample comparative data
        model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']
        rmse_scores = [3.45, 2.87, 2.41, 2.35, 2.68, 2.39]
        r2_scores = [0.71, 0.79, 0.83, 0.84, 0.80, 0.83]
        training_times = [2, 8, 25, 40, 15, 60]  # in seconds
        
        # Create comparison figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=rmse_scores,
            name='RMSE',
            marker_color='indianred'
        ))
        
        fig.update_layout(
            title='Model Error Comparison (RMSE)',
            xaxis_title='Model',
            yaxis_title='RMSE (lower is better)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # R² score visualization
        fig2 = px.bar(
            x=model_names, 
            y=r2_scores,
            title='R² Score Comparison',
            labels={'x': 'Model', 'y': 'R² Score (higher is better)'},
            color=r2_scores,
            color_continuous_scale='Viridis'
        )
        
        fig2.update_layout(yaxis_range=[0, 1])
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Training time visualization
        fig3 = px.bar(
            x=model_names, 
            y=training_times,
            title='Training Time Comparison',
            labels={'x': 'Model', 'y': 'Training Time (seconds)'},
            color=training_times,
            color_continuous_scale='Plasma'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        #### Key Insights:
        - **Gradient Boosting** typically provides the lowest error but has longer training times
        - **Linear Regression** is fastest to train but often has higher error for complex data
        - **Random Forest** and **Neural Networks** offer good balance of accuracy and complexity
        - **SVR** with the right kernel can perform well on specific types of data
        
        The optimal choice depends on your specific requirements for accuracy, training speed, and interpretability.
        """)
        
    with tabs[4]:
        st.subheader("Advanced Regression Techniques")
        
        st.markdown("""
        #### Advanced Models and Approaches
        
        Beyond standard regression models, these advanced techniques can further improve performance:
        
        1. **Ensemble Methods**:
           - **Stacking**: Combine predictions from multiple models using a meta-model
           - **Blending**: Similar to stacking but uses a hold-out set for training the meta-model
           
        2. **Time Series Specific Models**:
           - **ARIMA**: For univariate time series forecasting
           - **Prophet**: Facebook's tool for forecasting time series data
           - **LSTM/GRU**: Neural network architectures for sequence data
           
        3. **Gaussian Processes**:
           - Probabilistic approach that provides uncertainty estimates
           - Particularly useful when training data is limited
           
        4. **Quantile Regression**:
           - Predicts a range or interval rather than a single point
           - Useful for modeling uncertainty in weather predictions
        """)
        
        # Example visualization of ensemble method
        st.subheader("Example: Stacking Ensemble")
        
        # Create sample data
        methods = ['Single Best Model', 'Average Ensemble', 'Weighted Ensemble', 'Stacking']
        rmse = [2.35, 2.21, 2.15, 1.98]
        
        fig = px.bar(
            x=methods,
            y=rmse,
            title="Ensemble Method Performance",
            labels={'x': 'Method', 'y': 'RMSE (lower is better)'},
            color=rmse,
            color_continuous_scale='RdBu_r'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction intervals
        st.subheader("Example: Prediction Intervals")
        
        # Create sample data for prediction intervals
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 5 + 0.5 * x + np.random.normal(0, 1, 50)
        
        # Create prediction lines
        x_pred = np.linspace(0, 10, 100)
        y_pred = 5 + 0.5 * x_pred
        y_lower = y_pred - 1.96
        y_upper = y_pred + 1.96
        
        # Create figure
        fig = go.Figure()
        
        # Add scattered points
        fig.add_trace(go.Scatter(
            x=x, 
            y=y, 
            mode='markers', 
            name='Observations',
            marker=dict(color='blue', size=8)
        ))
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=x_pred, 
            y=y_pred, 
            mode='lines', 
            name='Prediction',
            line=dict(color='red', width=2)
        ))
        
        # Add prediction intervals
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_pred, x_pred[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(231,107,243,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Prediction Interval'
        ))
        
        fig.update_layout(
            title='Regression with Prediction Intervals',
            xaxis_title='X',
            yaxis_title='Y',
            legend=dict(x=0, y=1, traceorder='normal')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show sample predictions vs actual plot
        st.subheader("Sample Model Evaluation")
        
        # Create synthetic predictions and actual values
        np.random.seed(42)
        n_samples = 100
        actual = np.random.normal(20, 5, n_samples)
        predictions = actual + np.random.normal(0, 2, n_samples)
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'Actual': actual,
            'Predicted': predictions
        })
        
        # Create scatter plot
        fig = px.scatter(
            results_df,
            x='Actual',
            y='Predicted',
            title="Predictions vs Actual Values",
            labels={'Actual': "Actual Temperature", 'Predicted': "Predicted Temperature"}
        )
        
        # Add 45-degree reference line
        min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
        max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display metrics
        rmse = np.sqrt(np.mean((results_df['Actual'] - results_df['Predicted'])**2))
        mae = np.mean(np.abs(results_df['Actual'] - results_df['Predicted']))
        r2 = 1 - (np.sum((results_df['Actual'] - results_df['Predicted'])**2) / 
                 np.sum((results_df['Actual'] - results_df['Actual'].mean())**2))
        
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²'],
            'Value': [rmse, mae, r2]
        })
        st.dataframe(metrics_df)

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
    
    # Always load and process data on startup
    load_and_process_data()
    
    # Always engineer features on startup
    if st.session_state.data_processed:
        engineer_features()
    
    # Render navigation
    navigation()
    
    # Render the current page
    if st.session_state.current_page == "home":
        render_home_page()
    
    elif st.session_state.current_page == "data_exploration":
        render_data_exploration()
    
    elif st.session_state.current_page == "feature_engineering":
        render_feature_engineering()
    
    elif st.session_state.current_page == "classification_models":
        render_classification_models()
    
    elif st.session_state.current_page == "regression_models":
        render_regression_models()

# Run the app
if __name__ == "__main__":
    main()