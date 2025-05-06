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
        df = pd.read_csv("attached_assets/weather_prediction_dataset.csv")
        return df
    except Exception as e:
        logging.error(f"Error loading weather data: {e}")
        return None

def load_temperature_data():
    """Load temperature data from CSV."""
    try:
        df = pd.read_csv("attached_assets/BIG DATA GENERATED DATASET USED FOR ML.csv")
        return df
    except Exception as e:
        logging.error(f"Error loading temperature data: {e}")
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
    if not st.session_state.data_loaded:
        with st.spinner("Loading datasets..."):
            # Load weather and temperature data
            weather_df = load_weather_data()
            temp_df = load_temperature_data()
            
            if weather_df is not None and temp_df is not None:
                st.session_state.data_loaded = True
                st.session_state.weather_df = weather_df
                st.session_state.temp_df = temp_df
                st.success("Data loaded successfully!")
                
                logger.info("Data loaded")
            else:
                st.error("Failed to load data. Check logs for details.")
    
    if st.session_state.data_loaded and not st.session_state.data_processed:
        with st.spinner("Processing data..."):
            # Clean data
            cleaned_weather = clean_dataset(st.session_state.weather_df, "weather")
            cleaned_temp = clean_dataset(st.session_state.temp_df, "temperature")
            
            if cleaned_weather is not None and cleaned_temp is not None:
                st.session_state.cleaned_weather = cleaned_weather
                st.session_state.cleaned_temp = cleaned_temp
                st.session_state.data_processed = True
                st.session_state.data_cleaned = True
                st.success("Data processed successfully!")
                
                logger.info("Data processed")
            else:
                st.error("Failed to process data. Check logs for details.")

def engineer_features():
    """Engineer features if data is loaded and processed."""
    if st.session_state.data_loaded and st.session_state.data_processed and not st.session_state.features_engineered:
        with st.spinner("Engineering features..."):
            # Engineer features
            weather_features = engineer_weather_features(st.session_state.cleaned_weather)
            temp_features = engineer_temperature_features(st.session_state.cleaned_temp)
            
            if weather_features is not None and temp_features is not None:
                st.session_state.weather_features = weather_features
                st.session_state.temp_features = temp_features
                st.session_state.features_engineered = True
                st.success("Features engineered successfully!")
                
                logger.info("Features engineered")
            else:
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
    
    This simplified version demonstrates how classification modeling would work.
    In the full PySpark version, we would implement:
    
    - Logistic Regression
    - Random Forest
    - Naive Bayes
    - Gradient Boosted Trees
    
    For demonstration, we'll show how to set up a classification problem.
    """)
    
    # Create tabs for different aspects
    tabs = st.tabs(["Problem Definition", "Feature Selection", "Model Training"])
    
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
        st.subheader("Model Training Process")
        
        st.markdown("""
        #### Classification Model Training Steps
        
        1. **Data Preparation**:
           - Split data into training (80%) and testing (20%) sets
           - Encode categorical variables
           - Scale numerical features
        
        2. **Model Selection and Training**:
           - Train different classification models
           - Tune hyperparameters using cross-validation
        
        3. **Model Evaluation**:
           - Accuracy, Precision, Recall, F1 Score
           - Confusion Matrix
           - ROC Curve (for binary classification)
        
        4. **Feature Importance Analysis**:
           - Understand which features contribute most to predictions
        """)
        
        # Show a sample confusion matrix visualization
        st.subheader("Sample Model Evaluation Visualization")
        
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
    
    This simplified version demonstrates how regression modeling would work.
    In the full PySpark version, we would implement:
    
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosted Trees Regressor
    
    For demonstration, we'll show how to set up a regression problem.
    """)
    
    # Create tabs for different aspects
    tabs = st.tabs(["Problem Definition", "Feature Selection", "Model Training"])
    
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
        st.subheader("Model Training Process")
        
        st.markdown("""
        #### Regression Model Training Steps
        
        1. **Data Preparation**:
           - Split data into training (80%) and testing (20%) sets
           - Scale numerical features
           - Handle categorical variables
        
        2. **Model Selection and Training**:
           - Train different regression models
           - Tune hyperparameters using cross-validation
        
        3. **Model Evaluation**:
           - RMSE (Root Mean Squared Error)
           - MAE (Mean Absolute Error)
           - R² (Coefficient of Determination)
        
        4. **Residual Analysis**:
           - Check for patterns in residuals
           - Validate model assumptions
        """)
        
        # Show sample predictions vs actual plot
        st.subheader("Sample Model Evaluation Visualization")
        
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
    
    # Render navigation
    navigation()
    
    # Load and process data if needed
    if not st.session_state.data_processed:
        load_and_process_data()
    
    # Engineer features if needed
    if st.session_state.data_processed and not st.session_state.features_engineered:
        engineer_features()
    
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