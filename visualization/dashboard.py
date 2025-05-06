"""
Dashboard modules for rendering different pages in the W.A.R.P application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from datetime import datetime

# Import visualization utilities
from visualization.plot_utils import (
    plot_temperature_over_time,
    plot_temperature_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_weather_comparison,
    plot_predictions_vs_actual,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_residuals,
    plot_cluster_visualization,
    plot_time_series,
    plot_seasonal_decomposition,
    plot_weather_map,
    plot_association_rules
)

# Import data processing modules
from modules.eda import calculate_basic_stats, calculate_correlations, station_comparison, yearly_temperature_trends
from modules.data_cleaning import clean_dataset, standardize_features, detect_anomalies
from modules.feature_engineering import engineer_weather_features, engineer_temperature_features
from modules.feature_selection import select_features_chi_square, select_features_univariate, perform_pca
from modules.classification_models import (
    prepare_classification_data, 
    train_logistic_regression,
    train_random_forest,
    train_naive_bayes,
    train_gbt_classifier,
    compare_classification_models
)
from modules.regression_models import (
    prepare_regression_data,
    train_linear_regression,
    train_decision_tree_regressor,
    train_random_forest_regressor,
    train_gbt_regressor,
    compare_regression_models
)
from modules.clustering_models import (
    prepare_clustering_data,
    train_kmeans,
    train_bisecting_kmeans,
    train_gaussian_mixture,
    find_optimal_k
)
from modules.pattern_mining import mine_weather_patterns, mine_temperature_patterns
from modules.model_evaluation import (
    evaluate_binary_classifier,
    evaluate_multiclass_classifier,
    evaluate_regressor,
    evaluate_clustering,
    compare_models
)
from utils.spark_utils import read_from_delta
import config

# Render home page
def render_home_page():
    """Render the home page of the W.A.R.P dashboard."""
    st.title("W.A.R.P: Weather Analytics & Research Pipeline")
    
    st.markdown("""
    ### Welcome to the W.A.R.P Platform
    
    W.A.R.P (Weather Analytics & Research Pipeline) is a comprehensive platform for analyzing and predicting weather patterns 
    using PySpark and MLlib technologies. This platform provides tools for data exploration, feature engineering, 
    model training, and visualization of weather data.
    
    #### Key Features:
    - Data exploration and visualization of weather patterns
    - Feature engineering and selection for weather prediction
    - Multiple machine learning models for classification, regression, and clustering
    - Time series analysis and forecasting
    - Pattern mining and association rule discovery
    
    #### Getting Started:
    1. Use the sidebar to navigate between different sections
    2. Start with Data Exploration to understand the datasets
    3. Move to Feature Engineering to prepare data for modeling
    4. Train and evaluate different models in the Model sections
    
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
        if st.session_state.delta_tables_prepared:
            st.success("✅ Delta Tables Ready")
        else:
            st.error("❌ Delta Tables Not Ready")
            
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

# Render data exploration page
def render_data_exploration(spark):
    """Render the data exploration page."""
    st.title("Data Exploration")
    
    if not st.session_state.data_loaded:
        st.warning("Data not loaded yet. Please wait while we load the data.")
        return
    
    # Create tabs for different exploration options
    tabs = st.tabs(["Basic Statistics", "Correlations", "Station Comparison", "Temperature Trends", "Raw Data"])
    
    with tabs[0]:
        st.subheader("Basic Statistics")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather", "temperature"],
            key="basic_stats_dataset"
        )
        
        if st.button("Calculate Basic Statistics"):
            with st.spinner("Calculating basic statistics..."):
                stats = calculate_basic_stats(spark, dataset)
                
                if stats:
                    st.subheader(f"Basic Statistics for {dataset.capitalize()} Dataset")
                    
                    # Convert to DataFrame and display
                    stats_df = pd.DataFrame(stats)
                    st.dataframe(stats_df)
                else:
                    st.error("Failed to calculate statistics. Check logs for details.")
    
    with tabs[1]:
        st.subheader("Feature Correlations")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather", "temperature"],
            key="correlation_dataset"
        )
        
        limit = st.slider("Number of Correlations", min_value=5, max_value=50, value=15)
        
        if st.button("Calculate Correlations"):
            with st.spinner("Calculating correlations..."):
                corr_df = calculate_correlations(spark, dataset, limit=limit)
                
                if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                    st.subheader(f"Top {limit} Correlations for {dataset.capitalize()} Dataset")
                    
                    # Display correlation table
                    st.dataframe(corr_df)
                    
                    # Plot correlation heatmap
                    if len(corr_df) >= 2:
                        pivot_df = corr_df.pivot(index="feature1", columns="feature2", values="correlation")
                        fig = plot_correlation_heatmap(pivot_df, f"Correlation Heatmap - {dataset.capitalize()} Dataset")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to calculate correlations. Check logs for details.")
    
    with tabs[2]:
        st.subheader("Weather Station Comparison")
        
        metrics = st.multiselect(
            "Select Metrics to Compare",
            ["temperature", "humidity", "precipitation", "pressure", "wind_speed"],
            default=["temperature"]
        )
        
        if st.button("Compare Stations"):
            with st.spinner("Comparing stations..."):
                comparison = station_comparison(spark, metrics=metrics)
                
                if comparison:
                    st.subheader("Station Comparison Results")
                    
                    # Loop through metrics and create visualizations
                    for metric in metrics:
                        if metric in comparison:
                            fig = plot_weather_comparison(comparison, metric)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to compare stations. Check logs for details.")
    
    with tabs[3]:
        st.subheader("Yearly Temperature Trends")
        
        cities = st.multiselect(
            "Select Cities",
            ["Paris", "London", "Berlin", "Rome", "Madrid", "Amsterdam", "Vienna", "Warsaw", "Budapest", "Brussels"],
            default=["Paris", "London", "Berlin"]
        )
        
        start_year = st.number_input("Start Year", min_value=1800, max_value=2020, value=1900)
        end_year = st.number_input("End Year", min_value=1800, max_value=2020, value=2020)
        
        if st.button("Analyze Trends"):
            with st.spinner("Analyzing temperature trends..."):
                trends_df = yearly_temperature_trends(spark, cities, start_year, end_year)
                
                if isinstance(trends_df, pd.DataFrame) and not trends_df.empty:
                    st.subheader("Temperature Trends Analysis")
                    
                    # Create visualization
                    fig = plot_temperature_over_time(trends_df, start_year=start_year, end_year=end_year)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show trends data
                    st.dataframe(trends_df)
                else:
                    st.error("Failed to analyze trends. Check logs for details.")
    
    with tabs[4]:
        st.subheader("Raw Data Viewer")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather", "temperature", "weather_features", "temperature_features"],
            key="raw_data_dataset"
        )
        
        num_rows = st.slider("Number of Rows", min_value=5, max_value=100, value=20)
        
        if st.button("View Raw Data"):
            with st.spinner("Loading raw data..."):
                # Read from Delta Lake
                path = os.path.join(config.BRONZE_PATH if dataset in ["weather", "temperature"] else config.GOLD_PATH, dataset)
                df = read_from_delta(spark, path)
                
                if df is not None:
                    # Convert to Pandas for display
                    pdf = df.limit(num_rows).toPandas()
                    
                    st.subheader(f"Raw Data - {dataset.capitalize()} Dataset")
                    st.dataframe(pdf)
                else:
                    st.error(f"Failed to read {dataset} dataset. Check logs for details.")

# Render feature engineering page
def render_feature_engineering(spark):
    """Render the feature engineering page."""
    st.title("Feature Engineering & Selection")
    
    if not st.session_state.data_loaded:
        st.warning("Data not loaded yet. Please load data first.")
        return
    
    # Create tabs for different feature engineering options
    tabs = st.tabs(["Feature Engineering", "Feature Selection", "Dimensionality Reduction", "Data Cleaning"])
    
    with tabs[0]:
        st.subheader("Feature Engineering")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather", "temperature"],
            key="feature_engineering_dataset"
        )
        
        if st.button("Engineer Features"):
            with st.spinner("Engineering features..."):
                if dataset == "weather":
                    result = engineer_weather_features(spark)
                else:
                    result = engineer_temperature_features(spark)
                
                if result:
                    st.success(f"Successfully engineered features for {dataset} dataset!")
                    
                    # Show sample of engineered features
                    path = os.path.join(config.GOLD_PATH, f"{dataset}_features")
                    df = read_from_delta(spark, path)
                    
                    if df is not None:
                        st.subheader("Sample of Engineered Features")
                        st.dataframe(df.limit(10).toPandas())
                        
                        st.session_state.features_engineered = True
                else:
                    st.error(f"Failed to engineer features for {dataset} dataset. Check logs for details.")
    
    with tabs[1]:
        st.subheader("Feature Selection")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather_features", "temperature_features"],
            key="feature_selection_dataset"
        )
        
        method = st.selectbox(
            "Selection Method",
            ["Chi-Square", "Univariate", "Model-Based"],
            key="feature_selection_method"
        )
        
        num_features = st.slider("Number of Features to Select", min_value=3, max_value=20, value=10)
        
        if st.button("Select Features"):
            with st.spinner("Selecting features..."):
                # Read dataset to get columns
                path = os.path.join(config.GOLD_PATH, dataset)
                df = read_from_delta(spark, path)
                
                if df is not None:
                    # Get numeric feature columns
                    num_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['double', 'int', 'float']]
                    num_cols = [c for c in num_cols if not c.endswith('_index')]
                    
                    # Select target column based on dataset
                    if dataset == "weather_features":
                        label_col = "next_day_temp"
                    else:
                        label_col = "AvgYearlyTemp"
                    
                    # Remove label from features
                    feature_cols = [c for c in num_cols if c != label_col]
                    
                    if method == "Chi-Square":
                        selected_features, _ = select_features_chi_square(spark, dataset, feature_cols, label_col, num_features)
                    elif method == "Univariate":
                        selected_features, _ = select_features_univariate(spark, dataset, feature_cols, label_col, 
                                                                          feature_type="continuous", num_top_features=num_features)
                    else:  # Model-based
                        model_type = "regression" if label_col == "AvgYearlyTemp" else "classification"
                        selected_features = select_features_with_model(spark, dataset, feature_cols, label_col, 
                                                                      model_type=model_type, num_top_features=num_features)
                    
                    if selected_features:
                        st.success(f"Successfully selected {len(selected_features)} features!")
                        
                        # Display selected features
                        st.subheader("Selected Features")
                        
                        if isinstance(selected_features, list):
                            # For chi-square and univariate
                            features_df = pd.DataFrame({"Feature": selected_features})
                            st.dataframe(features_df)
                        else:
                            # For model-based (returns feature importance pairs)
                            st.dataframe(pd.DataFrame(selected_features, columns=["Feature", "Importance"]))
                            
                            # Plot feature importance
                            fig = plot_feature_importance(pd.DataFrame(selected_features, columns=["Feature", "Importance"]))
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to select features. Check logs for details.")
                else:
                    st.error(f"Failed to read {dataset} dataset. Check logs for details.")
    
    with tabs[2]:
        st.subheader("Dimensionality Reduction (PCA)")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather_features", "temperature_features"],
            key="pca_dataset"
        )
        
        num_components = st.slider("Number of Principal Components", min_value=2, max_value=15, value=5)
        
        if st.button("Perform PCA"):
            with st.spinner("Performing PCA..."):
                # Read dataset to get columns
                path = os.path.join(config.GOLD_PATH, dataset)
                df = read_from_delta(spark, path)
                
                if df is not None:
                    # Get numeric feature columns
                    num_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['double', 'int', 'float']]
                    num_cols = [c for c in num_cols if not c.endswith('_index')]
                    
                    # Remove target column if present
                    if dataset == "weather_features":
                        feature_cols = [c for c in num_cols if c != "next_day_temp"]
                    else:
                        feature_cols = [c for c in num_cols if c != "AvgYearlyTemp"]
                    
                    pca_df, pca_model, explained_variance = perform_pca(spark, dataset, feature_cols, num_components)
                    
                    if pca_df is not None and explained_variance is not None:
                        st.success(f"Successfully performed PCA with {num_components} components!")
                        
                        # Display explained variance
                        st.subheader("Explained Variance")
                        
                        # Create DataFrame for variance plot
                        variance_df = pd.DataFrame({
                            "Component": [f"PC{i+1}" for i in range(len(explained_variance))],
                            "Explained Variance": explained_variance,
                            "Cumulative Variance": np.cumsum(explained_variance)
                        })
                        
                        st.dataframe(variance_df)
                        
                        # Plot explained variance
                        fig = px.bar(
                            variance_df, 
                            x="Component", 
                            y="Explained Variance",
                            title="PCA Explained Variance by Component"
                        )
                        
                        # Add cumulative variance line
                        fig.add_trace(
                            go.Scatter(
                                x=variance_df["Component"],
                                y=variance_df["Cumulative Variance"],
                                mode="lines+markers",
                                name="Cumulative Variance",
                                yaxis="y2"
                            )
                        )
                        
                        # Update layout for dual y-axis
                        fig.update_layout(
                            yaxis2=dict(
                                title="Cumulative Variance",
                                overlaying="y",
                                side="right"
                            ),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show sample of PCA result
                        st.subheader("Sample PCA Result")
                        st.dataframe(pca_df.select("*").limit(10).toPandas())
                    else:
                        st.error("Failed to perform PCA. Check logs for details.")
                else:
                    st.error(f"Failed to read {dataset} dataset. Check logs for details.")
    
    with tabs[3]:
        st.subheader("Data Cleaning")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather", "temperature"],
            key="data_cleaning_dataset"
        )
        
        impute_method = st.selectbox(
            "Imputation Method", 
            ["mean", "median", "mode"],
            key="impute_method"
        )
        
        outlier_handling = st.selectbox(
            "Outlier Handling", 
            ["remove", "impute", "cap"],
            key="outlier_handling"
        )
        
        z_threshold = st.slider("Z-score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                result = clean_dataset(spark, dataset, impute_method, outlier_handling, z_threshold)
                
                if result is not None:
                    st.success(f"Successfully cleaned {dataset} dataset!")
                    
                    # Show sample of cleaned data
                    st.subheader("Sample of Cleaned Data")
                    st.dataframe(result.limit(10).toPandas())
                    
                    st.session_state.data_cleaned = True
                else:
                    st.error(f"Failed to clean {dataset} dataset. Check logs for details.")

# Render classification models page
def render_classification_models(spark):
    """Render the classification models page."""
    st.title("Classification Models")
    
    if not st.session_state.features_engineered:
        st.warning("Features not engineered yet. Please engineer features first.")
        return
    
    # Create tabs for different model options
    tabs = st.tabs(["Data Preparation", "Model Training", "Model Comparison", "Model Evaluation"])
    
    with tabs[0]:
        st.subheader("Data Preparation")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather_features", "temperature_features"],
            key="classification_dataset"
        )
        
        # Read dataset to get columns
        path = os.path.join(config.GOLD_PATH, dataset)
        df = read_from_delta(spark, path)
        
        if df is not None:
            # Get feature columns
            all_cols = df.columns
            
            # Select appropriate target variable
            if dataset == "weather_features":
                potential_targets = ["precipitation_category", "temp_category", "weather_condition"]
            else:
                potential_targets = ["temp_regime", "trend", "continent"]
            
            # Filter to actually available columns
            available_targets = [t for t in potential_targets if t in all_cols]
            
            if not available_targets:
                available_targets = ["(No categorical targets found)"]
            
            label_col = st.selectbox("Target Variable", available_targets)
            
            # Filter features based on selected target
            if label_col != "(No categorical targets found)":
                # Get numeric feature columns
                num_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['double', 'int', 'float']]
                num_cols = [c for c in num_cols if c != label_col and not c.endswith('_index')]
                
                # Let user select features
                selected_features = st.multiselect("Select Features", num_cols, default=num_cols[:5])
                
                train_ratio = st.slider("Training Data Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
                
                if st.button("Prepare Data"):
                    if selected_features and label_col != "(No categorical targets found)":
                        with st.spinner("Preparing classification data..."):
                            prepared_data = prepare_classification_data(spark, dataset, selected_features, label_col, train_ratio)
                            
                            if prepared_data:
                                st.success("Data prepared successfully!")
                                
                                # Store in session state for model training
                                st.session_state.classification_data = prepared_data
                                st.session_state.classification_features = selected_features
                                st.session_state.classification_target = label_col
                                
                                # Show data summary
                                train_count = prepared_data["train_df"].count()
                                test_count = prepared_data["test_df"].count()
                                
                                st.subheader("Data Summary")
                                st.write(f"Training set: {train_count} records")
                                st.write(f"Testing set: {test_count} records")
                                st.write(f"Features: {len(selected_features)} selected")
                                
                                # Show class distribution if available
                                if label_col != "(No categorical targets found)":
                                    class_counts = prepared_data["train_df"].groupBy(label_col).count().orderBy(label_col)
                                    class_df = class_counts.toPandas()
                                    
                                    st.subheader("Class Distribution")
                                    
                                    # Create bar chart
                                    fig = px.bar(
                                        class_df,
                                        x=label_col,
                                        y="count",
                                        title=f"Distribution of {label_col} in Training Data",
                                        labels={label_col: "Class", "count": "Count"}
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Failed to prepare data. Check logs for details.")
                    else:
                        st.error("Please select features and a valid target variable.")
            else:
                st.error("No suitable categorical target variables found in the dataset.")
        else:
            st.error(f"Failed to read {dataset} dataset. Check logs for details.")
    
    with tabs[1]:
        st.subheader("Model Training")
        
        if "classification_data" not in st.session_state:
            st.warning("Please prepare data first in the Data Preparation tab.")
            return
        
        model_type = st.selectbox(
            "Model Type",
            ["Logistic Regression", "Random Forest", "Naive Bayes", "Gradient Boosted Trees", "One-vs-Rest"],
            key="classification_model_type"
        )
        
        # Get whether it's a multiclass problem
        if "classification_target" in st.session_state:
            label_col = st.session_state.classification_target
            prepared_data = st.session_state.classification_data
            
            # Count distinct classes
            class_count = prepared_data["train_df"].select(label_col).distinct().count()
            is_multiclass = class_count > 2
            
            # Show model-specific parameters
            if model_type == "Random Forest":
                num_trees = st.slider("Number of Trees", min_value=10, max_value=100, value=20, step=5)
            
            if st.button("Train Model"):
                with st.spinner(f"Training {model_type} model..."):
                    if model_type == "Logistic Regression":
                        model_result = train_logistic_regression(prepared_data, multiclass=is_multiclass)
                    elif model_type == "Random Forest":
                        model_result = train_random_forest(prepared_data, num_trees=num_trees)
                    elif model_type == "Naive Bayes":
                        model_result = train_naive_bayes(prepared_data)
                    elif model_type == "Gradient Boosted Trees":
                        model_result = train_gbt_classifier(prepared_data)
                    else:  # One-vs-Rest
                        model_result = train_one_vs_rest(prepared_data)
                    
                    if model_result:
                        st.success(f"{model_type} model trained successfully!")
                        
                        # Store model result in session state
                        model_key = model_type.lower().replace(" ", "_")
                        if "models_trained" not in st.session_state:
                            st.session_state.models_trained = {}
                        
                        st.session_state.models_trained[model_key] = model_result
                        
                        # Show metrics
                        metrics = model_result["metrics"]
                        
                        st.subheader("Model Performance Metrics")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)
                        
                        # Show feature importance if available
                        if model_type in ["Logistic Regression", "Random Forest", "Gradient Boosted Trees"]:
                            st.subheader("Feature Importance")
                            
                            model = model_result["model"]
                            feature_cols = st.session_state.classification_features
                            
                            if model_type == "Logistic Regression":
                                # Get coefficients
                                coefficients = model.coefficients.toArray()
                                importance_df = pd.DataFrame({
                                    "Feature": feature_cols,
                                    "Importance": [abs(c) for c in coefficients]
                                })
                            else:
                                # For tree-based models
                                importances = model.featureImportances.toArray()
                                importance_df = pd.DataFrame({
                                    "Feature": feature_cols,
                                    "Importance": importances
                                })
                            
                            # Sort by importance
                            importance_df = importance_df.sort_values("Importance", ascending=False)
                            
                            # Show top 10 features
                            st.dataframe(importance_df.head(10))
                            
                            # Plot feature importance
                            fig = plot_feature_importance(importance_df, f"Feature Importance - {model_type}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Failed to train {model_type} model. Check logs for details.")
    
    with tabs[2]:
        st.subheader("Model Comparison")
        
        if "classification_data" not in st.session_state:
            st.warning("Please prepare data first in the Data Preparation tab.")
            return
        
        if st.button("Compare Models"):
            with st.spinner("Comparing classification models..."):
                prepared_data = st.session_state.classification_data
                comparison_results = compare_classification_models(prepared_data)
                
                if comparison_results:
                    st.success("Model comparison completed!")
                    
                    # Convert to DataFrame for display
                    models = list(comparison_results.keys())
                    metrics = list(comparison_results[models[0]].keys())
                    
                    comparison_data = []
                    for model in models:
                        row = {"Model": model}
                        for metric in metrics:
                            row[metric] = comparison_results[model][metric]
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.subheader("Model Comparison")
                    st.dataframe(comparison_df)
                    
                    # Plot comparison by accuracy
                    if "accuracy" in metrics:
                        fig = plot_model_comparison(comparison_df, "accuracy", higher_is_better=True)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot comparison by F1 score
                    if "f1" in metrics:
                        fig = plot_model_comparison(comparison_df, "f1", higher_is_better=True)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to compare models. Check logs for details.")
    
    with tabs[3]:
        st.subheader("Model Evaluation")
        
        if "models_trained" not in st.session_state or not st.session_state.models_trained:
            st.warning("Please train at least one model first.")
            return
        
        # Get available models
        available_models = list(st.session_state.models_trained.keys())
        
        if available_models:
            selected_model = st.selectbox("Select Model to Evaluate", available_models)
            
            if st.button("Evaluate Model"):
                with st.spinner("Evaluating model..."):
                    model_result = st.session_state.models_trained[selected_model]
                    predictions = model_result["predictions"]
                    label_col = st.session_state.classification_target
                    
                    # Check if binary or multiclass
                    class_count = predictions.select(label_col).distinct().count()
                    is_binary = class_count == 2
                    
                    if is_binary:
                        metrics = evaluate_binary_classifier(predictions, label_col)
                    else:
                        metrics = evaluate_multiclass_classifier(predictions, label_col)
                    
                    if metrics:
                        st.success("Model evaluation completed!")
                        
                        # Display metrics
                        st.subheader("Evaluation Metrics")
                        
                        # Filter out confusion matrix for display
                        display_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix" and k != "class_metrics"}
                        metrics_df = pd.DataFrame([display_metrics])
                        st.dataframe(metrics_df)
                        
                        # Display confusion matrix
                        st.subheader("Confusion Matrix")
                        confusion_matrix = metrics.get("confusion_matrix")
                        
                        if confusion_matrix:
                            cm_fig = plot_confusion_matrix(confusion_matrix, f"Confusion Matrix - {selected_model}")
                            st.plotly_chart(cm_fig, use_container_width=True)
                        
                        # Display class metrics for multiclass
                        if not is_binary and "class_metrics" in metrics:
                            st.subheader("Per-Class Metrics")
                            
                            class_metrics = metrics["class_metrics"]
                            class_data = []
                            
                            for class_name, class_metrics in class_metrics.items():
                                row = {"Class": class_name}
                                row.update(class_metrics)
                                class_data.append(row)
                            
                            class_df = pd.DataFrame(class_data)
                            st.dataframe(class_df)
                        
                        # Show sample predictions
                        st.subheader("Sample Predictions")
                        pred_df = predictions.select(label_col, "prediction", "probability").limit(10).toPandas()
                        st.dataframe(pred_df)
                    else:
                        st.error("Failed to evaluate model. Check logs for details.")

# Render regression models page
def render_regression_models(spark):
    """Render the regression models page."""
    st.title("Regression Models")
    
    if not st.session_state.features_engineered:
        st.warning("Features not engineered yet. Please engineer features first.")
        return
    
    # Create tabs for different model options
    tabs = st.tabs(["Data Preparation", "Model Training", "Model Comparison", "Model Evaluation"])
    
    with tabs[0]:
        st.subheader("Data Preparation")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather_features", "temperature_features"],
            key="regression_dataset"
        )
        
        # Read dataset to get columns
        path = os.path.join(config.GOLD_PATH, dataset)
        df = read_from_delta(spark, path)
        
        if df is not None:
            # Get feature columns
            all_cols = df.columns
            
            # Select appropriate target variable
            if dataset == "weather_features":
                potential_targets = ["next_day_temp", "next_day_humidity", "next_day_pressure"]
            else:
                potential_targets = ["AvgYearlyTemp", "TempChange", "TempVariability"]
            
            # Filter to actually available columns
            available_targets = [t for t in potential_targets if t in all_cols]
            
            if not available_targets:
                available_targets = ["(No numeric targets found)"]
            
            label_col = st.selectbox("Target Variable", available_targets)
            
            # Filter features based on selected target
            if label_col != "(No numeric targets found)":
                # Get numeric feature columns
                num_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['double', 'int', 'float']]
                num_cols = [c for c in num_cols if c != label_col and not c.endswith('_index')]
                
                # Let user select features
                selected_features = st.multiselect("Select Features", num_cols, default=num_cols[:5])
                
                train_ratio = st.slider("Training Data Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
                
                if st.button("Prepare Data"):
                    if selected_features and label_col != "(No numeric targets found)":
                        with st.spinner("Preparing regression data..."):
                            prepared_data = prepare_regression_data(spark, dataset, selected_features, label_col, train_ratio)
                            
                            if prepared_data:
                                st.success("Data prepared successfully!")
                                
                                # Store in session state for model training
                                st.session_state.regression_data = prepared_data
                                st.session_state.regression_features = selected_features
                                st.session_state.regression_target = label_col
                                
                                # Show data summary
                                train_count = prepared_data["train_df"].count()
                                test_count = prepared_data["test_df"].count()
                                
                                st.subheader("Data Summary")
                                st.write(f"Training set: {train_count} records")
                                st.write(f"Testing set: {test_count} records")
                                st.write(f"Features: {len(selected_features)} selected")
                                
                                # Show target distribution
                                target_stats = prepared_data["train_df"].select(label_col).summary().toPandas()
                                
                                st.subheader("Target Variable Statistics")
                                st.dataframe(target_stats)
                                
                                # Create histogram of target variable
                                target_values = prepared_data["train_df"].select(label_col).toPandas()
                                
                                fig = px.histogram(
                                    target_values,
                                    x=label_col,
                                    title=f"Distribution of {label_col} in Training Data",
                                    labels={label_col: label_col, "count": "Count"}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Failed to prepare data. Check logs for details.")
                    else:
                        st.error("Please select features and a valid target variable.")
            else:
                st.error("No suitable numeric target variables found in the dataset.")
        else:
            st.error(f"Failed to read {dataset} dataset. Check logs for details.")
    
    with tabs[1]:
        st.subheader("Model Training")
        
        if "regression_data" not in st.session_state:
            st.warning("Please prepare data first in the Data Preparation tab.")
            return
        
        model_type = st.selectbox(
            "Model Type",
            ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosted Trees", "Generalized Linear Model"],
            key="regression_model_type"
        )
        
        # Show model-specific parameters
        if model_type == "Random Forest":
            num_trees = st.slider("Number of Trees", min_value=10, max_value=100, value=20, step=5)
        elif model_type == "Generalized Linear Model":
            family = st.selectbox("Family", ["gaussian", "poisson", "gamma"])
            link = st.selectbox("Link Function", ["identity", "log", "inverse"])
        
        if st.button("Train Model"):
            with st.spinner(f"Training {model_type} model..."):
                prepared_data = st.session_state.regression_data
                
                if model_type == "Linear Regression":
                    model_result = train_linear_regression(prepared_data)
                elif model_type == "Decision Tree":
                    model_result = train_decision_tree_regressor(prepared_data)
                elif model_type == "Random Forest":
                    model_result = train_random_forest_regressor(prepared_data, num_trees=num_trees)
                elif model_type == "Gradient Boosted Trees":
                    model_result = train_gbt_regressor(prepared_data)
                else:  # Generalized Linear Model
                    model_result = train_generalized_linear_regression(prepared_data, family=family, link=link)
                
                if model_result:
                    st.success(f"{model_type} model trained successfully!")
                    
                    # Store model result in session state
                    model_key = model_type.lower().replace(" ", "_")
                    if "regression_models_trained" not in st.session_state:
                        st.session_state.regression_models_trained = {}
                    
                    st.session_state.regression_models_trained[model_key] = model_result
                    
                    # Show metrics
                    metrics = model_result["metrics"]
                    
                    st.subheader("Model Performance Metrics")
                    
                    # Filter out large arrays
                    display_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
                    metrics_df = pd.DataFrame([display_metrics])
                    st.dataframe(metrics_df)
                    
                    # Show coefficients if linear model
                    if model_type in ["Linear Regression", "Generalized Linear Model"]:
                        st.subheader("Model Coefficients")
                        
                        feature_cols = st.session_state.regression_features
                        coefficients = metrics.get("coefficients", [])
                        intercept = metrics.get("intercept", 0)
                        
                        # Create coefficients table
                        coef_data = [{"Feature": "intercept", "Coefficient": intercept}]
                        for i, feature in enumerate(feature_cols):
                            if i < len(coefficients):
                                coef_data.append({"Feature": feature, "Coefficient": coefficients[i]})
                        
                        coef_df = pd.DataFrame(coef_data)
                        st.dataframe(coef_df)
                        
                        # Plot coefficients
                        coef_plot_df = coef_df[coef_df["Feature"] != "intercept"]
                        fig = px.bar(
                            coef_plot_df,
                            x="Feature",
                            y="Coefficient",
                            title="Model Coefficients",
                            labels={"Feature": "Feature", "Coefficient": "Coefficient"}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance for tree-based models
                    if model_type in ["Decision Tree", "Random Forest", "Gradient Boosted Trees"]:
                        st.subheader("Feature Importance")
                        
                        model = model_result["model"]
                        feature_cols = st.session_state.regression_features
                        
                        # Get feature importance
                        importances = model.featureImportances.toArray()
                        importance_df = pd.DataFrame({
                            "Feature": feature_cols,
                            "Importance": importances
                        })
                        
                        # Sort by importance
                        importance_df = importance_df.sort_values("Importance", ascending=False)
                        
                        # Show top 10 features
                        st.dataframe(importance_df.head(10))
                        
                        # Plot feature importance
                        fig = plot_feature_importance(importance_df, f"Feature Importance - {model_type}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show predictions vs actual
                    st.subheader("Predictions vs Actual")
                    
                    label_col = st.session_state.regression_target
                    predictions = model_result["predictions"]
                    
                    # Convert to pandas for plotting
                    pred_df = predictions.select(label_col, "prediction").toPandas()
                    
                    # Create scatter plot
                    fig = plot_predictions_vs_actual(pred_df, label_col, "prediction", f"Predictions vs Actual - {model_type}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create residual plot
                    residuals_fig = plot_residuals(pred_df, label_col, "prediction", f"Residual Plot - {model_type}")
                    st.plotly_chart(residuals_fig, use_container_width=True)
                else:
                    st.error(f"Failed to train {model_type} model. Check logs for details.")
    
    with tabs[2]:
        st.subheader("Model Comparison")
        
        if "regression_data" not in st.session_state:
            st.warning("Please prepare data first in the Data Preparation tab.")
            return
        
        if st.button("Compare Models"):
            with st.spinner("Comparing regression models..."):
                prepared_data = st.session_state.regression_data
                comparison_results = compare_regression_models(prepared_data)
                
                if comparison_results:
                    st.success("Model comparison completed!")
                    
                    # Convert to DataFrame for display
                    models = list(comparison_results.keys())
                    
                    # Find common metrics across all models
                    common_metrics = set()
                    for model in models:
                        metrics = comparison_results[model]
                        for metric in metrics:
                            if not isinstance(metrics[metric], list) and not isinstance(metrics[metric], dict):
                                common_metrics.add(metric)
                    
                    comparison_data = []
                    for model in models:
                        row = {"Model": model}
                        for metric in common_metrics:
                            if metric in comparison_results[model]:
                                row[metric] = comparison_results[model][metric]
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.subheader("Model Comparison")
                    st.dataframe(comparison_df)
                    
                    # Plot comparison by RMSE (lower is better)
                    if "rmse" in common_metrics:
                        fig = plot_model_comparison(comparison_df, "rmse", higher_is_better=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot comparison by R-squared (higher is better)
                    if "r2" in common_metrics:
                        fig = plot_model_comparison(comparison_df, "r2", higher_is_better=True)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to compare models. Check logs for details.")
    
    with tabs[3]:
        st.subheader("Model Evaluation")
        
        if "regression_models_trained" not in st.session_state or not st.session_state.regression_models_trained:
            st.warning("Please train at least one model first.")
            return
        
        # Get available models
        available_models = list(st.session_state.regression_models_trained.keys())
        
        if available_models:
            selected_model = st.selectbox("Select Model to Evaluate", available_models)
            
            if st.button("Evaluate Model"):
                with st.spinner("Evaluating model..."):
                    model_result = st.session_state.regression_models_trained[selected_model]
                    predictions = model_result["predictions"]
                    label_col = st.session_state.regression_target
                    
                    metrics = evaluate_regressor(predictions, label_col)
                    
                    if metrics:
                        st.success("Model evaluation completed!")
                        
                        # Display metrics
                        st.subheader("Evaluation Metrics")
                        
                        # Filter out large arrays
                        display_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
                        metrics_df = pd.DataFrame([display_metrics])
                        st.dataframe(metrics_df)
                        
                        # Show predictions vs actual
                        st.subheader("Predictions vs Actual")
                        
                        # Convert to pandas for plotting
                        pred_df = predictions.select(label_col, "prediction").toPandas()
                        
                        # Create scatter plot
                        fig = plot_predictions_vs_actual(pred_df, label_col, "prediction", f"Predictions vs Actual - {selected_model}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create residual plot
                        residuals_fig = plot_residuals(pred_df, label_col, "prediction", f"Residual Plot - {selected_model}")
                        st.plotly_chart(residuals_fig, use_container_width=True)
                        
                        # Show residual histogram
                        residuals = metrics.get("residuals", [])
                        if residuals:
                            residuals_df = pd.DataFrame({"Residual": residuals})
                            
                            hist_fig = px.histogram(
                                residuals_df,
                                x="Residual",
                                title="Residual Distribution",
                                labels={"Residual": "Residual", "count": "Count"}
                            )
                            
                            st.plotly_chart(hist_fig, use_container_width=True)
                        
                        # Show sample predictions
                        st.subheader("Sample Predictions")
                        pred_sample_df = predictions.select(label_col, "prediction").limit(10).toPandas()
                        pred_sample_df["error"] = pred_sample_df[label_col] - pred_sample_df["prediction"]
                        st.dataframe(pred_sample_df)
                    else:
                        st.error("Failed to evaluate model. Check logs for details.")

# Render clustering models page
def render_clustering_models(spark):
    """Render the clustering models page."""
    st.title("Clustering Models")
    
    if not st.session_state.features_engineered:
        st.warning("Features not engineered yet. Please engineer features first.")
        return
    
    # Create tabs for different model options
    tabs = st.tabs(["Data Preparation", "Model Training", "Cluster Analysis"])
    
    with tabs[0]:
        st.subheader("Data Preparation")
        
        dataset = st.selectbox(
            "Select Dataset", 
            ["weather_features", "temperature_features"],
            key="clustering_dataset"
        )
        
        # Read dataset to get columns
        path = os.path.join(config.GOLD_PATH, dataset)
        df = read_from_delta(spark, path)
        
        if df is not None:
            # Get numeric feature columns
            num_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['double', 'int', 'float']]
            num_cols = [c for c in num_cols if not c.endswith('_index')]
            
            # Let user select features
            selected_features = st.multiselect("Select Features for Clustering", num_cols, default=num_cols[:5])
            
            use_scaling = st.checkbox("Scale Features", value=True, help="Standardize features to have mean 0 and unit variance")
            
            if st.button("Prepare Data"):
                if selected_features:
                    with st.spinner("Preparing clustering data..."):
                        prepared_data = prepare_clustering_data(spark, dataset, selected_features)
                        
                        if prepared_data:
                            st.success("Data prepared successfully!")
                            
                            # Store in session state for model training
                            st.session_state.clustering_data = prepared_data
                            st.session_state.clustering_features = selected_features
                            st.session_state.clustering_scaling = use_scaling
                            
                            # Show data summary
                            count = prepared_data["assembled_df"].count()
                            
                            st.subheader("Data Summary")
                            st.write(f"Total records: {count}")
                            st.write(f"Features: {len(selected_features)} selected")
                            st.write(f"Scaling: {'Enabled' if use_scaling else 'Disabled'}")
                            
                            # Show feature statistics
                            feature_stats = prepared_data["assembled_df"].select(selected_features).summary().toPandas()
                            
                            st.subheader("Feature Statistics")
                            st.dataframe(feature_stats)
                        else:
                            st.error("Failed to prepare data. Check logs for details.")
                else:
                    st.error("Please select features for clustering.")
        else:
            st.error(f"Failed to read {dataset} dataset. Check logs for details.")
    
    with tabs[1]:
        st.subheader("Model Training")
        
        if "clustering_data" not in st.session_state:
            st.warning("Please prepare data first in the Data Preparation tab.")
            return
        
        # Clustering method selection
        method = st.selectbox(
            "Clustering Method",
            ["K-Means", "Bisecting K-Means", "Gaussian Mixture"],
            key="clustering_method"
        )
        
        # Number of clusters selection
        k = st.slider("Number of Clusters (k)", min_value=2, max_value=20, value=3)
        
        if st.button("Find Optimal k"):
            with st.spinner("Finding optimal number of clusters..."):
                prepared_data = st.session_state.clustering_data
                use_scaled = st.session_state.clustering_scaling
                
                # Map method to internal name
                if method == "K-Means":
                    method_name = "kmeans"
                elif method == "Bisecting K-Means":
                    method_name = "bisecting_kmeans"
                else:
                    method_name = "gmm"
                
                # Find optimal k
                k_range = (2, 10)
                optimal_k_result = find_optimal_k(prepared_data, method=method_name, k_range=k_range, use_scaled=use_scaled)
                
                if optimal_k_result:
                    st.success("Optimal k analysis completed!")
                    
                    # Display results
                    optimal_k = optimal_k_result["optimal_k"]
                    best_silhouette = optimal_k_result["silhouette"]
                    
                    st.subheader("Optimal k Analysis")
                    st.write(f"Optimal number of clusters (k): {optimal_k}")
                    st.write(f"Silhouette score at k={optimal_k}: {best_silhouette:.4f}")
                    
                    # Plot silhouette scores
                    all_results = optimal_k_result["all_results"]
                    results_df = pd.DataFrame(all_results, columns=["k", "Silhouette"])
                    
                    fig = px.line(
                        results_df,
                        x="k",
                        y="Silhouette",
                        title="Silhouette Score by Number of Clusters",
                        markers=True
                    )
                    
                    # Add marker for optimal k
                    fig.add_trace(
                        go.Scatter(
                            x=[optimal_k],
                            y=[best_silhouette],
                            mode="markers",
                            marker=dict(color="red", size=12),
                            name=f"Optimal k={optimal_k}"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to find optimal k. Check logs for details.")
        
        if st.button("Train Clustering Model"):
            with st.spinner(f"Training {method} model with k={k}..."):
                prepared_data = st.session_state.clustering_data
                use_scaled = st.session_state.clustering_scaling
                
                # Train model based on method
                if method == "K-Means":
                    model_result = train_kmeans(prepared_data, k=k, use_scaled=use_scaled)
                elif method == "Bisecting K-Means":
                    model_result = train_bisecting_kmeans(prepared_data, k=k, use_scaled=use_scaled)
                else:  # Gaussian Mixture
                    model_result = train_gaussian_mixture(prepared_data, k=k, use_scaled=use_scaled)
                
                if model_result:
                    st.success(f"{method} model trained successfully!")
                    
                    # Store model result in session state
                    model_key = method.lower().replace(" ", "_").replace("-", "_")
                    if "clustering_models_trained" not in st.session_state:
                        st.session_state.clustering_models_trained = {}
                    
                    st.session_state.clustering_models_trained[model_key] = model_result
                    
                    # Show metrics
                    metrics = model_result["metrics"]
                    
                    st.subheader("Clustering Metrics")
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df)
                    
                    # Show cluster sizes
                    predictions = model_result["predictions"]
                    cluster_counts = predictions.groupBy("cluster").count().orderBy("cluster").toPandas()
                    
                    st.subheader("Cluster Sizes")
                    
                    # Create bar chart of cluster sizes
                    fig = px.bar(
                        cluster_counts,
                        x="cluster",
                        y="count",
                        title="Number of Points in Each Cluster",
                        labels={"cluster": "Cluster", "count": "Count"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show 2D visualization with first two features
                    if len(st.session_state.clustering_features) >= 2:
                        feature1 = st.session_state.clustering_features[0]
                        feature2 = st.session_state.clustering_features[1]
                        
                        st.subheader("Cluster Visualization")
                        
                        # Get visualization data
                        vis_data = predictions.select(feature1, feature2, "cluster").toPandas()
                        
                        # Create scatter plot
                        cluster_fig = plot_cluster_visualization(vis_data, feature1, feature2, f"Cluster Visualization - {method}")
                        st.plotly_chart(cluster_fig, use_container_width=True)
                    
                    # Show cluster centers for K-Means
                    if method == "K-Means" and "centers" in model_result:
                        st.subheader("Cluster Centers")
                        
                        centers = model_result["centers"]
                        features = st.session_state.clustering_features
                        
                        # Create centers table
                        centers_data = []
                        for i, center in enumerate(centers):
                            row = {"Cluster": i}
                            for j, feature in enumerate(features):
                                if j < len(center):
                                    row[feature] = center[j]
                            centers_data.append(row)
                        
                        centers_df = pd.DataFrame(centers_data)
                        st.dataframe(centers_df)
                else:
                    st.error(f"Failed to train {method} model. Check logs for details.")
    
    with tabs[2]:
        st.subheader("Cluster Analysis")
        
        if "clustering_models_trained" not in st.session_state or not st.session_state.clustering_models_trained:
            st.warning("Please train at least one clustering model first.")
            return
        
        # Get available models
        available_models = list(st.session_state.clustering_models_trained.keys())
        
        if available_models:
            selected_model = st.selectbox("Select Model to Analyze", available_models)
            
            # Select features for analysis
            dataset = st.selectbox(
                "Analysis Dataset", 
                ["Original Features", "Additional Features"],
                key="analysis_dataset"
            )
            
            if st.button("Analyze Clusters"):
                with st.spinner("Analyzing clusters..."):
                    model_result = st.session_state.clustering_models_trained[selected_model]
                    clustering_features = st.session_state.clustering_features
                    
                    # Get original dataset
                    if "clustering_data" in st.session_state:
                        prepared_data = st.session_state.clustering_data
                        predictions = model_result["predictions"]
                        
                        if dataset == "Original Features":
                            # Analyze using original features
                            analysis_result = analyze_clusters(model_result, None, clustering_features, [])
                        else:
                            # Try to get the original dataset
                            dataset_name = st.selectbox(
                                "Select Dataset", 
                                ["weather_features", "temperature_features"],
                                key="analysis_original_dataset"
                            )
                            
                            path = os.path.join(config.GOLD_PATH, dataset_name)
                            original_df = read_from_delta(spark, path)
                            
                            if original_df is not None:
                                # Analyze using all columns
                                analysis_result = analyze_clusters(model_result, original_df, clustering_features, [])
                            else:
                                st.error(f"Failed to read {dataset_name} dataset for analysis.")
                                return
                        
                        if analysis_result:
                            st.success("Cluster analysis completed!")
                            
                            # Display cluster analysis
                            st.subheader("Cluster Characteristics")
                            
                            # Loop through clusters
                            for cluster, stats in analysis_result.items():
                                st.markdown(f"#### {cluster.replace('_', ' ').title()}")
                                st.write(f"Size: {stats['size']} points ({stats['percentage']:.2f}% of total)")
                                
                                # Show feature statistics
                                if "feature_stats" in stats:
                                    feature_stats = stats["feature_stats"]
                                    
                                    # Convert to DataFrame for display
                                    stats_data = []
                                    for feature, fstats in feature_stats.items():
                                        stats_data.append({
                                            "Feature": feature,
                                            "Mean": fstats["mean"],
                                            "StdDev": fstats["stddev"],
                                            "Min": fstats["min"],
                                            "Max": fstats["max"]
                                        })
                                    
                                    stats_df = pd.DataFrame(stats_data)
                                    st.dataframe(stats_df)
                            
                            # Create cluster comparison visualization
                            st.subheader("Cluster Comparison")
                            
                            # Extract means for selected features across clusters
                            comparison_data = []
                            for cluster, stats in analysis_result.items():
                                for feature, fstats in stats["feature_stats"].items():
                                    comparison_data.append({
                                        "Cluster": cluster,
                                        "Feature": feature,
                                        "Mean": fstats["mean"]
                                    })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Create clustered bar chart
                            if not comparison_df.empty:
                                fig = px.bar(
                                    comparison_df,
                                    x="Feature",
                                    y="Mean",
                                    color="Cluster",
                                    barmode="group",
                                    title="Mean Feature Values by Cluster"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Evaluate clustering
                            st.subheader("Clustering Evaluation")
                            
                            # Get features column name
                            features_col = "scaledFeatures" if st.session_state.clustering_scaling else "features"
                            
                            eval_metrics = evaluate_clustering(predictions, features_col)
                            
                            if eval_metrics:
                                # Display metrics
                                st.write(f"Silhouette Score: {eval_metrics['silhouette']:.4f}")
                                st.write(f"Number of Clusters: {eval_metrics['num_clusters']}")
                                
                                # Show cluster proportions
                                props = pd.DataFrame({
                                    "Cluster": list(eval_metrics["cluster_proportions"].keys()),
                                    "Proportion": list(eval_metrics["cluster_proportions"].values())
                                })
                                
                                # Create pie chart
                                fig = px.pie(
                                    props,
                                    values="Proportion",
                                    names="Cluster",
                                    title="Cluster Proportions"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to analyze clusters. Check logs for details.")
                    else:
                        st.error("Clustering data not found. Please prepare data first.")

# Render time series page
def render_time_series(spark):
    """Render the time series analysis page."""
    st.title("Time Series Analysis")
    
    if not st.session_state.features_engineered:
        st.warning("Features not engineered yet. Please engineer features first.")
        return
    
    st.subheader("Time Series Analysis")
    
    dataset = st.selectbox(
        "Select Dataset", 
        ["weather_features", "temperature_features"],
        key="time_series_dataset"
    )
    
    # Read dataset to get columns
    path = os.path.join(config.GOLD_PATH, dataset)
    df = read_from_delta(spark, path)
    
    if df is not None:
        # Get time-related columns
        if dataset == "weather_features":
            time_col = "DATE_PARSED"
            potential_metrics = ["BASEL_temp_mean", "DE_BILT_temp_mean", "BUDAPEST_temp_mean", 
                               "BASEL_humidity", "DE_BILT_humidity", "BUDAPEST_humidity",
                               "BASEL_precipitation", "DE_BILT_precipitation", "BUDAPEST_precipitation"]
        else:
            time_col = "Year"
            potential_metrics = ["AvgYearlyTemp", "TempChange", "TempVariability"]
        
        # Filter to actually available columns
        available_metrics = [m for m in potential_metrics if m in df.columns]
        
        if not available_metrics:
            st.error("No suitable time series metrics found in the dataset.")
            return
        
        selected_metric = st.selectbox("Select Metric", available_metrics)
        
        # Optional filtering
        if dataset == "temperature_features":
            # Add country/city filtering for temperature dataset
            countries = df.select("Country").distinct().rdd.flatMap(lambda x: x).collect()
            selected_country = st.selectbox("Filter by Country", ["All"] + countries)
        
        if st.button("Analyze Time Series"):
            with st.spinner("Analyzing time series..."):
                # Apply filters
                filtered_df = df
                
                if dataset == "temperature_features" and selected_country != "All":
                    filtered_df = df.filter(df.Country == selected_country)
                
                # Select relevant columns
                ts_df = filtered_df.select(time_col, selected_metric).orderBy(time_col)
                
                # Convert to pandas for visualization
                pd_df = ts_df.toPandas()
                
                if not pd_df.empty:
                    st.success("Time series analysis completed!")
                    
                    # Display time series plot
                    st.subheader("Time Series Plot")
                    
                    fig = plot_time_series(pd_df, time_col, selected_metric, f"{selected_metric} over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display rolling statistics
                    st.subheader("Rolling Statistics")
                    
                    # Calculate rolling mean and std
                    window_size = st.slider("Window Size", min_value=3, max_value=20, value=5)
                    
                    pd_df["Rolling Mean"] = pd_df[selected_metric].rolling(window=window_size).mean()
                    pd_df["Rolling Std"] = pd_df[selected_metric].rolling(window=window_size).std()
                    
                    # Create plot with rolling statistics
                    fig = go.Figure()
                    
                    # Add original series
                    fig.add_trace(
                        go.Scatter(
                            x=pd_df[time_col],
                            y=pd_df[selected_metric],
                            mode="lines",
                            name=selected_metric
                        )
                    )
                    
                    # Add rolling mean
                    fig.add_trace(
                        go.Scatter(
                            x=pd_df[time_col],
                            y=pd_df["Rolling Mean"],
                            mode="lines",
                            name=f"Rolling Mean (window={window_size})",
                            line=dict(color="red")
                        )
                    )
                    
                    # Add rolling std
                    fig.add_trace(
                        go.Scatter(
                            x=pd_df[time_col],
                            y=pd_df["Rolling Std"],
                            mode="lines",
                            name=f"Rolling Std (window={window_size})",
                            line=dict(color="green")
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Rolling Statistics for {selected_metric}",
                        xaxis_title=time_col,
                        yaxis_title="Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display seasonality analysis
                    if dataset == "weather_features":
                        st.subheader("Seasonality Analysis")
                        
                        # Group by month to show seasonality
                        if "MONTH_NUM" in filtered_df.columns:
                            monthly_df = filtered_df.groupBy("MONTH_NUM").agg(avg(col(selected_metric)).alias("avg_value"))
                            monthly_df = monthly_df.orderBy("MONTH_NUM")
                            
                            # Convert to pandas
                            monthly_pd = monthly_df.toPandas()
                            
                            # Create month labels
                            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                            monthly_pd["Month"] = monthly_pd["MONTH_NUM"].apply(lambda x: month_names[x-1])
                            
                            # Create seasonal plot
                            fig = px.line(
                                monthly_pd,
                                x="Month",
                                y="avg_value",
                                title=f"Seasonal Pattern of {selected_metric}",
                                labels={"Month": "Month", "avg_value": f"Average {selected_metric}"},
                                markers=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display autocorrelation
                    st.subheader("Autocorrelation Analysis")
                    
                    # Calculate autocorrelation
                    from pandas.plotting import autocorrelation_plot
                    import matplotlib.pyplot as plt
                    
                    # Create matplotlib figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    autocorrelation_plot(pd_df[selected_metric].dropna(), ax=ax)
                    ax.set_title(f"Autocorrelation Plot for {selected_metric}")
                    
                    st.pyplot(fig)
                else:
                    st.error("No data available for the selected filters.")
    else:
        st.error(f"Failed to read {dataset} dataset. Check logs for details.")

# Render advanced analytics page
def render_advanced_analytics(spark):
    """Render the advanced analytics page."""
    st.title("Advanced Analytics")
    
    if not st.session_state.features_engineered:
        st.warning("Features not engineered yet. Please engineer features first.")
        return
    
    # Create tabs for different advanced analytics options
    tabs = st.tabs(["Pattern Mining", "Collaborative Filtering", "Custom Query"])
    
    with tabs[0]:
        st.subheader("Pattern Mining")
        
        dataset_type = st.selectbox(
            "Select Dataset Type", 
            ["Weather", "Temperature"],
            key="pattern_dataset_type"
        )
        
        min_support = st.slider("Minimum Support", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
        min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
        
        if st.button("Mine Patterns"):
            with st.spinner("Mining patterns..."):
                if dataset_type == "Weather":
                    patterns = mine_weather_patterns(spark, min_support, min_confidence)
                else:
                    patterns = mine_temperature_patterns(spark, min_support, min_confidence)
                
                if patterns:
                    st.success(f"Successfully mined patterns in {dataset_type.lower()} data!")
                    
                    # Show frequent itemsets
                    st.subheader("Frequent Itemsets")
                    itemsets = patterns["frequent_itemsets"].toPandas()
                    
                    if not itemsets.empty:
                        st.dataframe(itemsets)
                    
                    # Show association rules
                    st.subheader("Association Rules")
                    rules = patterns["association_rules"].toPandas()
                    
                    if not rules.empty:
                        st.dataframe(rules)
                        
                        # Plot association rules
                        fig = plot_association_rules(rules, f"Association Rules for {dataset_type} Data")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display top rules in a more readable format
                        st.subheader("Top Association Rules")
                        
                        for i, row in rules.head(10).iterrows():
                            antecedent = row["antecedent"]
                            consequent = row["consequent"]
                            confidence = row["confidence"]
                            lift = row["lift"]
                            
                            antecedent_str = ", ".join([str(item) for item in antecedent])
                            consequent_str = ", ".join([str(item) for item in consequent])
                            
                            st.markdown(f"**Rule {i+1}**: {antecedent_str} → {consequent_str}")
                            st.write(f"Confidence: {confidence:.4f}, Lift: {lift:.4f}")
                    else:
                        st.info("No association rules found with the given thresholds.")
                else:
                    st.error(f"Failed to mine patterns in {dataset_type.lower()} data. Check logs for details.")
    
    with tabs[1]:
        st.subheader("Collaborative Filtering")
        
        dataset_type = st.selectbox(
            "Select Dataset Type", 
            ["Weather", "Temperature"],
            key="cf_dataset_type"
        )
        
        st.write("Collaborative filtering can be used to recommend weather stations or locations based on similar patterns.")
        
        # Get data for collaborative filtering
        if st.button("Create Recommendation Data"):
            with st.spinner("Creating recommendation data..."):
                from modules.collaborative_filtering import create_forecast_accuracy_data
                
                accuracy_data = create_forecast_accuracy_data(spark, dataset_type.lower())
                
                if accuracy_data:
                    st.success("Recommendation data created successfully!")
                    
                    # Store in session state
                    st.session_state.cf_data_created = True
                    
                    # Show sample data
                    st.subheader("Sample Recommendation Data")
                    st.dataframe(accuracy_data.limit(10).toPandas())
                else:
                    st.error("Failed to create recommendation data. Check logs for details.")
        
        # Train ALS model
        if "cf_data_created" in st.session_state and st.session_state.cf_data_created:
            from modules.collaborative_filtering import prepare_als_data, train_als_model, generate_recommendations
            
            # Set up parameters
            if dataset_type == "Weather":
                user_col = "DATE_PARSED"
                item_col = "station"
                rating_col = "accuracy"
            else:
                user_col = "Country"
                item_col = "City"
                rating_col = "accuracy"
            
            rank = st.slider("Rank", min_value=5, max_value=50, value=10, step=5)
            reg_param = st.slider("Regularization Parameter", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            
            if st.button("Train Recommendation Model"):
                with st.spinner("Training ALS model..."):
                    # Read data from Delta Lake
                    path = os.path.join(config.GOLD_PATH, "forecast_accuracy" if dataset_type == "Weather" else "temp_accuracy")
                    df = read_from_delta(spark, path)
                    
                    if df is not None:
                        # Prepare data
                        prepared_data = prepare_als_data(spark, 
                                                        "forecast_accuracy" if dataset_type == "Weather" else "temp_accuracy", 
                                                        user_col, item_col, rating_col)
                        
                        if prepared_data:
                            # Train model
                            model_result = train_als_model(prepared_data, rank=rank, reg_param=reg_param)
                            
                            if model_result:
                                st.success("ALS model trained successfully!")
                                
                                # Store in session state
                                st.session_state.als_model_result = model_result
                                
                                # Show metrics
                                metrics = model_result["metrics"]
                                
                                st.subheader("Model Metrics")
                                metrics_df = pd.DataFrame([metrics])
                                st.dataframe(metrics_df)
                                
                                # Generate recommendations
                                num_recommendations = st.slider("Number of Recommendations", min_value=3, max_value=20, value=5)
                                recommendations = generate_recommendations(model_result, num_recommendations)
                                
                                if recommendations is not None:
                                    st.subheader("Sample Recommendations")
                                    st.dataframe(recommendations.limit(20).toPandas())
                                else:
                                    st.error("Failed to generate recommendations.")
                            else:
                                st.error("Failed to train ALS model. Check logs for details.")
                        else:
                            st.error("Failed to prepare data for ALS. Check logs for details.")
                    else:
                        st.error(f"Failed to read recommendation data. Check logs for details.")
    
    with tabs[2]:
        st.subheader("Custom Query")
        
        st.write("Run custom SparkSQL queries on the available datasets.")
        
        # Show available tables
        st.subheader("Available Tables")
        
        # List available tables in Delta Lake
        tables = [
            "bronze.weather", "bronze.temperature",
            "silver.weather_clean", "silver.temperature_clean",
            "gold.weather_features", "gold.temperature_features",
            "gold.weather_patterns_itemsets", "gold.weather_patterns_rules",
            "gold.temperature_patterns_itemsets", "gold.temperature_patterns_rules"
        ]
        
        st.write("Available tables:")
        for table in tables:
            st.code(table)
        
        # Query input
        query = st.text_area("Enter SQL Query", height=150,
                            value="SELECT * FROM gold.weather_features LIMIT 10")
        
        if st.button("Run Query"):
            with st.spinner("Running query..."):
                try:
                    # Create temporary views
                    if st.session_state.data_loaded:
                        # Read datasets and create views
                        for table in tables:
                            layer, name = table.split(".")
                            path = os.path.join(f"delta_lake/{layer}", name)
                            
                            try:
                                df = read_from_delta(spark, path)
                                if df is not None:
                                    df.createOrReplaceTempView(f"{layer}_{name}")
                            except:
                                pass
                    
                    # Process query
                    query = query.replace("gold.", "gold_").replace("silver.", "silver_").replace("bronze.", "bronze_")
                    
                    # Run query
                    result = spark.sql(query)
                    
                    # Show results
                    st.subheader("Query Results")
                    st.dataframe(result.toPandas())
                    
                    # Show execution plan
                    st.subheader("Execution Plan")
                    plan = result._jdf.queryExecution().toString()
                    st.code(plan)
                    
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")