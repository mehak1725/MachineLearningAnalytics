"""
Module for feature selection and dimensionality reduction.
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, ChiSqSelector, UnivariateFeatureSelector, PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
import config
from utils.spark_utils import read_from_delta, write_to_delta

def select_features_chi_square(spark, dataset_name, feature_cols, label_col, num_top_features=10):
    """
    Select top features using Chi-square test.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        feature_cols: List of feature column names
        label_col: Label column name
        num_top_features: Number of top features to select
        
    Returns:
        List of selected feature names and transformer
    """
    try:
        # Load dataset
        path = os.path.join(config.GOLD_PATH, dataset_name)
        df = read_from_delta(spark, path)
        if df is None:
            return None, None
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        assembled_df = assembler.transform(df)
        
        # Create selector
        selector = ChiSqSelector(
            numTopFeatures=min(num_top_features, len(feature_cols)),
            featuresCol="features",
            outputCol="selectedFeatures",
            labelCol=label_col
        )
        
        # Fit selector
        selector_model = selector.fit(assembled_df)
        
        # Get indices of selected features
        selected_indices = selector_model.selectedFeatures
        
        # Map indices back to feature names
        selected_features = [feature_cols[i] for i in selected_indices]
        
        return selected_features, selector_model
    
    except Exception as e:
        logging.error(f"Error selecting features with chi square: {e}")
        return None, None

def select_features_univariate(spark, dataset_name, feature_cols, label_col, feature_type="continuous", num_top_features=10):
    """
    Select top features using univariate feature selection.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        feature_cols: List of feature column names
        label_col: Label column name
        feature_type: Type of features ('continuous' or 'categorical')
        num_top_features: Number of top features to select
        
    Returns:
        List of selected feature names and transformer
    """
    try:
        # Load dataset
        path = os.path.join(config.GOLD_PATH, dataset_name)
        df = read_from_delta(spark, path)
        if df is None:
            return None, None
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        assembled_df = assembler.transform(df)
        
        # Choose feature selection mode based on feature type
        if feature_type == "continuous":
            selector_type = "regression"
        else:
            selector_type = "classification"
        
        # Create selector
        selector = UnivariateFeatureSelector(
            featuresCol="features",
            outputCol="selectedFeatures",
            labelCol=label_col,
            featureType=selector_type,
            labelType=selector_type,
            numTopFeatures=min(num_top_features, len(feature_cols))
        )
        
        # Fit selector
        selector_model = selector.fit(assembled_df)
        
        # Get indices of selected features
        selected_indices = selector_model.selectedFeatures
        
        # Map indices back to feature names
        selected_features = [feature_cols[i] for i in selected_indices]
        
        return selected_features, selector_model
    
    except Exception as e:
        logging.error(f"Error selecting features with univariate: {e}")
        return None, None

def perform_pca(spark, dataset_name, feature_cols, num_components=5):
    """
    Perform PCA for dimensionality reduction.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        feature_cols: List of feature column names
        num_components: Number of principal components
        
    Returns:
        Transformed DataFrame with PCA features and transformer
    """
    try:
        # Load dataset
        path = os.path.join(config.GOLD_PATH, dataset_name)
        df = read_from_delta(spark, path)
        if df is None:
            return None, None
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        assembled_df = assembler.transform(df)
        
        # Create PCA transformer
        pca = PCA(
            k=min(num_components, len(feature_cols)),
            inputCol="features",
            outputCol="pcaFeatures"
        )
        
        # Fit PCA
        pca_model = pca.fit(assembled_df)
        
        # Transform data
        pca_df = pca_model.transform(assembled_df)
        
        # Get explained variance
        explained_variance = pca_model.explainedVariance.toArray()
        
        # Write to delta
        output_path = os.path.join(config.GOLD_PATH, f"{dataset_name}_pca")
        write_to_delta(pca_df, output_path)
        
        return pca_df, pca_model, explained_variance
    
    except Exception as e:
        logging.error(f"Error performing PCA: {e}")
        return None, None, None

def select_features_with_model(spark, dataset_name, feature_cols, label_col, model_type="classification", num_top_features=10):
    """
    Select top features based on model coefficients.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        feature_cols: List of feature column names
        label_col: Label column name
        model_type: Type of model ('classification' or 'regression')
        num_top_features: Number of top features to select
        
    Returns:
        List of selected feature names and their importance
    """
    try:
        # Load dataset
        path = os.path.join(config.GOLD_PATH, dataset_name)
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        assembled_df = assembler.transform(df)
        
        # Choose model type
        if model_type == "classification":
            model = LogisticRegression(
                featuresCol="features",
                labelCol=label_col,
                maxIter=10,
                standardization=True
            )
        else:
            model = LinearRegression(
                featuresCol="features",
                labelCol=label_col,
                maxIter=10,
                standardization=True
            )
        
        # Fit model
        model_fitted = model.fit(assembled_df)
        
        # Get coefficients
        if model_type == "classification":
            coef = model_fitted.coefficients.toArray()
        else:
            coef = model_fitted.coefficients.toArray()
        
        # Get feature importance
        feature_importance = [(feature, abs(importance)) for feature, importance in zip(feature_cols, coef)]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features
        top_features = feature_importance[:num_top_features]
        
        return top_features
    
    except Exception as e:
        logging.error(f"Error selecting features with model: {e}")
        return None
