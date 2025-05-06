"""
Module for regression models using PySpark MLlib.
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression, 
    DecisionTreeRegressor, 
    RandomForestRegressor, 
    GBTRegressor,
    GeneralizedLinearRegression
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import numpy as np
import config
from utils.spark_utils import read_from_delta, write_to_delta

def prepare_regression_data(spark, dataset_name, feature_cols, label_col, train_ratio=0.8):
    """
    Prepare data for regression models.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        feature_cols: List of feature column names
        label_col: Label column name
        train_ratio: Ratio of data for training
        
    Returns:
        Dictionary with train and test DataFrames and preprocessing stages
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
        
        # Transform data
        assembled_df = assembler.transform(df)
        
        # Split data
        train_df, test_df = assembled_df.randomSplit([train_ratio, 1 - train_ratio], seed=config.RANDOM_SEED)
        
        return {
            "train_df": train_df,
            "test_df": test_df,
            "label_col": label_col,
            "feature_assembler": assembler
        }
    
    except Exception as e:
        logging.error(f"Error preparing regression data: {e}")
        return None

def train_linear_regression(prepared_data):
    """
    Train a linear regression model.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_regression_data
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        regressor = LinearRegression(
            featuresCol="features",
            labelCol=label_col,
            maxIter=10,
            regParam=0.3,
            elasticNetParam=0.8,
            standardization=True
        )
        
        # Train model
        model = regressor.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("r2")
        r2 = evaluator.evaluate(predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "coefficients": model.coefficients.toArray().tolist(),
            "intercept": model.intercept
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training linear regression: {e}")
        return None

def train_decision_tree_regressor(prepared_data):
    """
    Train a decision tree regressor.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_regression_data
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        regressor = DecisionTreeRegressor(
            featuresCol="features",
            labelCol=label_col,
            maxDepth=5,
            seed=config.RANDOM_SEED
        )
        
        # Train model
        model = regressor.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("r2")
        r2 = evaluator.evaluate(predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training decision tree regressor: {e}")
        return None

def train_random_forest_regressor(prepared_data, num_trees=20):
    """
    Train a random forest regressor.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_regression_data
        num_trees: Number of trees in the forest
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        regressor = RandomForestRegressor(
            featuresCol="features",
            labelCol=label_col,
            numTrees=num_trees,
            maxDepth=5,
            seed=config.RANDOM_SEED
        )
        
        # Train model
        model = regressor.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("r2")
        r2 = evaluator.evaluate(predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training random forest regressor: {e}")
        return None

def train_gbt_regressor(prepared_data):
    """
    Train a gradient-boosted tree regressor.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_regression_data
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        regressor = GBTRegressor(
            featuresCol="features",
            labelCol=label_col,
            maxIter=10,
            maxDepth=5,
            seed=config.RANDOM_SEED
        )
        
        # Train model
        model = regressor.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("r2")
        r2 = evaluator.evaluate(predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training GBT regressor: {e}")
        return None

def train_generalized_linear_regression(prepared_data, family="gaussian", link="identity"):
    """
    Train a generalized linear regression model.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_regression_data
        family: Distribution family (gaussian, binomial, poisson, gamma)
        link: Link function (identity, log, inverse, logit)
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        regressor = GeneralizedLinearRegression(
            featuresCol="features",
            labelCol=label_col,
            family=family,
            link=link,
            maxIter=10,
            regParam=0.3
        )
        
        # Train model
        model = regressor.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("r2")
        r2 = evaluator.evaluate(predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "coefficients": model.coefficients.toArray().tolist(),
            "intercept": model.intercept
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training generalized linear regression: {e}")
        return None

def compare_regression_models(prepared_data):
    """
    Compare multiple regression models.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_regression_data
        
    Returns:
        Dictionary with model comparisons
    """
    try:
        # Define models to compare
        models = {
            "linear_regression": lambda: train_linear_regression(prepared_data),
            "decision_tree": lambda: train_decision_tree_regressor(prepared_data),
            "random_forest": lambda: train_random_forest_regressor(prepared_data),
            "gbt": lambda: train_gbt_regressor(prepared_data),
            "glm": lambda: train_generalized_linear_regression(prepared_data)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model_func in models.items():
            model_result = model_func()
            if model_result:
                results[name] = model_result["metrics"]
        
        return results
    
    except Exception as e:
        logging.error(f"Error comparing regression models: {e}")
        return None
