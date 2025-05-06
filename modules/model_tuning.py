"""
Module for model selection and hyperparameter tuning.
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)
import config
from utils.spark_utils import write_to_delta

def tune_classification_model(prepared_data, model_type="logistic_regression", is_binary=True):
    """
    Tune hyperparameters for classification models.
    
    Args:
        prepared_data: Dictionary with prepared data
        model_type: Type of model ('logistic_regression', 'random_forest', or 'gbt')
        is_binary: Whether this is a binary classification problem
        
    Returns:
        Dictionary with best model and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create base model based on type
        if model_type == "logistic_regression":
            model = LogisticRegression(
                featuresCol="features",
                labelCol=label_col
            )
            
            # Create parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam, [0.01, 0.1, 0.3]) \
                .addGrid(model.elasticNetParam, [0.0, 0.5, 0.8]) \
                .addGrid(model.maxIter, [10, 20, 50]) \
                .build()
                
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                featuresCol="features",
                labelCol=label_col,
                seed=config.RANDOM_SEED
            )
            
            # Create parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.numTrees, [10, 20, 50]) \
                .addGrid(model.maxDepth, [3, 5, 10]) \
                .addGrid(model.impurity, ["gini", "entropy"]) \
                .build()
                
        elif model_type == "gbt":
            model = GBTClassifier(
                featuresCol="features",
                labelCol=label_col,
                seed=config.RANDOM_SEED
            )
            
            # Create parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxIter, [10, 20, 50]) \
                .addGrid(model.maxDepth, [3, 5, 10]) \
                .addGrid(model.stepSize, [0.05, 0.1, 0.2]) \
                .build()
                
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create evaluator
        if is_binary:
            evaluator = BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
        else:
            evaluator = MulticlassClassificationEvaluator(
                labelCol=label_col,
                predictionCol="prediction",
                metricName="f1"
            )
        
        # Create cross validator
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=config.CV_FOLDS,
            seed=config.RANDOM_SEED
        )
        
        # Fit cross validator
        cv_model = cv.fit(train_df)
        
        # Get best model
        best_model = cv_model.bestModel
        
        # Make predictions
        predictions = best_model.transform(test_df)
        
        # Evaluate best model
        if is_binary:
            auc_roc = evaluator.evaluate(predictions)
            
            evaluator.setMetricName("areaUnderPR")
            auc_pr = evaluator.evaluate(predictions)
            
            metrics = {
                "auc_roc": auc_roc,
                "auc_pr": auc_pr
            }
        else:
            evaluator.setMetricName("accuracy")
            accuracy = evaluator.evaluate(predictions)
            
            evaluator.setMetricName("f1")
            f1 = evaluator.evaluate(predictions)
            
            metrics = {
                "accuracy": accuracy,
                "f1": f1
            }
        
        # Get best parameters
        best_params = {}
        for param in param_grid[0].keys():
            param_name = param.name
            param_value = best_model.getOrDefault(param_name)
            best_params[param_name] = param_value
        
        return {
            "model": best_model,
            "predictions": predictions,
            "metrics": metrics,
            "best_params": best_params
        }
    
    except Exception as e:
        logging.error(f"Error tuning classification model: {e}")
        return None

def tune_regression_model(prepared_data, model_type="linear_regression"):
    """
    Tune hyperparameters for regression models.
    
    Args:
        prepared_data: Dictionary with prepared data
        model_type: Type of model ('linear_regression', 'random_forest', or 'gbt')
        
    Returns:
        Dictionary with best model and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create base model based on type
        if model_type == "linear_regression":
            model = LinearRegression(
                featuresCol="features",
                labelCol=label_col
            )
            
            # Create parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam, [0.01, 0.1, 0.3]) \
                .addGrid(model.elasticNetParam, [0.0, 0.5, 0.8]) \
                .addGrid(model.maxIter, [10, 20, 50]) \
                .build()
                
        elif model_type == "random_forest":
            model = RandomForestRegressor(
                featuresCol="features",
                labelCol=label_col,
                seed=config.RANDOM_SEED
            )
            
            # Create parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.numTrees, [10, 20, 50]) \
                .addGrid(model.maxDepth, [3, 5, 10]) \
                .build()
                
        elif model_type == "gbt":
            model = GBTRegressor(
                featuresCol="features",
                labelCol=label_col,
                seed=config.RANDOM_SEED
            )
            
            # Create parameter grid
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxIter, [10, 20, 50]) \
                .addGrid(model.maxDepth, [3, 5, 10]) \
                .addGrid(model.stepSize, [0.05, 0.1, 0.2]) \
                .build()
                
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create evaluator
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="rmse"
        )
        
        # Create train-validation split
        tvs = TrainValidationSplit(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            trainRatio=0.8,
            seed=config.RANDOM_SEED
        )
        
        # Fit train-validation split
        tvs_model = tvs.fit(train_df)
        
        # Get best model
        best_model = tvs_model.bestModel
        
        # Make predictions
        predictions = best_model.transform(test_df)
        
        # Evaluate best model
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
        
        # Get best parameters
        best_params = {}
        for param in param_grid[0].keys():
            param_name = param.name
            param_value = best_model.getOrDefault(param_name)
            best_params[param_name] = param_value
        
        return {
            "model": best_model,
            "predictions": predictions,
            "metrics": metrics,
            "best_params": best_params
        }
    
    except Exception as e:
        logging.error(f"Error tuning regression model: {e}")
        return None

def save_model(model, model_name, model_type):
    """
    Save a trained model.
    
    Args:
        model: Trained model
        model_name: Name of the model
        model_type: Type of model ('classification', 'regression', or 'clustering')
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create the model directory if it doesn't exist
        model_dir = os.path.join(config.MODEL_SAVE_PATH, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, model_name)
        model.write().overwrite().save(model_path)
        
        logging.info(f"Model saved to {model_path}")
        
        return True
    
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        return False

def load_model(spark, model_name, model_type, model_class):
    """
    Load a saved model.
    
    Args:
        spark: SparkSession instance
        model_name: Name of the model
        model_type: Type of model ('classification', 'regression', or 'clustering')
        model_class: Model class to load
        
    Returns:
        Loaded model
    """
    try:
        # Get the model path
        model_path = os.path.join(config.MODEL_SAVE_PATH, model_type, model_name)
        
        # Load the model
        model = model_class.load(model_path)
        
        logging.info(f"Model loaded from {model_path}")
        
        return model
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
