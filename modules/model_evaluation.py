"""
Module for model evaluation and metrics computation.
"""
import logging
import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, when, count, sum, desc, expr, avg, stddev
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator
)
import config
from utils.spark_utils import read_from_delta, write_to_delta, convert_to_pandas

def evaluate_binary_classifier(predictions, label_col):
    """
    Evaluate a binary classifier.
    
    Args:
        predictions: DataFrame with predictions
        label_col: Label column name
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Binary classification evaluator for AUC
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol=label_col,
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc_roc = evaluator_auc.evaluate(predictions)
        
        evaluator_auc.setMetricName("areaUnderPR")
        auc_pr = evaluator_auc.evaluate(predictions)
        
        # Calculate confusion matrix and derived metrics
        # True positives, false positives, true negatives, false negatives
        tp = predictions.filter((col(label_col) == 1) & (col("prediction") == 1)).count()
        fp = predictions.filter((col(label_col) == 0) & (col("prediction") == 1)).count()
        tn = predictions.filter((col(label_col) == 0) & (col("prediction") == 0)).count()
        fn = predictions.filter((col(label_col) == 1) & (col("prediction") == 0)).count()
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Confusion matrix
        confusion_matrix = {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn
        }
        
        # Combine metrics
        metrics = {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion_matrix
        }
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error evaluating binary classifier: {e}")
        return None

def evaluate_multiclass_classifier(predictions, label_col):
    """
    Evaluate a multiclass classifier.
    
    Args:
        predictions: DataFrame with predictions
        label_col: Label column name
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Multiclass classification evaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("f1")
        f1 = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("weightedPrecision")
        precision = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("weightedRecall")
        recall = evaluator.evaluate(predictions)
        
        # Calculate confusion matrix
        # Get unique labels
        labels = predictions.select(label_col).distinct().rdd.flatMap(lambda x: x).collect()
        labels.sort()
        
        # Calculate confusion matrix
        confusion_matrix = {}
        
        for true_label in labels:
            confusion_matrix[true_label] = {}
            
            for pred_label in labels:
                count_val = predictions.filter(
                    (col(label_col) == true_label) & 
                    (col("prediction") == pred_label)
                ).count()
                
                confusion_matrix[true_label][pred_label] = count_val
        
        # Class-wise metrics
        class_metrics = {}
        
        for label in labels:
            # True positives for this class
            tp = confusion_matrix[label][label]
            
            # False positives (predicted as this class but actually another class)
            fp = sum(confusion_matrix[other_label][label] for other_label in labels if other_label != label)
            
            # False negatives (actually this class but predicted as another class)
            fn = sum(confusion_matrix[label][other_label] for other_label in labels if other_label != label)
            
            # Precision and recall for this class
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            class_metrics[f"class_{label}"] = {
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1
            }
        
        # Combine metrics
        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": confusion_matrix,
            "class_metrics": class_metrics
        }
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error evaluating multiclass classifier: {e}")
        return None

def evaluate_regressor(predictions, label_col):
    """
    Evaluate a regression model.
    
    Args:
        predictions: DataFrame with predictions
        label_col: Label column name
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Regression evaluator
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
        
        # Calculate additional metrics
        # Mean error and mean absolute percentage error
        from pyspark.sql.functions import abs as sql_abs
        error_df = predictions.withColumn(
            "error", 
            col("prediction") - col(label_col)
        ).withColumn(
            "abs_error",
            sql_abs(col("error"))
        ).withColumn(
            "squared_error",
            col("error") * col("error")
        ).withColumn(
            "abs_pct_error",
            when(col(label_col) != 0, 
                sql_abs(col("error") / col(label_col)) * 100
            ).otherwise(None)
        )
        
        # Calculate metrics
        metrics_df = error_df.select(
            avg("error").alias("mean_error"),
            avg("abs_error").alias("mae"),
            avg("squared_error").alias("mse"),
            avg("abs_pct_error").alias("mape")
        ).collect()[0]
        
        mean_error = metrics_df["mean_error"]
        mape = metrics_df["mape"]
        
        # Calculate residual statistics
        residuals = error_df.select("error").rdd.flatMap(lambda x: x).collect()
        
        # Combine metrics
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mean_error": float(mean_error),
            "mape": float(mape),
            "residuals": residuals
        }
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error evaluating regressor: {e}")
        return None

def evaluate_clustering(predictions, features_col="features"):
    """
    Evaluate a clustering model.
    
    Args:
        predictions: DataFrame with predictions
        features_col: Features column name
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Clustering evaluator
        evaluator = ClusteringEvaluator(
            featuresCol=features_col,
            predictionCol="prediction",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        silhouette = evaluator.evaluate(predictions)
        
        # Calculate cluster sizes
        cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction")
        
        # Convert to dictionary for easier access
        cluster_sizes_dict = {
            row["prediction"]: row["count"] 
            for row in cluster_sizes.collect()
        }
        
        # Calculate cluster proportions
        total_points = predictions.count()
        cluster_proportions = {
            cluster: count / total_points 
            for cluster, count in cluster_sizes_dict.items()
        }
        
        # Combine metrics
        metrics = {
            "silhouette": silhouette,
            "cluster_sizes": cluster_sizes_dict,
            "cluster_proportions": cluster_proportions,
            "num_clusters": len(cluster_sizes_dict)
        }
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error evaluating clustering: {e}")
        return None

def compare_models(model_results, evaluation_type):
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        model_results: Dictionary with model names as keys and evaluation results as values
        evaluation_type: Type of evaluation ('classification', 'regression', or 'clustering')
        
    Returns:
        DataFrame with model comparison
    """
    try:
        comparison = []
        
        # Extract relevant metrics based on evaluation type
        if evaluation_type == "binary_classification":
            for name, result in model_results.items():
                metrics = result.get("metrics", {})
                comparison.append({
                    "Model": name,
                    "AUC-ROC": metrics.get("auc_roc", 0),
                    "AUC-PR": metrics.get("auc_pr", 0),
                    "Accuracy": metrics.get("accuracy", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1": metrics.get("f1", 0)
                })
                
        elif evaluation_type == "multiclass_classification":
            for name, result in model_results.items():
                metrics = result.get("metrics", {})
                comparison.append({
                    "Model": name,
                    "Accuracy": metrics.get("accuracy", 0),
                    "F1": metrics.get("f1", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0)
                })
                
        elif evaluation_type == "regression":
            for name, result in model_results.items():
                metrics = result.get("metrics", {})
                comparison.append({
                    "Model": name,
                    "RMSE": metrics.get("rmse", 0),
                    "MAE": metrics.get("mae", 0),
                    "R^2": metrics.get("r2", 0),
                    "MAPE": metrics.get("mape", 0)
                })
                
        elif evaluation_type == "clustering":
            for name, result in model_results.items():
                metrics = result.get("metrics", {})
                comparison.append({
                    "Model": name,
                    "Silhouette": metrics.get("silhouette", 0),
                    "Num Clusters": metrics.get("num_clusters", 0)
                })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        return comparison_df
    
    except Exception as e:
        logging.error(f"Error comparing models: {e}")
        return None

def analyze_feature_importance(model, feature_cols, model_type):
    """
    Analyze feature importance from a trained model.
    
    Args:
        model: Trained model
        feature_cols: List of feature column names
        model_type: Type of model ('logistic_regression', 'linear_regression', 'random_forest', etc.)
        
    Returns:
        DataFrame with feature importance
    """
    try:
        feature_importance = []
        
        # Extract feature importance based on model type
        if model_type in ["logistic_regression", "linear_regression"]:
            # For linear models, use coefficients
            coefficients = model.coefficients.toArray()
            
            # Map coefficients to feature names
            for i, col_name in enumerate(feature_cols):
                if i < len(coefficients):
                    importance = abs(coefficients[i])
                    feature_importance.append({
                        "Feature": col_name,
                        "Importance": importance,
                        "Coefficient": coefficients[i]
                    })
                    
        elif model_type in ["random_forest", "gbt"]:
            # For tree-based models, use feature importance
            importances = model.featureImportances.toArray()
            
            # Map importances to feature names
            for i, col_name in enumerate(feature_cols):
                if i < len(importances):
                    importance = importances[i]
                    feature_importance.append({
                        "Feature": col_name,
                        "Importance": importance
                    })
        
        # Convert to DataFrame and sort by importance
        importance_df = pd.DataFrame(feature_importance)
        
        if not importance_df.empty:
            importance_df = importance_df.sort_values("Importance", ascending=False)
        
        return importance_df
    
    except Exception as e:
        logging.error(f"Error analyzing feature importance: {e}")
        return None

def create_evaluation_report(model_result, model_type, feature_cols=None):
    """
    Create a comprehensive evaluation report for a model.
    
    Args:
        model_result: Dictionary with model, predictions, and metrics
        model_type: Type of model ('binary_classification', 'multiclass_classification', 'regression', 'clustering')
        feature_cols: List of feature column names for feature importance
        
    Returns:
        Dictionary with evaluation report
    """
    try:
        model = model_result.get("model")
        predictions = model_result.get("predictions")
        metrics = model_result.get("metrics", {})
        
        # Basic report structure
        report = {
            "model_type": model_type,
            "metrics": metrics,
            "feature_importance": None,
            "prediction_examples": None
        }
        
        # Add feature importance if applicable
        if feature_cols and model and model_type in ["binary_classification", "multiclass_classification", "regression"]:
            # Map model_type to the underlying model type for feature importance
            if model_type == "binary_classification" or model_type == "multiclass_classification":
                if "LogisticRegression" in str(type(model)):
                    feature_importance = analyze_feature_importance(model, feature_cols, "logistic_regression")
                elif "RandomForest" in str(type(model)):
                    feature_importance = analyze_feature_importance(model, feature_cols, "random_forest")
                elif "GBT" in str(type(model)):
                    feature_importance = analyze_feature_importance(model, feature_cols, "gbt")
                else:
                    feature_importance = None
            else:  # regression
                if "LinearRegression" in str(type(model)):
                    feature_importance = analyze_feature_importance(model, feature_cols, "linear_regression")
                elif "RandomForest" in str(type(model)):
                    feature_importance = analyze_feature_importance(model, feature_cols, "random_forest")
                elif "GBT" in str(type(model)):
                    feature_importance = analyze_feature_importance(model, feature_cols, "gbt")
                else:
                    feature_importance = None
            
            report["feature_importance"] = feature_importance
        
        # Add prediction examples
        if predictions:
            # Convert to pandas for sample
            prediction_df = convert_to_pandas(predictions, limit=10)
            report["prediction_examples"] = prediction_df
        
        return report
    
    except Exception as e:
        logging.error(f"Error creating evaluation report: {e}")
        return None
