"""
Module for classification models using PySpark MLlib.
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import (
    LogisticRegression, 
    RandomForestClassifier, 
    NaiveBayes, 
    OneVsRest, 
    GBTClassifier,
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import config
from utils.spark_utils import read_from_delta, write_to_delta

def prepare_classification_data(spark, dataset_name, feature_cols, label_col, train_ratio=0.8):
    """
    Prepare data for classification models.
    
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
        
        # Check for string labels and apply indexing if needed
        label_indexer = None
        if df.select(label_col).dtypes[0][1] == 'string':
            label_indexer = StringIndexer(
                inputCol=label_col,
                outputCol=f"{label_col}_index",
                handleInvalid="keep"
            )
            df = label_indexer.fit(df).transform(df)
            label_col = f"{label_col}_index"
        
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
            "label_indexer": label_indexer,
            "feature_assembler": assembler
        }
    
    except Exception as e:
        logging.error(f"Error preparing classification data: {e}")
        return None

def train_logistic_regression(prepared_data, multiclass=False):
    """
    Train a logistic regression classifier.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_classification_data
        multiclass: Whether this is a multiclass problem
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        if multiclass:
            classifier = LogisticRegression(
                featuresCol="features",
                labelCol=label_col,
                maxIter=20,
                regParam=0.3,
                elasticNetParam=0.8,
                family="multinomial"
            )
        else:
            classifier = LogisticRegression(
                featuresCol="features",
                labelCol=label_col,
                maxIter=20,
                regParam=0.3,
                elasticNetParam=0.8
            )
        
        # Train model
        model = classifier.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        if multiclass:
            evaluator = MulticlassClassificationEvaluator(
                labelCol=label_col,
                predictionCol="prediction",
                metricName="accuracy"
            )
            accuracy = evaluator.evaluate(predictions)
            
            evaluator.setMetricName("f1")
            f1 = evaluator.evaluate(predictions)
            
            metrics = {
                "accuracy": accuracy,
                "f1": f1
            }
        else:
            evaluator = BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            auc_roc = evaluator.evaluate(predictions)
            
            evaluator.setMetricName("areaUnderPR")
            auc_pr = evaluator.evaluate(predictions)
            
            metrics = {
                "auc_roc": auc_roc,
                "auc_pr": auc_pr
            }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training logistic regression: {e}")
        return None

def train_random_forest(prepared_data, num_trees=20):
    """
    Train a random forest classifier.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_classification_data
        num_trees: Number of trees in the forest
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        classifier = RandomForestClassifier(
            featuresCol="features",
            labelCol=label_col,
            numTrees=num_trees,
            maxDepth=5,
            seed=config.RANDOM_SEED
        )
        
        # Train model
        model = classifier.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("f1")
        f1 = evaluator.evaluate(predictions)
        
        metrics = {
            "accuracy": accuracy,
            "f1": f1
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training random forest: {e}")
        return None

def train_naive_bayes(prepared_data):
    """
    Train a Naive Bayes classifier.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_classification_data
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        classifier = NaiveBayes(
            featuresCol="features",
            labelCol=label_col,
            smoothing=1.0,
            modelType="multinomial"
        )
        
        # Train model
        model = classifier.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("f1")
        f1 = evaluator.evaluate(predictions)
        
        metrics = {
            "accuracy": accuracy,
            "f1": f1
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training Naive Bayes: {e}")
        return None

def train_gbt_classifier(prepared_data):
    """
    Train a Gradient-Boosted Trees classifier.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_classification_data
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create model
        classifier = GBTClassifier(
            featuresCol="features",
            labelCol=label_col,
            maxIter=10,
            maxDepth=5,
            seed=config.RANDOM_SEED
        )
        
        # Train model
        model = classifier.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("f1")
        f1 = evaluator.evaluate(predictions)
        
        metrics = {
            "accuracy": accuracy,
            "f1": f1
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training GBT classifier: {e}")
        return None

def train_one_vs_rest(prepared_data, base_classifier=None):
    """
    Train a One-vs-Rest multiclass classifier.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_classification_data
        base_classifier: Base classifier to use (if None, use LogisticRegression)
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        label_col = prepared_data["label_col"]
        
        # Create base classifier if not provided
        if base_classifier is None:
            base_classifier = LogisticRegression(
                featuresCol="features",
                labelCol=label_col,
                maxIter=10,
                regParam=0.1
            )
        
        # Create One-vs-Rest classifier
        classifier = OneVsRest(
            classifier=base_classifier,
            labelCol=label_col
        )
        
        # Train model
        model = classifier.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("f1")
        f1 = evaluator.evaluate(predictions)
        
        metrics = {
            "accuracy": accuracy,
            "f1": f1
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training One-vs-Rest classifier: {e}")
        return None

def compare_classification_models(prepared_data):
    """
    Compare multiple classification models.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_classification_data
        
    Returns:
        Dictionary with model comparisons
    """
    try:
        # Define models to compare
        models = {
            "logistic_regression": lambda: train_logistic_regression(prepared_data),
            "random_forest": lambda: train_random_forest(prepared_data),
            "naive_bayes": lambda: train_naive_bayes(prepared_data),
            "gbt": lambda: train_gbt_classifier(prepared_data)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model_func in models.items():
            model_result = model_func()
            if model_result:
                results[name] = model_result["metrics"]
        
        return results
    
    except Exception as e:
        logging.error(f"Error comparing classification models: {e}")
        return None
