"""
Module for collaborative filtering using Alternating Least Squares (ALS).
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_list, avg, count
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import config
from utils.spark_utils import read_from_delta, write_to_delta

def prepare_als_data(spark, dataset_name, user_col, item_col, rating_col, train_ratio=0.8):
    """
    Prepare data for ALS collaborative filtering.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        user_col: Column name for users
        item_col: Column name for items
        rating_col: Column name for ratings
        train_ratio: Ratio of data for training
        
    Returns:
        Dictionary with train and test DataFrames
    """
    try:
        # Load dataset
        path = os.path.join(config.GOLD_PATH, dataset_name)
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Split data
        train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=config.RANDOM_SEED)
        
        return {
            "train_df": train_df,
            "test_df": test_df,
            "user_col": user_col,
            "item_col": item_col,
            "rating_col": rating_col
        }
    
    except Exception as e:
        logging.error(f"Error preparing ALS data: {e}")
        return None

def create_forecast_accuracy_data(spark, dataset_type="weather"):
    """
    Create a dataset for forecast accuracy recommendation.
    
    Args:
        spark: SparkSession instance
        dataset_type: Type of dataset ('weather' or 'temperature')
        
    Returns:
        DataFrame with forecast accuracy data
    """
    try:
        if dataset_type == "weather":
            path = os.path.join(config.GOLD_PATH, "weather_features")
            id_cols = ["DATE_PARSED", "YEAR", "MONTH_NUM", "DAY"]
        else:
            path = os.path.join(config.GOLD_PATH, "temperature_features")
            id_cols = ["Country", "City", "Year"]
        
        # Load dataset
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        if dataset_type == "weather":
            # Extract stations and their metrics
            stations = set()
            metrics = set()
            
            for col_name in df.columns:
                if "_" in col_name:
                    parts = col_name.split("_")
                    station = parts[0]
                    metric = "_".join(parts[1:])
                    stations.add(station)
                    metrics.add(metric)
            
            # Create accuracy data
            # For this example, we'll use temperature prediction accuracy
            # We'll simulate forecast error as the difference between stations
            
            # First, find temperature columns
            temp_cols = [c for c in df.columns if "temp_mean" in c and "_lag" not in c]
            
            if not temp_cols:
                return None
            
            # Create a base for comparison
            base_station = list(stations)[0]
            base_col = f"{base_station}_temp_mean"
            
            if base_col not in df.columns:
                for c in temp_cols:
                    if c.endswith("_temp_mean"):
                        base_col = c
                        base_station = c.split("_")[0]
                        break
            
            # Calculate "errors" between stations
            accuracy_data = []
            
            for station in stations:
                if station != base_station:
                    station_col = f"{station}_temp_mean"
                    
                    if station_col in df.columns:
                        # Calculate error
                        accuracy_df = df.select(
                            *id_cols,
                            col(base_col).alias("actual"),
                            col(station_col).alias("predicted"),
                            (col(base_col) - col(station_col)).alias("error")
                        )
                        
                        # Add station info
                        accuracy_df = accuracy_df.withColumn("station", lit(station))
                        
                        # Calculate accuracy (inverse of absolute error)
                        accuracy_df = accuracy_df.withColumn(
                            "accuracy",
                            1.0 / (1.0 + abs(col("error")))
                        )
                        
                        accuracy_data.append(accuracy_df)
            
            if not accuracy_data:
                return None
            
            # Union all accuracy data
            from functools import reduce
            from pyspark.sql import DataFrame
            
            combined_df = reduce(DataFrame.unionAll, accuracy_data)
            
            # Write to gold layer
            output_path = os.path.join(config.GOLD_PATH, "forecast_accuracy")
            write_to_delta(combined_df, output_path)
            
            return combined_df
        
        else:
            # For temperature dataset, create a country-city-year accuracy measure
            # This will be a synthetic measure for demonstration
            from pyspark.sql.functions import rand, lit
            
            # Create an artificial accuracy metric
            accuracy_df = df.select(
                "Country",
                "City",
                "Year",
                "AvgYearlyTemp",
                "PrevTemp",
                abs(col("AvgYearlyTemp") - col("PrevTemp")).alias("error")
            )
            
            # Calculate accuracy (inverse of absolute error)
            accuracy_df = accuracy_df.withColumn(
                "accuracy",
                1.0 / (1.0 + col("error"))
            )
            
            # Write to gold layer
            output_path = os.path.join(config.GOLD_PATH, "temp_accuracy")
            write_to_delta(accuracy_df, output_path)
            
            return accuracy_df
    
    except Exception as e:
        logging.error(f"Error creating forecast accuracy data: {e}")
        return None

def train_als_model(prepared_data, rank=10, max_iter=10, reg_param=0.01):
    """
    Train an ALS collaborative filtering model.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_als_data
        rank: Rank of the factorization
        max_iter: Maximum number of iterations
        reg_param: Regularization parameter
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        user_col = prepared_data["user_col"]
        item_col = prepared_data["item_col"]
        rating_col = prepared_data["rating_col"]
        
        # Create model
        als = ALS(
            userCol=user_col,
            itemCol=item_col,
            ratingCol=rating_col,
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            coldStartStrategy="drop",
            seed=config.RANDOM_SEED
        )
        
        # Train model
        model = als.fit(train_df)
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol=rating_col,
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "rank": rank,
            "max_iter": max_iter,
            "reg_param": reg_param
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training ALS model: {e}")
        return None

def tune_als_model(prepared_data):
    """
    Tune ALS model hyperparameters.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_als_data
        
    Returns:
        Dictionary with best model and hyperparameters
    """
    try:
        train_df = prepared_data["train_df"]
        test_df = prepared_data["test_df"]
        user_col = prepared_data["user_col"]
        item_col = prepared_data["item_col"]
        rating_col = prepared_data["rating_col"]
        
        # Create model
        als = ALS(
            userCol=user_col,
            itemCol=item_col,
            ratingCol=rating_col,
            coldStartStrategy="drop",
            seed=config.RANDOM_SEED
        )
        
        # Create parameter grid
        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [5, 10, 20]) \
            .addGrid(als.maxIter, [5, 10]) \
            .addGrid(als.regParam, [0.01, 0.1]) \
            .build()
        
        # Create evaluator
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol=rating_col,
            predictionCol="prediction"
        )
        
        # Create cross validator
        cv = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=config.RANDOM_SEED
        )
        
        # Fit cross validator
        cv_model = cv.fit(train_df)
        
        # Get best model
        best_model = cv_model.bestModel
        
        # Make predictions
        predictions = best_model.transform(test_df)
        
        # Evaluate best model
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        # Get best parameters
        best_rank = best_model.getOrDefault("rank")
        best_max_iter = best_model.getOrDefault("maxIter")
        best_reg_param = best_model.getOrDefault("regParam")
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "best_rank": best_rank,
            "best_max_iter": best_max_iter,
            "best_reg_param": best_reg_param
        }
        
        return {
            "model": best_model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error tuning ALS model: {e}")
        return None

def generate_recommendations(model_result, num_recommendations=10):
    """
    Generate recommendations for all users.
    
    Args:
        model_result: Dictionary with model and predictions from train_als_model
        num_recommendations: Number of recommendations per user
        
    Returns:
        DataFrame with recommendations
    """
    try:
        model = model_result["model"]
        
        # Generate recommendations
        recommendations = model.recommendForAllUsers(num_recommendations)
        
        # Explode recommendations
        from pyspark.sql.functions import explode
        
        recommendations = recommendations.select(
            col("user"),
            explode(col("recommendations")).alias("recommendation")
        )
        
        recommendations = recommendations.select(
            col("user"),
            col("recommendation.item").alias("item"),
            col("recommendation.rating").alias("predicted_rating")
        )
        
        return recommendations
    
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return None
