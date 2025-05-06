"""
Module for Exploratory Data Analysis (EDA) on weather datasets.
"""
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, corr, mean, stddev, min, max, count, isnan
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import numpy as np
import os
import config
from utils.spark_utils import read_from_delta

def calculate_basic_stats(spark, dataset):
    """
    Calculate basic statistics for a dataset.
    
    Args:
        spark: SparkSession instance
        dataset: Name of the dataset ('weather' or 'temperature')
        
    Returns:
        Dictionary with basic statistics
    """
    if dataset == 'weather':
        path = os.path.join(config.SILVER_PATH, "weather_clean")
        id_cols = ["DATE", "DATE_PARSED", "YEAR", "MONTH", "MONTH_NUM", "DAY"]
    else:
        path = os.path.join(config.SILVER_PATH, "temperature_clean")
        id_cols = ["Country", "City", "Year"]
    
    try:
        # Load data
        df = read_from_delta(spark, path)
        if df is None:
            return {"error": f"Could not load {dataset} dataset"}
        
        # Get numeric columns
        numeric_cols = [col_name for col_name, dtype in df.dtypes 
                        if dtype in ('double', 'float', 'int', 'bigint', 'decimal')
                        and col_name not in id_cols]
        
        if not numeric_cols:
            return {"error": "No numeric columns found for analysis"}
        
        # Calculate statistics
        stats = df.select([
            mean(col(c)).alias(f"{c}_mean"),
            stddev(col(c)).alias(f"{c}_stddev"),
            min(col(c)).alias(f"{c}_min"),
            max(col(c)).alias(f"{c}_max"),
            count(when(col(c).isNull() | isnan(col(c)), c)).alias(f"{c}_nulls")
        ] for c in numeric_cols).collect()[0].asDict()
        
        # Format results
        result = {}
        for c in numeric_cols:
            result[c] = {
                "mean": stats.get(f"{c}_mean"),
                "stddev": stats.get(f"{c}_stddev"),
                "min": stats.get(f"{c}_min"),
                "max": stats.get(f"{c}_max"),
                "nulls": stats.get(f"{c}_nulls")
            }
        
        return result
    
    except Exception as e:
        logging.error(f"Error calculating basic stats: {e}")
        return {"error": str(e)}

def calculate_correlations(spark, dataset, columns=None, limit=15):
    """
    Calculate correlations between variables.
    
    Args:
        spark: SparkSession instance
        dataset: Name of the dataset ('weather' or 'temperature')
        columns: List of columns to include (if None, use all numeric)
        limit: Maximum number of correlations to return
        
    Returns:
        DataFrame with correlation values
    """
    if dataset == 'weather':
        path = os.path.join(config.SILVER_PATH, "weather_clean")
        id_cols = ["DATE", "DATE_PARSED", "YEAR", "MONTH", "MONTH_NUM", "DAY"]
    else:
        path = os.path.join(config.SILVER_PATH, "temperature_clean")
        id_cols = ["Country", "City", "Year"]
    
    try:
        # Load data
        df = read_from_delta(spark, path)
        if df is None:
            return pd.DataFrame({"error": [f"Could not load {dataset} dataset"]})
        
        # Get numeric columns
        numeric_cols = [col_name for col_name, dtype in df.dtypes 
                       if dtype in ('double', 'float', 'int', 'bigint', 'decimal')
                       and col_name not in id_cols]
        
        if not numeric_cols:
            return pd.DataFrame({"error": ["No numeric columns found for analysis"]})
        
        # Filter columns if specified
        if columns:
            numeric_cols = [c for c in numeric_cols if c in columns]
        
        # Limit to most common columns to avoid excessive computation
        numeric_cols = numeric_cols[:20]
        
        # Calculate pairwise correlations
        correlations = []
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_val = df.select(corr(col1, col2).alias('correlation')).first()['correlation']
                if corr_val is not None:
                    correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_val
                    })
        
        # Convert to pandas DataFrame
        corr_df = pd.DataFrame(correlations)
        
        # Sort by absolute correlation and limit
        if not corr_df.empty:
            corr_df['abs_corr'] = corr_df['correlation'].abs()
            corr_df = corr_df.sort_values('abs_corr', ascending=False).head(limit)
            corr_df = corr_df.drop('abs_corr', axis=1)
        
        return corr_df
    
    except Exception as e:
        logging.error(f"Error calculating correlations: {e}")
        return pd.DataFrame({"error": [str(e)]})

def station_comparison(spark, stations=None, metrics=None):
    """
    Compare weather metrics across different stations.
    
    Args:
        spark: SparkSession instance
        stations: List of stations to include (if None, use all)
        metrics: List of metrics to compare (if None, use common metrics)
        
    Returns:
        Dictionary with comparison data
    """
    try:
        # Load data
        path = os.path.join(config.SILVER_PATH, "weather_clean")
        df = read_from_delta(spark, path)
        if df is None:
            return {"error": "Could not load weather dataset"}
        
        # Get all available stations from column names
        all_stations = set()
        common_metrics = set()
        
        for col_name in df.columns:
            if "_" in col_name:
                parts = col_name.split("_")
                station = parts[0]
                metric = "_".join(parts[1:])
                all_stations.add(station)
                common_metrics.add(metric)
        
        if not stations:
            stations = list(all_stations)
        else:
            # Filter to only include existing stations
            stations = [s for s in stations if s in all_stations]
        
        if not metrics:
            # Use some common metrics that many stations would have
            common_metrics = ["temp_mean", "humidity", "pressure"]
            # Filter to metrics that exist
            available_metrics = []
            for metric in common_metrics:
                if any(f"{station}_{metric}" in df.columns for station in stations):
                    available_metrics.append(metric)
            metrics = available_metrics
        
        # Prepare comparison data
        comparison = {}
        for metric in metrics:
            metric_data = {"stations": [], "values": []}
            for station in stations:
                col_name = f"{station}_{metric}"
                if col_name in df.columns:
                    avg_value = df.select(mean(col(col_name))).first()[0]
                    if avg_value is not None:
                        metric_data["stations"].append(station)
                        metric_data["values"].append(avg_value)
            
            if metric_data["stations"]:
                comparison[metric] = metric_data
        
        return comparison
    
    except Exception as e:
        logging.error(f"Error comparing stations: {e}")
        return {"error": str(e)}

def yearly_temperature_trends(spark, cities=None, start_year=None, end_year=None):
    """
    Analyze yearly temperature trends for select cities.
    
    Args:
        spark: SparkSession instance
        cities: List of cities to include (if None, use top cities)
        start_year: Starting year for trend analysis
        end_year: Ending year for trend analysis
        
    Returns:
        DataFrame with trend data
    """
    try:
        # Load data
        path = os.path.join(config.SILVER_PATH, "temperature_clean")
        df = read_from_delta(spark, path)
        if df is None:
            return pd.DataFrame({"error": ["Could not load temperature dataset"]})
        
        # Apply year filter if specified
        if start_year:
            df = df.filter(col("Year") >= start_year)
        if end_year:
            df = df.filter(col("Year") <= end_year)
        
        # Get top cities by data points if not specified
        if not cities:
            city_counts = df.groupBy("City").count().orderBy(col("count").desc())
            top_cities = city_counts.limit(5).select("City").rdd.flatMap(lambda x: x).collect()
            cities = top_cities
        
        # Filter to selected cities
        df = df.filter(col("City").isin(cities))
        
        # Get trend data
        trend_df = df.select("City", "Year", "AvgYearlyTemp").orderBy("City", "Year")
        
        # Convert to pandas for easier plotting
        trends = trend_df.toPandas()
        
        return trends
    
    except Exception as e:
        logging.error(f"Error analyzing temperature trends: {e}")
        return pd.DataFrame({"error": [str(e)]})

def weather_patterns_by_month(spark, stations=None, metrics=None):
    """
    Analyze seasonal weather patterns by month.
    
    Args:
        spark: SparkSession instance
        stations: List of stations to include (if None, use top stations)
        metrics: List of metrics to analyze (if None, use common metrics)
        
    Returns:
        Dictionary with pattern data by month
    """
    try:
        # Load data
        path = os.path.join(config.SILVER_PATH, "weather_clean")
        df = read_from_delta(spark, path)
        if df is None:
            return {"error": "Could not load weather dataset"}
        
        # Get all stations from column names if not specified
        all_stations = set()
        for col_name in df.columns:
            if "_" in col_name:
                station = col_name.split("_")[0]
                all_stations.add(station)
        
        if not stations:
            # Use first few stations
            stations = list(all_stations)[:3]
        else:
            # Filter to only include existing stations
            stations = [s for s in stations if s in all_stations]
        
        if not metrics:
            # Use some common metrics
            metrics = ["temp_mean", "humidity", "precipitation"]
        
        # Build analysis data by month
        monthly_patterns = {}
        
        for month in range(1, 13):
            month_df = df.filter(col("MONTH_NUM") == month)
            
            if month_df.count() == 0:
                continue
            
            month_data = {}
            
            for station in stations:
                station_data = {}
                
                for metric in metrics:
                    col_name = f"{station}_{metric}"
                    
                    if col_name in df.columns:
                        stats = month_df.select(
                            mean(col(col_name)).alias("mean"),
                            stddev(col(col_name)).alias("stddev"),
                            min(col(col_name)).alias("min"),
                            max(col(col_name)).alias("max")
                        ).first()
                        
                        if stats["mean"] is not None:
                            station_data[metric] = {
                                "mean": stats["mean"],
                                "stddev": stats["stddev"],
                                "min": stats["min"],
                                "max": stats["max"]
                            }
                
                if station_data:
                    month_data[station] = station_data
            
            if month_data:
                monthly_patterns[month] = month_data
        
        return monthly_patterns
    
    except Exception as e:
        logging.error(f"Error analyzing weather patterns: {e}")
        return {"error": str(e)}
