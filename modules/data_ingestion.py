"""
Module for data ingestion and Delta Lake integration.
"""
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan
import config
from utils.spark_utils import write_to_delta, read_from_delta

def ingest_and_profile_data(spark, data_df, dataset_name):
    """
    Ingest data and create a data profile report.
    
    Args:
        spark: SparkSession instance
        data_df: DataFrame to profile
        dataset_name: Name of the dataset for report identification
        
    Returns:
        Dictionary with data profile information
    """
    if data_df is None:
        return {"error": "No data provided"}
    
    # Get basic statistics
    row_count = data_df.count()
    column_count = len(data_df.columns)
    
    # Analyze column types
    column_types = {col_name: str(dtype) for col_name, dtype in data_df.dtypes}
    
    # Calculate missing values
    null_counts = {}
    for col_name in data_df.columns:
        null_count = data_df.filter(
            col(col_name).isNull() | 
            isnan(col(col_name)) | 
            (col(col_name) == "")
        ).count()
        null_counts[col_name] = null_count
    
    # Get sample values for each column
    sample_values = {}
    for col_name in data_df.columns:
        sample = data_df.select(col_name).limit(5).rdd.flatMap(lambda x: x).collect()
        sample_values[col_name] = sample
    
    # Compile profile
    profile = {
        "dataset_name": dataset_name,
        "row_count": row_count,
        "column_count": column_count,
        "column_types": column_types,
        "null_counts": null_counts,
        "sample_values": sample_values
    }
    
    return profile

def prepare_delta_tables(spark):
    """
    Prepare Delta Lake tables for the medallion architecture.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        Dictionary with Delta Lake tables information
    """
    # Create directories if they don't exist
    os.makedirs(config.BRONZE_PATH, exist_ok=True)
    os.makedirs(config.SILVER_PATH, exist_ok=True)
    os.makedirs(config.GOLD_PATH, exist_ok=True)
    
    # Check if Delta tables exist
    bronze_tables = []
    silver_tables = []
    gold_tables = []
    
    # Check for bronze tables
    try:
        bronze_weather_exists = os.path.exists(os.path.join(config.BRONZE_PATH, "weather_data"))
        bronze_temp_exists = os.path.exists(os.path.join(config.BRONZE_PATH, "temperature_data"))
        
        if bronze_weather_exists:
            bronze_tables.append("weather_data")
        if bronze_temp_exists:
            bronze_tables.append("temperature_data")
    except Exception as e:
        logging.error(f"Error checking bronze tables: {e}")
    
    return {
        "bronze_tables": bronze_tables,
        "silver_tables": silver_tables,
        "gold_tables": gold_tables
    }

def create_silver_tables(spark):
    """
    Create Silver layer Delta Tables with cleaned and processed data.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        Boolean indicating success
    """
    try:
        # Load bronze data
        weather_bronze = read_from_delta(spark, os.path.join(config.BRONZE_PATH, "weather_data"))
        temp_bronze = read_from_delta(spark, os.path.join(config.BRONZE_PATH, "temperature_data"))
        
        if weather_bronze is None or temp_bronze is None:
            return False
        
        # Process weather data - filter out rows with too many nulls
        weather_columns = [col for col in weather_bronze.columns if col not in ["DATE", "DATE_PARSED"]]
        weather_silver = weather_bronze.dropna(thresh=len(weather_columns) - 10)  # Allow some missing values
        
        # Process temperature data - remove rows with missing years or temps
        temp_silver = temp_bronze.dropna(subset=["Year", "AvgYearlyTemp"])
        
        # Write to silver layer
        write_to_delta(weather_silver, os.path.join(config.SILVER_PATH, "weather_clean"), partition_by="YEAR")
        write_to_delta(temp_silver, os.path.join(config.SILVER_PATH, "temperature_clean"), partition_by="Country")
        
        return True
    
    except Exception as e:
        logging.error(f"Error creating silver tables: {e}")
        return False

def get_delta_tables_info(spark):
    """
    Get information about existing Delta tables.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        Dictionary with table information
    """
    delta_info = {
        "bronze": [],
        "silver": [],
        "gold": []
    }
    
    # Check Bronze tables
    bronze_paths = [
        {"name": "weather_data", "path": os.path.join(config.BRONZE_PATH, "weather_data")},
        {"name": "temperature_data", "path": os.path.join(config.BRONZE_PATH, "temperature_data")}
    ]
    
    for table in bronze_paths:
        try:
            if os.path.exists(table["path"]):
                df = read_from_delta(spark, table["path"])
                if df is not None:
                    row_count = df.count()
                    delta_info["bronze"].append({
                        "name": table["name"],
                        "rows": row_count,
                        "columns": len(df.columns)
                    })
        except Exception as e:
            logging.error(f"Error checking bronze table {table['name']}: {e}")
    
    # Check Silver tables
    silver_paths = [
        {"name": "weather_clean", "path": os.path.join(config.SILVER_PATH, "weather_clean")},
        {"name": "temperature_clean", "path": os.path.join(config.SILVER_PATH, "temperature_clean")}
    ]
    
    for table in silver_paths:
        try:
            if os.path.exists(table["path"]):
                df = read_from_delta(spark, table["path"])
                if df is not None:
                    row_count = df.count()
                    delta_info["silver"].append({
                        "name": table["name"],
                        "rows": row_count,
                        "columns": len(df.columns)
                    })
        except Exception as e:
            logging.error(f"Error checking silver table {table['name']}: {e}")
    
    # Check Gold tables (to be created by other modules)
    gold_paths = [
        {"name": "weather_features", "path": os.path.join(config.GOLD_PATH, "weather_features")},
        {"name": "temperature_predictions", "path": os.path.join(config.GOLD_PATH, "temperature_predictions")},
        {"name": "weather_clusters", "path": os.path.join(config.GOLD_PATH, "weather_clusters")}
    ]
    
    for table in gold_paths:
        try:
            if os.path.exists(table["path"]):
                df = read_from_delta(spark, table["path"])
                if df is not None:
                    row_count = df.count()
                    delta_info["gold"].append({
                        "name": table["name"],
                        "rows": row_count,
                        "columns": len(df.columns)
                    })
        except Exception as e:
            logging.error(f"Error checking gold table {table['name']}: {e}")
    
    return delta_info
