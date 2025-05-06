"""
Data loading utilities for the W.A.R.P. project.
"""
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, year, dayofmonth, hour
import config
from utils.spark_utils import convert_date_udf, register_temp_view, write_to_delta

def load_weather_data(spark):
    """
    Load and parse the weather prediction dataset.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        DataFrame with parsed weather data
    """
    try:
        # Load data
        df = spark.read.option("header", "true").csv(config.WEATHER_DATA_PATH)
        
        # Convert date string to date type
        convert_date = convert_date_udf()
        df = df.withColumn("DATE_PARSED", convert_date(col("DATE")))
        
        # Extract date components
        df = df.withColumn("YEAR", year(col("DATE_PARSED"))) \
               .withColumn("MONTH_NUM", month(col("DATE_PARSED"))) \
               .withColumn("DAY", dayofmonth(col("DATE_PARSED")))
        
        # Convert numeric columns to double
        numeric_cols = [col_name for col_name in df.columns 
                        if col_name not in ["DATE", "DATE_PARSED", "MONTH", "YEAR", "MONTH_NUM", "DAY"]]
        
        for col_name in numeric_cols:
            df = df.withColumn(col_name, col(col_name).cast("double"))
        
        # Register as temp view for SQL queries
        register_temp_view(df, "weather_raw")
        
        # Write to Delta Lake Bronze layer
        write_to_delta(df, os.path.join(config.BRONZE_PATH, "weather_data"), partition_by="YEAR")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading weather data: {e}")
        return None

def load_temperature_data(spark):
    """
    Load and parse the historical temperature dataset.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        DataFrame with parsed temperature data
    """
    try:
        # Load data
        df = spark.read.option("header", "true").csv(config.TEMP_DATA_PATH)
        
        # Convert numeric columns to appropriate types
        df = df.withColumn("Year", col("Year").cast("integer")) \
               .withColumn("AvgYearlyTemp", col("AvgYearlyTemp").cast("double")) \
               .withColumn("PrevTemp", col("PrevTemp").cast("double")) \
               .withColumn("TempChange", col("TempChange").cast("double"))
        
        # Register as temp view for SQL queries
        register_temp_view(df, "temperature_raw")
        
        # Write to Delta Lake Bronze layer
        write_to_delta(df, os.path.join(config.BRONZE_PATH, "temperature_data"), partition_by="Country")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading temperature data: {e}")
        return None

def get_cities(spark):
    """Get unique list of cities from the temperature dataset."""
    try:
        temp_df = spark.read.format("delta").load(os.path.join(config.BRONZE_PATH, "temperature_data"))
        cities = temp_df.select("City").distinct().rdd.flatMap(lambda x: x).collect()
        return sorted(cities)
    except Exception as e:
        logging.error(f"Error getting cities: {e}")
        # Fall back to reading from CSV
        try:
            temp_df = spark.read.option("header", "true").csv(config.TEMP_DATA_PATH)
            cities = temp_df.select("City").distinct().rdd.flatMap(lambda x: x).collect()
            return sorted(cities)
        except:
            return ["No cities found"]

def get_weather_stations(spark):
    """Get unique list of weather stations from the weather dataset."""
    try:
        weather_df = spark.read.format("delta").load(os.path.join(config.BRONZE_PATH, "weather_data"))
        # Extract station names from column names (they are prefixed with station name and underscore)
        columns = weather_df.columns
        stations = set()
        for col in columns:
            if "_" in col:
                station = col.split("_")[0]
                stations.add(station)
        return sorted(list(stations))
    except Exception as e:
        logging.error(f"Error getting weather stations: {e}")
        return ["No stations found"]
