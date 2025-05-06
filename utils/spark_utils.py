"""
Utilities for PySpark session management and common operations.
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType, DateType
from pyspark import SparkConf
import datetime
import logging

def create_spark_session(app_name="WARP_Weather_Analytics"):
    """Create and configure a Spark session."""
    conf = SparkConf()
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.sql.shuffle.partitions", "4")  # For local development
    conf.set("spark.driver.memory", "4g")
    
    # Create session
    spark = SparkSession.builder \
        .appName(app_name) \
        .config(conf=conf) \
        .getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def stop_spark_session(spark):
    """Stop the Spark session."""
    if spark:
        spark.stop()

def convert_date_udf():
    """UDF for converting date strings to date objects."""
    def convert_date(date_str):
        if date_str is None:
            return None
        try:
            # Assuming format YYYYMMDD
            return datetime.datetime.strptime(date_str, "%Y%m%d").date()
        except:
            return None
    
    return udf(convert_date, DateType())

def register_temp_view(df, name):
    """Register DataFrame as a temp view for SQL queries."""
    if df is not None:
        df.createOrReplaceTempView(name)
        return True
    return False

def convert_to_pandas(spark_df, limit=10000):
    """
    Convert Spark DataFrame to Pandas DataFrame with a limit to prevent OOM errors.
    For visualization purposes only.
    """
    if spark_df is None:
        return None
    
    # Apply limit and convert
    return spark_df.limit(limit).toPandas()

def write_to_delta(df, path, mode="overwrite", partition_by=None):
    """Write DataFrame to Delta Lake format."""
    try:
        if partition_by:
            df.write.format("delta").mode(mode).partitionBy(partition_by).save(path)
        else:
            df.write.format("delta").mode(mode).save(path)
        return True
    except Exception as e:
        logging.error(f"Error writing to Delta Lake: {e}")
        return False

def read_from_delta(spark, path):
    """Read DataFrame from Delta Lake format."""
    try:
        return spark.read.format("delta").load(path)
    except Exception as e:
        logging.error(f"Error reading from Delta Lake: {e}")
        return None
