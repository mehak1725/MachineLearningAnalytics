"""
Module for data cleaning and outlier handling.
"""
import logging
import os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, isnan, when, count, avg, stddev, min, max, udf, lag, lead
from pyspark.sql.types import DoubleType, BooleanType
from pyspark.ml.feature import Imputer, StandardScaler
from pyspark.ml.feature import VectorAssembler
import config
from utils.spark_utils import read_from_delta, write_to_delta

def identify_outliers(spark, dataset, method="zscore", threshold=3.0):
    """
    Identify outliers in a dataset.
    
    Args:
        spark: SparkSession instance
        dataset: Name of the dataset ('weather' or 'temperature')
        method: Method for outlier detection ('zscore' or 'iqr')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outlier flags
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
            return None
        
        # Get numeric columns
        numeric_cols = [col_name for col_name, dtype in df.dtypes 
                       if dtype in ('double', 'float', 'int', 'bigint', 'decimal')
                       and col_name not in id_cols]
        
        if not numeric_cols:
            return None
        
        # Create result dataframe with ID columns
        result_df = df.select(id_cols)
        
        # Calculate outliers for each numeric column
        if method == "zscore":
            # Z-score method
            for col_name in numeric_cols:
                # Calculate z-score
                stats = df.select(
                    avg(col(col_name)).alias("mean"),
                    stddev(col(col_name)).alias("stddev")
                ).first()
                
                if stats["stddev"] is not None and stats["stddev"] > 0:
                    # Create z-score UDF
                    def calc_zscore(value, mean, stddev):
                        if value is None:
                            return None
                        return abs((value - mean) / stddev)
                    
                    zscore_udf = udf(lambda x: calc_zscore(x, stats["mean"], stats["stddev"]), DoubleType())
                    
                    # Apply z-score and flag outliers
                    result_df = result_df.join(
                        df.select(
                            *id_cols,
                            when(
                                zscore_udf(col(col_name)) > threshold,
                                True
                            ).otherwise(False).alias(f"{col_name}_outlier")
                        ),
                        on=id_cols,
                        how="left"
                    )
        
        elif method == "iqr":
            # IQR method
            for col_name in numeric_cols:
                # Calculate quartiles
                quantiles = df.stat.approxQuantile(col_name, [0.25, 0.75], 0.01)
                
                if len(quantiles) == 2 and None not in quantiles:
                    q1, q3 = quantiles
                    iqr = q3 - q1
                    lower_bound = q1 - (threshold * iqr)
                    upper_bound = q3 + (threshold * iqr)
                    
                    # Flag outliers
                    result_df = result_df.join(
                        df.select(
                            *id_cols,
                            when(
                                (col(col_name) < lower_bound) | (col(col_name) > upper_bound),
                                True
                            ).otherwise(False).alias(f"{col_name}_outlier")
                        ),
                        on=id_cols,
                        how="left"
                    )
        
        return result_df
    
    except Exception as e:
        logging.error(f"Error identifying outliers: {e}")
        return None

def clean_dataset(spark, dataset, impute_method="mean", outlier_handling="remove", z_threshold=3.0):
    """
    Clean a dataset by handling missing values and outliers.
    
    Args:
        spark: SparkSession instance
        dataset: Name of the dataset ('weather' or 'temperature')
        impute_method: Method for imputation ('mean', 'median', or 'mode')
        outlier_handling: Strategy for outliers ('remove', 'impute', or 'cap')
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with cleaned data
    """
    if dataset == 'weather':
        path = os.path.join(config.SILVER_PATH, "weather_clean")
        id_cols = ["DATE", "DATE_PARSED", "YEAR", "MONTH", "MONTH_NUM", "DAY"]
        output_path = os.path.join(config.GOLD_PATH, "weather_cleaned")
    else:
        path = os.path.join(config.SILVER_PATH, "temperature_clean")
        id_cols = ["Country", "City", "Year"]
        output_path = os.path.join(config.GOLD_PATH, "temperature_cleaned")
    
    try:
        # Load data
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Get numeric columns
        numeric_cols = [col_name for col_name, dtype in df.dtypes 
                       if dtype in ('double', 'float', 'int', 'bigint', 'decimal')
                       and col_name not in id_cols]
        
        if not numeric_cols:
            return None
        
        # Handle outliers
        if outlier_handling != "keep":
            for col_name in numeric_cols:
                # Calculate statistics for outlier detection
                stats = df.select(
                    avg(col(col_name)).alias("mean"),
                    stddev(col(col_name)).alias("stddev")
                ).first()
                
                if stats["stddev"] is not None and stats["stddev"] > 0:
                    # Identify outliers using z-score
                    zscore = (col(col_name) - stats["mean"]) / stats["stddev"]
                    
                    if outlier_handling == "remove":
                        # Filter out rows with outliers
                        df = df.filter(abs(zscore) <= z_threshold)
                    
                    elif outlier_handling == "cap":
                        # Cap outliers at threshold values
                        upper_bound = stats["mean"] + (z_threshold * stats["stddev"])
                        lower_bound = stats["mean"] - (z_threshold * stats["stddev"])
                        
                        df = df.withColumn(
                            col_name,
                            when(col(col_name) > upper_bound, upper_bound)
                            .when(col(col_name) < lower_bound, lower_bound)
                            .otherwise(col(col_name))
                        )
                    
                    elif outlier_handling == "impute":
                        # Mark outliers as null for later imputation
                        df = df.withColumn(
                            col_name,
                            when(abs(zscore) > z_threshold, None)
                            .otherwise(col(col_name))
                        )
        
        # Handle missing values
        imputer = Imputer(
            inputCols=numeric_cols,
            outputCols=numeric_cols,
            strategy=impute_method
        )
        
        imputer_model = imputer.fit(df)
        df_imputed = imputer_model.transform(df)
        
        # Write results to Gold layer
        write_to_delta(df_imputed, output_path)
        
        return df_imputed
    
    except Exception as e:
        logging.error(f"Error cleaning dataset: {e}")
        return None

def standardize_features(spark, dataset):
    """
    Standardize numeric features to have mean 0 and unit variance.
    
    Args:
        spark: SparkSession instance
        dataset: Name of the dataset ('weather' or 'temperature')
        
    Returns:
        DataFrame with standardized features
    """
    if dataset == 'weather':
        path = os.path.join(config.GOLD_PATH, "weather_cleaned")
        id_cols = ["DATE", "DATE_PARSED", "YEAR", "MONTH", "MONTH_NUM", "DAY"]
        output_path = os.path.join(config.GOLD_PATH, "weather_standardized")
    else:
        path = os.path.join(config.GOLD_PATH, "temperature_cleaned")
        id_cols = ["Country", "City", "Year"]
        output_path = os.path.join(config.GOLD_PATH, "temperature_standardized")
    
    try:
        # Load data
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Get numeric columns
        numeric_cols = [col_name for col_name, dtype in df.dtypes 
                       if dtype in ('double', 'float', 'int', 'bigint', 'decimal')
                       and col_name not in id_cols]
        
        if not numeric_cols:
            return None
        
        # Assemble features vector
        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        df_assembled = assembler.transform(df)
        
        # Standardize features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaledFeatures",
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)
        
        # Write results to Gold layer
        write_to_delta(df_scaled, output_path)
        
        return df_scaled
    
    except Exception as e:
        logging.error(f"Error standardizing features: {e}")
        return None

def detect_anomalies(spark, dataset, columns=None, window_size=5):
    """
    Detect anomalies in time series data using moving window statistics.
    
    Args:
        spark: SparkSession instance
        dataset: Name of the dataset ('weather' or 'temperature')
        columns: List of columns to check for anomalies
        window_size: Size of moving window for anomaly detection
        
    Returns:
        DataFrame with anomaly flags
    """
    if dataset == 'weather':
        path = os.path.join(config.SILVER_PATH, "weather_clean")
        id_cols = ["DATE", "DATE_PARSED", "YEAR", "MONTH", "MONTH_NUM", "DAY"]
        date_col = "DATE_PARSED"
    else:
        path = os.path.join(config.SILVER_PATH, "temperature_clean")
        id_cols = ["Country", "City", "Year"]
        date_col = "Year"
    
    try:
        # Load data
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Get numeric columns if not specified
        if not columns:
            numeric_cols = [col_name for col_name, dtype in df.dtypes 
                           if dtype in ('double', 'float', 'int', 'bigint', 'decimal')
                           and col_name not in id_cols]
            
            # Limit to first few columns to avoid excessive computation
            columns = numeric_cols[:5]
        
        if not columns:
            return None
        
        # Create window for moving statistics
        if dataset == 'weather':
            # For weather data, use time-based window
            window = Window.orderBy(date_col).rangeBetween(-window_size, window_size)
        else:
            # For temperature data, use partition by city and then window by year
            window = Window.partitionBy("City").orderBy("Year").rangeBetween(-window_size, window_size)
        
        # Detect anomalies for each column
        result_df = df.select(id_cols)
        
        for col_name in columns:
            if col_name in df.columns:
                # Calculate moving window statistics
                stats_df = df.select(
                    *id_cols,
                    col(col_name),
                    avg(col(col_name)).over(window).alias(f"{col_name}_avg"),
                    stddev(col(col_name)).over(window).alias(f"{col_name}_stddev")
                )
                
                # Identify anomalies (values more than 2 stddev from moving average)
                anomaly_df = stats_df.select(
                    *id_cols,
                    when(
                        (col(f"{col_name}_stddev").isNotNull()) &
                        (abs(col(col_name) - col(f"{col_name}_avg")) > 2 * col(f"{col_name}_stddev")),
                        True
                    ).otherwise(False).alias(f"{col_name}_anomaly")
                )
                
                # Join with result
                result_df = result_df.join(anomaly_df, on=id_cols, how="left")
        
        return result_df
    
    except Exception as e:
        logging.error(f"Error detecting anomalies: {e}")
        return None
