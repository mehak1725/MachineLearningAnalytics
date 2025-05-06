"""
Module for feature engineering on weather datasets.
"""
import logging
import os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, when, lag, lead, datediff, date_format, 
    hour, dayofweek, month, year, expr, udf,
    sin, cos, to_date, lit, concat, unix_timestamp
)
from pyspark.sql.types import DoubleType, StringType, FloatType
from pyspark.ml.feature import SQLTransformer, Bucketizer, QuantileDiscretizer
import math
import config
from utils.spark_utils import read_from_delta, write_to_delta

def engineer_weather_features(spark):
    """
    Engineer features for the weather dataset.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        DataFrame with engineered features
    """
    try:
        # Load clean data
        path = os.path.join(config.SILVER_PATH, "weather_clean")
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # 1. Add date-based features
        df = df.withColumn("dayofweek", dayofweek(col("DATE_PARSED")))
        
        # 2. Add cyclical encoding for month and day of week
        # This preserves the cyclical nature of these features
        df = df.withColumn("month_sin", sin(col("MONTH_NUM") * (2.0 * math.pi/12)))
        df = df.withColumn("month_cos", cos(col("MONTH_NUM") * (2.0 * math.pi/12)))
        df = df.withColumn("dayofweek_sin", sin(col("dayofweek") * (2.0 * math.pi/7)))
        df = df.withColumn("dayofweek_cos", cos(col("dayofweek") * (2.0 * math.pi/7)))
        
        # 3. Create temperature difference features (daily temperature swing)
        # Find temperature columns
        temp_cols = [c for c in df.columns if "_temp_" in c]
        max_temp_cols = [c for c in temp_cols if "temp_max" in c]
        min_temp_cols = [c for c in temp_cols if "temp_min" in c]
        
        # For each station with both min and max, create swing feature
        for max_col in max_temp_cols:
            station = max_col.split("_")[0]
            min_col = f"{station}_temp_min"
            if min_col in df.columns:
                df = df.withColumn(
                    f"{station}_temp_swing", 
                    col(max_col) - col(min_col)
                )
        
        # 4. Create lagged features for time series analysis
        # Define window
        time_window = Window.orderBy("DATE_PARSED")
        
        # Create lags for important features
        lag_features = []
        for station in ["BASEL", "DE_BILT", "BUDAPEST"]:  # Example stations
            for feature in ["temp_mean", "humidity", "pressure"]:
                col_name = f"{station}_{feature}"
                if col_name in df.columns:
                    # Create lags of 1, 2, 3 days
                    for lag_days in range(1, 4):
                        lag_col_name = f"{col_name}_lag{lag_days}"
                        df = df.withColumn(lag_col_name, lag(col_name, lag_days).over(time_window))
                        lag_features.append(lag_col_name)
        
        # 5. Create interaction terms between important features
        # Example: temp × humidity interaction
        for station in ["BASEL", "DE_BILT", "BUDAPEST"]:  # Example stations
            temp_col = f"{station}_temp_mean"
            humidity_col = f"{station}_humidity"
            if temp_col in df.columns and humidity_col in df.columns:
                df = df.withColumn(
                    f"{station}_temp_humidity_interaction",
                    col(temp_col) * col(humidity_col)
                )
        
        # 6. Create precipitation indicators
        precip_cols = [c for c in df.columns if "_precipitation" in c]
        for precip_col in precip_cols:
            station = precip_col.split("_")[0]
            df = df.withColumn(
                f"{station}_has_rain",
                when(col(precip_col) > 0, 1).otherwise(0)
            )
        
        # 7. Create temperature buckets
        # Use quantile discretizer for adaptive binning
        for station in ["BASEL", "DE_BILT", "BUDAPEST"]:  # Example stations
            temp_col = f"{station}_temp_mean"
            if temp_col in df.columns:
                discretizer = QuantileDiscretizer(
                    numBuckets=5,
                    inputCol=temp_col,
                    outputCol=f"{station}_temp_bucket",
                    handleInvalid="skip"
                )
                
                df = discretizer.fit(df).transform(df)
        
        # Write results to gold layer
        output_path = os.path.join(config.GOLD_PATH, "weather_features")
        write_to_delta(df, output_path, partition_by="YEAR")
        
        return df
    
    except Exception as e:
        logging.error(f"Error engineering weather features: {e}")
        return None

def engineer_temperature_features(spark):
    """
    Engineer features for the temperature dataset.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        DataFrame with engineered features
    """
    try:
        # Load clean data
        path = os.path.join(config.SILVER_PATH, "temperature_clean")
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # 1. Create decade feature
        df = df.withColumn("Decade", (col("Year") / 10).cast("int") * 10)
        
        # 2. Create temperature change features
        # Window for time-based operations
        time_window = Window.partitionBy("Country", "City").orderBy("Year")
        
        # Add rolling average
        for window_size in [5, 10]:
            # Create rolling average over previous years
            avg_col = f"avg_temp_last_{window_size}years"
            df = df.withColumn(
                avg_col,
                expr(f"avg(AvgYearlyTemp) OVER (PARTITION BY Country, City ORDER BY Year ROWS BETWEEN {window_size} PRECEDING AND 1 PRECEDING)")
            )
            
            # Create difference from rolling average
            df = df.withColumn(
                f"diff_from_{window_size}yr_avg",
                col("AvgYearlyTemp") - col(avg_col)
            )
        
        # 3. Create trend indicators
        # Add lag for previous year not in dataset
        df = df.withColumn("prev_year_temp", lag("AvgYearlyTemp", 1).over(time_window))
        
        # Calculate year-over-year change
        df = df.withColumn(
            "yoy_change", 
            col("AvgYearlyTemp") - col("prev_year_temp")
        )
        
        # Create trend indicator
        df = df.withColumn(
            "trend", 
            when(col("yoy_change") > 0.1, "warming")
            .when(col("yoy_change") < -0.1, "cooling")
            .otherwise("stable")
        )
        
        # 4. Create change acceleration
        df = df.withColumn("prev_yoy_change", lag("yoy_change", 1).over(time_window))
        df = df.withColumn(
            "temp_acceleration", 
            col("yoy_change") - col("prev_yoy_change")
        )
        
        # 5. Create relative temperature features
        # Get global average for each year
        global_avg = df.groupBy("Year").agg({"AvgYearlyTemp": "avg"}).withColumnRenamed("avg(AvgYearlyTemp)", "global_avg_temp")
        
        # Join global average
        df = df.join(global_avg, on="Year", how="left")
        
        # Calculate deviation from global average
        df = df.withColumn(
            "deviation_from_global_avg", 
            col("AvgYearlyTemp") - col("global_avg_temp")
        )
        
        # 6. Create categorical features
        # Temperature regime categorization
        df = df.withColumn(
            "temp_regime",
            when(col("AvgYearlyTemp") < 0, "freezing")
            .when(col("AvgYearlyTemp") < 10, "cold")
            .when(col("AvgYearlyTemp") < 20, "moderate")
            .when(col("AvgYearlyTemp") < 25, "warm")
            .otherwise("hot")
        )
        
        # 7. Create interaction features
        # City-specific year interaction (to capture city-specific trends)
        # First encode city as numeric
        cities = df.select("City").distinct().collect()
        city_mapping = {city.City: idx for idx, city in enumerate(cities)}
        
        # Create city index UDF
        city_index_udf = udf(lambda city: float(city_mapping.get(city, 0)), FloatType())
        df = df.withColumn("city_index", city_index_udf(col("City")))
        
        # Create interaction term
        df = df.withColumn(
            "city_year_interaction", 
            col("city_index") * (col("Year") - 1700)  # Normalize year
        )
        
        # Write results to gold layer
        output_path = os.path.join(config.GOLD_PATH, "temperature_features")
        write_to_delta(df, output_path, partition_by="Country")
        
        return df
    
    except Exception as e:
        logging.error(f"Error engineering temperature features: {e}")
        return None

def create_prediction_datasets(spark):
    """
    Create prediction-ready datasets with features and targets.
    
    Args:
        spark: SparkSession instance
        
    Returns:
        Dictionary with prediction datasets
    """
    try:
        results = {}
        
        # 1. Temperature prediction dataset
        # Load weather features
        weather_path = os.path.join(config.GOLD_PATH, "weather_features")
        weather_df = read_from_delta(spark, weather_path)
        
        if weather_df is not None:
            # Choose features and target for temperature prediction
            target_cols = []
            feature_cols = []
            
            # Get all temp_mean columns as potential targets
            for col_name in weather_df.columns:
                if "temp_mean" in col_name and "_lag" not in col_name:
                    target_cols.append(col_name)
                    
                # Get features that aren't targets or their direct derivatives
                if (
                    ("_lag" in col_name or "_humidity" in col_name or "_pressure" in col_name) and
                    "temp_mean" not in col_name
                ):
                    feature_cols.append(col_name)
            
            # Add time features
            feature_cols.extend(["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"])
            
            # Create dataset for each target
            for target in target_cols[:3]:  # Limit to first few to avoid excessive computation
                station = target.split("_")[0]
                
                # Get relevant features for this station
                station_features = [f for f in feature_cols if station in f or "_sin" in f or "_cos" in f]
                
                if station_features:
                    # Select columns
                    pred_df = weather_df.select(
                        ["DATE_PARSED", "YEAR", "MONTH_NUM", "DAY"] + 
                        station_features + 
                        [target]
                    )
                    
                    # Store in results
                    results[f"{station}_temp_prediction"] = pred_df
        
        # 2. Rain prediction dataset (classification)
        if weather_df is not None:
            # Get all precipitation columns
            precip_cols = [col_name for col_name in weather_df.columns if "_precipitation" in col_name]
            
            for precip_col in precip_cols[:3]:  # Limit to first few
                station = precip_col.split("_")[0]
                
                # Create binary target
                rain_df = weather_df.withColumn(
                    f"{station}_will_rain",
                    when(col(precip_col) > 0, 1).otherwise(0)
                )
                
                # Select features (similar to temperature but add humidity and pressure)
                station_features = [
                    f for f in weather_df.columns 
                    if (station in f and 
                        "precipitation" not in f and 
                        "will_rain" not in f) or 
                       "_sin" in f or 
                       "_cos" in f
                ]
                
                if station_features:
                    # Select columns
                    pred_df = rain_df.select(
                        ["DATE_PARSED", "YEAR", "MONTH_NUM", "DAY"] + 
                        station_features + 
                        [f"{station}_will_rain"]
                    )
                    
                    # Store in results
                    results[f"{station}_rain_prediction"] = pred_df
        
        # 3. Long-term temperature trend dataset
        temp_path = os.path.join(config.GOLD_PATH, "temperature_features")
        temp_df = read_from_delta(spark, temp_path)
        
        if temp_df is not None:
            # Select features for temperature trend prediction
            feature_cols = [
                "City", "Country", "Year", "Decade", 
                "avg_temp_last_5years", "avg_temp_last_10years",
                "yoy_change", "temp_acceleration", "global_avg_temp"
            ]
            
            # Filter to columns that exist
            feature_cols = [col_name for col_name in feature_cols if col_name in temp_df.columns]
            
            if feature_cols:
                # Select columns
                trend_df = temp_df.select(
                    feature_cols + ["AvgYearlyTemp"]
                )
                
                # Store in results
                results["long_term_temp_prediction"] = trend_df
        
        # Write prediction datasets to gold layer
        for name, df in results.items():
            output_path = os.path.join(config.GOLD_PATH, name)
            write_to_delta(df, output_path)
        
        return results
    
    except Exception as e:
        logging.error(f"Error creating prediction datasets: {e}")
        return {}
