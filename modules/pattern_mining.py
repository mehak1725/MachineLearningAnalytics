"""
Module for frequent pattern mining using PySpark MLlib.
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml.fpm import FPGrowth
import config
from utils.spark_utils import read_from_delta, write_to_delta

def prepare_pattern_data(spark, dataset_name, categorical_cols, min_support=0.1):
    """
    Prepare data for pattern mining by creating item arrays from categorical columns.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        categorical_cols: List of categorical column names
        min_support: Minimum support threshold
        
    Returns:
        DataFrame with items column for pattern mining
    """
    try:
        # Load dataset
        path = os.path.join(config.GOLD_PATH, dataset_name)
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Filter columns
        selected_cols = []
        for col_name in categorical_cols:
            if col_name in df.columns:
                selected_cols.append(col_name)
        
        if not selected_cols:
            return None
        
        # Create item array by combining categorical columns with values
        items_expr = array()
        
        for col_name in selected_cols:
            # Create string representation of column and value
            # e.g., "temp_regime=hot"
            item_expr = col(col_name).cast("string").alias(col_name)
            df = df.withColumn(col_name + "_item", col_name + "=" + item_expr)
            items_expr = items_expr.add(col(col_name + "_item"))
        
        # Create items column
        df = df.withColumn("items", items_expr)
        
        return df
    
    except Exception as e:
        logging.error(f"Error preparing pattern data: {e}")
        return None

def mine_frequent_patterns(spark, items_df, min_support=0.1, min_confidence=0.5):
    """
    Mine frequent patterns using FP-Growth algorithm.
    
    Args:
        spark: SparkSession instance
        items_df: DataFrame with items column
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary with frequent itemsets and association rules
    """
    try:
        if items_df is None or "items" not in items_df.columns:
            return None
        
        # Create FPGrowth model
        fpgrowth = FPGrowth(
            itemsCol="items",
            minSupport=min_support,
            minConfidence=min_confidence
        )
        
        # Train model
        model = fpgrowth.fit(items_df)
        
        # Get frequent itemsets
        frequent_itemsets = model.freqItemsets
        
        # Get association rules
        association_rules = model.associationRules
        
        # Convert to more readable format
        readable_itemsets = frequent_itemsets.select(
            "items",
            "freq",
            (col("freq") / items_df.count()).alias("support")
        ).orderBy("support", ascending=False)
        
        readable_rules = association_rules.select(
            "antecedent",
            "consequent",
            "confidence",
            "lift"
        ).orderBy("confidence", ascending=False)
        
        return {
            "model": model,
            "frequent_itemsets": readable_itemsets,
            "association_rules": readable_rules
        }
    
    except Exception as e:
        logging.error(f"Error mining frequent patterns: {e}")
        return None

def mine_weather_patterns(spark, min_support=0.05, min_confidence=0.3):
    """
    Mine patterns in weather data.
    
    Args:
        spark: SparkSession instance
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary with weather patterns
    """
    try:
        # Load weather features
        path = os.path.join(config.GOLD_PATH, "weather_features")
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Create categorical features for pattern mining
        from pyspark.sql.functions import when, col
        
        # Add more meaningful categorical columns
        # Temperature categories
        for station in ["BASEL", "DE_BILT", "BUDAPEST"]:
            temp_col = f"{station}_temp_mean"
            if temp_col in df.columns:
                df = df.withColumn(
                    f"{station}_temp_category",
                    when(col(temp_col) < 0, "freezing")
                    .when(col(temp_col) < 5, "very_cold")
                    .when(col(temp_col) < 10, "cold")
                    .when(col(temp_col) < 15, "mild")
                    .when(col(temp_col) < 20, "warm")
                    .when(col(temp_col) < 25, "hot")
                    .otherwise("very_hot")
                )
        
        # Humidity categories
        for station in ["BASEL", "DE_BILT", "BUDAPEST"]:
            humidity_col = f"{station}_humidity"
            if humidity_col in df.columns:
                df = df.withColumn(
                    f"{station}_humidity_category",
                    when(col(humidity_col) < 0.3, "very_dry")
                    .when(col(humidity_col) < 0.5, "dry")
                    .when(col(humidity_col) < 0.7, "moderate")
                    .when(col(humidity_col) < 0.8, "humid")
                    .otherwise("very_humid")
                )
        
        # Precipitation categories
        for station in ["BASEL", "DE_BILT", "BUDAPEST"]:
            precip_col = f"{station}_precipitation"
            if precip_col in df.columns:
                df = df.withColumn(
                    f"{station}_precip_category",
                    when(col(precip_col) == 0, "no_rain")
                    .when(col(precip_col) < 1, "light_rain")
                    .when(col(precip_col) < 5, "moderate_rain")
                    .otherwise("heavy_rain")
                )
        
        # Season categories
        df = df.withColumn(
            "season",
            when((col("MONTH_NUM") >= 3) & (col("MONTH_NUM") <= 5), "spring")
            .when((col("MONTH_NUM") >= 6) & (col("MONTH_NUM") <= 8), "summer")
            .when((col("MONTH_NUM") >= 9) & (col("MONTH_NUM") <= 11), "fall")
            .otherwise("winter")
        )
        
        # Get categorical columns
        categorical_cols = [c for c in df.columns if "_category" in c or c == "season"]
        
        # Prepare data for pattern mining
        items_df = prepare_pattern_data(spark, "weather_features", categorical_cols)
        
        if items_df is None:
            return None
        
        # Mine patterns
        patterns = mine_frequent_patterns(spark, items_df, min_support, min_confidence)
        
        if patterns is None:
            return None
        
        # Write patterns to gold layer
        write_to_delta(patterns["frequent_itemsets"], os.path.join(config.GOLD_PATH, "weather_patterns_itemsets"))
        write_to_delta(patterns["association_rules"], os.path.join(config.GOLD_PATH, "weather_patterns_rules"))
        
        return patterns
    
    except Exception as e:
        logging.error(f"Error mining weather patterns: {e}")
        return None

def mine_temperature_patterns(spark, min_support=0.05, min_confidence=0.3):
    """
    Mine patterns in temperature data.
    
    Args:
        spark: SparkSession instance
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary with temperature patterns
    """
    try:
        # Load temperature features
        path = os.path.join(config.GOLD_PATH, "temperature_features")
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Add more meaningful categorical columns
        from pyspark.sql.functions import when, col
        
        # Temperature regime (already exists in engineered features)
        if "temp_regime" not in df.columns:
            df = df.withColumn(
                "temp_regime",
                when(col("AvgYearlyTemp") < 0, "freezing")
                .when(col("AvgYearlyTemp") < 10, "cold")
                .when(col("AvgYearlyTemp") < 20, "moderate")
                .when(col("AvgYearlyTemp") < 25, "warm")
                .otherwise("hot")
            )
        
        # Temperature change categories
        df = df.withColumn(
            "temp_change_category",
            when(col("TempChange") < -1, "significant_cooling")
            .when(col("TempChange") < -0.2, "cooling")
            .when(col("TempChange") < 0.2, "stable")
            .when(col("TempChange") < 1, "warming")
            .otherwise("significant_warming")
        )
        
        # Decade categories
        df = df.withColumn(
            "decade",
            (col("Year") / 10).cast("int") * 10
        )
        
        # Get categorical columns
        categorical_cols = ["Country", "temp_regime", "temp_change_category", "decade", "trend"]
        
        # Prepare data for pattern mining
        items_df = prepare_pattern_data(spark, "temperature_features", categorical_cols)
        
        if items_df is None:
            return None
        
        # Mine patterns
        patterns = mine_frequent_patterns(spark, items_df, min_support, min_confidence)
        
        if patterns is None:
            return None
        
        # Write patterns to gold layer
        write_to_delta(patterns["frequent_itemsets"], os.path.join(config.GOLD_PATH, "temperature_patterns_itemsets"))
        write_to_delta(patterns["association_rules"], os.path.join(config.GOLD_PATH, "temperature_patterns_rules"))
        
        return patterns
    
    except Exception as e:
        logging.error(f"Error mining temperature patterns: {e}")
        return None
