"""
Configuration settings for the W.A.R.P. project.
"""
import os
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, DateType

# Paths
WEATHER_DATA_PATH = "attached_assets/weather_prediction_dataset.csv"
TEMP_DATA_PATH = "attached_assets/BIG DATA GENERATED DATASET USED FOR ML.csv"

# Delta Lake paths
DELTA_LAKE_PATH = "delta_lake"
BRONZE_PATH = os.path.join(DELTA_LAKE_PATH, "bronze")
SILVER_PATH = os.path.join(DELTA_LAKE_PATH, "silver")
GOLD_PATH = os.path.join(DELTA_LAKE_PATH, "gold")

# Schema definitions
WEATHER_SCHEMA = StructType([
    StructField("DATE", StringType(), True),
    StructField("MONTH", IntegerType(), True),
    # Many other fields will be inferred from data
])

TEMP_SCHEMA = StructType([
    StructField("Country", StringType(), True),
    StructField("City", StringType(), True),
    StructField("Year", IntegerType(), True),
    StructField("AvgYearlyTemp", FloatType(), True),
    StructField("PrevTemp", FloatType(), True),
    StructField("TempChange", FloatType(), True)
])

# Models configuration
MODEL_SAVE_PATH = "models"

# Training parameters
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42
CV_FOLDS = 3

# Feature Engineering settings
LOOKBACK_DAYS = 7  # Number of days to look back for time series features
FORECAST_DAYS = 3  # Number of days to forecast
