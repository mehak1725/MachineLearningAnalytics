"""
Module for clustering models using PySpark MLlib.
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
import config
from utils.spark_utils import read_from_delta, write_to_delta

def prepare_clustering_data(spark, dataset_name, feature_cols):
    """
    Prepare data for clustering models.
    
    Args:
        spark: SparkSession instance
        dataset_name: Name of the dataset in gold layer
        feature_cols: List of feature column names
        
    Returns:
        Dictionary with assembled and scaled DataFrames
    """
    try:
        # Load dataset
        path = os.path.join(config.GOLD_PATH, dataset_name)
        df = read_from_delta(spark, path)
        if df is None:
            return None
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        # Transform data
        assembled_df = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaledFeatures",
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(assembled_df)
        scaled_df = scaler_model.transform(assembled_df)
        
        return {
            "assembled_df": assembled_df,
            "scaled_df": scaled_df,
            "feature_assembler": assembler,
            "feature_scaler": scaler_model
        }
    
    except Exception as e:
        logging.error(f"Error preparing clustering data: {e}")
        return None

def train_kmeans(prepared_data, k=3, use_scaled=True):
    """
    Train a KMeans clustering model.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_clustering_data
        k: Number of clusters
        use_scaled: Whether to use scaled features
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        # Choose input data based on whether to use scaled features
        if use_scaled:
            df = prepared_data["scaled_df"]
            features_col = "scaledFeatures"
        else:
            df = prepared_data["assembled_df"]
            features_col = "features"
        
        # Create model
        kmeans = KMeans(
            k=k,
            featuresCol=features_col,
            predictionCol="cluster",
            seed=config.RANDOM_SEED,
            initMode="k-means||",
            initSteps=5,
            tol=1e-4,
            maxIter=20
        )
        
        # Train model
        model = kmeans.fit(df)
        
        # Make predictions
        predictions = model.transform(df)
        
        # Evaluate model
        evaluator = ClusteringEvaluator(
            featuresCol=features_col,
            predictionCol="cluster",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        silhouette = evaluator.evaluate(predictions)
        
        metrics = {
            "silhouette": silhouette,
            "k": k,
            "cost": model.summary.trainingCost
        }
        
        # Get cluster centers
        centers = model.clusterCenters()
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
            "centers": centers
        }
    
    except Exception as e:
        logging.error(f"Error training KMeans: {e}")
        return None

def train_bisecting_kmeans(prepared_data, k=3, use_scaled=True):
    """
    Train a Bisecting KMeans clustering model.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_clustering_data
        k: Number of clusters
        use_scaled: Whether to use scaled features
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        # Choose input data based on whether to use scaled features
        if use_scaled:
            df = prepared_data["scaled_df"]
            features_col = "scaledFeatures"
        else:
            df = prepared_data["assembled_df"]
            features_col = "features"
        
        # Create model
        bkmeans = BisectingKMeans(
            k=k,
            featuresCol=features_col,
            predictionCol="cluster",
            seed=config.RANDOM_SEED,
            maxIter=20
        )
        
        # Train model
        model = bkmeans.fit(df)
        
        # Make predictions
        predictions = model.transform(df)
        
        # Evaluate model
        evaluator = ClusteringEvaluator(
            featuresCol=features_col,
            predictionCol="cluster",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        silhouette = evaluator.evaluate(predictions)
        
        metrics = {
            "silhouette": silhouette,
            "k": k,
            "cost": model.summary.trainingCost
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training Bisecting KMeans: {e}")
        return None

def train_gaussian_mixture(prepared_data, k=3, use_scaled=True):
    """
    Train a Gaussian Mixture clustering model.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_clustering_data
        k: Number of clusters
        use_scaled: Whether to use scaled features
        
    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    try:
        # Choose input data based on whether to use scaled features
        if use_scaled:
            df = prepared_data["scaled_df"]
            features_col = "scaledFeatures"
        else:
            df = prepared_data["assembled_df"]
            features_col = "features"
        
        # Create model
        gmm = GaussianMixture(
            k=k,
            featuresCol=features_col,
            predictionCol="cluster",
            seed=config.RANDOM_SEED,
            maxIter=20
        )
        
        # Train model
        model = gmm.fit(df)
        
        # Make predictions
        predictions = model.transform(df)
        
        # Evaluate model
        evaluator = ClusteringEvaluator(
            featuresCol=features_col,
            predictionCol="cluster",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        silhouette = evaluator.evaluate(predictions)
        
        metrics = {
            "silhouette": silhouette,
            "k": k
        }
        
        return {
            "model": model,
            "predictions": predictions,
            "metrics": metrics
        }
    
    except Exception as e:
        logging.error(f"Error training Gaussian Mixture: {e}")
        return None

def find_optimal_k(prepared_data, method="kmeans", k_range=(2, 10), use_scaled=True):
    """
    Find the optimal number of clusters.
    
    Args:
        prepared_data: Dictionary with prepared data from prepare_clustering_data
        method: Clustering method ('kmeans', 'bisecting_kmeans', or 'gmm')
        k_range: Range of k values to try
        use_scaled: Whether to use scaled features
        
    Returns:
        Dictionary with optimal k and evaluation metrics
    """
    try:
        results = []
        
        for k in range(k_range[0], k_range[1] + 1):
            if method == "kmeans":
                result = train_kmeans(prepared_data, k, use_scaled)
            elif method == "bisecting_kmeans":
                result = train_bisecting_kmeans(prepared_data, k, use_scaled)
            elif method == "gmm":
                result = train_gaussian_mixture(prepared_data, k, use_scaled)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            if result:
                results.append((k, result["metrics"]["silhouette"]))
        
        # Find optimal k
        if results:
            optimal_k, best_silhouette = max(results, key=lambda x: x[1])
            
            return {
                "optimal_k": optimal_k,
                "silhouette": best_silhouette,
                "all_results": results
            }
        else:
            return None
    
    except Exception as e:
        logging.error(f"Error finding optimal k: {e}")
        return None

def analyze_clusters(clustering_result, original_df, feature_cols, id_cols):
    """
    Analyze the characteristics of each cluster.
    
    Args:
        clustering_result: Dictionary with clustering model and predictions
        original_df: Original DataFrame with all columns
        feature_cols: List of feature column names used for clustering
        id_cols: List of ID column names
        
    Returns:
        Dictionary with cluster analysis
    """
    try:
        predictions = clustering_result["predictions"]
        
        # Join predictions with original data
        if original_df is None:
            df = predictions
        else:
            # Get common ID columns
            common_cols = [col_name for col_name in id_cols if col_name in predictions.columns]
            
            if not common_cols:
                return None
            
            # Join on common ID columns
            df = predictions.join(original_df, on=common_cols, how="inner")
        
        # Analyze each cluster
        num_clusters = int(df.select("cluster").rdd.flatMap(lambda x: x).max()) + 1
        cluster_stats = {}
        
        for i in range(num_clusters):
            # Filter to current cluster
            cluster_df = df.filter(col("cluster") == i)
            
            # Calculate statistics for each feature
            feature_stats = {}
            
            for feature in feature_cols:
                if feature in df.columns:
                    stats = cluster_df.select(feature).summary().collect()
                    feature_stats[feature] = {
                        "count": float(stats[0][1]),
                        "mean": float(stats[1][1]),
                        "stddev": float(stats[2][1]),
                        "min": float(stats[3][1]),
                        "max": float(stats[7][1])
                    }
            
            # Store cluster statistics
            cluster_stats[f"cluster_{i}"] = {
                "size": cluster_df.count(),
                "percentage": (cluster_df.count() / df.count()) * 100,
                "feature_stats": feature_stats
            }
        
        return cluster_stats
    
    except Exception as e:
        logging.error(f"Error analyzing clusters: {e}")
        return None
