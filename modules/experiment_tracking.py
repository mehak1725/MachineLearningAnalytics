"""
Module for experiment tracking using MLflow.
"""
import logging
import os
import json
import pandas as pd
from datetime import datetime
import config
from utils.spark_utils import write_to_delta

# MLflow is optional in this implementation
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Falling back to basic experiment tracking.")

class ExperimentTracker:
    """A class for tracking ML experiments with or without MLflow."""
    
    def __init__(self, experiment_name="warp_experiments", use_mlflow=True):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            use_mlflow: Whether to use MLflow if available
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        # Create experiments directory if it doesn't exist
        self.experiments_dir = os.path.join("experiments")
        os.makedirs(self.experiments_dir, exist_ok=True)
        
        # Set up MLflow if available and requested
        if self.use_mlflow:
            # Set tracking URI to local file system
            mlflow_dir = os.path.join(self.experiments_dir, "mlflow")
            mlflow.set_tracking_uri(f"file:{mlflow_dir}")
            
            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=os.path.join(mlflow_dir, self.experiment_name)
                )
            except:
                self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
                
            mlflow.set_experiment(self.experiment_name)
            
            self.client = MlflowClient()
            logging.info(f"MLflow experiment '{self.experiment_name}' set up with ID {self.experiment_id}")
        else:
            # Simple tracking without MLflow
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = os.path.join(self.experiments_dir, self.experiment_name)
            os.makedirs(self.experiment_dir, exist_ok=True)
            logging.info(f"Simple experiment tracking set up in {self.experiment_dir}")
    
    def start_run(self, run_name=None):
        """
        Start a new run.
        
        Args:
            run_name: Name of the run
            
        Returns:
            Run ID
        """
        if self.use_mlflow:
            # Start MLflow run
            mlflow.start_run(run_name=run_name)
            run_id = mlflow.active_run().info.run_id
            logging.info(f"Started MLflow run '{run_name}' with ID {run_id}")
        else:
            # Simple run
            if run_name is None:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            run_id = run_name
            run_dir = os.path.join(self.experiment_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            # Store run info
            run_info = {
                "run_id": run_id,
                "start_time": datetime.now().isoformat(),
                "status": "RUNNING"
            }
            
            with open(os.path.join(run_dir, "run_info.json"), 'w') as f:
                json.dump(run_info, f, indent=2)
                
            logging.info(f"Started simple run '{run_name}' in {run_dir}")
        
        return run_id
    
    def log_param(self, key, value):
        """
        Log a parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.use_mlflow:
            mlflow.log_param(key, value)
        else:
            # Get current run
            active_run = self._get_active_run()
            if active_run:
                run_dir = os.path.join(self.experiment_dir, active_run)
                
                # Load existing params or create new
                params_file = os.path.join(run_dir, "params.json")
                if os.path.exists(params_file):
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                else:
                    params = {}
                
                # Add new param
                params[key] = value
                
                # Save params
                with open(params_file, 'w') as f:
                    json.dump(params, f, indent=2)
    
    def log_metric(self, key, value, step=None):
        """
        Log a metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step value
        """
        if self.use_mlflow:
            mlflow.log_metric(key, value, step=step)
        else:
            # Get current run
            active_run = self._get_active_run()
            if active_run:
                run_dir = os.path.join(self.experiment_dir, active_run)
                
                # Load existing metrics or create new
                metrics_file = os.path.join(run_dir, "metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                
                # Add new metric
                if key not in metrics:
                    metrics[key] = []
                    
                metric_value = {
                    "value": value,
                    "step": step,
                    "timestamp": datetime.now().isoformat()
                }
                
                metrics[key].append(metric_value)
                
                # Save metrics
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
    
    def log_model(self, model, model_name, flavor=None, input_example=None):
        """
        Log a model.
        
        Args:
            model: Model to log
            model_name: Name of the model
            flavor: Model flavor for MLflow
            input_example: Example input for model signature
        """
        if self.use_mlflow:
            if flavor == "spark":
                mlflow.spark.log_model(model, model_name)
            else:
                # Generic logging
                mlflow.pyfunc.log_model(model, model_name)
        else:
            # Get current run
            active_run = self._get_active_run()
            if active_run:
                run_dir = os.path.join(self.experiment_dir, active_run)
                models_dir = os.path.join(run_dir, "models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Save model info
                model_info = {
                    "model_name": model_name,
                    "model_type": str(type(model)),
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(os.path.join(models_dir, f"{model_name}_info.json"), 'w') as f:
                    json.dump(model_info, f, indent=2)
                    
                # For Spark models, save to a directory
                try:
                    model_path = os.path.join(models_dir, model_name)
                    model.write().overwrite().save(model_path)
                    logging.info(f"Model '{model_name}' saved to {model_path}")
                except Exception as e:
                    logging.error(f"Could not save model: {e}")
    
    def log_artifacts(self, artifacts_dir):
        """
        Log artifacts from a directory.
        
        Args:
            artifacts_dir: Directory containing artifacts
        """
        if self.use_mlflow:
            mlflow.log_artifacts(artifacts_dir)
        else:
            # Get current run
            active_run = self._get_active_run()
            if active_run:
                run_dir = os.path.join(self.experiment_dir, active_run)
                artifacts_dest = os.path.join(run_dir, "artifacts")
                os.makedirs(artifacts_dest, exist_ok=True)
                
                # Copy artifacts
                import shutil
                for item in os.listdir(artifacts_dir):
                    source = os.path.join(artifacts_dir, item)
                    dest = os.path.join(artifacts_dest, item)
                    
                    if os.path.isdir(source):
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source, dest)
    
    def end_run(self, status="FINISHED"):
        """
        End the current run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', etc.)
        """
        if self.use_mlflow:
            mlflow.end_run(status=status)
        else:
            # Get current run
            active_run = self._get_active_run()
            if active_run:
                run_dir = os.path.join(self.experiment_dir, active_run)
                
                # Update run info
                run_info_file = os.path.join(run_dir, "run_info.json")
                if os.path.exists(run_info_file):
                    with open(run_info_file, 'r') as f:
                        run_info = json.load(f)
                    
                    run_info["end_time"] = datetime.now().isoformat()
                    run_info["status"] = status
                    
                    with open(run_info_file, 'w') as f:
                        json.dump(run_info, f, indent=2)
    
    def get_run_info(self, run_id):
        """
        Get information about a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Dictionary with run information
        """
        if self.use_mlflow:
            run = self.client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
                "params": run.data.params,
                "metrics": run.data.metrics
            }
        else:
            run_dir = os.path.join(self.experiment_dir, run_id)
            
            if not os.path.exists(run_dir):
                return None
            
            # Load run info
            run_info_file = os.path.join(run_dir, "run_info.json")
            if os.path.exists(run_info_file):
                with open(run_info_file, 'r') as f:
                    run_info = json.load(f)
            else:
                run_info = {
                    "run_id": run_id,
                    "status": "UNKNOWN"
                }
            
            # Load params
            params_file = os.path.join(run_dir, "params.json")
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = json.load(f)
            else:
                params = {}
            
            # Load metrics
            metrics_file = os.path.join(run_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                # Extract last value for each metric
                last_metrics = {}
                for key, values in metrics.items():
                    if values:
                        last_metrics[key] = values[-1]["value"]
            else:
                last_metrics = {}
            
            return {
                **run_info,
                "params": params,
                "metrics": last_metrics
            }
    
    def list_runs(self):
        """
        List all runs in the experiment.
        
        Returns:
            List of run information
        """
        if self.use_mlflow:
            runs = self.client.search_runs(self.experiment_id)
            return [{
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status
            } for run in runs]
        else:
            runs = []
            if os.path.exists(self.experiment_dir):
                for run_id in os.listdir(self.experiment_dir):
                    run_dir = os.path.join(self.experiment_dir, run_id)
                    if os.path.isdir(run_dir):
                        # Load run info
                        run_info_file = os.path.join(run_dir, "run_info.json")
                        if os.path.exists(run_info_file):
                            with open(run_info_file, 'r') as f:
                                run_info = json.load(f)
                                runs.append(run_info)
                        else:
                            runs.append({
                                "run_id": run_id,
                                "status": "UNKNOWN"
                            })
            
            return runs
    
    def _get_active_run(self):
        """
        Get the active run ID from run_info.json files.
        
        Returns:
            Active run ID or None
        """
        if os.path.exists(self.experiment_dir):
            for run_id in os.listdir(self.experiment_dir):
                run_dir = os.path.join(self.experiment_dir, run_id)
                if os.path.isdir(run_dir):
                    # Check run info
                    run_info_file = os.path.join(run_dir, "run_info.json")
                    if os.path.exists(run_info_file):
                        with open(run_info_file, 'r') as f:
                            run_info = json.load(f)
                            
                        if run_info.get("status") == "RUNNING":
                            return run_id
        
        return None

def log_experiment(experiment_name, model_info, params, metrics, model=None, flavor=None):
    """
    Log an experiment with the given model information.
    
    Args:
        experiment_name: Name of the experiment
        model_info: Dictionary with model information
        params: Dictionary with model parameters
        metrics: Dictionary with evaluation metrics
        model: Model object (optional)
        flavor: Model flavor for MLflow (optional)
        
    Returns:
        Run ID
    """
    try:
        # Create experiment tracker
        tracker = ExperimentTracker(experiment_name)
        
        # Start run
        run_name = model_info.get("model_name", "unnamed_model")
        run_id = tracker.start_run(run_name)
        
        # Log parameters
        for key, value in params.items():
            tracker.log_param(key, value)
        
        # Log model info parameters
        for key, value in model_info.items():
            if key != "model_name":
                tracker.log_param(f"info_{key}", value)
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tracker.log_metric(key, value)
        
        # Log model if provided
        if model is not None:
            tracker.log_model(model, run_name, flavor)
        
        # End run
        tracker.end_run()
        
        logging.info(f"Experiment logged with run ID {run_id}")
        
        return run_id
    
    except Exception as e:
        logging.error(f"Error logging experiment: {e}")
        return None

def get_experiment_summary(experiment_name=None):
    """
    Get a summary of all experiments or a specific experiment.
    
    Args:
        experiment_name: Name of the experiment (if None, get all experiments)
        
    Returns:
        DataFrame with experiment summary
    """
    try:
        if experiment_name:
            # Get specific experiment
            tracker = ExperimentTracker(experiment_name)
            runs = tracker.list_runs()
            
            # Get detailed info for each run
            detailed_runs = []
            for run in runs:
                run_id = run.get("run_id")
                if run_id:
                    run_info = tracker.get_run_info(run_id)
                    if run_info:
                        detailed_runs.append(run_info)
            
            # Convert to DataFrame
            if detailed_runs:
                # Flatten the structure
                flat_runs = []
                for run in detailed_runs:
                    flat_run = {
                        "run_id": run.get("run_id"),
                        "status": run.get("status"),
                        "start_time": run.get("start_time"),
                        "end_time": run.get("end_time")
                    }
                    
                    # Add params
                    for param_name, param_value in run.get("params", {}).items():
                        flat_run[f"param_{param_name}"] = param_value
                    
                    # Add metrics
                    for metric_name, metric_value in run.get("metrics", {}).items():
                        flat_run[f"metric_{metric_name}"] = metric_value
                    
                    flat_runs.append(flat_run)
                
                return pd.DataFrame(flat_runs)
            else:
                return pd.DataFrame()
        else:
            # List all experiment directories
            experiments_dir = os.path.join("experiments")
            if os.path.exists(experiments_dir):
                experiments = []
                for exp_name in os.listdir(experiments_dir):
                    exp_dir = os.path.join(experiments_dir, exp_name)
                    if os.path.isdir(exp_dir) and exp_name != "mlflow":
                        # Get run count
                        run_count = 0
                        for item in os.listdir(exp_dir):
                            if os.path.isdir(os.path.join(exp_dir, item)):
                                run_count += 1
                        
                        experiments.append({
                            "experiment_name": exp_name,
                            "run_count": run_count
                        })
                
                return pd.DataFrame(experiments)
            else:
                return pd.DataFrame()
    
    except Exception as e:
        logging.error(f"Error getting experiment summary: {e}")
        return pd.DataFrame()
