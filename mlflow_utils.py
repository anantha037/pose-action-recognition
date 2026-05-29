"""
MLflow helper utilities.
Contains helper functions for managing MLflow runs, downloading models, and setting up the tracking URI.
"""

import os
import shutil
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "pose-action-recognition") -> None:
    """
    Configures MLflow tracking URI and experiment.
    
    If tracking_uri is None, it defaults to a local './mlruns' directory, 
    which is suitable for local and Google Colab execution.
    
    Args:
        tracking_uri (Optional[str]): The MLflow tracking server URI.
        experiment_name (str): The name of the experiment.
    """
    if tracking_uri is None:
        # Default to local mlruns if no remote tracking server is provided
        # Use absolute path for reliability
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tracking_uri = f"file:///{os.path.join(current_dir, 'mlruns')}"
        
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI set to: {tracking_uri}")
    print(f"MLflow experiment set to: {experiment_name}")

def download_best_model(
    experiment_name: str = "pose-action-recognition", 
    download_dir: str = "models", 
    metric: str = "metrics.test_accuracy"
) -> Optional[str]:
    """
    Finds the best model artifact from the given experiment based on the specified metric 
    and downloads it locally.
    
    Args:
        experiment_name (str): The name of the MLflow experiment.
        download_dir (str): Local directory to save the downloaded model.
        metric (str): The metric to use for sorting runs to find the best model.
        
    Returns:
        Optional[str]: Local path to the downloaded model, or None if no runs/artifacts are found.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found.")
        return None
        
    # Search for runs, ordered by the specified metric descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{metric} DESC"],
        max_results=1
    )
    
    if not runs:
        print(f"Error: No runs found in experiment '{experiment_name}'.")
        return None
        
    best_run = runs[0]
    run_id = best_run.info.run_id
    best_metric_val = best_run.data.metrics.get(metric.replace('metrics.', ''), 'N/A')
    
    print(f"Found best run (ID: {run_id}) with {metric}: {best_metric_val}")
    
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Assume the artifact is named 'best_lstm.pt' or 'best_transformer.pt' based on typical logs.
    # List artifacts to find the .pt file.
    artifacts = client.list_artifacts(run_id)
    pt_artifacts = [a for a in artifacts if a.path.endswith('.pt')]
    
    if not pt_artifacts:
        print(f"Error: No .pt model artifacts found in run {run_id}.")
        return None
        
    # Download the first .pt artifact found (usually the best model checkpoint)
    artifact_path = pt_artifacts[0].path
    print(f"Downloading artifact '{artifact_path}'...")
    
    local_artifact_path = client.download_artifacts(run_id, artifact_path, dst_path=download_dir)
    print(f"Successfully downloaded best model to: {local_artifact_path}")
    
    return local_artifact_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download the best MLflow model.")
    parser.add_argument("--experiment", type=str, default="pose-action-recognition", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()
    
    setup_mlflow(experiment_name=args.experiment)
    download_best_model(experiment_name=args.experiment, download_dir=args.output_dir)
