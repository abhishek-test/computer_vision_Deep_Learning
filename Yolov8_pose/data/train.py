'''
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-pose.pt')
    model.train(data='C:\\Abhishek Data\\VSCode_Workspace\\Python\\DL_Practice\\Yolov8_pose\\data\\config.yaml', epochs=500, imgsz=640)
'''

import mlflow
import mlflow.pyfunc
import torch
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    # Initialize MLflow
    mlflow.set_tracking_uri("file:/C:/Abhishek Data/VSCode_Workspace/Python/DL_Practice/Yolov8_pose/data/mlruns")  # Update this path
    mlflow.set_experiment("YOLOv8 Training")

    # Define your dataset and model parameters
    model_path   = "yolov8n-pose.pt"  # YOLOv8 small model
    data_yaml    = "C:/Abhishek Data/VSCode_Workspace/Python/DL_Practice/Yolov8_pose/data/config.yaml"    # Dataset configuration file
    project_name = "yolo_experiments"
    run_name     = "yolov8s_coco3"
    epochs       = 75
    img_size     = 640 

    # Start MLflow logging
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model",    model_path)
        mlflow.log_param("data",     data_yaml)
        mlflow.log_param("epochs",   epochs)
        mlflow.log_param("img_size", img_size)

        # Initialize YOLOv8 model
        model = YOLO(model_path)

        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            project=project_name,
            name=run_name,
            verbose=True
        )

        metrics = results.results_dict
        mlflow.log_metric("precision_B", metrics.get("metrics/precision(B)", 0))
        mlflow.log_metric("recall_B",    metrics.get("metrics/recall(B)",    0))
        mlflow.log_metric("mAP50_B",     metrics.get("metrics/mAP50(B)",     0))
        mlflow.log_metric("mAP50-95_B",  metrics.get("metrics/mAP50-95(B)",  0))
        mlflow.log_metric("precision_P", metrics.get("metrics/precision(P)", 0))
        mlflow.log_metric("recall_P",    metrics.get("metrics/recall(P)",    0))
        mlflow.log_metric("mAP50_P",     metrics.get("metrics/mAP50(P)",     0))
        mlflow.log_metric("mAP50-95_P",  metrics.get("metrics/mAP50-95(P)",  0))
        mlflow.log_metric("fitness",     metrics.get("fitness",              0))

        # Log artifacts (model and results)
        run_dir = Path(project_name) / run_name
        mlflow.log_artifacts(str(run_dir / "weights"),  artifact_path="weights")
        mlflow.log_artifacts(str(run_dir),              artifact_path="results")

        '''
        # Save and log the best model
        best_model_path = run_dir / "weights" / "best.pt"
        mlflow.pytorch.log_model(torch.load(str(best_model_path)), artifact_path="best_model")
        '''

    # End of script
    print("Training completed and logged to MLflow!")