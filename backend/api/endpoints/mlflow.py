"""MLflow integration endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import mlflow
import mlflow.pytorch
import mlflow.transformers
from datetime import datetime, timedelta

router = APIRouter()

# MLflow配置
MLFLOW_TRACKING_URI = "http://localhost:5000"

@router.get("/status")
async def mlflow_status():
    """检查MLflow连接状态"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        client.search_experiments()
        return {"status": "connected", "tracking_uri": MLFLOW_TRACKING_URI}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}

@router.get("/experiments")
async def list_mlflow_experiments():
    """获取MLflow实验列表"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        
        return {
            "experiments": [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/runs")
async def get_mlflow_runs(experiment_id: str, max_results: int = 100):
    """获取实验运行记录"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment_id], max_results=max_results)
        
        return {
            "runs": [
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                    "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    "metrics": {k: v for k, v in run.data.metrics.items()},
                    "params": {k: v for k, v in run.data.params.items()}
                }
                for run in runs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LogMetricsRequest(BaseModel):
    run_id: str
    metrics: dict

@router.post("/experiments/{experiment_id}/log")
async def log_to_mlflow(experiment_id: str, request: LogMetricsRequest):
    """记录指标到MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        with mlflow.start_run(run_id=request.run_id):
            for metric_name, value in request.metrics.items():
                mlflow.log_metric(metric_name, value)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RegisterModelRequest(BaseModel):
    run_id: str
    model_name: str
    stage: str = "Staging"

@router.post("/models/register")
async def register_model(request: RegisterModelRequest):
    """注册模型到MLflow Model Registry"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        
        # 获取模型URI
        model_uri = f"runs:/{request.run_id}/model"
        
        # 注册模型
        result = client.create_registered_model(request.model_name)
        
        # 创建模型版本
        version = client.create_model_version(
            name=request.model_name,
            source=model_uri,
            run_id=request.run_id
        )
        
        # 转换阶段
        client.transition_model_version_stage(
            name=request.model_name,
            version=version.version,
            stage=request.stage
        )
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "version": version.version,
            "stage": request.stage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """获取模型版本列表"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        
        versions = client.get_latest_versions(model_name)
        
        return {
            "model_name": model_name,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id
                }
                for v in versions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
