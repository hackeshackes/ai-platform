"""
ray.py - AI Platform v2.3
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

# 直接导入模块
import importlib.util
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'ray/manager.py')

spec = importlib.util.spec_from_file_location("gateway_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    ray_manager = module.ray_manager
except Exception as e:
    print(f"Failed to import module: {e}")
    ray_manager = None

from api.endpoints.auth import get_current_user

router = APIRouter()
class CreateClusterModel(BaseModel):
    name: str
    head_node: str
    worker_nodes: Optional[List[str]] = None

class RegisterDatasetModel(BaseModel):
    name: str
    uri: str
    format: str  # parquet, json, csv
    schema: Optional[Dict] = None

class SubmitJobModel(BaseModel):
    name: str
    cluster_id: str
    script: str
    resources: Optional[Dict] = None

class AutoscalingModel(BaseModel):
    cluster_id: str
    min_workers: int = 0
    max_workers: int = 10
    cpu_target: float = 0.5
    memory_target: float = 0.5

@router.get("/clusters")
async def list_clusters(status: Optional[str] = None):
    """
    列出Ray集群
    
    v2.3: Ray Data
    """
    clusters = ray_manager.list_clusters(status=status)
    
    return {
        "total": len(clusters),
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "name": c.name,
                "status": c.status,
                "workers_count": len(c.worker_nodes),
                "resources": c.resources,
                "created_at": c.created_at.isoformat()
            }
            for c in clusters
        ]
    }

@router.post("/clusters")
async def create_cluster(request: CreateClusterModel):
    """
    创建Ray集群
    
    v2.3: Ray Data
    """
    cluster = ray_manager.create_cluster(
        name=request.name,
        head_node=request.head_node,
        worker_nodes=request.worker_nodes
    )
    
    return {
        "cluster_id": cluster.cluster_id,
        "name": cluster.name,
        "message": "Cluster created"
    }

@router.post("/clusters/{cluster_id}/start")
async def start_cluster(cluster_id: str):
    """
    启动集群
    
    v2.3: Ray Data
    """
    result = ray_manager.start_cluster(cluster_id)
    if not result:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    return {"message": "Cluster started"}

@router.post("/clusters/{cluster_id}/stop")
async def stop_cluster(cluster_id: str):
    """
    停止集群
    
    v2.3: Ray Data
    """
    result = ray_manager.stop_cluster(cluster_id)
    if not result:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    return {"message": "Cluster stopped"}

@router.get("/clusters/{cluster_id}/resources")
async def get_cluster_resources(cluster_id: str):
    """
    获取集群资源
    
    v2.3: Ray Data
    """
    resources = ray_manager.get_cluster_resources(cluster_id)
    if not resources:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    return resources

@router.get("/clusters/{cluster_id}/metrics")
async def get_cluster_metrics(cluster_id: str):
    """
    获取集群指标
    
    v2.3: Ray Data
    """
    metrics = ray_manager.get_cluster_metrics(cluster_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    return metrics

@router.get("/datasets")
async def list_datasets():
    """
    列出Ray数据集
    
    v2.3: Ray Data
    """
    datasets = ray_manager.list_datasets()
    
    return {
        "total": len(datasets),
        "datasets": [
            {
                "dataset_id": d.dataset_id,
                "name": d.name,
                "uri": d.uri,
                "format": d.format,
                "num_blocks": d.num_blocks,
                "size_bytes": d.size_bytes
            }
            for d in datasets
        ]
    }

@router.post("/datasets")
async def register_dataset(request: RegisterDatasetModel):
    """
    注册Ray数据集
    
    v2.3: Ray Data
    """
    dataset = ray_manager.register_dataset(
        name=request.name,
        uri=request.uri,
        format=request.format,
        schema=request.schema
    )
    
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "message": "Dataset registered"
    }

@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """
    获取数据集详情
    
    v2.3: Ray Data
    """
    dataset = ray_manager.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "uri": dataset.uri,
        "format": dataset.format,
        "size_bytes": dataset.size_bytes,
        "num_blocks": dataset.num_blocks,
        "schema": dataset.schema,
        "created_at": dataset.created_at.isoformat()
    }

@router.get("/datasets/{dataset_id}/summary")
async def get_dataset_summary(dataset_id: str):
    """
    获取数据集摘要
    
    v2.3: Ray Data
    """
    summary = ray_manager.get_dataset_summary(dataset_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return summary

@router.post("/jobs")
async def submit_job(request: SubmitJobModel):
    """
    提交Ray任务
    
    v2.3: Ray Data
    """
    job = ray_manager.submit_job(
        name=request.name,
        cluster_id=request.cluster_id,
        script=request.script,
        resources=request.resources
    )
    
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "created_at": job.created_at.isoformat()
    }

@router.get("/jobs")
async def list_jobs(
    cluster_id: Optional[str] = None,
    status: Optional[str] = None
):
    """
    列出Ray任务
    
    v2.3: Ray Data
    """
    jobs = ray_manager.list_jobs(cluster_id=cluster_id, status=status)
    
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j.job_id,
                "name": j.name,
                "cluster_id": j.cluster_id,
                "status": j.status,
                "resources": j.resources,
                "created_at": j.created_at.isoformat()
            }
            for j in jobs
        ]
    }

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    获取任务详情
    
    v2.3: Ray Data
    """
    job = ray_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "name": job.name,
        "cluster_id": job.cluster_id,
        "script": job.script,
        "status": job.status,
        "resources": job.resources,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }

@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, tail: int = 100):
    """
    获取任务日志
    
    v2.3: Ray Data
    """
    logs = ray_manager.get_job_logs(job_id, tail=tail)
    
    return {
        "job_id": job_id,
        "logs": logs
    }

@router.post("/autoscaling")
async def configure_autoscaling(request: AutoscalingModel):
    """
    配置自动扩缩容
    
    v2.3: Ray Data
    """
    config = ray_manager.configure_autoscaling(
        cluster_id=request.cluster_id,
        min_workers=request.min_workers,
        max_workers=request.max_workers,
        cpu_target=request.cpu_target,
        memory_target=request.memory_target
    )
    
    return {
        "config_id": config.config_id,
        "cluster_id": config.cluster_id,
        "message": "Autoscaling configured"
    }

@router.get("/autoscaling/{cluster_id}")
async def get_autoscaling_status(cluster_id: str):
    """
    获取自动扩缩容状态
    
    v2.3: Ray Data
    """
    status = ray_manager.get_autoscaling_status(cluster_id)
    
    return status

@router.get("/health")
async def ray_health():
    """
    Ray健康检查
    
    v2.3: Ray Data
    """
    clusters = list(ray_manager.clusters.values())
    running_clusters = [c for c in clusters if c.status == "running"]
    datasets = list(ray_manager.datasets.values())
    jobs = list(ray_manager.jobs.values())
    
    return {
        "status": "healthy",
        "total_clusters": len(clusters),
        "running_clusters": len(running_clusters),
        "total_datasets": len(datasets),
        "total_jobs": len(jobs)
    }
