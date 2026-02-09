"""
分布式训练API端点 v2.2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from distributed.trainer import distributed_trainer
from api.endpoints.auth import get_current_user

router = APIRouter()

class RegisterWorkerModel(BaseModel):
    hostname: str
    gpu_count: int
    gpu_info: Optional[List[Dict]] = None

class CreateJobModel(BaseModel):
    name: str
    script: str
    world_size: int
    worker_ids: Optional[List[str]] = None

@router.post("/workers")
async def register_worker(request: RegisterWorkerModel):
    """
    注册Worker节点
    
    v2.2: 分布式训练
    """
    worker = await distributed_trainer.register_worker(
        hostname=request.hostname,
        gpu_count=request.gpu_count,
        gpu_info=request.gpu_info
    )
    
    return {
        "worker_id": worker.worker_id,
        "hostname": worker.hostname,
        "gpu_count": worker.gpu_count,
        "status": worker.status
    }

@router.get("/workers")
async def list_workers(status: Optional[str] = None):
    """
    列出Worker节点
    
    v2.2: 分布式训练
    """
    workers = distributed_trainer.list_workers(status=status)
    
    return {
        "total": len(workers),
        "workers": [
            {
                "worker_id": w.worker_id,
                "hostname": w.hostname,
                "gpu_count": w.gpu_count,
                "status": w.status,
                "last_heartbeat": w.last_heartbeat.isoformat()
            }
            for w in workers
        ]
    }

@router.get("/workers/{worker_id}")
async def get_worker(worker_id: str):
    """
    获取Worker详情
    
    v2.2: 分布式训练
    """
    worker = distributed_trainer.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    return {
        "worker_id": worker.worker_id,
        "hostname": worker.hostname,
        "gpu_count": worker.gpu_count,
        "gpu_info": worker.gpu_info,
        "status": worker.status,
        "last_heartbeat": worker.last_heartbeat.isoformat()
    }

@router.post("/jobs")
async def create_job(
    request: CreateJobModel,
    current_user = Depends(get_current_user)
):
    """
    创建分布式训练任务
    
    v2.2: 分布式训练
    """
    try:
        job = await distributed_trainer.create_job(
            name=request.name,
            script=request.script,
            world_size=request.world_size,
            worker_ids=request.worker_ids
        )
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "world_size": job.world_size,
            "status": job.status,
            "workers": job.workers,
            "created_at": job.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100
):
    """
    列出训练任务
    
    v2.2: 分布式训练
    """
    jobs = distributed_trainer.list_jobs(status=status, limit=limit)
    
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j.job_id,
                "name": j.name,
                "world_size": j.world_size,
                "status": j.status,
                "created_at": j.created_at.isoformat()
            }
            for j in jobs
        ]
    }

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    获取任务详情
    
    v2.2: 分布式训练
    """
    job = distributed_trainer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "name": job.name,
        "script": job.script,
        "world_size": job.world_size,
        "workers": job.workers,
        "status": job.status,
        "metrics": job.metrics,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }

@router.post("/jobs/{job_id}/start")
async def start_job(job_id: str):
    """
    启动训练任务
    
    v2.2: 分布式训练
    """
    result = await distributed_trainer.start_job(job_id)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to start job")
    
    return {"message": "Job started"}

@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """
    停止训练任务
    
    v2.2: 分布式训练
    """
    result = await distributed_trainer.stop_job(job_id)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to stop job")
    
    return {"message": "Job stopped"}

@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, tail: int = 100):
    """
    获取任务日志
    
    v2.2: 分布式训练
    """
    job = distributed_trainer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logs = distributed_trainer.get_job_logs(job_id, tail=tail)
    
    return {
        "job_id": job_id,
        "logs": logs
    }

@router.post("/clusters")
async def create_cluster(
    name: str,
    worker_ids: List[str],
    scheduler: str = "ray"
):
    """
    创建集群配置
    
    v2.2: 分布式训练
    """
    cluster = await distributed_trainer.create_cluster(
        name=name,
        worker_ids=worker_ids,
        scheduler=scheduler
    )
    
    return {
        "cluster_id": cluster.config_id,
        "name": cluster.name,
        "workers": cluster.workers,
        "scheduler": cluster.scheduler
    }

@router.get("/clusters")
async def list_clusters():
    """
    列出集群
    
    v2.2: 分布式训练
    """
    clusters = distributed_trainer.list_clusters()
    
    return {
        "total": len(clusters),
        "clusters": [
            {
                "cluster_id": c.config_id,
                "name": c.name,
                "workers": c.workers,
                "scheduler": c.scheduler
            }
            for c in clusters
        ]
    }

@router.get("/resources")
async def get_cluster_resources():
    """
    获取集群资源统计
    
    v2.2: 分布式训练
    """
    resources = distributed_trainer.get_cluster_resources()
    
    return resources

@router.get("/jobs/{job_id}/resources")
async def get_job_resources(job_id: str):
    """
    获取任务资源使用
    
    v2.2: 分布式训练
    """
    resources = distributed_trainer.get_job_resources(job_id)
    if not resources:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return resources
