"""
Model Serving API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'serving/engine.py')

spec = importlib.util.spec_from_file_location("serving_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    serving_engine = module.serving_engine
    ServingStatus = module.ServingStatus
    DeploymentStrategy = module.DeploymentStrategy
    BatchStrategy = module.BatchStrategy
except Exception as e:
    print(f"Failed to import serving module: {e}")
    serving_engine = None
    ServingStatus = None
    DeploymentStrategy = None
    BatchStrategy = None

router = APIRouter()

from pydantic import BaseModel

class CreateEndpointModel(BaseModel):
    name: str
    model_id: str
    model_version: str
    replicas: int = 1
    resource_config: Optional[Dict] = None

class TrafficSplitModel(BaseModel):
    model_version: str
    weight: float
    description: Optional[str] = None

class ShadowConfigModel(BaseModel):
    shadow_model_id: str
    shadow_version: str
    mirror_percent: float = 10.0

class BatchingConfigModel(BaseModel):
    batch_size: int = 32
    max_batch_size: int = 128
    batch_timeout_ms: int = 1000

class RollbackModel(BaseModel):
    target_version: str

@router.get("/endpoints")
async def list_endpoints(status: Optional[str] = None):
    """列出模型端点"""
    sstatus = ServingStatus(status) if status else None
    endpoints = serving_engine.list_endpoints(status=sstatus)
    
    return {
        "total": len(endpoints),
        "endpoints": [
            {
                "endpoint_id": e.endpoint_id,
                "name": e.name,
                "model_id": e.model_id,
                "version": e.model_version,
                "status": e.status.value,
                "replicas": e.replicas,
                "url": e.url
            }
            for e in endpoints
        ]
    }

@router.post("/endpoints")
async def create_endpoint(request: CreateEndpointModel):
    """创建模型端点"""
    endpoint = serving_engine.create_endpoint(
        name=request.name,
        model_id=request.model_id,
        model_version=request.model_version,
        replicas=request.replicas,
        resource_config=request.resource_config
    )
    
    return {
        "endpoint_id": endpoint.endpoint_id,
        "name": endpoint.name,
        "message": "Endpoint created"
    }

@router.get("/endpoints/{endpoint_id}")
async def get_endpoint(endpoint_id: str):
    """获取端点详情"""
    endpoint = serving_engine.get_endpoint(endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    metrics = serving_engine.get_metrics(endpoint_id)
    
    return {
        "endpoint_id": endpoint.endpoint_id,
        "name": endpoint.name,
        "model_id": endpoint.model_id,
        "model_version": endpoint.model_version,
        "status": endpoint.status.value,
        "replicas": endpoint.replicas,
        "resource_config": endpoint.resource_config,
        "metrics": {
            "request_count": metrics.request_count if metrics else 0,
            "avg_latency_ms": metrics.avg_latency_ms if metrics else 0
        }
    }

@router.post("/endpoints/{endpoint_id}/start")
async def start_endpoint(endpoint_id: str):
    """启动端点"""
    result = serving_engine.start_endpoint(endpoint_id)
    if not result:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {"message": "Endpoint started"}

@router.post("/endpoints/{endpoint_id}/stop")
async def stop_endpoint(endpoint_id: str):
    """停止端点"""
    result = serving_engine.stop_endpoint(endpoint_id)
    if not result:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {"message": "Endpoint stopped"}

@router.delete("/endpoints/{endpoint_id}")
async def delete_endpoint(endpoint_id: str):
    """删除端点"""
    result = serving_engine.delete_endpoint(endpoint_id)
    if not result:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {"message": "Endpoint deleted"}

@router.post("/endpoints/{endpoint_id}/traffic")
async def set_traffic_split(endpoint_id: str, splits: List[TrafficSplitModel]):
    """设置流量分割"""
    split_dicts = [s.model_dump() for s in splits]
    result = serving_engine.set_traffic_split(endpoint_id, split_dicts)
    if not result:
        raise HTTPException(status_code=400, detail="Invalid traffic split")
    return {"message": "Traffic split configured"}

@router.get("/endpoints/{endpoint_id}/traffic")
async def get_traffic_split(endpoint_id: str):
    """获取流量分割"""
    splits = serving_engine.get_traffic_split(endpoint_id)
    return {
        "splits": [
            {
                "model_version": s.model_version,
                "weight": s.weight
            }
            for s in splits
        ]
    }

@router.post("/endpoints/{endpoint_id}/shadow")
async def configure_shadow(endpoint_id: str, request: ShadowConfigModel):
    """配置Shadow Mode"""
    shadow = serving_engine.configure_shadow(
        endpoint_id=endpoint_id,
        shadow_model_id=request.shadow_model_id,
        shadow_version=request.shadow_version,
        mirror_percent=request.mirror_percent
    )
    return {
        "shadow_id": shadow.shadow_id,
        "message": "Shadow mode configured"
    }

@router.post("/endpoints/{endpoint_id}/batching")
async def configure_batching(endpoint_id: str, request: BatchingConfigModel):
    """配置批处理"""
    config = serving_engine.configure_batching(
        endpoint_id=endpoint_id,
        batch_size=request.batch_size,
        max_batch_size=request.max_batch_size,
        batch_timeout_ms=request.batch_timeout_ms
    )
    return {
        "batch_size": config.batch_size,
        "message": "Batching configured"
    }

@router.post("/endpoints/{endpoint_id}/rollback")
async def rollback_version(endpoint_id: str, request: RollbackModel):
    """回滚版本"""
    result = serving_engine.rollback_version(endpoint_id, request.target_version)
    if not result:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {"message": f"Rolled back to version {request.target_version}"}

@router.post("/inference/{endpoint_id}")
async def inference(endpoint_id: str, input_data: Dict):
    """执行推理"""
    try:
        result = serving_engine.inference(endpoint_id, input_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/endpoints/{endpoint_id}/metrics")
async def get_metrics(endpoint_id: str):
    """获取服务指标"""
    metrics = serving_engine.get_metrics(endpoint_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    return {
        "request_count": metrics.request_count,
        "avg_latency_ms": metrics.avg_latency_ms,
        "throughput": metrics.throughput,
        "gpu_utilization": metrics.gpu_utilization
    }

@router.get("/summary")
async def get_summary():
    """获取服务统计"""
    return serving_engine.get_summary()

@router.get("/health")
async def serving_health():
    """健康检查"""
    return {
        "status": "healthy",
        "endpoints": len(serving_engine.endpoints)
    }
