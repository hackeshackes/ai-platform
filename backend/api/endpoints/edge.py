"""
Edge Inference API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'edge/engine.py')

spec = importlib.util.spec_from_file_location("edge_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    edge_inference_engine = module.edge_inference_engine
    ExportFormat = module.ExportFormat
    DeviceType = module.DeviceType
except Exception as e:
    print(f"Failed to import edge module: {e}")
    edge_inference_engine = None
    ExportFormat = None
    DeviceType = None

router = APIRouter()

class RegisterModelModel(BaseModel):
    model_id: str
    model_path: str
    model_type: str = "pytorch"
    metadata: Optional[Dict] = None

class ExportConfigModel(BaseModel):
    model_id: str
    model_version: str
    export_format: str
    target_device: str
    optimization_level: int = 1
    quantize: bool = False
    input_shape: Optional[List[int]] = None

class QuickExportModel(BaseModel):
    model_id: str
    export_format: str = "onnx"
    quantize: bool = True

class CompatibilityModel(BaseModel):
    model_id: str
    export_format: str
    device: str

class CreateDeploymentModel(BaseModel):
    name: str
    model_id: str
    export_config_id: str
    device_type: str
    device_url: str
    config: Optional[Dict] = None

# ==================== 模型注册 ====================

@router.post("/models/register")
async def register_model(request: RegisterModelModel):
    """注册模型"""
    model = edge_inference_engine.register_model(
        model_id=request.model_id,
        model_path=request.model_path,
        model_type=request.model_type,
        metadata=request.metadata
    )
    return {
        "model_id": model["model_id"],
        "message": "Model registered"
    }

@router.get("/models/{model_id}")
async def get_registered_model(model_id: str):
    """获取注册模型"""
    model = edge_inference_engine.get_registered_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

# ==================== 导出配置 ====================

@router.post("/export/config")
async def create_export_config(request: ExportConfigModel):
    """创建导出配置"""
    try:
        fmt = ExportFormat(request.export_format)
        device = DeviceType(request.target_device)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    config = edge_inference_engine.create_export_config(
        model_id=request.model_id,
        model_version=request.model_version,
        export_format=fmt,
        target_device=device,
        optimization_level=request.optimization_level,
        quantize=request.quantize,
        input_shape=request.input_shape
    )
    
    return {
        "config_id": config.config_id,
        "format": config.export_format.value,
        "device": config.target_device.value,
        "message": "Export config created"
    }

@router.post("/export/{config_id}")
async def export_model(config_id: str):
    """导出模型"""
    try:
        result = edge_inference_engine.export_model(config_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/export/quick")
async def quick_export(request: QuickExportModel):
    """快速导出"""
    try:
        fmt = ExportFormat(request.export_format)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid format: {request.export_format}")
    
    result = edge_inference_engine.quick_export(
        model_id=request.model_id,
        format=fmt,
        quantize=request.quantize
    )
    return result

@router.get("/export/compatibility")
async def check_compatibility(request: CompatibilityModel):
    """检查兼容性"""
    try:
        fmt = ExportFormat(request.export_format)
        dev = DeviceType(request.device)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    result = edge_inference_engine.check_compatibility(request.model_id, fmt, dev)
    return result

# ==================== 边缘部署 ====================

@router.post("/deployments")
async def create_deployment(request: CreateDeploymentModel):
    """创建部署"""
    try:
        device = DeviceType(request.device_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid device: {request.device_type}")
    
    deployment = edge_inference_engine.create_deployment(
        name=request.name,
        model_id=request.model_id,
        export_config_id=request.export_config_id,
        device_type=device,
        device_url=request.device_url,
        config=request.config
    )
    
    return {
        "deployment_id": deployment.deployment_id,
        "name": deployment.name,
        "message": "Deployment created"
    }

@router.get("/deployments")
async def list_deployments(
    status: Optional[str] = None,
    device_type: Optional[str] = None
):
    """列出部署"""
    dstatus = status
    ddevice = DeviceType(device_type) if device_type else None
    
    deployments = edge_inference_engine.list_deployments(
        status=dstatus,
        device_type=ddevice
    )
    
    return {
        "total": len(deployments),
        "deployments": [
            {
                "deployment_id": d.deployment_id,
                "name": d.name,
                "device_type": d.device_type.value,
                "status": d.status
            }
            for d in deployments
        ]
    }

@router.get("/deployments/{deployment_id}")
async def get_deployment(deployment_id: str):
    """获取部署详情"""
    deployment = edge_inference_engine.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    info = edge_inference_engine.get_device_info(deployment_id)
    return info

@router.post("/deployments/{deployment_id}/deploy")
async def deploy_to_device(deployment_id: str):
    """部署到设备"""
    result = edge_inference_engine.deploy_to_device(deployment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {"message": "Deployed to device"}

@router.delete("/deployments/{deployment_id}")
async def remove_deployment(deployment_id: str):
    """移除部署"""
    result = edge_inference_engine.remove_deployment(deployment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {"message": "Deployment removed"}

# ==================== 边缘推理 ====================

@router.post("/inference/{deployment_id}")
async def edge_inference(deployment_id: str, input_data: Dict):
    """执行边缘推理"""
    try:
        result = edge_inference_engine.inference(deployment_id, input_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/inference/{deployment_id}/batch")
async def batch_inference(deployment_id: str, input_batch: List[Dict]):
    """批量推理"""
    try:
        results = edge_inference_engine.batch_inference(deployment_id, input_batch)
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/deployments/{deployment_id}/metrics")
async def get_edge_metrics(deployment_id: str):
    """获取设备指标"""
    metrics = edge_inference_engine.get_metrics(deployment_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    return {
        "inference_count": metrics.inference_count,
        "avg_latency_ms": metrics.avg_latency_ms,
        "throughput": metrics.throughput,
        "memory_usage_mb": metrics.memory_usage_mb
    }

@router.get("/summary")
async def get_summary():
    """获取统计"""
    return edge_inference_engine.get_summary()

@router.get("/health")
async def edge_health():
    """健康检查"""
    return {
        "status": "healthy",
        "deployments": len(edge_inference_engine.deployments)
    }
