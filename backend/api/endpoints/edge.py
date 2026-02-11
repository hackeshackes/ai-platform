"""
边缘AI部署 - API端点
提供边缘部署相关的REST API
"""

import os
import sys
import uuid
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

# 添加backend路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from enum import Enum

from edge.optimizer import ModelOptimizer, OptimizationConfig, create_optimizer
from edge.tensorrt import TensorRTEngine, TensorRTConfig, create_tensorrt_engine
from edge.device import DeviceManager, DeviceStatus, DeviceType, create_device_manager
from edge.offline import OfflineInferenceEngine, create_inference_engine

logger = logging.getLogger(__name__)

# 创建路由器
edge_router = APIRouter(prefix="/edge", tags=["Edge AI"])

# 全局实例
_device_manager: Optional[DeviceManager] = None
_optimizer: Optional[ModelOptimizer] = None
_tensorrt_engine: Optional[TensorRTEngine] = None
_inference_engine: Optional[OfflineInferenceEngine] = None


def get_device_manager() -> DeviceManager:
    """获取设备管理器实例"""
    global _device_manager
    if _device_manager is None:
        _device_manager = create_device_manager()
    return _device_manager


def get_optimizer() -> ModelOptimizer:
    """获取模型优化器实例"""
    global _optimizer
    if _optimizer is None:
        _optimizer = create_optimizer()
    return _optimizer


def get_tensorrt_engine() -> TensorRTEngine:
    """获取TensorRT引擎实例"""
    global _tensorrt_engine
    if _tensorrt_engine is None:
        _tensorrt_engine = create_tensorrt_engine()
    return _tensorrt_engine


def get_inference_engine() -> OfflineInferenceEngine:
    """获取推理引擎实例"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = create_inference_engine()
    return _inference_engine


# ============ 请求/响应模型 ============

class ONNXExportRequest(BaseModel):
    """ONNX导出请求"""
    model_path: str = Field(..., description="源模型路径")
    output_path: Optional[str] = Field(None, description="输出路径")
    input_shape: Optional[List[int]] = Field(None, description="输入形状")
    opset_version: int = Field(11, ge=11, le=15, description="ONNX opset版本")
    optimize_for_inference: bool = Field(True, description="是否优化推理")


class TensorRTBuildRequest(BaseModel):
    """TensorRT构建请求"""
    onnx_path: str = Field(..., description="ONNX模型路径")
    output_path: Optional[str] = Field(None, description="输出路径")
    max_batch_size: int = Field(32, ge=1, description="最大批处理大小")
    precision_mode: str = Field("fp16", pattern="^(fp32|fp16|int8|mixed)$", description="精度模式")
    max_workspace_size: int = Field(8589934592, description="最大工作空间大小")


class DeviceRegisterRequest(BaseModel):
    """设备注册请求"""
    device_name: str = Field(..., min_length=1, description="设备名称")
    device_type: str = Field(..., description="设备类型")
    ip_address: Optional[str] = Field(None, description="IP地址")
    tags: List[str] = Field(default_factory=list, description="标签")


class ModelDeployRequest(BaseModel):
    """模型部署请求"""
    model_path: str = Field(..., description="模型文件路径")
    device_id: str = Field(..., description="目标设备ID")
    config: Optional[Dict[str, Any]] = Field(None, description="部署配置")


class InferenceRequest(BaseModel):
    """推理请求"""
    model_id: str = Field(..., description="模型ID")
    inputs: Dict[str, Any] = Field(..., description="输入数据")
    mode: str = Field("sync", pattern="^(sync|async|batch)$", description="推理模式")


class DataSyncRequest(BaseModel):
    """数据同步请求"""
    source_path: str = Field(..., description="源路径")
    target_device_id: str = Field(..., description="目标设备ID")
    direction: str = Field("upload", pattern="^(upload|download|sync)$", description="同步方向")
    recursive: bool = Field(True, description="是否递归")


# ============ ONNX优化端点 ============

@edge_router.post("/optimize/onnx")
async def export_onnx(request: ONNXExportRequest) -> Dict[str, Any]:
    """
    导出模型到ONNX格式
    
    将PyTorch或TensorFlow模型导出为ONNX格式
    """
    optimizer = get_optimizer()
    
    # 检查模型文件是否存在
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # 创建优化配置
    config = OptimizationConfig(
        input_shape=request.input_shape,
        opset_version=request.opset_version,
        optimize_for_inference=request.optimize_for_inference
    )
    
    # 加载模型（这里需要实际的模型加载逻辑）
    model = optimizer.load_model(request.model_path)
    
    # 导出到ONNX
    result = optimizer.export_to_onnx(model, config, request.output_path)
    
    return result


@edge_router.post("/optimize/quantize")
async def quantize_model(
    model_path: str,
    mode: str = "int8"
) -> Dict[str, Any]:
    """
    量化ONNX模型
    
    对ONNX模型进行INT8或FP16量化
    """
    optimizer = get_optimizer()
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    result = optimizer.quantize_model(model_path, mode)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    
    return result


@edge_router.post("/optimize/validate")
async def validate_onnx(model_path: str) -> Dict[str, Any]:
    """验证ONNX模型"""
    optimizer = get_optimizer()
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    result = optimizer.validate_onnx_model(model_path)
    return result


# ============ TensorRT端点 ============

@edge_router.post("/optimize/tensorrt")
async def build_tensorrt_engine(request: TensorRTBuildRequest) -> Dict[str, Any]:
    """
    构建TensorRT引擎
    
    将ONNX模型转换为TensorRT优化引擎
    """
    tensorrt = get_tensorrt_engine()
    
    if not os.path.exists(request.onnx_path):
        raise HTTPException(status_code=404, detail="ONNX model not found")
    
    # 解析精度模式
    from edge.tensorrt import PrecisionMode
    precision_map = {
        "fp32": PrecisionMode.FP32,
        "fp16": PrecisionMode.FP16,
        "int8": PrecisionMode.INT8,
        "mixed": PrecisionMode.MIXED
    }
    
    config = TensorRTConfig(
        max_batch_size=request.max_batch_size,
        precision_mode=precision_map.get(request.precision_mode, PrecisionMode.FP16),
        max_workspace_size=request.max_workspace_size
    )
    
    result = tensorrt.convert_onnx_to_engine(
        request.onnx_path,
        config,
        request.output_path
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    
    return result


@edge_router.post("/tensorrt/benchmark")
async def benchmark_tensorrt(
    engine_path: str,
    warmup: int = 10,
    iterations: int = 100
) -> Dict[str, Any]:
    """基准测试TensorRT引擎"""
    tensorrt = get_tensorrt_engine()
    
    if not os.path.exists(engine_path):
        raise HTTPException(status_code=404, detail="Engine file not found")
    
    result = tensorrt.benchmark_engine(engine_path, warmup, iterations)
    return result


@edge_router.get("/tensorrt/info/{engine_path:path}")
async def get_tensorrt_info(engine_path: str) -> Dict[str, Any]:
    """获取TensorRT引擎信息"""
    tensorrt = get_tensorrt_engine()
    
    if not os.path.exists(engine_path):
        raise HTTPException(status_code=404, detail="Engine file not found")
    
    result = tensorrt.get_engine_info(engine_path)
    return result


# ============ 设备管理端点 ============

@edge_router.get("/devices")
async def list_devices(
    status: Optional[str] = None,
    device_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取设备列表
    
    返回所有已注册的边缘设备
    """
    dm = get_device_manager()
    
    # 筛选条件
    status_filter = DeviceStatus(status) if status else None
    type_filter = DeviceType(device_type) if device_type else None
    
    devices = dm.list_devices(status=status_filter, device_type=type_filter)
    
    return {
        "devices": [device.to_dict() for device in devices],
        "total": len(devices),
        "stats": dm.get_device_stats()
    }


@edge_router.post("/devices/register")
async def register_device(request: DeviceRegisterRequest) -> Dict[str, Any]:
    """
    注册新设备
    
    将新边缘设备注册到管理系统
    """
    dm = get_device_manager()
    
    try:
        device_type = DeviceType(request.device_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device type: {request.device_type}"
        )
    
    device = dm.register_device(
        device_name=request.device_name,
        device_type=device_type,
        ip_address=request.ip_address,
        metadata={"tags": request.tags}
    )
    
    return {
        "status": "registered",
        "device": device.to_dict()
    }


@edge_router.delete("/devices/{device_id}")
async def unregister_device(device_id: str) -> Dict[str, Any]:
    """注销设备"""
    dm = get_device_manager()
    
    success = dm.unregister_device(device_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"status": "unregistered", "device_id": device_id}


@edge_router.get("/devices/{device_id}")
async def get_device(device_id: str) -> Dict[str, Any]:
    """获取设备详情"""
    dm = get_device_manager()
    
    device = dm.get_device(device_id)
    
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return device.to_dict()


@edge_router.get("/devices/{device_id}/metrics")
async def get_device_metrics(device_id: str, limit: int = 100) -> Dict[str, Any]:
    """获取设备性能指标"""
    dm = get_device_manager()
    
    device = dm.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    metrics = dm.get_device_metrics(device_id, limit)
    
    return {
        "device_id": device_id,
        "metrics": [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "gpu_usage": m.gpu_usage,
                "temperature": m.temperature
            }
            for m in metrics
        ]
    }


@edge_router.post("/devices/{device_id}/heartbeat")
async def device_heartbeat(
    device_id: str,
    metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """设备心跳"""
    dm = get_device_manager()
    
    from edge.device import DeviceMetrics
    
    device_metrics = None
    if metrics:
        device_metrics = DeviceMetrics(
            device_id=device_id,
            cpu_usage=metrics.get("cpu_usage", 0),
            memory_usage=metrics.get("memory_usage", 0),
            gpu_usage=metrics.get("gpu_usage", 0),
            temperature=metrics.get("temperature", 0)
        )
    
    success = dm.update_device_status(
        device_id,
        DeviceStatus.ONLINE,
        metrics=device_metrics
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"status": "heartbeat_received"}


# ============ 模型部署端点 ============

@edge_router.post("/deploy")
async def deploy_model(request: ModelDeployRequest) -> Dict[str, Any]:
    """
    部署模型到设备
    
    将优化后的模型部署到指定的边缘设备
    """
    dm = get_device_manager()
    
    # 检查设备
    device = dm.get_device(request.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # 检查模型文件
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # TODO: 实现实际的模型部署逻辑
    # 这里需要与设备上的Agent通信
    
    return {
        "status": "deploying",
        "model_path": request.model_path,
        "device_id": request.device_id,
        "message": "Deployment initiated"
    }


@edge_router.get("/deploy/status/{job_id}")
async def get_deploy_status(job_id: str) -> Dict[str, Any]:
    """获取部署状态"""
    # TODO: 实现部署状态查询
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100
    }


# ============ 数据同步端点 ============

@edge_router.post("/sync")
async def sync_data(request: DataSyncRequest) -> Dict[str, Any]:
    """
    同步数据到设备
    
    在云端和边缘设备之间同步模型、数据等
    """
    dm = get_device_manager()
    
    device = dm.get_device(request.target_device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # TODO: 实现实际的数据同步逻辑
    
    return {
        "status": "syncing",
        "source": request.source_path,
        "target_device": request.target_device_id,
        "direction": request.direction
    }


@edge_router.get("/sync/history")
async def get_sync_history(
    device_id: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """获取同步历史"""
    # TODO: 实现同步历史查询
    return {
        "history": [],
        "total": 0
    }


# ============ 离线推理端点 ============

@edge_router.post("/inference")
async def run_inference(request: InferenceRequest) -> Dict[str, Any]:
    """
    执行离线推理
    
    在本地推理引擎上运行模型推理
    """
    engine = get_inference_engine()
    
    from edge.offline import InferenceMode
    
    mode_map = {
        "sync": InferenceMode.SYNC,
        "async": InferenceMode.ASYNC,
        "batch": InferenceMode.BATCH
    }
    
    response = engine.infer(
        request.model_id,
        request.inputs,
        mode=mode_map.get(request.mode, InferenceMode.SYNC)
    )
    
    return {
        "request_id": response.request_id,
        "outputs": response.outputs,
        "latency_ms": response.latency_ms,
        "success": response.success,
        "error": response.error_message
    }


@edge_router.post("/inference/batch")
async def batch_inference(
    model_id: str,
    inputs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """批量推理"""
    engine = get_inference_engine()
    
    results = engine.batch_infer(model_id, inputs)
    
    return {
        "model_id": model_id,
        "total_requests": len(results),
        "successful": sum(1 for r in results if r.success),
        "results": [
            {
                "outputs": r.outputs,
                "latency_ms": r.latency_ms,
                "success": r.success
            }
            for r in results
        ]
    }


@edge_router.get("/inference/models")
async def list_loaded_models() -> Dict[str, Any]:
    """列出已加载的模型"""
    engine = get_inference_engine()
    return engine.get_engine_status()


@edge_router.post("/inference/models/load")
async def load_inference_model(
    model_id: str,
    model_path: str,
    model_type: str = "onnx",
    device: str = "cpu"
) -> Dict[str, Any]:
    """加载模型到推理引擎"""
    engine = get_inference_engine()
    
    result = engine.load_model(model_id, model_path, model_type, device)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    
    return result


@edge_router.delete("/inference/models/{model_id}")
async def unload_inference_model(model_id: str) -> Dict[str, Any]:
    """卸载模型"""
    engine = get_inference_engine()
    
    success = engine.unload_model(model_id)
    
    return {"success": success}


@edge_router.get("/inference/cache")
async def get_cache_status() -> Dict[str, Any]:
    """获取推理缓存状态"""
    engine = get_inference_engine()
    return engine.cache.get_cache_stats()


# ============ 健康检查端点 ============

@edge_router.get("/health")
async def edge_health() -> Dict[str, Any]:
    """边缘服务健康检查"""
    dm = get_device_manager()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "device_manager": {
                "status": "ready",
                "devices_count": len(dm.list_devices())
            },
            "optimizer": {
                "status": "ready"
            },
            "inference_engine": {
                "status": "ready",
                "models_count": len(engine.get_engine_status().get("loaded_models", []))
            }
        }
    }


# ============ 任务跟踪 ============

# 简单的内存任务存储
_pending_tasks: Dict[str, Dict[str, Any]] = {}


@edge_router.get("/tasks")
async def list_tasks(status: Optional[str] = None) -> Dict[str, Any]:
    """列出所有任务"""
    tasks = list(_pending_tasks.values())
    
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    
    return {"tasks": tasks, "total": len(tasks)}


@edge_router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """获取任务详情"""
    if task_id not in _pending_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return _pending_tasks[task_id]


# ============ 导出路由器 ============

def include_edge_routers(app):
    """将边缘路由器包含到主应用"""
    app.include_router(edge_router)
