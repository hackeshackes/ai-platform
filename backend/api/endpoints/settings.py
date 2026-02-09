"""Settings and configuration endpoints"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# 系统配置
class SystemConfig(BaseModel):
    site_name: str = "AI Platform"
    site_description: str = "大模型全生命周期管理平台"
    language: str = "zh-CN"
    theme: str = "light"
    timezone: str = "Asia/Shanghai"

# GPU阈值配置
class GPUThresholds(BaseModel):
    warning_utilization: int = 80
    critical_utilization: int = 95
    warning_memory: int = 80
    critical_memory: int = 95

# 用户配置
class UserConfig(BaseModel):
    theme: str = "light"
    language: str = "zh-CN"
    notifications: bool = True

# 存储配置
class StorageConfig(BaseModel):
    max_dataset_size_gb: int = 10
    max_model_size_gb: int = 50
    default_storage_path: str = "/data"

# 训练配置
class TrainingConfig(BaseModel):
    default_learning_rate: float = 2e-5
    default_batch_size: int = 4
    default_epochs: int = 3
    max_concurrent_tasks: int = 2
    gpu_required_min: int = 1

@router.get("/system")
async def get_system_config():
    """获取系统配置"""
    return {
        "site_name": "AI Platform",
        "site_description": "大模型全生命周期管理平台",
        "version": "1.0.0",
        "language": "zh-CN",
        "theme": "light",
        "timezone": "Asia/Shanghai",
        "features": {
            "gpu_monitoring": True,
            "distributed_training": True,
            "model_registry": True,
            "inference_service": True
        }
    }

@router.get("/gpu-thresholds")
async def get_gpu_thresholds():
    """获取GPU阈值配置"""
    return {
        "warning_utilization": 80,
        "critical_utilization": 95,
        "warning_memory": 80,
        "critical_memory": 95
    }

@router.post("/gpu-thresholds")
async def update_gpu_thresholds(data: GPUThresholds):
    """更新GPU阈值配置"""
    return data.dict()

@router.get("/storage")
async def get_storage_config():
    """获取存储配置"""
    return {
        "max_dataset_size_gb": 10,
        "max_model_size_gb": 50,
        "default_storage_path": "/data",
        "used_storage_gb": 2.5,
        "total_storage_gb": 100
    }

@router.get("/training")
async def get_training_config():
    """获取训练默认配置"""
    return {
        "default_learning_rate": 2e-5,
        "default_batch_size": 4,
        "default_epochs": 3,
        "max_concurrent_tasks": 2,
        "supported_frameworks": ["transformers", "peft", "deepspeed", "accelerate"]
    }

@router.get("/user")
async def get_user_config():
    """获取用户偏好配置"""
    return {
        "theme": "light",
        "language": "zh-CN",
        "notifications": True,
        "default_project": None,
        "gpu_alerts": True,
        "task_completion_alerts": True
    }
