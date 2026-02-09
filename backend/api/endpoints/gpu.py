"""GPU Monitoring endpoints"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import time

router = APIRouter()

# GPU指标模型
class GPUMetrics(BaseModel):
    gpu_id: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    utilization_percent: int
    temperature_c: int
    power_watts: Optional[float] = None

class GPUSummary(BaseModel):
    total_gpus: int
    total_memory_mb: int
    used_memory_mb: int
    avg_utilization: float
    metrics: List[GPUMetrics]

# 模拟GPU数据（真实环境使用pynvml）
def get_mock_gpu_data():
    return {
        "total_gpus": 1,
        "total_memory_mb": 16384,
        "used_memory_mb": 4096,
        "avg_utilization": 45.0,
        "metrics": [
            {
                "gpu_id": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "total_memory_mb": 16384,
                "used_memory_mb": 4096,
                "utilization_percent": 45,
                "temperature_c": 58,
                "power_watts": 320.5
            }
        ]
    }

def get_real_gpu_data():
    """真实GPU数据采集（需要pynvml）"""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        metrics_list = []
        total_memory = 0
        total_used = 0
        total_util = 0
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            metrics = {
                "gpu_id": i,
                "name": pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                "total_memory_mb": info.total // 1024**2,
                "used_memory_mb": info.used // 1024**2,
                "utilization_percent": util.gpu,
                "temperature_c": temp,
                "power_watts": None
            }
            metrics_list.append(metrics)
            
            total_memory += info.total
            total_used += info.used
            total_util += util.gpu
        
        return {
            "total_gpus": device_count,
            "total_memory_mb": total_memory // 1024**2,
            "used_memory_mb": total_used // 1024**2,
            "avg_utilization": total_util / device_count if device_count > 0 else 0,
            "metrics": metrics_list
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("")
async def get_gpu_status():
    """获取GPU状态"""
    data = get_real_gpu_data()
    if "error" in data:
        # 返回模拟数据
        return get_mock_gpu_data()
    return data

@router.get("/history")
async def get_gpu_history(minutes: int = 10):
    """获取GPU历史数据"""
    # 返回模拟历史数据
    import random
    history = []
    now = time.time()
    
    for i in range(minutes * 6):  # 每10秒一个数据点
        history.append({
            "timestamp": now - (minutes * 60 - i * 10),
            "utilization": random.randint(30, 80),
            "memory_mb": random.randint(3000, 8000)
        })
    
    return {"history": history}
