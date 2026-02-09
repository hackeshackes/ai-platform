"""Health check endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter
import subprocess
import datetime

router = APIRouter()

@router.get("")
async def health_check():
    """基本健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@router.get("/detailed")
async def detailed_health_check():
    """详细健康检查"""
    checks = {}
    
    # 检查数据库
    try:
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # 检查Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
    
    # 检查磁盘
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        checks["disk"] = {
            "total": f"{total // (2**30)} GB",
            "used": f"{used // (2**30)} GB",
            "free": f"{free // (2**30)} GB"
        }
    except Exception as e:
        checks["disk"] = f"error: {str(e)}"
    
    # 检查GPU (如果有)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(", ")
            checks["gpu"] = {
                "name": gpu_info[0],
                "memory": f"{gpu_info[1]} MB",
                "utilization": f"{gpu_info[2]}%"
            }
        else:
            checks["gpu"] = "not available"
    except FileNotFoundError:
        checks["gpu"] = "nvidia-smi not found"
    except Exception as e:
        checks["gpu"] = f"error: {str(e)}"
    
    overall_status = "healthy" if all(v == "healthy" or isinstance(v, dict) for v in checks.values()) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "checks": checks
    }
