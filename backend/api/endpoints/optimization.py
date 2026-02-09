"""
optimization.py - AI Platform v2.3
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
module_path = os.path.join(backend_dir, 'optimization/performance.py')

spec = importlib.util.spec_from_file_location("gateway_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    performance_optimizer = module.performance_optimizer
except Exception as e:
    print(f"Failed to import module: {e}")
    performance_optimizer = None

from api.endpoints.auth import get_current_user

router = APIRouter()
"""
optimization.py - AI Platform v2.3
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
module_path = os.path.join(backend_dir, 'optimization/performance.py')

spec = importlib.util.spec_from_file_location("optimization_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    performance_optimizer = module.performance_optimizer
except Exception as e:
    print(f"Failed to import {module_name} module: {e}")
    performance_optimizer = None

from api.endpoints.auth import get_current_user

router = APIRouter()
"""
性能优化API端点 v2.3
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

# 动态导入optimization模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from optimization.performance import performance_optimizer
except ImportError:
    performance_optimizer = None

from api.endpoints.auth import get_current_user

router = APIRouter()

@router.get("/cache/stats")
async def get_cache_stats():
    """
    获取缓存统计
    
    v2.3: 性能优化
    """
    stats = performance_optimizer.get_cache_stats()
    return stats

@router.post("/cache/clear")
async def clear_cache():
    """
    清空缓存
    
    v2.3: 性能优化
    """
    await performance_optimizer.clear()
    return {"message": "Cache cleared"}

@router.delete("/cache/{key}")
async def delete_cache(key: str):
    """
    删除缓存键
    
    v2.3: 性能优化
    """
    result = await performance_optimizer.delete(key)
    if not result:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"message": "Cache key deleted"}

@router.get("/queries/slow")
async def get_slow_queries(threshold_ms: float = 100):
    """
    获取慢查询
    
    v2.3: 性能优化
    """
    slow_queries = performance_optimizer.get_slow_queries(threshold_ms)
    return {
        "total": len(slow_queries),
        "queries": slow_queries
    }

@router.get("/queries/stats")
async def get_query_stats():
    """
    获取查询统计
    
    v2.3: 性能优化
    """
    stats = performance_optimizer.get_query_stats()
    return {
        "total": len(stats),
        "stats": stats
    }

@router.get("/connections/pool")
async def get_connection_pool_stats():
    """
    获取连接池统计
    
    v2.3: 性能优化
    """
    stats = await performance_optimizer.get_connection_pool_stats()
    return stats

@router.get("/summary")
async def get_performance_summary():
    """
    获取性能摘要
    
    v2.3: 性能优化
    """
    summary = performance_optimizer.get_performance_summary()
    return summary

@router.get("/health")
async def performance_health():
    """
    性能优化健康检查
    
    v2.3: 性能优化
    """
    return {
        "status": "healthy",
        "cache_available": True,
        "optimization_enabled": True
    }
