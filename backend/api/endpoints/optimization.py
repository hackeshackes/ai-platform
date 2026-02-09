"""
性能优化API端点 v2.3
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from backend.optimization.performance import performance_optimizer

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
