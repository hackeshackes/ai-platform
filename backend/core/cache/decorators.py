"""
缓存装饰器
"""
from functools import wraps
from typing import Callable, Any
import hashlib
import json

def cached(
    ttl: int = 3600,
    key_builder: Callable = None,
    prefix: str = "api"
):
    """
    API响应缓存装饰器
    
    用法:
        @cached(ttl=300, prefix="projects")
        async def get_projects():
            return await project_service.list()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from .manager import cache_manager
            
            # 构建缓存key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # 默认key生成
                key_data = f"{func.__module__}:{func.__name__}:{args}:{kwargs}"
                key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
                cache_key = f"aip:{prefix}:{func.__name__}:{key_hash}"
            
            # 尝试获取缓存
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行原函数
            result = await func(*args, **kwargs)
            
            # 缓存结果
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_invalidate(pattern: str):
    """
    缓存失效装饰器
    
    用法:
        @cache_invalidate(pattern="projects")
        async def update_project(id: str, data: dict):
            return await project_service.update(id, data)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from .manager import cache_manager
            
            # 执行原函数
            result = await func(*args, **kwargs)
            
            # 失效相关缓存
            cache_manager.delete_pattern(pattern)
            
            return result
        return wrapper
    return decorator
