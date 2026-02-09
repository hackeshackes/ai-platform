"""
缓存装饰器 - API缓存集成
"""
from functools import wraps
from typing import Callable, Any, Optional
from datetime import timedelta
from core.cache import cache

def cached(
    ttl_key: str,
    key_builder: Optional[Callable] = None,
    cache_result: bool = True
):
    """
    缓存装饰器
    
    用法:
        @cached(ttl_key="project_list", key_builder=lambda f, u: f"projects:{u}")
        async def get_user_projects(user_id: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 构建缓存键
            if key_builder:
                cache_key = key_builder(func, *args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(frozenset(kwargs.items()))}"
            
            # 尝试从缓存获取
            if cache_result:
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 缓存结果
            if cache_result and result is not None:
                cache.set(cache_key, result, ttl_key)
            
            return result
        
        return wrapper
    return decorator

def invalidate_cache(*ttl_keys: str):
    """
    缓存失效装饰器
    
    用法:
        @invalidate_cache("project_list", "project_detail")
        async def update_project(id: int, data: dict):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # 清除相关缓存
            for key in ttl_keys:
                cache.invalidate_pattern(key)
            
            return result
        
        return wrapper
    return decorator

class CacheKeys:
    """缓存键构建器"""
    
    @staticmethod
    def project_list(user_id: int) -> str:
        return f"projects:list:{user_id}"
    
    @staticmethod
    def project_detail(project_id: int) -> str:
        return f"projects:detail:{project_id}"
    
    @staticmethod
    def task_list(user_id: int) -> str:
        return f"tasks:list:{user_id}"
    
    @staticmethod
    def task_detail(task_id: int) -> str:
        return f"tasks:detail:{task_id}"
    
    @staticmethod
    def dataset_list(user_id: int) -> str:
        return f"datasets:list:{user_id}"
    
    @staticmethod
    def dataset_detail(dataset_id: int) -> str:
        return f"datasets:detail:{dataset_id}"
    
    @staticmethod
    def gpu_metrics() -> str:
        return "gpu:metrics"
    
    @staticmethod
    def user_session(user_id: int) -> str:
        return f"user:session:{user_id}"
    
    @staticmethod
    def system_config() -> str:
        return "system:config"
