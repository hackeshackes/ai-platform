"""
缓存模块
"""
from .manager import RedisManager, cache_manager
from .decorators import cached, cache_invalidate
from .config import CacheConfig

# 便捷函数别名 (兼容旧API)
cache = cache_manager

__all__ = [
    'RedisManager',
    'cache_manager',
    'cached',
    'cache_invalidate',
    'CacheConfig',
    'cache'  # 兼容旧API
]
