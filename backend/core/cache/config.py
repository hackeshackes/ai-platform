"""
Redis缓存配置
"""
from typing import Optional
import os

class CacheConfig:
    """缓存配置"""
    
    # Redis连接配置
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD', None)
    
    # 连接池配置
    MAX_CONNECTIONS: int = 50
    CONNECT_TIMEOUT: int = 5
    SOCKET_TIMEOUT: int = 5
    
    # 缓存策略配置
    DEFAULT_TTL: int = 3600  # 默认过期时间(秒)
    MAX_MEMORY: str = "256mb"
    
    # LRU配置
    MAX_ITEMS: int = 10000
    
    # 监控配置
    ENABLE_STATS: bool = True
