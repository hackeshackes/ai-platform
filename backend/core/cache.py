"""
Redis缓存管理器 v2.0
"""
import json
import redis
from typing import Optional, Any
from datetime import timedelta

class CacheManager:
    """Redis缓存管理器"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        self.redis = redis.Redis(
            host=host, port=port, db=db, password=password, decode_responses=True
        )
        self.prefix = "aiplatform:v2:"
        
        # TTL配置
        self.TTL = {
            "session": timedelta(hours=24),
            "user": timedelta(hours=1),
            "project_list": timedelta(minutes=5),
            "task_status": timedelta(seconds=30),
            "gpu_metrics": timedelta(seconds=5),
            "dataset_info": timedelta(hours=1),
            "model_info": timedelta(hours=1),
            "system_config": timedelta(hours=24),
        }
    
    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        data = self.redis.get(self._key(key))
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return None
    
    def set(self, key: str, value: Any, ttl_key: str = "system_config"):
        """设置缓存"""
        ttl = self.TTL.get(ttl_key, timedelta(minutes=5))
        data = json.dumps(value, default=str)
        self.redis.setex(self._key(key), ttl, data)
    
    def delete(self, key: str):
        """删除缓存"""
        self.redis.delete(self._key(key))
    
    def invalidate_pattern(self, pattern: str):
        """批量删除"""
        keys = self.redis.keys(self._key(pattern))
        if keys:
            self.redis.delete(*keys)
    
    def ping(self) -> bool:
        """检查连接"""
        try:
            return self.redis.ping()
        except redis.ConnectionError:
            return False

# 全局缓存实例
cache = CacheManager()
