"""
Redis缓存管理器
"""
import json
import time
import hashlib
from typing import Any, Optional, Dict, List
from functools import wraps
import threading

class RedisManager:
    """Redis缓存管理器"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._local_cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._connected = False
        
        # 尝试连接Redis
        try:
            import redis
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.client.ping()
            self._connected = True
            print(f"✅ Redis connected: {host}:{port}")
        except Exception as e:
            print(f"⚠️ Redis not available, using local cache: {e}")
            self.client = None
            self._connected = False
    
    def _make_key(self, prefix: str, *args) -> str:
        """生成缓存key"""
        key_data = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"aip:{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # 先查本地缓存
        with self._lock:
            if key in self._local_cache:
                value, expire_at = self._local_cache[key]
                if time.time() < expire_at:
                    return value
                del self._local_cache[key]
        
        # 查Redis
        if self._connected and self.client:
            try:
                data = self.client.get(key)
                if data:
                    value = json.loads(data)
                    # 同步到本地
                    with self._lock:
                        self._local_cache[key] = (value, time.time() + 300)
                    return value
            except Exception:
                pass
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存"""
        # 保存到本地
        with self._lock:
            self._local_cache[key] = (value, time.time() + min(ttl, 300))
        
        # 保存到Redis
        if self._connected and self.client:
            try:
                self.client.setex(key, ttl, json.dumps(value))
                return True
            except Exception:
                pass
        
        return True
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            self._local_cache.pop(key, None)
        
        if self._connected and self.client:
            try:
                self.client.delete(key)
                return True
            except Exception:
                pass
        
        return False
    
    def delete_pattern(self, pattern: str) -> int:
        """删除匹配的所有key"""
        count = 0
        with self._lock:
            keys_to_delete = [k for k in self._local_cache if pattern in k]
            for k in keys_to_delete:
                del self._local_cache[k]
            count = len(keys_to_delete)
        
        if self._connected and self.client:
            try:
                keys = self.client.keys(f"*{pattern}*")
                count += self.client.delete(*keys)
            except Exception:
                pass
        
        return count
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._local_cache.clear()
        
        if self._connected and self.client:
            try:
                self.client.flushdb()
            except Exception:
                pass
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            "local_cache_size": len(self._local_cache),
            "connected": self._connected,
            "redis_enabled": self._connected
        }

# 全局实例
cache_manager = RedisManager()

# 便捷函数
def get_cache(key: str) -> Optional[Any]:
    return cache_manager.get(key)

def set_cache(key: str, value: Any, ttl: int = 3600):
    return cache_manager.set(key, value, ttl)

def delete_cache(key: str):
    return cache_manager.delete(key)

def invalidate_pattern(pattern: str):
    return cache_manager.delete_pattern(pattern)
