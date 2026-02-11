"""
请求限流中间件
"""
import time
from typing import Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import threading

@dataclass
class RateLimitConfig:
    """限流配置"""
    requests: int = 100  # 每窗口请求数
    window: int = 60      # 窗口时间(秒)
    
@dataclass
class RateLimitEntry:
    """限流记录"""
    count: int = 0
    window_start: float = field(default_factory=time.time)
    remaining: int = 100
    reset_time: int = 60

class RateLimiter:
    """令牌桶限流器"""
    
    def __init__(self, default_config: RateLimitConfig = None):
        self.config = default_config or RateLimitConfig()
        self._buckets: Dict[str, RateLimitEntry] = {}
        self._lock = threading.Lock()
    
    def _get_bucket(self, key: str) -> RateLimitEntry:
        """获取或创建桶"""
        with self._lock:
            now = time.time()
            entry = self._buckets.get(key)
            
            if entry is None:
                entry = RateLimitEntry(
                    count=0,
                    window_start=now,
                    remaining=self.config.requests
                )
                self._buckets[key] = entry
                return entry
            
            # 检查窗口是否过期
            if now - entry.window_start >= self.config.window:
                entry.count = 0
                entry.window_start = now
                entry.remaining = self.config.requests
            
            return entry
    
    def allow_request(self, key: str) -> Tuple[bool, Dict]:
        """检查是否允许请求"""
        now = time.time()
        entry = self._get_bucket(key)
        
        with self._lock:
            # 检查是否超限
            if entry.count >= self.config.requests:
                reset_in = int(self.config.window - (now - entry.window_start))
                return False, {
                    "remaining": 0,
                    "reset_in": reset_in,
                    "limit": self.config.requests
                }
            
            entry.count += 1
            entry.remaining = self.config.requests - entry.count
            reset_in = int(self.config.window - (now - entry.window_start))
            
            return True, {
                "remaining": entry.remaining,
                "reset_in": reset_in,
                "limit": self.config.requests
            }
    
    def get_headers(self, key: str) -> Dict:
        """获取限流头信息"""
        entry = self._get_bucket(key)
        now = time.time()
        reset_in = int(self.config.window - (now - entry.window_start))
        
        return {
            "X-RateLimit-Limit": str(self.config.requests),
            "X-RateLimit-Remaining": str(entry.remaining),
            "X-RateLimit-Reset-In": str(reset_in)
        }
    
    def reset(self, key: str = None):
        """重置限流"""
        with self._lock:
            if key:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()

# 全局限流器
rate_limiter = RateLimiter(RateLimitConfig(requests=100, window=60))

# 便捷函数
def check_rate_limit(key: str) -> Tuple[bool, Dict]:
    return rate_limiter.allow_request(key)

def get_rate_limit_headers(key: str) -> Dict:
    return rate_limiter.get_headers(key)
