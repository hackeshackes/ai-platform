"""
Rate Limiter - 请求限流器

提供多种限流算法：
- 令牌桶算法 (Token Bucket)
- 滑动窗口算法 (Sliding Window)
- 固定窗口算法 (Fixed Window)
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import threading
import math

class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"

@dataclass
class RateLimitConfig:
    """限流配置"""
    requests: int  # 允许的请求数
    window_seconds: int  # 时间窗口（秒）
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_multiplier: float = 1.0  # 突发倍率

@dataclass
class RateLimitResult:
    """限流结果"""
    allowed: bool
    remaining: int
    reset_at: float
    limit: int
    current: int
    retry_after: Optional[float] = None

class TokenBucket:
    """令牌桶限流器"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """尝试消费令牌"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                remaining = int(self.tokens)
                # 计算下次 refill 时间
                time_until_full = (self.capacity - self.tokens) / self.refill_rate
                return True, time_until_full
            else:
                # 计算需要等待的时间
                time_until_full = (tokens - self.tokens) / self.refill_rate
                return False, time_until_full
    
    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def get_remaining(self) -> int:
        """获取剩余令牌"""
        self._refill()
        return int(self.tokens)
    
    def get_reset_time(self) -> float:
        """获取桶满时间"""
        self._refill()
        if self.tokens >= self.capacity:
            return time.time()
        return time.time() + (self.capacity - self.tokens) / self.refill_rate

class SlidingWindowCounter:
    """滑动窗口限流器"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}  # key -> [timestamps]
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> Tuple[bool, int, float]:
        """检查是否允许请求"""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            if key not in self.requests:
                self.requests[key] = []
            
            # 清理过期请求
            self.requests[key] = [t for t in self.requests[key] if t > window_start]
            
            current = len(self.requests[key])
            
            if current < self.max_requests:
                self.requests[key].append(now)
                remaining = self.max_requests - current - 1
                reset_at = now + self.window_seconds
                return True, remaining, reset_at
            
            # 找到最早的请求时间
            oldest = min(self.requests[key]) if self.requests[key] else now
            retry_after = oldest + self.window_seconds - now
            return False, 0, retry_after
    
    def get_current(self, key: str) -> int:
        """获取当前窗口的请求数"""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            if key not in self.requests:
                return 0
            
            return len([t for t in self.requests[key] if t > window_start])
    
    def cleanup(self):
        """清理过期数据"""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            for key in list(self.requests.keys()):
                self.requests[key] = [t for t in self.requests[key] if t > window_start]
                if not self.requests[key]:
                    del self.requests[key]

class FixedWindowCounter:
    """固定窗口限流器"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.windows: Dict[str, Tuple[int, float]] = {}  # key -> (count, window_start)
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> Tuple[bool, int, float]:
        """检查是否允许请求"""
        with self.lock:
            now = time.time()
            window_start = (now // self.window_seconds) * self.window_seconds
            
            if key not in self.windows:
                self.windows[key] = (0, window_start)
            
            count, stored_start = self.windows[key]
            
            if stored_start != window_start:
                # 新窗口
                count = 0
                stored_start = window_start
            
            if count < self.max_requests:
                self.windows[key] = (count + 1, window_start)
                remaining = self.max_requests - count - 1
                reset_at = window_start + self.window_seconds
                return True, remaining, reset_at
            
            reset_at = window_start + self.window_seconds
            retry_after = reset_at - now
            return False, 0, retry_after

class RateLimiter:
    """主限流器"""
    
    def __init__(self):
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindowCounter] = {}
        self.fixed_windows: Dict[str, FixedWindowCounter] = {}
        self.default_config = RateLimitConfig(
            requests=100,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        self.lock = threading.Lock()
    
    def create_limiter(self, key: str, config: RateLimitConfig) -> RateLimitAlgorithm:
        """为key创建限流器"""
        with self.lock:
            if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                refill_rate = config.requests / config.window_seconds
                burst_capacity = int(config.requests * config.burst_multiplier)
                self.token_buckets[key] = TokenBucket(burst_capacity, refill_rate)
            elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                self.sliding_windows[key] = SlidingWindowCounter(
                    config.requests, config.window_seconds
                )
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                self.fixed_windows[key] = FixedWindowCounter(
                    config.requests, config.window_seconds
                )
            return config.algorithm
    
    def check_rate_limit(
        self,
        key: str,
        config: Optional[RateLimitConfig] = None
    ) -> RateLimitResult:
        """检查限流"""
        if config is None:
            config = self.default_config
        
        # 确保限流器存在
        if key not in self.token_buckets:
            if key not in self.sliding_windows:
                if key not in self.fixed_windows:
                    self.create_limiter(key, config)
        
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            limiter = self.token_buckets.get(key)
            if limiter:
                allowed, reset_after = limiter.consume()
                remaining = limiter.get_remaining()
                reset_at = time.time() + reset_after
                return RateLimitResult(
                    allowed=allowed,
                    remaining=max(0, remaining),
                    reset_at=reset_at,
                    limit=limiter.capacity,
                    current=limiter.capacity - remaining,
                    retry_after=reset_after if not allowed else None
                )
        
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            limiter = self.sliding_windows.get(key)
            if limiter:
                allowed, remaining, reset_at = limiter.is_allowed(key)
                current = config.requests - remaining
                return RateLimitResult(
                    allowed=allowed,
                    remaining=max(0, remaining),
                    reset_at=reset_at,
                    limit=config.requests,
                    current=current,
                    retry_after=reset_at - time.time() if not allowed else None
                )
        
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            limiter = self.fixed_windows.get(key)
            if limiter:
                allowed, remaining, reset_at = limiter.is_allowed(key)
                current = config.requests - remaining
                return RateLimitResult(
                    allowed=allowed,
                    remaining=max(0, remaining),
                    reset_at=reset_at,
                    limit=config.requests,
                    current=current,
                    retry_after=reset_at - time.time() if not allowed else None
                )
        
        # 默认允许
        return RateLimitResult(
            allowed=True,
            remaining=config.requests,
            reset_at=time.time() + config.window_seconds,
            limit=config.requests,
            current=0
        )
    
    def get_usage(self, key: str, config: Optional[RateLimitConfig] = None) -> Dict:
        """获取使用统计"""
        if config is None:
            config = self.default_config
        
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            limiter = self.token_buckets.get(key)
            if limiter:
                return {
                    "remaining": limiter.get_remaining(),
                    "capacity": limiter.capacity,
                    "refill_rate": limiter.refill_rate
                }
        
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            limiter = self.sliding_windows.get(key)
            if limiter:
                return {
                    "current": limiter.get_current(key),
                    "limit": config.requests,
                    "window_seconds": config.window_seconds
                }
        
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            limiter = self.fixed_windows.get(key)
            if limiter:
                now = time.time()
                window_start = (now // config.window_seconds) * config.window_seconds
                count = limiter.windows.get(key, (0, window_start))[0] if key in limiter.windows else 0
                return {
                    "current": count,
                    "limit": config.requests,
                    "window_seconds": config.window_seconds
                }
        
        return {"remaining": config.requests, "limit": config.requests}
    
    def reset(self, key: str):
        """重置限流器"""
        self.token_buckets.pop(key, None)
        self.sliding_windows.pop(key, None)
        self.fixed_windows.pop(key, None)
    
    def cleanup_expired(self):
        """清理过期数据"""
        for limiter in self.sliding_windows.values():
            limiter.cleanup()

# 全局限流器实例
limiter = RateLimiter()
