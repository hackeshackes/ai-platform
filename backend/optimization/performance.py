"""
性能优化模块 v2.3
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from redis import Redis
import json

@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    keys_count: int = 0

@dataclass
class QueryStats:
    """查询统计"""
    query: str
    count: int = 0
    avg_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis: Optional[Redis] = None
        self.cache_stats = CacheStats()
        self.query_stats: List[QueryStats] = []
        self.query_counts: Dict[str, int] = {}
        self.query_times: Dict[str, List[float]] = {}
        
        try:
            self.redis = Redis.from_url(redis_url, decode_responses=True)
        except Exception:
            print("Redis not available, using in-memory cache")
            self._memory_cache: Dict[str, Any] = {}
    
    # 缓存管理
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if self.redis:
            value = self.redis.get(key)
            if value:
                self.cache_stats.hits += 1
                return json.loads(value)
            else:
                self.cache_stats.misses += 1
                return None
        else:
            value = self._memory_cache.get(key)
            if value:
                self.cache_stats.hits += 1
                return value
            else:
                self.cache_stats.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        if self.redis:
            self.redis.setex(key, ttl, json.dumps(value))
        else:
            self._memory_cache[key] = {
                "value": value,
                "expire": datetime.utcnow().timestamp() + ttl
            }
        self.cache_stats.size += 1
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        if self.redis:
            return self.redis.delete(key) > 0
        else:
            if key in self._memory_cache:
                del self._memory_cache[key]
                return True
            return False
    
    async def clear(self):
        """清空缓存"""
        if self.redis:
            self.redis.flushdb()
        else:
            self._memory_cache.clear()
        self.cache_stats = CacheStats()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.cache_stats.hits + self.cache_stats.misses
        hit_rate = (self.cache_stats.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.cache_stats.hits,
            "misses": self.cache_stats.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": self.cache_stats.size
        }
    
    # 查询优化
    async def track_query(self, query: str, time_ms: float):
        """追踪查询"""
        # 更新计数
        self.query_counts[query] = self.query_counts.get(query, 0) + 1
        
        # 更新时间
        if query not in self.query_times:
            self.query_times[query] = []
        self.query_times[query].append(time_ms)
        
        # 限制历史
        if len(self.query_times[query]) > 100:
            self.query_times[query] = self.query_times[query][-100:]
    
    def get_slow_queries(self, threshold_ms: float = 100) -> List[Dict]:
        """获取慢查询"""
        slow_queries = []
        for query, times in self.query_times.items():
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > threshold_ms:
                    slow_queries.append({
                        "query": query[:100],
                        "count": len(times),
                        "avg_time_ms": round(avg_time, 2),
                        "max_time_ms": round(max(times), 2)
                    })
        
        slow_queries.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        return slow_queries[:20]
    
    def get_query_stats(self) -> List[Dict]:
        """获取查询统计"""
        stats = []
        for query, times in self.query_times.items():
            if times:
                stats.append({
                    "query": query[:100],
                    "count": len(times),
                    "avg_time_ms": round(sum(times) / len(times), 2),
                    "min_time_ms": round(min(times), 2),
                    "max_time_ms": round(max(times), 2)
                })
        
        stats.sort(key=lambda x: x["count"], reverse=True)
        return stats[:50]
    
    # 连接池管理
    async def get_connection_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        # 模拟连接池统计
        return {
            "total_connections": 10,
            "active_connections": 5,
            "idle_connections": 5,
            "wait_count": 0
        }
    
    # 性能指标
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "cache": self.get_cache_stats(),
            "total_queries": sum(len(times) for times in self.query_times.values()),
            "slow_query_count": len(self.get_slow_queries()),
            "avg_query_time": self._calc_avg_query_time()
        }
    
    def _calc_avg_query_time(self) -> float:
        """计算平均查询时间"""
        all_times = []
        for times in self.query_times.values():
            all_times.extend(times)
        
        if all_times:
            return round(sum(all_times) / len(all_times), 2)
        return 0.0

# PerformanceOptimizer实例
performance_optimizer = PerformanceOptimizer()
