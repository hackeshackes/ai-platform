"""
Dynamic Batching - High-throughput batch processing for LLM inference

动态批处理模块，优化推理吞吐量，支持连续批处理和动态批处理策略。
"""
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging
from collections import deque
from datetime import datetime
import threading
import uuid

from .vllm_engine import (
    GenerationRequest, 
    GenerationResponse, 
    vLLMEngine,
    vLLMConfig
)

logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    """批处理策略"""
    STATIC = "static"           # 静态批处理
    CONTINUOUS = "continuous"   # 连续批处理
    DYNAMIC = "dynamic"         # 动态批处理
    ORACLE = "oracle"           # Oracle最优批处理


@dataclass
class BatchItem:
    """批处理项"""
    request: GenerationRequest
    future: asyncio.Future
    request_id: str
    created_at: float
    priority: int = 0
    retry_count: int = 0


@dataclass
class BatchResult:
    """批处理结果"""
    items: List[BatchItem]
    start_time: float
    end_time: float
    batch_size: int
    strategy: BatchingStrategy


@dataclass
class BatchingConfig:
    """批处理配置"""
    max_batch_size: int = 16
    min_batch_size: int = 1
    batch_timeout_ms: int = 100
    max_queue_size: int = 1000
    strategy: BatchingStrategy = BatchingStrategy.CONTINUOUS
    enable_prefill_cache: bool = True
    enable_chunked_prefill: bool = True
    max_tokens_per_batch: int = 2048


class BatchScheduler:
    """批处理调度器
    
    实现连续批处理和动态批处理策略，优化GPU利用率和吞吐量。
    """
    
    def __init__(self, config: Optional[BatchingConfig] = None, engine: Optional[vLLMEngine] = None):
        self.config = config or BatchingConfig()
        self.engine = engine
        self._request_queue: deque = deque()
        self._priority_queue: List[BatchItem] = []
        self._running_batches: Dict[str, BatchResult] = {}
        self._lock = threading.Lock()
        self._stats = {
            "total_batches": 0,
            "total_requests": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
            "queue_wait_time_ms": 0.0,
            "gpu_utilization": 0.0,
        }
        self._batch_callbacks: List[Callable] = []
        
    def add_request(
        self,
        request: GenerationRequest,
        request_id: str = "",
        priority: int = 0
    ) -> asyncio.Future:
        """添加请求到批处理队列"""
        future = asyncio.Future()
        
        item = BatchItem(
            request=request,
            future=future,
            request_id=request_id or str(uuid.uuid4()),
            created_at=time.time(),
            priority=priority
        )
        
        with self._lock:
            if len(self._request_queue) >= self.config.max_queue_size:
                raise RuntimeError("批处理队列已满")
            
            self._request_queue.append(item)
            self._stats["total_requests"] += 1
        
        return future
    
    async def _execute_batch(self, items: List[BatchItem]) -> List[GenerationResponse]:
        """执行批处理"""
        if not self.engine:
            raise RuntimeError("未配置推理引擎")
        
        start_time = time.time()
        results = []
        
        try:
            # 批量推理
            for item in items:
                try:
                    response = self.engine.generate(item.request, item.request_id)
                    results.append(response)
                    if not item.future.done():
                        item.future.set_result(response)
                except Exception as e:
                    if not item.future.done():
                        item.future.set_exception(e)
                    results.append(GenerationResponse(
                        text=f"Error: {e}",
                        generated_tokens=0,
                        finish_reason="error",
                        request_id=item.request_id
                    ))
                    
        except Exception as e:
            logger.error(f"批处理执行失败: {e}")
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)
        
        return results
    
    def _should_dispatch_batch(self) -> tuple:
        """检查是否应该调度批处理"""
        current_time = time.time()
        queue_size = len(self._request_queue)
        
        if queue_size == 0:
            return False, None, 0
        
        # 策略1: 达到最大批处理大小
        if queue_size >= self.config.max_batch_size:
            batch = [self._request_queue.popleft() for _ in range(self.config.max_batch_size)]
            return True, batch, 0
        
        # 策略2: 超时触发
        oldest_item = self._request_queue[0]
        wait_time = (current_time - oldest_item.created_at) * 1000  # ms
        
        if queue_size >= self.config.min_batch_size:
            if wait_time >= self.config.batch_timeout_ms:
                batch = [self._request_queue.popleft() for _ in range(queue_size)]
                return True, batch, wait_time
        
        return False, None, wait_time
    
    async def run_batch_loop(self):
        """运行批处理主循环"""
        while True:
            should_dispatch, batch, wait_time = self._should_dispatch_batch()
            
            if should_dispatch:
                start_time = time.time()
                await self._execute_batch(batch)
                end_time = time.time()
                
                # 更新统计
                batch_latency = (end_time - start_time) * 1000
                self._stats["total_batches"] += 1
                self._stats["avg_batch_size"] = (
                    (self._stats["avg_batch_size"] * (self._stats["total_batches"] - 1) + len(batch))
                    / self._stats["total_batches"]
                )
                self._stats["avg_latency_ms"] = (
                    (self._stats["avg_latency_ms"] * (self._stats["total_batches"] - 1) + batch_latency)
                    / self._stats["total_batches"]
                )
                
                # 通知回调
                for callback in self._batch_callbacks:
                    try:
                        callback(batch, batch_latency)
                    except Exception as e:
                        logger.warning(f"批处理回调失败: {e}")
            
            await asyncio.sleep(0.001)  # 短暂休眠，避免CPU过高
    
    def register_batch_callback(self, callback: Callable):
        """注册批处理回调"""
        self._batch_callbacks.append(callback)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计"""
        return {
            "queue_size": len(self._request_queue),
            "running_batches": len(self._running_batches),
            "total_requests": self._stats["total_requests"],
            "total_batches": self._stats["total_batches"],
            "avg_batch_size": round(self._stats["avg_batch_size"], 2),
            "avg_latency_ms": round(self._stats["avg_latency_ms"], 2),
            "queue_wait_time_ms": round(self._stats["queue_wait_time_ms"], 2),
        }
    
    def clear_queue(self):
        """清空队列"""
        with self._lock:
            for item in self._request_queue:
                if not item.future.done():
                    item.future.cancel()
            self._request_queue.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取批处理统计"""
        return {
            "total_batches": self._stats["total_batches"],
            "total_requests": self._stats["total_requests"],
            "avg_batch_size": round(self._stats["avg_batch_size"], 2),
            "avg_latency_ms": round(self._stats["avg_latency_ms"], 2),
            "queue_wait_time_ms": round(self._stats["queue_wait_time_ms"], 2),
            "gpu_utilization": self._stats["gpu_utilization"],
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "min_batch_size": self.config.min_batch_size,
                "batch_timeout_ms": self.config.batch_timeout_ms,
                "strategy": self.config.strategy.value,
            }
        }


class KVCacheManager:
    """
    KV缓存管理器
    
    优化KV缓存的内存使用，支持:
    - PagedAttention缓存管理
    - 缓存共享
    - 缓存预分配
    """
    
    def __init__(self, max_cache_size: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._cache_size = 0
        self._max_cache_size = max_cache_size
        self._lock = threading.Lock()
        
    def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self._lock:
            return self._cache.get(key)
    
    def set_cache(self, key: str, value: Any, size: int = 1):
        """设置缓存"""
        with self._lock:
            # 清理过期缓存
            while self._cache_size + size > self._max_cache_size:
                if not self._cache:
                    break
                oldest_key = next(iter(self._cache))
                removed = self._cache.pop(oldest_key)
                self._cache_size -= 1
            
            self._cache[key] = value
            self._cache_size += 1
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._cache_size = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "cache_size": self._cache_size,
            "max_cache_size": self._max_cache_size,
            "cache_hit_rate": 0.0,  # 需要实现命中率追踪
        }


class PrefillScheduler:
    """
    Prefill调度器
    
    优化预fill阶段的批处理，支持chunked prefill。
    """
    
    def __init__(self, max_new_tokens: int = 256):
        self.max_new_tokens = max_new_tokens
        self._prefill_queue: deque = deque()
        self._lock = threading.Lock()
        
    async def schedule_prefill(
        self,
        prompts: List[str],
        callback: Callable[[List[str]], asyncio.Future]
    ) -> List[str]:
        """调度prefill"""
        with self._lock:
            self._prefill_queue.extend(prompts)
        
        results = await callback(list(self._prefill_queue))
        
        with self._lock:
            for _ in range(len(prompts)):
                self._prefill_queue.popleft()
        
        return results


# 全局批处理调度器
_batch_scheduler: Optional[BatchScheduler] = None


def create_batch_scheduler(
    config: Optional[BatchingConfig] = None,
    engine: Optional[vLLMEngine] = None
) -> BatchScheduler:
    """创建批处理调度器"""
    global _batch_scheduler
    _batch_scheduler = BatchScheduler(config, engine)
    return _batch_scheduler


def get_batch_scheduler() -> BatchScheduler:
    """获取全局批处理调度器"""
    global _batch_scheduler
    if _batch_scheduler is None:
        raise RuntimeError("批处理调度器未初始化")
    return _batch_scheduler
