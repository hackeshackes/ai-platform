"""
配置管理 - Performance Tuner v12
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class OptimizationStrategy(Enum):
    """优化策略"""
    CONSERVATIVE = "conservative"  # 保守策略，最小变更
    MODERATE = "moderate"          # 中等优化
    AGGRESSIVE = "aggressive"      # 激进优化


@dataclass
class PerformanceConfig:
    """性能配置"""
    # 分析配置
    analysis_interval: int = 60           # 分析间隔(秒)
    metrics_sample_rate: float = 1.0     # 指标采样率
    analysis_depth: str = "medium"       # 分析深度: shallow/medium/deep

    # 自动调优配置
    strategy: OptimizationStrategy = OptimizationStrategy.MODERATE
    max_downtime: int = 300              # 最大停机时间(秒)
    auto_apply: bool = False             # 是否自动应用优化
    validation_required: bool = True     # 是否需要验证

    # 优化配置
    db_optimization: bool = True         # 数据库优化
    cache_optimization: bool = True      # 缓存优化
    api_optimization: bool = True        # API优化
    memory_optimization: bool = True     # 内存优化
    network_optimization: bool = True    # 网络优化

    # 目标配置
    target_cpu_reduction: float = 0.3    # CPU降低目标
    target_memory_reduction: float = 0.2 # 内存降低目标
    target_latency_reduction: float = 0.3 # 延迟降低目标
    target_throughput_increase: float = 0.3  # 吞吐量提升目标

    # 约束配置
    max_retries: int = 3                 # 最大重试次数
    timeout_per_optimization: int = 60   # 单次优化超时(秒)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 基准配置
    warmup_requests: int = 10            # 预热请求数
    benchmark_duration: int = 60          # 基准测试持续时间(秒)
    concurrent_users: int = 10           # 并发用户数

    # 负载测试配置
    load_test_duration: int = 300         # 负载测试持续时间
    ramp_up_time: int = 60                # 爬坡时间
    max_concurrent: int = 100             # 最大并发

    # 压力测试配置
    stress_test_duration: int = 120       # 压力测试持续时间
    max_requests_per_second: int = 1000   # 最大每秒请求数

    # 稳定性测试配置
    stability_test_duration: int = 3600   # 稳定性测试持续时间(1小时)
    error_rate_threshold: float = 0.01    # 错误率阈值


@dataclass
class DatabaseOptimizationConfig:
    """数据库优化配置"""
    # 索引优化
    index_usage_threshold: float = 0.1    # 索引使用率阈值
    unused_index_retention_days: int = 30  # 未使用索引保留天数

    # 查询优化
    slow_query_threshold_ms: int = 1000   # 慢查询阈值(ms)
    max_query_complexity: int = 10         # 最大查询复杂度

    # 连接池
    min_connections: int = 5              # 最小连接数
    max_connections: int = 50             # 最大连接数
    connection_timeout: int = 30           # 连接超时(秒)

    # 读写分离
    read_replica_count: int = 2           # 读副本数量
    write_read_delay_ms: int = 5          # 读写延迟(ms)


@dataclass
class CacheOptimizationConfig:
    """缓存优化配置"""
    # 缓存策略
    default_ttl: int = 3600               # 默认TTL(秒)
    max_cache_size_mb: int = 512          # 最大缓存大小(MB)
    cache_hit_threshold: float = 0.8      # 缓存命中阈值

    # 预热配置
    preheat_enabled: bool = True          # 是否启用预热
    preheat_keys: List[str] = None        # 预热键列表

    # 淘汰策略
    eviction_policy: str = "lru"          # 淘汰策略: lru/lfu/fifo
    eviction_threshold: float = 0.9       # 淘汰阈值

    # 分布式缓存
    distributed_enabled: bool = False     # 是否启用分布式缓存
    redis_cluster: bool = False           # 是否使用Redis集群


@dataclass
class APIOptimizationConfig:
    """API优化配置"""
    # 并发配置
    max_concurrent_requests: int = 1000   # 最大并发请求数
    request_queue_size: int = 100         # 请求队列大小

    # 批处理
    batch_size: int = 100                 # 批处理大小
    batch_timeout_ms: int = 50            # 批处理超时(ms)

    # 压缩
    compression_enabled: bool = True      # 是否启用压缩
    compression_min_size: int = 1024      # 最小压缩大小(bytes)

    # 连接池
    connection_pool_size: int = 50        # 连接池大小
    keep_alive_timeout: int = 30         # 保持连接超时(秒)


@dataclass
class MemoryOptimizationConfig:
    """内存优化配置"""
    # 对象池
    object_pool_enabled: bool = True      # 是否启用对象池
    pool_size: int = 100                  # 池大小

    # 压缩
    compression_enabled: bool = True      # 是否启用压缩
    compression_threshold: int = 10000    # 压缩阈值(bytes)

    # 垃圾回收
    gc_frequency: int = 60                # GC频率(秒)
    gc_threshold_mb: int = 100            # GC阈值(MB)

    # 内存限制
    memory_limit_mb: int = 1024          # 内存限制(MB)
    memory_warning_threshold: float = 0.8 # 内存警告阈值


@dataclass
class NetworkOptimizationConfig:
    """网络优化配置"""
    # 连接池
    connection_pool_size: int = 100       # 连接池大小
    connection_timeout: int = 30          # 连接超时(秒)
    keep_alive_timeout: int = 60         # 保持连接超时(秒)

    # 压缩
    gzip_enabled: bool = True             # 是否启用gzip
    brotli_enabled: bool = False          # 是否启用brotli
    compression_level: int = 6            # 压缩级别(1-9)

    # 超时配置
    read_timeout: int = 30               # 读取超时(秒)
    write_timeout: int = 30               # 写入超时(秒)
    total_timeout: int = 60               # 总超时(秒)

    # 重试配置
    max_retries: int = 3                  # 最大重试次数
    retry_backoff_ms: int = 100           # 重试退避时间(ms)


# 默认配置实例
DEFAULT_CONFIG = PerformanceConfig()
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()
DEFAULT_DB_CONFIG = DatabaseOptimizationConfig()
DEFAULT_CACHE_CONFIG = CacheOptimizationConfig()
DEFAULT_API_CONFIG = APIOptimizationConfig()
DEFAULT_MEMORY_CONFIG = MemoryOptimizationConfig()
DEFAULT_NETWORK_CONFIG = NetworkOptimizationConfig()
