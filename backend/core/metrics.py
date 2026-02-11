"""
性能指标模块 - MetricsCollector

收集和展示系统性能指标
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 计数器
    GAUGE = "gauge"          # 瞬时值
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"      # 摘要


@dataclass
class Metric:
    """指标数据"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


class MetricsCollector:
    """
    性能指标收集器
    
    支持计数器、瞬时值、直方图等指标类型
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化指标收集器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._initialized = False
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._histogram_buckets = self.config.get('histogram_buckets', [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        self._max_histogram_samples = self.config.get('max_histogram_samples', 1000)
        
        # API指标
        self._request_count = 0
        self._request_latencies: List[float] = []
        self._error_count = 0
        self._active_connections = 0
        
        # 数据库指标
        self._db_query_count = 0
        self._db_query_latencies: List[float] = []
        self._slow_queries = 0
        self._db_connections_active = 0
        self._db_connections_idle = 0
    
    async def initialize(self) -> None:
        """初始化指标收集器"""
        self._initialized = True
        logger.info("指标收集器初始化完成")
    
    async def shutdown(self) -> None:
        """关闭指标收集器"""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._initialized = False
        logger.info("指标收集器已关闭")
    
    # ==================== 基础指标操作 ====================
    
    def counter_inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        增加计数器
        
        Args:
            name: 指标名称
            value: 增加的值
            labels: 标签
        """
        key = self._make_key(name, labels)
        self._counters[key] += value
        
        # 记录指标
        self._metrics[name].append(Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self._counters[key],
            labels=labels or {}
        ))
    
    def counter_dec(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        减少计数器
        
        Args:
            name: 指标名称
            value: 减少的值
            labels: 标签
        """
        key = self._make_key(name, labels)
        self._counters[key] = max(0, self._counters[key] - value)
    
    def gauge_set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        设置瞬时值
        
        Args:
            name: 指标名称
            value: 值
            labels: 标签
        """
        key = self._make_key(name, labels)
        self._gauges[key] = value
        
        # 记录指标
        self._metrics[name].append(Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {}
        ))
    
    def gauge_inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """增加瞬时值"""
        key = self._make_key(name, labels)
        self._gauges[key] = self._gauges.get(key, 0) + value
    
    def gauge_dec(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """减少瞬时值"""
        key = self._make_key(name, labels)
        self._gauges[key] = self._gauges.get(key, 0) - value
    
    def histogram_observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        观察直方图值
        
        Args:
            name: 指标名称
            value: 观察值
            labels: 标签
        """
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        
        # 限制样本数量
        if len(self._histograms[key]) > self._max_histogram_samples:
            self._histograms[key] = self._histograms[key][-self._max_histogram_samples:]
        
        # 记录指标
        self._metrics[name].append(Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {}
        ))
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """生成指标键"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}[{label_str}]"
        return name
    
    # ==================== 系统指标 ====================
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """
        收集系统指标
        
        Returns:
            系统指标字典
        """
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # 更新瞬时值
            self.gauge_set("system_cpu_percent", cpu_percent)
            self.gauge_set("system_memory_percent", memory.percent)
            self.gauge_set("system_disk_percent", disk.percent)
            self.gauge_set("system_network_bytes_sent", network.bytes_sent)
            self.gauge_set("system_network_bytes_recv", network.bytes_recv)
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free // (1024 * 1024 * 1024),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            }
        except ImportError:
            # psutil 未安装时返回模拟数据
            return {
                "cpu_percent": 25,
                "memory_percent": 45,
                "memory_available_mb": 4096,
                "disk_percent": 35,
                "disk_free_gb": 200,
                "network_bytes_sent": 1024000,
                "network_bytes_recv": 2048000,
                "mock": True
            }
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return {"error": str(e)}
    
    # ==================== API指标 ====================
    
    def record_request(self, latency: float, success: bool = True) -> None:
        """
        记录API请求
        
        Args:
            latency: 响应延迟（秒）
            success: 是否成功
        """
        self._request_count += 1
        self._request_latencies.append(latency)
        self.histogram_observe("api_request_latency", latency, {"method": "all"})
        
        if not success:
            self._error_count += 1
        
        # 记录计数器
        self.counter_inc("api_requests_total", 1.0, {"status": "success" if success else "error"})
    
    def record_db_query(self, latency: float, slow: bool = False) -> None:
        """
        记录数据库查询
        
        Args:
            latency: 查询延迟（秒）
            slow: 是否为慢查询
        """
        self._db_query_count += 1
        self._db_query_latencies.append(latency)
        self.histogram_observe("db_query_latency", latency)
        
        if slow:
            self._slow_queries += 1
            self.counter_inc("db_slow_queries_total")
    
    def connection_opened(self) -> None:
        """记录新连接"""
        self._active_connections += 1
        self.gauge_inc("api_active_connections")
    
    def connection_closed(self) -> None:
        """记录连接关闭"""
        self._active_connections = max(0, self._active_connections - 1)
        self.gauge_dec("api_active_connections")
    
    # ==================== 聚合指标 ====================
    
    def _calculate_latency_percentiles(self, latencies: List[float]) -> Dict[str, float]:
        """计算延迟百分位"""
        if not latencies:
            return {"p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        def percentile(p: float) -> float:
            idx = int(n * p)
            return sorted_latencies[min(idx, n - 1)]
        
        return {
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99)
        }
    
    def _calculate_histogram_percentiles(self, values: List[float]) -> Dict[str, float]:
        """计算直方图百分位"""
        if not values:
            return {"min": 0, "max": 0, "avg": 0, **self._calculate_latency_percentiles(values)}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        def percentile(p: float) -> float:
            idx = int(n * p)
            return sorted_values[min(idx, n - 1)]
        
        return {
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(values) / n,
            **self._calculate_latency_percentiles(values)
        }
    
    # ==================== 收集所有指标 ====================
    
    async def collect_all(self) -> Dict[str, Any]:
        """
        收集所有性能指标
        
        Returns:
            完整指标字典
        """
        # 系统指标
        system_metrics = await self.collect_system_metrics()
        
        # API指标
        latency_percentiles = self._calculate_latency_percentiles(self._request_latencies)
        error_rate = (self._error_count / self._request_count * 100) if self._request_count > 0 else 0
        
        api_metrics = {
            "requests_total": self._request_count,
            "requests_active": self._active_connections,
            "errors_total": self._error_count,
            "error_rate_percent": round(error_rate, 2),
            "latency": {
                **self._calculate_histogram_percentiles(self._request_latencies),
                "unit": "seconds"
            }
        }
        
        # 数据库指标
        db_latency_percentiles = self._calculate_latency_percentiles(self._db_query_latencies)
        db_metrics = {
            "queries_total": self._db_query_count,
            "slow_queries_total": self._slow_queries,
            "connections_active": self._db_connections_active,
            "connections_idle": self._db_connections_idle,
            "latency": {
                **self._calculate_histogram_percentiles(self._db_query_latencies),
                "unit": "seconds"
            }
        }
        
        # 计数器
        counters = {
            name: value for name, value in self._counters.items()
        }
        
        # 瞬时值
        gauges = {
            name: value for name, value in self._gauges.items()
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_metrics,
            "api": api_metrics,
            "database": db_metrics,
            "counters": counters,
            "gauges": gauges,
            "histograms": {
                name: self._calculate_histogram_percentiles(values)
                for name, values in self._histograms.items()
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取指标收集器状态
        
        Returns:
            状态信息
        """
        return {
            "initialized": self._initialized,
            "metrics_count": sum(len(v) for v in self._metrics.values()),
            "counters_count": len(self._counters),
            "gauges_count": len(self._gauges),
            "histograms_count": len(self._histograms),
            "request_count": self._request_count,
            "db_query_count": self._db_query_count
        }
    
    # ==================== Prometheus格式 ====================
    
    def to_prometheus_format(self) -> str:
        """
        导出为Prometheus格式
        
        Returns:
            Prometheus格式的指标字符串
        """
        lines = []
        
        # 计数器
        for name, value in self._counters.items():
            key = name.replace("[", "{").replace("]", "}").replace("=", "=\"")
            if key.endswith("}"):
                key = key[:-1] + '"}'
            lines.append(f"# TYPE {name.split('[')[0]} counter")
            lines.append(f"{key} {value}")
        
        # 瞬时值
        for name, value in self._gauges.items():
            key = name.replace("[", "{").replace("]", "}").replace("=", "=\"")
            if key.endswith("}"):
                key = key[:-1] + '"}'
            lines.append(f"# TYPE {name.split('[')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # 直方图
        for name, values in self._histograms.items():
            base_name = name.split('[')[0]
            lines.append(f"# TYPE {base_name} histogram")
            
            for value in values:
                lines.append(f"{base_name}_bucket{{le=\"{value}\"}} 0")
            lines.append(f"{base_name}_bucket{{le=\"+Inf\"}} {len(values)}")
            lines.append(f"{base_name}_sum {sum(values)}")
            lines.append(f"{base_name}_count {len(values)}")
        
        return "\n".join(lines)
