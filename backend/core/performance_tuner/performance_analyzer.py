"""
性能分析器 - Performance Analyzer v12

功能:
- 瓶颈识别
- 资源分析
- 代码分析
- 数据库分析
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    CPU = "cpu"
    MEMORY = "memory"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    API = "api"


class BottleneckSeverity(Enum):
    """瓶颈严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricSample:
    """指标样本"""
    metric_type: MetricType
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Bottleneck:
    """性能瓶颈"""
    id: str
    name: str
    description: str
    severity: BottleneckSeverity
    metric_type: MetricType
    location: str
    impact_score: float
    suggested_fix: str
    affected_components: List[str]
    detected_at: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """性能报告"""
    id: str
    target: str
    generated_at: float
    duration: float

    # 摘要
    overall_score: float
    health_status: str

    # 指标汇总
    metrics_summary: Dict[str, Dict[str, float]]

    # 瓶颈列表
    bottlenecks: List[Bottleneck]

    # 资源使用情况
    resource_usage: Dict[str, Any]

    # 建议
    recommendations: List[str]

    # 趋势分析
    trends: Dict[str, List[Tuple[float, float]]]


class PerformanceAnalyzer:
    """
    性能分析器

    提供全面的性能分析功能，包括瓶颈识别、资源分析、代码分析和数据库分析。
    """

    def __init__(
        self,
        sample_interval: float = 1.0,
        max_samples: int = 3600,
        enable_deep_analysis: bool = False
    ):
        """
        初始化性能分析器

        Args:
            sample_interval: 采样间隔(秒)
            max_samples: 最大样本数
            enable_deep_analysis: 是否启用深度分析
        """
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.enable_deep_analysis = enable_deep_analysis

        # 指标存储
        self._samples: Dict[MetricType, List[MetricSample]] = defaultdict(list)
        self._lock = threading.Lock()

        # 监控系统
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # 瓶颈检测规则
        self._bottleneck_rules = self._init_bottleneck_rules()

    def _init_bottleneck_rules(self) -> Dict[str, Dict]:
        """初始化瓶颈检测规则"""
        return {
            "high_cpu": {
                "metric": MetricType.CPU,
                "condition": lambda v: v > 80,
                "severity": BottleneckSeverity.HIGH,
                "name": "CPU使用率过高",
                "fix": "考虑增加CPU资源或优化CPU密集型操作"
            },
            "high_memory": {
                "metric": MetricType.MEMORY,
                "condition": lambda v: v > 85,
                "severity": BottleneckSeverity.HIGH,
                "name": "内存使用率过高",
                "fix": "考虑增加内存或优化内存使用"
            },
            "high_latency": {
                "metric": MetricType.LATENCY,
                "condition": lambda v: v > 1000,
                "severity": BottleneckSeverity.MEDIUM,
                "name": "响应延迟过高",
                "fix": "优化代码逻辑或增加缓存"
            },
            "low_throughput": {
                "metric": MetricType.THROUGHPUT,
                "condition": lambda v: v < 100,
                "severity": BottleneckSeverity.MEDIUM,
                "name": "吞吐量过低",
                "fix": "考虑增加实例数量或优化处理逻辑"
            },
            "slow_query": {
                "metric": MetricType.DATABASE,
                "condition": lambda v: v > 1000,
                "severity": BottleneckSeverity.HIGH,
                "name": "数据库查询过慢",
                "fix": "添加索引或优化查询语句"
            },
            "low_cache_hit": {
                "metric": MetricType.CACHE,
                "condition": lambda v: v < 0.5,
                "severity": BottleneckSeverity.MEDIUM,
                "name": "缓存命中率过低",
                "fix": "调整缓存策略或增加缓存容量"
            }
        }

    async def start_monitoring(self, target: str = "system"):
        """开始监控"""
        if self._monitoring:
            logger.warning("监控已在运行")
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(target))

    async def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self, target: str):
        """监控循环"""
        while self._monitoring:
            try:
                # 收集系统指标
                self._collect_system_metrics()

                # 收集应用指标
                await self._collect_application_metrics(target)

                await asyncio.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")

    def _collect_system_metrics(self):
        """收集系统指标"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_sample(MetricSample(
            metric_type=MetricType.CPU,
            value=cpu_percent,
            timestamp=time.time(),
            unit="percent"
        ))

        # 内存
        memory = psutil.virtual_memory()
        self._add_sample(MetricSample(
            metric_type=MetricType.MEMORY,
            value=memory.percent,
            timestamp=time.time(),
            unit="percent"
        ))

        # 磁盘
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        self._add_sample(MetricSample(
            metric_type=MetricType.DISK,
            value=disk_percent,
            timestamp=time.time(),
            unit="percent"
        ))

        # 网络
        net_io = psutil.net_io_counters()
        self._add_sample(MetricSample(
            metric_type=MetricType.NETWORK,
            value=net_io.bytes_sent + net_io.bytes_recv,
            timestamp=time.time(),
            unit="bytes"
        ))

    async def _collect_application_metrics(self, target: str):
        """收集应用指标"""
        # 模拟应用指标收集
        # 实际实现中，这里会收集特定应用的指标

        # 延迟 (模拟)
        latency = self._measure_latency(target)
        self._add_sample(MetricSample(
            metric_type=MetricType.LATENCY,
            value=latency,
            timestamp=time.time(),
            unit="ms"
        ))

        # 吞吐量 (模拟)
        throughput = self._measure_throughput(target)
        self._add_sample(MetricSample(
            metric_type=MetricType.THROUGHPUT,
            value=throughput,
            timestamp=time.time(),
            unit="req/s"
        ))

    def _measure_latency(self, target: str) -> float:
        """测量延迟"""
        # 模拟测量
        return 50.0 + (hash(target) % 100)

    def _measure_throughput(self, target: str) -> float:
        """测量吞吐量"""
        # 模拟测量
        return 500.0 + (hash(target) % 200)

    def _add_sample(self, sample: MetricSample):
        """添加指标样本"""
        with self._lock:
            self._samples[sample.metric_type].append(sample)
            # 限制样本数量
            if len(self._samples[sample.metric_type]) > self.max_samples:
                self._samples[sample.metric_type].pop(0)

    async def analyze(
        self,
        target: str,
        metrics: Optional[List[str]] = None,
        duration: Optional[int] = None
    ) -> PerformanceReport:
        """
        执行性能分析

        Args:
            target: 分析目标
            metrics: 指标列表
            duration: 分析持续时间(秒)

        Returns:
            PerformanceReport: 性能报告
        """
        start_time = time.time()

        # 收集指标
        if metrics is None:
            metrics = ["cpu", "memory", "latency", "throughput"]

        metric_types = [MetricType(m) for m in metrics]

        # 收集深度分析数据
        if self.enable_deep_analysis:
            await self._collect_deep_metrics(target)

        # 生成报告
        report = PerformanceReport(
            id=f"report_{int(start_time)}",
            target=target,
            generated_at=start_time,
            duration=time.time() - start_time,
            overall_score=self._calculate_overall_score(),
            health_status=self._calculate_health_status(),
            metrics_summary=self._summarize_metrics(metric_types),
            bottlenecks=self._detect_bottlenecks(),
            resource_usage=self._analyze_resource_usage(),
            recommendations=self._generate_recommendations(),
            trends=self._analyze_trends()
        )

        return report

    def _calculate_overall_score(self) -> float:
        """计算整体评分 (0-100)"""
        scores = []

        for metric_type, samples in self._samples.items():
            if samples:
                latest = samples[-1]
                if metric_type == MetricType.CPU:
                    scores.append(100 - latest.value)
                elif metric_type == MetricType.MEMORY:
                    scores.append(100 - latest.value)
                elif metric_type == MetricType.LATENCY:
                    scores.append(max(0, 100 - latest.value / 10))
                elif metric_type == MetricType.THROUGHPUT:
                    scores.append(min(100, latest.value / 10))

        return sum(scores) / len(scores) if scores else 50.0

    def _calculate_health_status(self) -> str:
        """计算健康状态"""
        score = self._calculate_overall_score()
        if score >= 80:
            return "healthy"
        elif score >= 60:
            return "warning"
        elif score >= 40:
            return "degraded"
        else:
            return "critical"

    def _summarize_metrics(self, metric_types: List[MetricType]) -> Dict[str, Dict[str, float]]:
        """汇总指标"""
        summary = {}

        for metric_type in metric_types:
            samples = self._samples.get(metric_type, [])
            if samples:
                values = [s.value for s in samples]
                summary[metric_type.value] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                    "std": self._calculate_std(values)
                }

        return summary

    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def _detect_bottlenecks(self) -> List[Bottleneck]:
        """检测瓶颈"""
        bottlenecks = []

        for rule_name, rule in self._bottleneck_rules.items():
            metric_type = rule["metric"]
            samples = self._samples.get(metric_type, [])

            if samples:
                latest_value = samples[-1].value

                if rule["condition"](latest_value):
                    bottleneck = Bottleneck(
                        id=f"bn_{rule_name}_{int(time.time())}",
                        name=rule["name"],
                        description=f"检测到{rule['name']}, 当前值: {latest_value:.2f}",
                        severity=rule["severity"],
                        metric_type=metric_type,
                        location="system",
                        impact_score=latest_value / 100,
                        suggested_fix=rule["fix"],
                        affected_components=[metric_type.value],
                        detected_at=time.time(),
                        evidence={"current_value": latest_value}
                    )
                    bottlenecks.append(bottleneck)

        return bottlenecks

    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """分析资源使用"""
        return {
            "cpu": {
                "percent": psutil.cpu_percent(),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "network": psutil.net_io_counters()._asdict(),
            "process": self._get_process_info()
        }

    def _get_process_info(self) -> Dict[str, Any]:
        """获取进程信息"""
        try:
            process = psutil.Process()
            return {
                "memory_info": process.memory_info()._asdict(),
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "threads": len(process.threads())
            }
        except Exception:
            return {}

    def _analyze_trends(self) -> Dict[str, List[Tuple[float, float]]]:
        """分析趋势"""
        trends = {}

        for metric_type, samples in self._samples.items():
            if len(samples) >= 10:
                values = [(s.timestamp, s.value) for s in samples[-10:]]
                trends[metric_type.value] = values

        return trends

    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []

        bottlenecks = self._detect_bottlenecks()

        for bottleneck in bottlenecks:
            if bottleneck.suggested_fix:
                recommendations.append(bottleneck.suggested_fix)

        # 基于指标生成建议
        for metric_type, samples in self._samples.items():
            if samples:
                latest = samples[-1]
                if metric_type == MetricType.CPU and latest.value > 70:
                    recommendations.append("考虑优化CPU密集型操作或增加CPU资源")
                if metric_type == MetricType.MEMORY and latest.value > 80:
                    recommendations.append("建议进行内存分析，优化内存使用")

        return recommendations[:10]  # 限制建议数量

    async def _collect_deep_metrics(self, target: str):
        """收集深度指标"""
        # 数据库分析
        await self._analyze_database(target)

        # 缓存分析
        await self._analyze_cache(target)

        # API分析
        await self._analyze_api(target)

    async def _analyze_database(self, target: str):
        """数据库分析"""
        # 模拟数据库指标
        self._add_sample(MetricSample(
            metric_type=MetricType.DATABASE,
            value=50.0,  # 平均查询时间(ms)
            timestamp=time.time(),
            unit="ms"
        ))

    async def _analyze_cache(self, target: str):
        """缓存分析"""
        # 模拟缓存命中率
        self._add_sample(MetricSample(
            metric_type=MetricType.CACHE,
            value=0.85,  # 85% 命中率
            timestamp=time.time(),
            unit="percent"
        ))

    async def _analyze_api(self, target: str):
        """API分析"""
        # API指标已在基本收集中处理

    def get_metrics(self, metric_type: MetricType) -> List[MetricSample]:
        """获取特定类型的指标"""
        return self._samples.get(metric_type, [])

    def get_all_metrics(self) -> Dict[MetricType, List[MetricSample]]:
        """获取所有指标"""
        return dict(self._samples)

    def clear_metrics(self):
        """清除所有指标"""
        with self._lock:
            self._samples.clear()
