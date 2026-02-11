"""
Monitoring Dashboard - AI Platform v4

监控仪表盘 - 实时显示推理成本、延迟、Token使用量等核心指标
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid


class MetricType(Enum):
    """指标类型"""
    COST = "cost"
    LATENCY = "latency"
    TOKENS = "tokens"
    REQUESTS = "requests"
    ERRORS = "errors"
    THRUPUT = "throughput"


class TimeRange(Enum):
    """时间范围"""
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class CostMetrics:
    """成本指标"""
    total_cost: float = 0.0
    cost_by_provider: Dict[str, float] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_trend: List[MetricPoint] = field(default_factory=list)
    currency: str = "USD"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    latency_trend: List[MetricPoint] = field(default_factory=list)


@dataclass
class TokenMetrics:
    """Token使用指标"""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_by_provider: Dict[str, int] = field(default_factory=dict)
    tokens_by_model: Dict[str, int] = field(default_factory=list)
    token_trend: List[MetricPoint] = field(default_factory=list)


@dataclass
class RequestMetrics:
    """请求指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    request_trend: List[MetricPoint] = field(default_factory=list)


class MonitoringDashboard:
    """
    监控仪表盘 - 聚合所有监控数据
    
    功能:
    - 实时成本追踪
    - 性能指标聚合
    - Token使用统计
    - 多维度数据展示
    """
    
    def __init__(self):
        self._metrics_history: Dict[str, List[MetricPoint]] = {}
        self._cost_aggregator: Dict[str, float] = {}
        self._performance_aggregator: Dict[str, Any] = {}
        self._token_aggregator: Dict[str, Any] = {}
        self._request_aggregator: Dict[str, Any] = {}
        self._alerts: List[Dict] = []
        
    async def get_dashboard_data(self, time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """
        获取仪表盘完整数据
        
        Args:
            time_range: 时间范围
            
        Returns:
            仪表盘数据字典
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "time_range": time_range.value,
            "overview": await self._get_overview(),
            "cost": await self._get_cost_metrics(time_range),
            "performance": await self._get_performance_metrics(time_range),
            "tokens": await self._get_token_metrics(time_range),
            "requests": await self._get_request_metrics(time_range),
            "providers": await self._get_provider_breakdown(),
            "models": await self._get_model_breakdown()
        }
    
    async def _get_overview(self) -> Dict[str, Any]:
        """获取概览数据"""
        return {
            "total_cost_today": self._calculate_total_cost_today(),
            "total_requests_today": self._get_total_requests_today(),
            "avg_latency_ms": self._get_avg_latency(),
            "error_rate_percent": self._get_error_rate(),
            "active_providers": len(self._cost_aggregator),
            "status": "healthy" if self._get_error_rate() < 1.0 else "warning"
        }
    
    async def _get_cost_metrics(self, time_range: TimeRange) -> Dict[str, Any]:
        """获取成本指标"""
        return {
            "total_cost": sum(self._cost_aggregator.values()),
            "cost_by_provider": self._cost_aggregator.copy(),
            "cost_by_model": self._get_cost_by_model(),
            "trend": self._get_metric_trend(MetricType.COST, time_range),
            "currency": "USD",
            "forecast": self._forecast_cost(time_range)
        }
    
    async def _get_performance_metrics(self, time_range: TimeRange) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "latency": {
                "avg_ms": self._performance_aggregator.get("avg_latency", 0),
                "p50_ms": self._performance_aggregator.get("p50_latency", 0),
                "p95_ms": self._performance_aggregator.get("p95_latency", 0),
                "p99_ms": self._performance_aggregator.get("p99_latency", 0)
            },
            "throughput_rps": self._performance_aggregator.get("throughput", 0),
            "trend": self._get_metric_trend(MetricType.LATENCY, time_range)
        }
    
    async def _get_token_metrics(self, time_range: TimeRange) -> Dict[str, Any]:
        """获取Token使用指标"""
        return {
            "total_tokens": self._token_aggregator.get("total", 0),
            "prompt_tokens": self._token_aggregator.get("prompt", 0),
            "completion_tokens": self._token_aggregator.get("completion", 0),
            "by_provider": self._token_aggregator.get("by_provider", {}),
            "by_model": self._token_aggregator.get("by_model", {}),
            "trend": self._get_metric_trend(MetricType.TOKENS, time_range)
        }
    
    async def _get_request_metrics(self, time_range: TimeRange) -> Dict[str, Any]:
        """获取请求指标"""
        return {
            "total_requests": self._request_aggregator.get("total", 0),
            "successful": self._request_aggregator.get("successful", 0),
            "failed": self._request_aggregator.get("failed", 0),
            "error_rate_percent": self._get_error_rate(),
            "trend": self._get_metric_trend(MetricType.REQUESTS, time_range)
        }
    
    async def _get_provider_breakdown(self) -> List[Dict[str, Any]]:
        """获取按提供商分解的数据"""
        providers = []
        for provider, cost in self._cost_aggregator.items():
            providers.append({
                "name": provider,
                "cost": cost,
                "requests": self._request_aggregator.get(provider, {}).get("requests", 0),
                "avg_latency_ms": self._performance_aggregator.get(provider, {}).get("avg_latency", 0)
            })
        return sorted(providers, key=lambda x: x["cost"], reverse=True)
    
    async def _get_model_breakdown(self) -> List[Dict[str, Any]]:
        """获取按模型分解的数据"""
        return [
            {
                "name": model,
                "cost": data.get("cost", 0),
                "requests": data.get("requests", 0),
                "tokens": data.get("tokens", 0)
            }
            for model, data in self._token_aggregator.get("by_model", {}).items()
        ]
    
    # ============ 数据更新方法 ============
    
    async def record_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost: float,
        success: bool
    ):
        """记录一次请求"""
        # 更新成本
        self._cost_aggregator[provider] = self._cost_aggregator.get(provider, 0) + cost
        
        # 更新Token
        self._token_aggregator["total"] = self._token_aggregator.get("total", 0) + prompt_tokens + completion_tokens
        self._token_aggregator["prompt"] = self._token_aggregator.get("prompt", 0) + prompt_tokens
        self._token_aggregator["completion"] = self._token_aggregator.get("completion", 0) + completion_tokens
        
        # 更新请求统计
        self._request_aggregator["total"] = self._request_aggregator.get("total", 0) + 1
        if success:
            self._request_aggregator["successful"] = self._request_aggregator.get("successful", 0) + 1
        else:
            self._request_aggregator["failed"] = self._request_aggregator.get("failed", 0) + 1
        
        # 更新性能指标
        self._update_latency_stats(latency_ms)
        
        # 记录指标历史
        self._record_metric_point(MetricType.COST, cost, {"provider": provider, "model": model})
        self._record_metric_point(MetricType.LATENCY, latency_ms, {"provider": provider})
        self._record_metric_point(MetricType.TOKENS, prompt_tokens + completion_tokens, {"provider": provider})
        self._record_metric_point(MetricType.REQUESTS, 1, {"success": str(success)})
    
    def _record_metric_point(self, metric_type: MetricType, value: float, labels: Dict[str, str]):
        """记录指标数据点"""
        key = f"{metric_type.value}"
        if key not in self._metrics_history:
            self._metrics_history[key] = []
        self._metrics_history[key].append(MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels
        ))
        
        # 保留最近10000个点
        if len(self._metrics_history[key]) > 10000:
            self._metrics_history[key] = self._metrics_history[key][-10000:]
    
    def _update_latency_stats(self, latency_ms: float):
        """更新延迟统计"""
        latencies = self._performance_aggregator.get("latencies", [])
        latencies.append(latency_ms)
        
        # 保留最近1000个样本
        if len(latencies) > 1000:
            latencies = latencies[-1000:]
        
        self._performance_aggregator["latencies"] = latencies
        self._performance_aggregator["avg_latency"] = sum(latencies) / len(latencies)
        self._performance_aggregator["p50_latency"] = self._percentile(latencies, 50)
        self._performance_aggregator["p95_latency"] = self._percentile(latencies, 95)
        self._performance_aggregator["p99_latency"] = self._percentile(latencies, 99)
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    # ============ 辅助计算方法 ============
    
    def _calculate_total_cost_today(self) -> float:
        """计算今日总成本"""
        return sum(self._cost_aggregator.values())
    
    def _get_total_requests_today(self) -> int:
        """获取今日总请求数"""
        return self._request_aggregator.get("total", 0)
    
    def _get_avg_latency(self) -> float:
        """获取平均延迟"""
        return self._performance_aggregator.get("avg_latency", 0)
    
    def _get_error_rate(self) -> float:
        """获取错误率"""
        total = self._request_aggregator.get("total", 0)
        failed = self._request_aggregator.get("failed", 0)
        if total == 0:
            return 0
        return (failed / total) * 100
    
    def _get_cost_by_model(self) -> Dict[str, float]:
        """获取按模型分类的成本"""
        # 从历史数据中聚合
        cost_by_model = {}
        for point in self._metrics_history.get(MetricType.COST.value, []):
            model = point.labels.get("model", "unknown")
            cost_by_model[model] = cost_by_model.get(model, 0) + point.value
        return cost_by_model
    
    def _get_metric_trend(self, metric_type: MetricType, time_range: TimeRange) -> List[Dict]:
        """获取指标趋势数据"""
        key = metric_type.value
        points = self._metrics_history.get(key, [])
        
        # 按时间范围过滤
        now = datetime.now()
        hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}[time_range.value]
        cutoff = now - timedelta(hours=hours)
        
        filtered = [p for p in points if p.timestamp >= cutoff]
        
        # 转换为响应格式
        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "value": p.value
            }
            for p in filtered
        ]
    
    def _forecast_cost(self, time_range: TimeRange) -> Dict[str, Any]:
        """预测成本"""
        total = sum(self._cost_aggregator.values())
        hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}[time_range.value]
        
        # 简单线性预测
        forecast = total * (hours / 24)  # 假设基于24小时
        
        return {
            "predicted_total": round(forecast, 2),
            "confidence": "medium",
            "assumption": "Linear extrapolation based on current rate"
        }
    
    def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """获取提供商状态"""
        return {
            "provider": provider,
            "cost": self._cost_aggregator.get(provider, 0),
            "requests": self._request_aggregator.get(provider, {}).get("requests", 0),
            "avg_latency_ms": self._performance_aggregator.get(provider, {}).get("avg_latency", 0),
            "status": "healthy"
        }


# 创建全局仪表盘实例
dashboard = MonitoringDashboard()


def get_dashboard() -> MonitoringDashboard:
    """获取仪表盘实例"""
    return dashboard
