"""
AI Platform Backend - Monitoring System

监控体系模块

提供完整的系统监控、健康检查、性能指标、日志追踪和告警功能。
"""

__version__ = "1.0.0"

from .monitoring import MonitoringSystem, get_monitoring_system, init_monitoring, shutdown_monitoring
from .health import HealthChecker, HealthStatus, HealthCheckResult
from .metrics import MetricsCollector, Metric, MetricType
from .logger import StructuredLogger, get_logger, init_logger
from .alerts import AlertManager, AlertRule, Alert, AlertSeverity, AlertStatus
from .tracing import TracingManager, Span, SpanContext, SpanKind, SpanStatus

__all__ = [
    # 监控主模块
    "MonitoringSystem",
    "get_monitoring_system",
    "init_monitoring",
    "shutdown_monitoring",
    
    # 健康检查
    "HealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    
    # 性能指标
    "MetricsCollector",
    "Metric",
    "MetricType",
    
    # 结构化日志
    "StructuredLogger",
    "get_logger",
    "init_logger",
    
    # 告警管理
    "AlertManager",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    
    # 链路追踪
    "TracingManager",
    "Span",
    "SpanContext",
    "SpanKind",
    "SpanStatus"
]
