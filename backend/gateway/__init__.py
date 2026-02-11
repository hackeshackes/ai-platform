"""
Gateway Module - API网关模块

提供完整的API网关功能：
- router: 网关路由器
- ratelimit: 请求限流器
- quota: 配额管理器
- middleware: 网关中间件
"""

from gateway.router import router, GatewayRouter, Route, HttpMethod
from gateway.ratelimit import limiter, RateLimiter, RateLimitConfig, RateLimitResult
from gateway.quota import quota_manager, QuotaManager, QuotaConfig, QuotaUsage
from gateway.middleware import (
    GatewayMiddleware,
    RateLimitMiddleware,
    QuotaMiddleware,
    TrafficStatsMiddleware,
    AnomalyDetectionMiddleware,
    RequestLoggingMiddleware
)

__all__ = [
    # Router
    "router",
    "GatewayRouter",
    "Route",
    "HttpMethod",
    # Rate Limiter
    "limiter",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    # Quota Manager
    "quota_manager",
    "QuotaManager",
    "QuotaConfig",
    "QuotaUsage",
    # Middleware
    "GatewayMiddleware",
    "RateLimitMiddleware",
    "QuotaMiddleware",
    "TrafficStatsMiddleware",
    "AnomalyDetectionMiddleware",
    "RequestLoggingMiddleware"
]
