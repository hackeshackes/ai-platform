"""
Gateway Middleware - 网关中间件

提供网关所需的各种中间件：
- 请求限流中间件
- 配额检查中间件
- 流量统计中间件
- 异常检测中间件
- 请求日志中间件
- 认证中间件
"""
from typing import Callable, Dict, Optional, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import logging
import asyncio

from gateway.router import router, Route
from gateway.ratelimit import limiter, RateLimitConfig, RateLimitAlgorithm
from gateway.quota import quota_manager, QuotaConfig, QuotaPeriod

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""
    
    def __init__(
        self,
        app,
        default_requests: int = 100,
        window_seconds: int = 60,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    ):
        super().__init__(app)
        self.default_config = RateLimitConfig(
            requests=default_requests,
            window_seconds=window_seconds,
            algorithm=algorithm
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 获取客户端IP作为限流key
        client_ip = self._get_client_ip(request)
        path = request.url.path
        
        # 构建限流key
        rate_limit_key = f"{client_ip}:{path}"
        
        # 检查是否有路由级别的限流配置
        route = router.get_route(path, request.method)
        if route and route.rate_limit:
            config = RateLimitConfig(
                requests=route.rate_limit,
                window_seconds=60,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            )
        else:
            config = self.default_config
        
        # 检查限流
        result = limiter.check_rate_limit(rate_limit_key, config)
        
        if not result.allowed:
            logger.warning(f"Rate limit exceeded for {rate_limit_key}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded",
                    "retry_after": result.retry_after,
                    "limit": result.limit,
                    "remaining": 0
                },
                headers={
                    "Retry-After": str(int(result.retry_after)) if result.retry_after else "60",
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result.reset_at))
                }
            )
        
        response = await call_next(request)
        
        # 添加限流头
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"


class QuotaMiddleware(BaseHTTPMiddleware):
    """配额检查中间件"""
    
    def __init__(
        self,
        app,
        default_quota: int = 1000,
        period: QuotaPeriod = QuotaPeriod.DAILY
    ):
        super().__init__(app)
        self.default_config = QuotaConfig(
            limit=default_quota,
            period=period
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 获取用户ID或API密钥
        api_key = request.headers.get("X-API-Key") or \
                  request.query_params.get("api_key")
        user_id = request.headers.get("X-User-ID")
        
        quota_key = api_key or user_id or "anonymous"
        
        # 检查配额
        usage = quota_manager.get_usage(quota_key)
        
        if usage.is_exceeded:
            logger.warning(f"Quota exceeded for {quota_key}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Quota Exceeded",
                    "message": "API quota has been exhausted",
                    "used": usage.used,
                    "limit": usage.limit,
                    "reset_at": usage.reset_at
                }
            )
        
        response = await call_next(request)
        
        # 消耗配额
        quota_manager.consume(quota_key)
        
        # 添加配额头
        response.headers["X-Quota-Limit"] = str(usage.limit)
        response.headers["X-Quota-Remaining"] = str(usage.remaining)
        response.headers["X-Quota-Reset"] = str(int(usage.reset_at))
        
        return response


class TrafficStatsMiddleware(BaseHTTPMiddleware):
    """流量统计中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        path = request.url.path
        method = request.method
        
        response = await call_next(request)
        
        duration = time.time() - start
        status_code = response.status_code
        
        # 更新统计
        self._update_stats(path, method, status_code, duration)
        
        return response
    
    def _update_stats(
        self,
        path: str,
        method: str,
        status_code: int,
        duration: float
    ):
        """更新统计信息"""
        key = f"{method}:{path}"
        
        if key not in self.stats:
            self.stats[key] = {
                "path": path,
                "method": method,
                "requests": 0,
                "errors": 0,
                "total_duration": 0,
                "min_duration": float("inf"),
                "max_duration": 0,
                "status_codes": {}
            }
        
        stats = self.stats[key]
        stats["requests"] += 1
        stats["total_duration"] += duration
        stats["min_duration"] = min(stats["min_duration"], duration)
        stats["max_duration"] = max(stats["max_duration"], duration)
        
        if status_code >= 400:
            stats["errors"] += 1
        
        status_key = str(status_code)
        stats["status_codes"][status_key] = stats["status_codes"].get(status_key, 0) + 1
        
        self.total_requests += 1
        if status_code >= 400:
            self.total_errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": (
                self.total_errors / self.total_requests * 100
                if self.total_requests > 0 else 0
            ),
            "routes": list(self.stats.values()),
            "average_requests_per_second": (
                self.total_requests / uptime if uptime > 0 else 0
            )
        }


class AnomalyDetectionMiddleware(BaseHTTPMiddleware):
    """异常检测中间件"""
    
    def __init__(
        self,
        app,
        error_threshold: float = 0.1,  # 10%错误率阈值
        latency_threshold_ms: float = 1000,  # 1秒延迟阈值
        window_seconds: int = 60
    ):
        super().__init__(app)
        self.error_threshold = error_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.window_seconds = window_seconds
        self.request_window: Dict[str, list] = {}  # key -> [(timestamp, status_code, duration_ms)]
        self.alerts: list = []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        path = request.url.path
        method = request.method
        
        response = await call_next(request)
        
        duration_ms = (time.time() - start) * 1000
        status_code = response.status_code
        
        # 检测异常
        await self._detect_anomaly(path, method, status_code, duration_ms)
        
        return response
    
    async def _detect_anomaly(
        self,
        path: str,
        method: str,
        status_code: int,
        duration_ms: float
    ):
        """检测异常"""
        key = f"{method}:{path}"
        now = time.time()
        
        if key not in self.request_window:
            self.request_window[key] = []
        
        # 添加当前请求
        self.request_window[key].append({
            "timestamp": now,
            "status_code": status_code,
            "duration_ms": duration_ms
        })
        
        # 清理过期数据
        window_start = now - self.window_seconds
        self.request_window[key] = [
            r for r in self.request_window[key]
            if r["timestamp"] > window_start
        ]
        
        # 计算错误率
        window_requests = self.request_window[key]
        if len(window_requests) < 5:
            return  # 需要至少5个请求
        
        errors = sum(1 for r in window_requests if r["status_code"] >= 400)
        error_rate = errors / len(window_requests)
        
        avg_latency = sum(r["duration_ms"] for r in window_requests) / len(window_requests)
        
        # 检测高错误率
        if error_rate > self.error_threshold:
            self._create_alert(
                key,
                "high_error_rate",
                f"Error rate {error_rate*100:.1f}% exceeds threshold {self.error_threshold*100:.1f}%",
                {"error_rate": error_rate, "requests": len(window_requests)}
            )
        
        # 检测高延迟
        if avg_latency > self.latency_threshold_ms:
            self._create_alert(
                key,
                "high_latency",
                f"Average latency {avg_latency:.0f}ms exceeds threshold {self.latency_threshold_ms:.0f}ms",
                {"avg_latency": avg_latency, "requests": len(window_requests)}
            )
    
    def _create_alert(
        self,
        key: str,
        alert_type: str,
        message: str,
        details: Dict
    ):
        """创建告警"""
        alert = {
            "timestamp": time.time(),
            "key": key,
            "type": alert_type,
            "message": message,
            "details": details
        }
        
        # 避免重复告警（5分钟内）
        recent_duplicate = any(
            a["key"] == key and a["type"] == alert_type and
            time.time() - a["timestamp"] < 300
            for a in self.alerts
        )
        
        if not recent_duplicate:
            self.alerts.append(alert)
            logger.warning(f"Gateway Alert: {message}")
    
    def get_alerts(self, limit: int = 50) -> list:
        """获取告警列表"""
        return self.alerts[-limit:]
    
    def clear_alerts(self):
        """清除告警"""
        self.alerts.clear()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    def __init__(self, app, logger_name: str = "gateway"):
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        path = request.url.path
        method = request.method
        
        # 记录请求开始
        self.logger.info(f"Request started: {method} {path}")
        
        response = await call_next(request)
        
        duration = time.time() - start
        
        # 记录请求完成
        self.logger.info(
            f"Request completed: {method} {path} -> {response.status_code} "
            f"({duration*1000:.2f}ms)"
        )
        
        return response


class GatewayMiddleware:
    """网关中间件管理器"""
    
    def __init__(self):
        self.middlewares = []
    
    def add_rate_limit(
        self,
        default_requests: int = 100,
        window_seconds: int = 60,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    ):
        """添加限流中间件"""
        return lambda app: RateLimitMiddleware(
            app, default_requests, window_seconds, algorithm
        )
    
    def add_quota(
        self,
        default_quota: int = 1000,
        period: QuotaPeriod = QuotaPeriod.DAILY
    ):
        """添加配额中间件"""
        return lambda app: QuotaMiddleware(app, default_quota, period)
    
    def add_traffic_stats(self):
        """添加流量统计中间件"""
        return lambda app: TrafficStatsMiddleware(app)
    
    def add_anomaly_detection(
        self,
        error_threshold: float = 0.1,
        latency_threshold_ms: float = 1000,
        window_seconds: int = 60
    ):
        """添加异常检测中间件"""
        return lambda app: AnomalyDetectionMiddleware(
            app, error_threshold, latency_threshold_ms, window_seconds
        )
    
    def add_request_logging(self, logger_name: str = "gateway"):
        """添加请求日志中间件"""
        return lambda app: RequestLoggingMiddleware(app, logger_name)
    
    def get_all_middlewares(self, app):
        """获取所有中间件（按顺序）"""
        return [
            RequestLoggingMiddleware(app),
            AnomalyDetectionMiddleware(app),
            TrafficStatsMiddleware(app),
            QuotaMiddleware(app),
            RateLimitMiddleware(app),
        ]
