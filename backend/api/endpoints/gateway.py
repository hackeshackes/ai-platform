"""
Gateway API - 网关管理API

提供网关配置和监控的REST API端点。
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import time

from backend.gateway.router import router as gateway_router, Route, HttpMethod
from backend.gateway.ratelimit import limiter, RateLimitConfig, RateLimitAlgorithm
from backend.gateway.quota import quota_manager, QuotaConfig, QuotaPeriod, QuotaRule
from backend.gateway.middleware import TrafficStatsMiddleware, AnomalyDetectionMiddleware

router = APIRouter()

# ============ 请求模型 ============

class AddRouteRequest(BaseModel):
    """添加路由请求"""
    path: str = Field(..., description="路由路径，如 /api/v1/service/*")
    target_url: str = Field(..., description="目标服务URL")
    methods: List[str] = Field(default=["GET", "POST"], description="支持的HTTP方法")
    strip_prefix: bool = Field(default=False, description="是否移除前缀")
    timeout: int = Field(default=30, ge=1, le=300, description="超时时间（秒）")
    retries: int = Field(default=3, ge=0, le=10, description="重试次数")
    weight: int = Field(default=100, ge=1, le=100, description="负载均衡权重")
    rate_limit: Optional[int] = Field(None, ge=1, description="限流（请求/秒）")
    quota_limit: Optional[int] = Field(None, ge=1, description="每日配额限制")
    description: str = Field(default="", description="路由描述")
    headers: Dict[str, str] = Field(default_factory=dict, description="附加头")
    auth_required: bool = Field(default=True, description="是否需要认证")

class UpdateRouteRequest(BaseModel):
    """更新路由请求"""
    path: Optional[str] = None
    target_url: Optional[str] = None
    methods: Optional[List[str]] = None
    enabled: Optional[bool] = None
    rate_limit: Optional[int] = None
    quota_limit: Optional[int] = None
    description: Optional[str] = None
    timeout: Optional[int] = None
    retries: Optional[int] = None

class SetRateLimitRequest(BaseModel):
    """设置限流请求"""
    key: str = Field(..., description="限流标识（IP、用户ID等）")
    requests: int = Field(..., ge=1, description="允许的请求数")
    window_seconds: int = Field(default=60, ge=1, description="时间窗口（秒）")
    algorithm: str = Field(default="token_bucket", description="限流算法")

class AddQuotaRuleRequest(BaseModel):
    """添加配额规则请求"""
    name: str = Field(..., description="规则名称")
    key_pattern: str = Field(..., description="匹配模式")
    limit: int = Field(..., ge=1, description="配额上限")
    period: str = Field(default="daily", description="周期 daily/weekly/monthly")
    priority: int = Field(default=0, description="优先级")
    description: str = Field(default="", description="规则描述")

class SetQuotaRequest(BaseModel):
    """设置配额请求"""
    key: str = Field(..., description="配额标识")
    limit: int = Field(..., ge=1, description="配额上限")
    period: str = Field(default="daily", description="周期")

# ============ 路由管理端点 ============

@router.post("/route", tags=["Gateway"])
async def add_route(request: AddRouteRequest):
    """
    添加路由配置
    
    添加新的API路由，将指定路径的请求转发到目标服务。
    """
    route = Route(
        path=request.path,
        target_url=request.target_url,
        methods=request.methods,
        strip_prefix=request.strip_prefix,
        timeout=request.timeout,
        retries=request.retries,
        weight=request.weight,
        rate_limit=request.rate_limit,
        quota_limit=request.quota_limit,
        description=request.description,
        headers=request.headers,
        auth_required=request.auth_required
    )
    
    route_id = gateway_router.add_route(route)
    
    return {
        "success": True,
        "route_id": route_id,
        "message": "Route added successfully"
    }

@router.get("/routes", tags=["Gateway"])
async def list_routes():
    """
    列出所有路由
    
    返回所有已配置的路由列表。
    """
    routes = gateway_router.list_routes()
    return {
        "success": True,
        "total": len(routes),
        "routes": routes
    }

@router.put("/route/{route_id}", tags=["Gateway"])
async def update_route(route_id: str, request: UpdateRouteRequest):
    """
    更新路由配置
    
    更新指定路由的配置参数。
    """
    update_data = request.model_dump(exclude_unset=True)
    
    route = gateway_router.update_route(route_id, **update_data)
    
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    return {
        "success": True,
        "message": "Route updated successfully"
    }

@router.delete("/route/{route_id}", tags=["Gateway"])
async def delete_route(route_id: str):
    """
    删除路由
    
    删除指定的路由配置。
    """
    success = gateway_router.delete_route(route_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Route not found")
    
    return {
        "success": True,
        "message": "Route deleted successfully"
    }

# ============ 限流管理端点 ============

@router.post("/limit", tags=["Gateway"])
async def set_rate_limit(request: SetRateLimitRequest):
    """
    设置限流
    
    为指定的标识配置限流规则。
    """
    try:
        algorithm = RateLimitAlgorithm(request.algorithm)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm. Choose from: token_bucket, sliding_window, fixed_window"
        )
    
    config = RateLimitConfig(
        requests=request.requests,
        window_seconds=request.window_seconds,
        algorithm=algorithm
    )
    
    limiter.create_limiter(request.key, config)
    
    return {
        "success": True,
        "message": "Rate limit configured successfully",
        "key": request.key,
        "config": {
            "requests": request.requests,
            "window_seconds": request.window_seconds,
            "algorithm": request.algorithm
        }
    }

@router.get("/limit/{key}", tags=["Gateway"])
async def get_rate_limit(key: str):
    """
    获取限流状态
    
    查询指定标识的当前限流状态。
    """
    result = limiter.check_rate_limit(key)
    
    return {
        "success": True,
        "key": key,
        "allowed": result.allowed,
        "remaining": result.remaining,
        "limit": result.limit,
        "reset_at": result.reset_at,
        "retry_after": result.retry_after
    }

@router.delete("/limit/{key}", tags=["Gateway"])
async def reset_rate_limit(key: str):
    """
    重置限流
    
    重置指定标识的限流计数器。
    """
    limiter.reset(key)
    
    return {
        "success": True,
        "message": "Rate limit reset successfully"
    }

# ============ 配额管理端点 ============

@router.post("/quota/rule", tags=["Gateway"])
async def add_quota_rule(request: AddQuotaRuleRequest):
    """
    添加配额规则
    
    为特定模式的用户或API添加配额规则。
    """
    try:
        period = QuotaPeriod(request.period)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period. Choose from: daily, weekly, monthly"
        )
    
    rule = QuotaRule(
        name=request.name,
        key_pattern=request.key_pattern,
        config=QuotaConfig(
            limit=request.limit,
            period=period
        ),
        priority=request.priority,
        description=request.description
    )
    
    quota_manager.add_rule(rule)
    
    return {
        "success": True,
        "message": "Quota rule added successfully"
    }

@router.get("/quota/rules", tags=["Gateway"])
async def list_quota_rules():
    """
    列出配额规则
    
    返回所有已配置的配额规则。
    """
    rules = quota_manager.list_rules()
    
    return {
        "success": True,
        "total": len(rules),
        "rules": rules
    }

@router.delete("/quota/rule/{rule_id}", tags=["Gateway"])
async def delete_quota_rule(rule_id: str):
    """
    删除配额规则
    
    删除指定的配额规则。
    """
    success = quota_manager.delete_rule(rule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {
        "success": True,
        "message": "Quota rule deleted successfully"
    }

@router.get("/quota", tags=["Gateway"])
async def get_quota(
    key: str = Query(..., description="配额标识（用户ID或API Key）")
):
    """
    查询配额
    
    查询指定标识的配额使用情况。
    """
    usage = quota_manager.get_usage(key)
    
    return {
        "success": True,
        "key": usage.key,
        "used": usage.used,
        "limit": usage.limit,
        "remaining": usage.remaining,
        "period": usage.period,
        "percentage": usage.percentage,
        "reset_at": usage.reset_at,
        "is_exceeded": usage.is_exceeded
    }

@router.post("/quota/consume", tags=["Gateway"])
async def consume_quota(
    key: str = Query(..., description="配额标识")
):
    """
    消耗配额
    
    为指定标识消耗一个配额单位。
    """
    usage = quota_manager.consume(key)
    
    return {
        "success": True,
        "key": usage.key,
        "used": usage.used,
        "limit": usage.limit,
        "remaining": usage.remaining,
        "percentage": usage.percentage,
        "is_exceeded": usage.is_exceeded
    }

@router.put("/quota/{key}", tags=["Gateway"])
async def set_quota(key: str, request: SetQuotaRequest):
    """
    设置配额
    
    重置指定标识的配额（需要配合规则使用）。
    """
    quota_manager.reset(key)
    
    return {
        "success": True,
        "message": f"Quota for '{key}' has been reset"
    }

# ============ 流量统计端点 ============

@router.get("/usage", tags=["Gateway"])
async def get_usage_stats():
    """
    获取使用统计
    
    返回网关的流量统计信息。
    """
    # 从中间件获取统计
    stats = {
        "timestamp": time.time(),
        "uptime_seconds": time.time(),  # 简化的计算
        "total_requests": 0,
        "total_errors": 0,
        "error_rate": 0,
        "routes": []
    }
    
    # 获取限流统计
    rate_limit_stats = {
        "active_limiters": len(limiter.token_buckets) + 
                          len(limiter.sliding_windows) + 
                          len(limiter.fixed_windows)
    }
    
    # 获取配额统计
    quota_usage = quota_manager.get_all_usage()
    quota_stats = {
        "total_users": len(quota_usage),
        "exceeded_users": sum(1 for u in quota_usage if u.is_exceeded)
    }
    
    return {
        "success": True,
        "gateway_stats": stats,
        "rate_limit_stats": rate_limit_stats,
        "quota_stats": quota_stats
    }

@router.get("/alerts", tags=["Gateway"])
async def get_alerts(limit: int = Query(default=50, ge=1, le=100)):
    """
    获取告警
    
    返回异常检测产生的告警列表。
    """
    # 简化实现
    return {
        "success": True,
        "alerts": [],
        "total": 0
    }

@router.get("/health", tags=["Gateway"])
async def gateway_health():
    """
    网关健康检查
    
    返回网关组件的健康状态。
    """
    return {
        "status": "healthy",
        "components": {
            "router": len(gateway_router.routes),
            "rate_limiter": {
                "token_bucket": len(limiter.token_buckets),
                "sliding_window": len(limiter.sliding_windows),
                "fixed_window": len(limiter.fixed_windows)
            },
            "quota_manager": {
                "rules": len(quota_manager.rules),
                "active_quotas": len(quota_manager.usage)
            }
        },
        "timestamp": time.time()
    }
