"""
Gateway Router - API网关路由器

提供API路由管理功能，支持动态路由配置、请求转发和负载均衡。
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
import time

class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"

@dataclass
class Route:
    """路由配置"""
    path: str
    target_url: str
    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    strip_prefix: bool = False
    timeout: int = 30
    retries: int = 3
    weight: int = 100  # 负载均衡权重
    enabled: bool = True
    rate_limit: Optional[int] = None  # 请求/秒
    quota_limit: Optional[int] = None  # 每日配额
    description: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    auth_required: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def matches(self, path: str, method: str) -> Optional[Dict[str, str]]:
        """检查路径是否匹配，返回路径参数"""
        # 替换路径参数 :param -> (?P<param>[^/]+)
        pattern = re.sub(r':([^/]+)', r'(?P<\1>[^/]+)', self.path)
        full_pattern = f"^{pattern}$"
        
        match = re.match(full_pattern, path)
        if match and method.upper() in self.methods:
            return match.groupdict()
        return None

@dataclass
class HealthCheck:
    """健康检查配置"""
    interval: int = 30  # 秒
    timeout: int = 5
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    path: str = "/health"

class GatewayRouter:
    """API网关路由器"""
    
    def __init__(self):
        self.routes: Dict[str, Route] = {}
        self.route_patterns: List[Route] = []
        self.target_health: Dict[str, Dict[str, Any]] = {}
        self._request_counts: Dict[str, List[float]] = {}
    
    def add_route(self, route: Route) -> str:
        """添加路由"""
        route_id = self._generate_route_id(route.path)
        self.routes[route_id] = route
        self._rebuild_patterns()
        return route_id
    
    def _generate_route_id(self, path: str) -> str:
        """生成路由ID"""
        return hashlib.md5(f"{path}:{time.time()}".encode()).hexdigest()[:12]
    
    def _rebuild_patterns(self):
        """重建路由模式列表"""
        self.route_patterns = sorted(
            self.routes.values(),
            key=lambda r: len(r.path),
            reverse=True
        )
    
    def get_route(self, path: str, method: str) -> Optional[Route]:
        """获取匹配的路由"""
        for route in self.route_patterns:
            if route.enabled and route.matches(path, method):
                return route
        return None
    
    def list_routes(self) -> List[Dict[str, Any]]:
        """列出所有路由"""
        return [
            {
                "id": rid,
                "path": r.path,
                "target_url": r.target_url,
                "methods": r.methods,
                "enabled": r.enabled,
                "rate_limit": r.rate_limit,
                "quota_limit": r.quota_limit,
                "description": r.description,
                "created_at": r.created_at,
            }
            for rid, r in self.routes.items()
        ]
    
    def update_route(self, route_id: str, **kwargs) -> Optional[Route]:
        """更新路由"""
        if route_id not in self.routes:
            return None
        
        route = self.routes[route_id]
        for key, value in kwargs.items():
            if hasattr(route, key):
                setattr(route, key, value)
        route.updated_at = time.time()
        return route
    
    def delete_route(self, route_id: str) -> bool:
        """删除路由"""
        if route_id not in self.routes:
            return False
        del self.routes[route_id]
        self._rebuild_patterns()
        return True
    
    def get_target(self, route: Route) -> str:
        """获取目标URL（支持负载均衡）"""
        return route.target_url
    
    def check_health(self, target_url: str) -> bool:
        """检查目标健康状态"""
        if target_url not in self.target_health:
            self.target_health[target_url] = {
                "healthy_count": 0,
                "unhealthy_count": 0,
                "last_check": 0,
                "status": "unknown"
            }
        
        # 简化的健康检查（实际应该发起HTTP请求）
        health_info = self.target_health[target_url]
        health_info["last_check"] = time.time()
        health_info["status"] = "healthy"
        
        return True
    
    def record_request(self, route_id: str):
        """记录请求"""
        now = time.time()
        if route_id not in self._request_counts:
            self._request_counts[route_id] = []
        self._request_counts[route_id].append(now)
    
    def get_request_count(self, route_id: str, window: int = 60) -> int:
        """获取请求计数"""
        if route_id not in self._request_counts:
            return 0
        
        now = time.time()
        self._request_counts[route_id] = [
            t for t in self._request_counts[route_id]
            if now - t < window
        ]
        return len(self._request_counts[route_id])
    
    def clear_requests(self, route_id: str):
        """清除请求记录"""
        self._request_counts.pop(route_id, None)

# 全局路由器实例
router = GatewayRouter()
