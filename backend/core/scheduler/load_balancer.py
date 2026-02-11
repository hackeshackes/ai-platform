"""
智能调度系统 - 负载均衡器

智能路由/健康检查/会话保持/灰度发布
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import hashlib
import random

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"


class HealthCheckType(Enum):
    TCP = "tcp"
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"


@dataclass
class BackendServer:
    """后端服务器"""
    id: str
    host: str
    port: int
    weight: int = 100
    max_connections: int = 1000
    current_connections: int = 0
    health_check_interval: int = 30  # 秒
    is_healthy: bool = True
    last_health_check: datetime = field(default_factory=datetime.now)
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Session:
    """会话信息"""
    session_id: str
    user_id: str
    backend_id: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    metadata: Dict = field(default_factory=dict)


@dataclass
class GrayReleaseRule:
    """灰度发布规则"""
    name: str
    backend_id: str
    traffic_percentage: float  # 0-100
    conditions: List[Dict] = field(default_factory=list)  # 如 [{"header": "version", "value": "beta"}]
    version: str = "v1"


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN,
        health_check_interval: int = 30,
        session_timeout: int = 86400,
        enable_health_check: bool = True
    ):
        self.algorithm = algorithm
        self.health_check_interval = health_check_interval
        self.session_timeout = session_timeout
        self.enable_health_check = enable_health_check
        
        self.backends: Dict[str, BackendServer] = {}
        self.sessions: Dict[str, Session] = {}
        self.gray_rules: List[GrayReleaseRule] = []
        
        self.request_count = 0
        self.error_count = 0
        
        # 算法状态
        self._round_robin_index = {}
        self._connection_counts = {}
        self._adaptive_scores = {}
        
    def register_backend(self, backend: BackendServer) -> bool:
        """注册后端服务器"""
        self.backends[backend.id] = backend
        self._round_robin_index[backend.id] = 0
        self._connection_counts[backend.id] = 0
        self._adaptive_scores[backend.id] = 100.0
        logger.info(f"后端服务器注册成功: {backend.id} ({backend.host}:{backend.port})")
        return True
    
    def unregister_backend(self, backend_id: str) -> bool:
        """注销后端服务器"""
        if backend_id in self.backends:
            del self.backends[backend_id]
            # 清理相关会话
            sessions_to_remove = [
                sid for sid, session in self.sessions.items()
                if session.backend_id == backend_id
            ]
            for sid in sessions_to_remove:
                del self.sessions[sid]
            logger.info(f"后端服务器已注销: {backend_id}")
            return True
        return False
    
    def route_request(
        self,
        request_id: str,
        client_ip: str,
        path: str = "",
        headers: Optional[Dict] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[Optional[str], Dict]:
        """
        路由请求到后端服务器
        
        Returns:
            (backend_id, response_headers)
        """
        self.request_count += 1
        headers = headers or {}
        
        # 检查灰度发布规则
        gray_backend = self._check_gray_rules(headers, client_ip)
        if gray_backend:
            self._connection_counts[gray_backend] = self._connection_counts.get(gray_backend, 0) + 1
            return gray_backend, {"X-Backend": gray_backend, "X-Route": "gray"}
        
        # 会话保持
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if session.expires_at > datetime.now():
                backend_id = session.backend_id
                if backend_id in self.backends and self.backends[backend_id].is_healthy:
                    self._connection_counts[backend_id] = self._connection_counts.get(backend_id, 0) + 1
                    return backend_id, {"X-Backend": backend_id, "X-Route": "sticky"}
        
        # 选择后端服务器
        healthy_backends = self._get_healthy_backends()
        
        if not healthy_backends:
            self.error_count += 1
            logger.error("无可用后端服务器")
            return None, {"X-Error": "No healthy backends"}
        
        backend_id = self._select_backend(healthy_backends, client_ip, request_id)
        
        # 创建会话(如果需要)
        if user_id:
            self._create_session(request_id, user_id, backend_id)
        
        self._connection_counts[backend_id] = self._connection_counts.get(backend_id, 0) + 1
        return backend_id, {"X-Backend": backend_id, "X-Route": self.algorithm.value}
    
    def _get_healthy_backends(self) -> List[BackendServer]:
        """获取健康的后端服务器列表"""
        if not self.enable_health_check:
            return list(self.backends.values())
        
        return [
            backend for backend in self.backends.values()
            if backend.is_healthy
        ]
    
    def _select_backend(
        self,
        backends: List[BackendServer],
        client_ip: str,
        request_id: str
    ) -> str:
        """根据算法选择后端"""
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_select(backends)
        
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(backends)
        
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_select(backends)
        
        elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
            return self._ip_hash_select(backends, client_ip)
        
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            return self._random_select(backends)
        
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            return self._consistent_hash_select(backends, request_id)
        
        elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            return self._adaptive_select(backends)
        
        return backends[0].id
    
    def _round_robin_select(self, backends: List[BackendServer]) -> str:
        """轮询选择"""
        backend = backends[0]
        self._round_robin_index[backend.id] = (self._round_robin_index.get(backend.id, 0) + 1) % len(backends)
        return backend.id
    
    def _weighted_round_robin_select(self, backends: List[BackendServer]) -> str:
        """加权轮询选择"""
        total_weight = sum(b.weight for b in backends)
        if total_weight == 0:
            return backends[0].id
        
        random_weight = random.randint(1, total_weight)
        cum_weight = 0
        
        for backend in backends:
            cum_weight += backend.weight
            if random_weight <= cum_weight:
                return backend.id
        
        return backends[0].id
    
    def _least_connections_select(self, backends: List[BackendServer]) -> str:
        """最少连接数选择"""
        return min(backends, key=lambda b: b.current_connections).id
    
    def _ip_hash_select(self, backends: List[BackendServer], client_ip: str) -> str:
        """IP哈希选择"""
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return backends[hash_value % len(backends)].id
    
    def _random_select(self, backends: List[BackendServer]) -> str:
        """随机选择"""
        return random.choice(backends).id
    
    def _consistent_hash_select(self, backends: List[BackendServer], request_id: str) -> str:
        """一致性哈希选择"""
        hash_value = int(hashlib.sha256(request_id.encode()).hexdigest(), 16)
        return backends[hash_value % len(backends)].id
    
    def _adaptive_select(self, backends: List[BackendServer]) -> str:
        """自适应选择 - 基于实时指标"""
        for backend in backends:
            # 计算综合得分
            health_score = 100 if backend.is_healthy else 0
            connection_score = (1 - backend.current_connections / backend.max_connections) * 100
            response_score = max(0, 100 - backend.response_time_avg)
            error_penalty = backend.error_rate * 100
            
            self._adaptive_scores[backend.id] = (
                health_score * 0.4 +
                connection_score * 0.3 +
                response_score * 0.2 -
                error_penalty * 0.1
            )
        
        return max(backends, key=lambda b: self._adaptive_scores.get(b.id, 0)).id
    
    def _check_gray_rules(self, headers: Dict, client_ip: str) -> Optional[str]:
        """检查灰度发布规则"""
        for rule in self.gray_rules:
            # 检查流量百分比
            if random.random() * 100 < rule.traffic_percentage:
                # 检查条件
                conditions_met = True
                for condition in rule.conditions:
                    header = condition.get("header")
                    value = condition.get("value")
                    if headers.get(header) != value:
                        conditions_met = False
                        break
                
                if conditions_met and rule.backend_id in self.backends:
                    return rule.backend_id
        
        return None
    
    def _create_session(self, session_id: str, user_id: str, backend_id: str) -> Session:
        """创建会话"""
        session = Session(
            session_id=session_id,
            user_id=user_id,
            backend_id=backend_id,
            expires_at=datetime.now() + timedelta(seconds=self.session_timeout)
        )
        self.sessions[session_id] = session
        return session
    
    def add_gray_release_rule(self, rule: GrayReleaseRule) -> bool:
        """添加灰度发布规则"""
        self.gray_rules.append(rule)
        logger.info(f"灰度发布规则已添加: {rule.name}")
        return True
    
    def health_check(self, backend_id: str, is_healthy: bool, response_time: float = 0):
        """执行健康检查"""
        if backend_id in self.backends:
            backend = self.backends[backend_id]
            backend.is_healthy = is_healthy
            backend.last_health_check = datetime.now()
            
            # 更新响应时间和错误率
            if backend.response_time_avg == 0:
                backend.response_time_avg = response_time
            else:
                backend.response_time_avg = (backend.response_time_avg + response_time) / 2
            
            logger.info(f"健康检查完成: {backend_id}, healthy={is_healthy}, response_time={response_time}ms")
    
    def auto_health_check(self, check_func: Callable[[BackendServer], Tuple[bool, float]]):
        """自动健康检查"""
        for backend_id, backend in self.backends.items():
            is_healthy, response_time = check_func(backend)
            self.health_check(backend_id, is_healthy, response_time)
    
    def get_load_balancer_stats(self) -> Dict:
        """获取负载均衡统计"""
        healthy_count = sum(1 for b in self.backends.values() if b.is_healthy)
        total_connections = sum(b.current_connections for b in self.backends.values())
        avg_response_time = sum(b.response_time_avg for b in self.backends.values()) / len(self.backends) if self.backends else 0
        
        return {
            "total_backends": len(self.backends),
            "healthy_backends": healthy_count,
            "unhealthy_backends": len(self.backends) - healthy_count,
            "total_connections": total_connections,
            "active_sessions": len(self.sessions),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "average_response_time": avg_response_time,
            "algorithm": self.algorithm.value,
            "gray_rules_count": len(self.gray_rules)
        }
    
    def get_backend_stats(self, backend_id: str) -> Optional[Dict]:
        """获取单个后端统计"""
        if backend_id not in self.backends:
            return None
        
        backend = self.backends[backend_id]
        return {
            "id": backend.id,
            "host": backend.host,
            "port": backend.port,
            "weight": backend.weight,
            "current_connections": backend.current_connections,
            "max_connections": backend.max_connections,
            "connection_utilization": backend.current_connections / backend.max_connections,
            "is_healthy": backend.is_healthy,
            "avg_response_time": backend.response_time_avg,
            "error_rate": backend.error_rate,
            "last_health_check": backend.last_health_check.isoformat()
        }
    
    def gray_release_canary(self, backend_id: str, traffic_percentage: float) -> bool:
        """金丝雀发布"""
        if backend_id not in self.backends:
            return False
        
        rule = GrayReleaseRule(
            name=f"canary-{backend_id}",
            backend_id=backend_id,
            traffic_percentage=traffic_percentage
        )
        self.gray_rules.append(rule)
        return True
