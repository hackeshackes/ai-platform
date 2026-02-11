#!/usr/bin/env python3
"""
资源管理器 - resource_manager.py

功能:
- 自动扩容
- 负载均衡
- 健康检查
"""

import asyncio
import psutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class ScalingPolicy(Enum):
    MANUAL = "manual"
    AUTO = "auto"
    SCHEDULED = "scheduled"


class LoadBalancerAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"


@dataclass
class ResourceMetrics:
    """资源指标"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    active_connections: int
    request_rate: float
    error_rate: float
    latency_p99: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ScalingRule:
    """扩容规则"""
    metric: str  # cpu, memory, requests, errors
    operator: str  # gt, lt, gte, lte
    threshold: float
    action: str  # scale_up, scale_down
    step: int = 1
    cooldown_seconds: int = 60
    min_replicas: int = 1
    max_replicas: int = 10


@dataclass
class LoadBalancerConfig:
    """负载均衡配置"""
    algorithm: LoadBalancerAlgorithm = LoadBalancerAlgorithm.ROUND_ROBIN
    health_check_path: str = "/health"
    health_check_interval: int = 10
    health_check_timeout: int = 5
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    sticky_sessions: bool = False
    ssl_termination: bool = True


class ResourceManager:
    """资源管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scaling_rules: List[ScalingRule] = []
        self.lb_config = LoadBalancerConfig()
        self.target_deployments: Dict[str, Dict] = {}
        self.metrics_history: Dict[str, deque] = {}
        self.last_scaling_time: Dict[str, float] = {}
        self._running = False
        
    def add_scaling_rule(self, rule: ScalingRule):
        """添加扩容规则"""
        self.scaling_rules.append(rule)
    
    def add_deployment(self, deployment_id: str, endpoints: List[str], replicas: int = 1):
        """添加要管理的部署"""
        self.target_deployments[deployment_id] = {
            "endpoints": endpoints,
            "replicas": replicas,
            "healthy_replicas": replicas,
            "status": "healthy",
            "last_check": None
        }
        self.metrics_history[deployment_id] = deque(maxlen=100)
    
    async def start(self):
        """启动资源管理"""
        self._running = True
        print("[ResourceManager] 启动资源管理器")
        
        while self._running:
            try:
                await self._collect_metrics()
                await self._check_health()
                await self._evaluate_scaling()
                await self._update_load_balancer()
            except Exception as e:
                print(f"[ResourceManager] 错误: {e}")
            
            await asyncio.sleep(10)  # 每10秒检查一次
    
    def stop(self):
        """停止资源管理"""
        self._running = False
        print("[ResourceManager] 停止资源管理器")
    
    async def _collect_metrics(self):
        """收集资源指标"""
        for deployment_id in self.target_deployments:
            metrics = await self._get_deployment_metrics(deployment_id)
            self.metrics_history[deployment_id].append(metrics)
    
    async def _get_deployment_metrics(self, deployment_id: str) -> ResourceMetrics:
        """获取部署的资源指标"""
        # 模拟指标收集
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io={"bytes_sent": net_io.bytes_sent, "bytes_recv": net_io.bytes_recv},
            active_connections=0,
            request_rate=100.0,
            error_rate=0.1,
            latency_p99=50.0
        )
    
    async def _check_health(self):
        """健康检查"""
        for deployment_id, deployment in self.target_deployments.items():
            healthy_count = 0
            
            for endpoint in deployment["endpoints"]:
                is_healthy = await self._health_check(endpoint)
                if is_healthy:
                    healthy_count += 1
            
            deployment["healthy_replicas"] = healthy_count
            deployment["last_check"] = datetime.now().isoformat()
            
            if healthy_count == 0:
                deployment["status"] = "unhealthy"
            elif healthy_count < deployment["replicas"]:
                deployment["status"] = "degraded"
            else:
                deployment["status"] = "healthy"
    
    async def _health_check(self, endpoint: str) -> bool:
        """执行健康检查"""
        try:
            # 模拟健康检查
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False
    
    async def _evaluate_scaling(self):
        """评估是否需要扩容"""
        current_time = time.time()
        
        for deployment_id in self.target_deployments:
            if deployment_id not in self.metrics_history:
                continue
            
            # 获取最新指标
            metrics_list = list(self.metrics_history[deployment_id])
            if not metrics_list:
                continue
            
            avg_metrics = self._calculate_average_metrics(metrics_list)
            
            for rule in self.scaling_rules:
                # 检查冷却时间
                last_time = self.last_scaling_time.get(deployment_id, 0)
                if current_time - last_time < rule.cooldown_seconds:
                    continue
                
                should_scale, scale_action = self._evaluate_rule(avg_metrics, rule)
                
                if should_scale:
                    current_replicas = self.target_deployments[deployment_id]["replicas"]
                    
                    if scale_action == "scale_up":
                        new_replicas = min(current_replicas + rule.step, rule.max_replicas)
                    else:
                        new_replicas = max(current_replicas - rule.step, rule.min_replicas)
                    
                    if new_replicas != current_replicas:
                        print(f"[ResourceManager] {deployment_id}: {scale_action} "
                              f"{current_replicas} -> {new_replicas} replicas")
                        
                        await self._execute_scaling(deployment_id, new_replicas)
                        self.last_scaling_time[deployment_id] = current_time
    
    def _calculate_average_metrics(self, metrics_list: List[ResourceMetrics]) -> ResourceMetrics:
        """计算平均指标"""
        if not metrics_list:
            return ResourceMetrics(0, 0, 0, {}, 0, 0, 0, 0)
        
        return ResourceMetrics(
            cpu_percent=sum(m.cpu_percent for m in metrics_list) / len(metrics_list),
            memory_percent=sum(m.memory_percent for m in metrics_list) / len(metrics_list),
            disk_percent=sum(m.disk_percent for m in metrics_list) / len(metrics_list),
            network_io={},
            active_connections=sum(m.active_connections for m in metrics_list) // len(metrics_list),
            request_rate=sum(m.request_rate for m in metrics_list) / len(metrics_list),
            error_rate=sum(m.error_rate for m in metrics_list) / len(metrics_list),
            latency_p99=sum(m.latency_p99 for m in metrics_list) / len(metrics_list)
        )
    
    def _evaluate_rule(self, metrics: ResourceMetrics, rule: ScalingRule) -> tuple:
        """评估扩容规则"""
        metric_value = getattr(metrics, rule.metric, 0)
        
        if rule.operator == "gt":
            condition = metric_value > rule.threshold
        elif rule.operator == "gte":
            condition = metric_value >= rule.threshold
        elif rule.operator == "lt":
            condition = metric_value < rule.threshold
        elif rule.operator == "lte":
            condition = metric_value <= rule.threshold
        else:
            condition = False
        
        if condition:
            return True, rule.action
        return False, None
    
    async def _execute_scaling(self, deployment_id: str, replicas: int):
        """执行扩容操作"""
        self.target_deployments[deployment_id]["replicas"] = replicas
        # 实际扩容逻辑由deployer处理
        
        await self._notify_scaling_event(deployment_id, replicas)
    
    async def _notify_scaling_event(self, deployment_id: str, replicas: int):
        """通知扩容事件"""
        print(f"[ResourceManager] 扩容事件: {deployment_id} -> {replicas} 副本")
    
    async def _update_load_balancer(self):
        """更新负载均衡器配置"""
        for deployment_id, deployment in self.target_deployments.items():
            # 计算权重
            weights = self._calculate_weights(deployment)
            
            deployment["weights"] = weights
    
    def _calculate_weights(self, deployment: Dict) -> Dict[str, int]:
        """计算负载均衡权重"""
        weights = {}
        for i, endpoint in enumerate(deployment["endpoints"]):
            weights[endpoint] = 100 // len(deployment["endpoints"])
        return weights
    
    def get_next_endpoint(self, deployment_id: str) -> Optional[str]:
        """获取下一个要路由的端点（负载均衡）"""
        if deployment_id not in self.target_deployments:
            return None
        
        deployment = self.target_deployments[deployment_id]
        healthy_endpoints = deployment["endpoints"][:deployment["healthy_replicas"]]
        
        if not healthy_endpoints:
            return None
        
        if self.lb_config.algorithm == LoadBalancerAlgorithm.ROUND_ROBIN:
            return healthy_endpoints[0]
        elif self.lb_config.algorithm == LoadBalancerAlgorithm.LEAST_CONNECTIONS:
            return min(healthy_endpoints, key=lambda e: self._get_connection_count(e))
        elif self.lb_config.algorithm == LoadBalancerAlgorithm.IP_HASH:
            return healthy_endpoints[hash(healthy_endpoints[0]) % len(healthy_endpoints)]
        
        return healthy_endpoints[0]
    
    def _get_connection_count(self, endpoint: str) -> int:
        """获取连接数"""
        return 0  # 模拟
    
    async def manual_scale(self, deployment_id: str, replicas: int) -> Dict:
        """手动扩容"""
        if deployment_id not in self.target_deployments:
            return {"success": False, "error": "部署不存在"}
        
        self.target_deployments[deployment_id]["replicas"] = replicas
        
        return {
            "success": True,
            "deployment_id": deployment_id,
            "replicas": replicas
        }
    
    def get_resource_status(self, deployment_id: str = None) -> Dict:
        """获取资源状态"""
        if deployment_id:
            return self.target_deployments.get(deployment_id, {})
        return {
            "deployments": self.target_deployments,
            "scaling_rules": len(self.scaling_rules),
            "active_metrics": len(self.metrics_history)
        }
    
    def get_metrics_history(self, deployment_id: str, limit: int = 10) -> List[Dict]:
        """获取指标历史"""
        if deployment_id not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[deployment_id])[-limit:]
        return [m.__dict__ for m in history]


# CLI入口
if __name__ == "__main__":
    import json
    
    async def main():
        manager = ResourceManager()
        
        # 添加扩容规则
        manager.add_scaling_rule(ScalingRule(
            metric="cpu_percent",
            operator="gt",
            threshold=80,
            action="scale_up",
            step=1,
            cooldown_seconds=60
        ))
        
        manager.add_scaling_rule(ScalingRule(
            metric="cpu_percent",
            operator="lt",
            threshold=30,
            action="scale_down",
            step=1,
            cooldown_seconds=120
        ))
        
        # 添加测试部署
        manager.add_deployment("test-001", ["http://localhost:8080", "http://localhost:8081"], 2)
        
        print("[ResourceManager] 启动监控...")
        await manager.start()
    
    asyncio.run(main())
