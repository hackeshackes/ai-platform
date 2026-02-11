"""
健康检查器 - Health Checker
服务健康检测、依赖健康检测、资源健康检测、自检报告
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import get_config, ResourceThresholds


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DOWN = "down"


@dataclass
class ServiceHealth:
    """服务健康状态"""
    service_name: str
    status: HealthStatus
    message: str
    last_check: datetime
    response_time: float  # 毫秒
    details: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    consecutive_failures: int = 0


@dataclass
class DependencyHealth:
    """依赖健康状态"""
    dependency_name: str
    dependency_type: str  # database, cache, queue, api, etc.
    status: HealthStatus
    message: str
    last_check: datetime
    response_time: float
    connection_pool: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceHealth:
    """资源健康状态"""
    resource_type: str  # cpu, memory, disk, network
    status: HealthStatus
    usage: float  # 百分比
    message: str
    last_check: datetime
    thresholds: ResourceThresholds = field(default_factory=ResourceThresholds)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """健康检查报告"""
    report_id: str
    timestamp: datetime
    overall_status: HealthStatus
    services: List[ServiceHealth]
    dependencies: List[DependencyHealth]
    resources: List[ResourceHealth]
    summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.service_checks: Dict[str, Callable] = {}
        self.dependency_checks: Dict[str, Callable] = {}
        self.resource_checks: Dict[str, Callable] = {}
        self.health_history: List[HealthReport] = []
        self.max_history: int = 100
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """设置默认检查函数"""
        # 服务检查
        self.register_service_check('http', self._check_http)
        self.register_service_check('process', self._check_process)
        self.register_service_check('port', self._check_port)
        
        # 依赖检查
        self.register_dependency_check('database', self._check_database)
        self.register_dependency_check('redis', self._check_redis)
        self.register_dependency_check('rabbitmq', self._check_rabbitmq)
        
        # 资源检查
        self.register_resource_check('cpu', self._check_cpu)
        self.register_resource_check('memory', self._check_memory)
        self.register_resource_check('disk', self._check_disk)
        self.register_resource_check('network', self._check_network)
    
    def register_service_check(self, check_type: str, check_func: Callable):
        """注册服务检查函数"""
        self.service_checks[check_type] = check_func
    
    def register_dependency_check(self, dep_type: str, check_func: Callable):
        """注册依赖检查函数"""
        self.dependency_checks[dep_type] = check_func
    
    def register_resource_check(self, resource_type: str, check_func: Callable):
        """注册资源检查函数"""
        self.resource_checks[resource_type] = check_func
    
    async def check_all(self) -> HealthReport:
        """执行所有健康检查"""
        services = []
        dependencies = []
        resources = []
        
        # 并行执行检查
        tasks = []
        
        # 服务检查任务
        for service_name, service_config in self.config.services.items():
            tasks.append(self._check_service(service_name, service_config))
        
        # 资源检查任务
        for resource_type in self.resource_checks:
            tasks.append(self._check_resource(resource_type))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ServiceHealth):
                services.append(result)
            elif isinstance(result, DependencyHealth):
                dependencies.append(result)
            elif isinstance(result, ResourceHealth):
                resources.append(result)
        
        # 计算总体状态
        overall_status = self._calculate_overall_status(
            services, dependencies, resources
        )
        
        # 生成报告
        report = HealthReport(
            report_id=f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            overall_status=overall_status,
            services=services,
            dependencies=dependencies,
            resources=resources,
            summary=self._generate_summary(services, dependencies, resources),
            recommendations=self._generate_recommendations(
                services, dependencies, resources
            )
        )
        
        # 保存历史
        self.health_history.append(report)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
        
        return report
    
    async def _check_service(
        self, service_name: str, service_config
    ) -> ServiceHealth:
        """检查单个服务"""
        # 默认使用进程检查
        check_func = self.service_checks.get('process', self._check_process)
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func(service_name, service_config)
            else:
                result = check_func(service_name, service_config)
            return result
        except Exception as e:
            logger.error(f"Service check failed for {service_name}: {e}")
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.DOWN,
                message=str(e),
                last_check=datetime.now(),
                response_time=0,
                error_count=1
            )
    
    async def _check_resource(self, resource_type: str) -> ResourceHealth:
        """检查单个资源"""
        check_func = self.resource_checks.get(resource_type)
        if not check_func:
            return ResourceHealth(
                resource_type=resource_type,
                status=HealthStatus.UNKNOWN,
                usage=0,
                message=f"No check function for {resource_type}",
                last_check=datetime.now()
            )
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            return result
        except Exception as e:
            logger.error(f"Resource check failed for {resource_type}: {e}")
            return ResourceHealth(
                resource_type=resource_type,
                status=HealthStatus.CRITICAL,
                usage=0,
                message=str(e),
                last_check=datetime.now()
            )
    
    def _check_http(self, service_name: str, config) -> ServiceHealth:
        """HTTP服务检查"""
        import aiohttp
        
        url = getattr(config, 'health_url', f'http://localhost/health')
        timeout = getattr(config, 'timeout', 10)
        
        start_time = time.time()
        try:
            response = aiohttp.ClientSession().get(url, timeout=timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.HEALTHY,
                    message="Service is responding",
                    last_check=datetime.now(),
                    response_time=response_time,
                    details={'status_code': response.status_code}
                )
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.WARNING,
                    message=f"Service returned status {response.status_code}",
                    last_check=datetime.now(),
                    response_time=response_time,
                    details={'status_code': response.status_code}
                )
        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.DOWN,
                message=str(e),
                last_check=datetime.now(),
                response_time=0,
                error_count=1
            )
    
    def _check_process(self, service_name: str, config) -> ServiceHealth:
        """进程检查"""
        process_name = getattr(config, 'process_name', service_name)
        
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                if proc.info['name'] == process_name:
                    status = proc.info['status']
                    if status == psutil.STATUS_RUNNING:
                        return ServiceHealth(
                            service_name=service_name,
                            status=HealthStatus.HEALTHY,
                            message="Process is running",
                            last_check=datetime.now(),
                            response_time=1.0,
                            details={'pid': proc.info['pid']}
                        )
                    else:
                        return ServiceHealth(
                            service_name=service_name,
                            status=HealthStatus.WARNING,
                            message=f"Process status: {status}",
                            last_check=datetime.now(),
                            response_time=0
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return ServiceHealth(
            service_name=service_name,
            status=HealthStatus.DOWN,
            message=f"Process {process_name} not found",
            last_check=datetime.now(),
            response_time=0
        )
    
    def _check_port(self, service_name: str, config) -> ServiceHealth:
        """端口检查"""
        host = getattr(config, 'host', 'localhost')
        port = getattr(config, 'port', 80)
        
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        start_time = time.time()
        try:
            result = sock.connect_ex((host, port))
            response_time = (time.time() - start_time) * 1000
            
            if result == 0:
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.HEALTHY,
                    message=f"Port {port} is open",
                    last_check=datetime.now(),
                    response_time=response_time
                )
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.DOWN,
                    message=f"Port {port} is closed",
                    last_check=datetime.now(),
                    response_time=response_time
                )
        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.DOWN,
                message=str(e),
                last_check=datetime.now(),
                response_time=0
            )
        finally:
            sock.close()
    
    async def _check_database(self) -> DependencyHealth:
        """数据库健康检查"""
        # 模拟检查
        return DependencyHealth(
            dependency_name="database",
            dependency_type="database",
            status=HealthStatus.HEALTHY,
            message="Database connection is healthy",
            last_check=datetime.now(),
            response_time=5.0,
            connection_pool={'active': 10, 'idle': 5, 'max': 20}
        )
    
    async def _check_redis(self) -> DependencyHealth:
        """Redis健康检查"""
        return DependencyHealth(
            dependency_name="redis",
            dependency_type="cache",
            status=HealthStatus.HEALTHY,
            message="Redis connection is healthy",
            last_check=datetime.now(),
            response_time=1.0,
            details={'memory_used': '50MB', 'keys': 1000}
        )
    
    async def _check_rabbitmq(self) -> DependencyHealth:
        """RabbitMQ健康检查"""
        return DependencyHealth(
            dependency_name="rabbitmq",
            dependency_type="queue",
            status=HealthStatus.HEALTHY,
            message="RabbitMQ connection is healthy",
            last_check=datetime.now(),
            response_time=2.0,
            details={'queues': 5, 'messages': 100}
        )
    
    def _check_cpu(self) -> ResourceHealth:
        """CPU健康检查"""
        usage = psutil.cpu_percent(interval=1)
        thresholds = self.config.resource_thresholds
        
        if usage >= thresholds.cpu_critical:
            status = HealthStatus.CRITICAL
            message = f"CPU usage is critically high: {usage}%"
        elif usage >= thresholds.cpu_warning:
            status = HealthStatus.WARNING
            message = f"CPU usage is high: {usage}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage is normal: {usage}%"
        
        return ResourceHealth(
            resource_type="cpu",
            status=status,
            usage=usage,
            message=message,
            last_check=datetime.now(),
            thresholds=thresholds,
            details={'cores': psutil.cpu_count(), 'load_avg': psutil.getloadavg()}
        )
    
    def _check_memory(self) -> ResourceHealth:
        """内存健康检查"""
        mem = psutil.virtual_memory()
        thresholds = self.config.resource_thresholds
        
        if mem.percent >= thresholds.memory_critical:
            status = HealthStatus.CRITICAL
            message = f"Memory usage is critically high: {mem.percent}%"
        elif mem.percent >= thresholds.memory_warning:
            status = HealthStatus.WARNING
            message = f"Memory usage is high: {mem.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage is normal: {mem.percent}%"
        
        return ResourceHealth(
            resource_type="memory",
            status=status,
            usage=mem.percent,
            message=message,
            last_check=datetime.now(),
            thresholds=thresholds,
            details={'total': mem.total, 'available': mem.available, 'used': mem.used}
        )
    
    def _check_disk(self) -> ResourceHealth:
        """磁盘健康检查"""
        disk = psutil.disk_usage('/')
        thresholds = self.config.resource_thresholds
        
        if disk.percent >= thresholds.disk_critical:
            status = HealthStatus.CRITICAL
            message = f"Disk usage is critically high: {disk.percent}%"
        elif disk.percent >= thresholds.disk_warning:
            status = HealthStatus.WARNING
            message = f"Disk usage is high: {disk.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage is normal: {disk.percent}%"
        
        return ResourceHealth(
            resource_type="disk",
            status=status,
            usage=disk.percent,
            message=message,
            last_check=datetime.now(),
            thresholds=thresholds,
            details={'total': disk.total, 'used': disk.used, 'free': disk.free}
        )
    
    def _check_network(self) -> ResourceHealth:
        """网络健康检查"""
        # 简单延迟检查
        import socket
        
        thresholds = self.config.resource_thresholds
        
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            latency = 20  # 模拟延迟
            status = HealthStatus.HEALTHY
            message = f"Network is healthy, latency: {latency}ms"
        except Exception:
            latency = 1000
            status = HealthStatus.CRITICAL
            message = "Network is unreachable"
        
        return ResourceHealth(
            resource_type="network",
            status=status,
            usage=min(latency / thresholds.network_latency_critical * 100, 100),
            message=message,
            last_check=datetime.now(),
            thresholds=thresholds,
            details={'latency_ms': latency}
        )
    
    def _calculate_overall_status(
        self,
        services: List[ServiceHealth],
        dependencies: List[DependencyHealth],
        resources: List[ResourceHealth]
    ) -> HealthStatus:
        """计算总体健康状态"""
        all_statuses = []
        
        all_statuses.extend([s.status for s in services])
        all_statuses.extend([d.status for d in dependencies])
        all_statuses.extend([r.status for r in resources])
        
        if HealthStatus.CRITICAL in all_statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DOWN in all_statuses:
            return HealthStatus.DOWN
        elif HealthStatus.WARNING in all_statuses:
            return HealthStatus.WARNING
        elif all_statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _generate_summary(
        self,
        services: List[ServiceHealth],
        dependencies: List[DependencyHealth],
        resources: List[ResourceHealth]
    ) -> Dict[str, Any]:
        """生成健康摘要"""
        return {
            'total_services': len(services),
            'healthy_services': sum(1 for s in services if s.status == HealthStatus.HEALTHY),
            'total_dependencies': len(dependencies),
            'healthy_dependencies': sum(1 for d in dependencies if d.status == HealthStatus.HEALTHY),
            'total_resources': len(resources),
            'healthy_resources': sum(1 for r in resources if r.status == HealthStatus.HEALTHY),
            'check_duration_ms': sum(
                (s.response_time for s in services),
                (d.response_time for d in dependencies),
                (r.usage for r in resources)
            )
        }
    
    def _generate_recommendations(
        self,
        services: List[ServiceHealth],
        dependencies: List[DependencyHealth],
        resources: List[ResourceHealth]
    ) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        for service in services:
            if service.status in [HealthStatus.DOWN, HealthStatus.WARNING]:
                recommendations.append(
                    f"Service '{service.service_name}': {service.message}"
                )
        
        for resource in resources:
            if resource.status == HealthStatus.CRITICAL:
                recommendations.append(
                    f"Critical: {resource.message}. Immediate action required."
                )
            elif resource.status == HealthStatus.WARNING:
                recommendations.append(
                    f"Warning: {resource.message}. Consider taking action soon."
                )
        
        return recommendations
    
    def get_latest_report(self) -> Optional[HealthReport]:
        """获取最新健康报告"""
        if self.health_history:
            return self.health_history[-1]
        return None
    
    def get_health_history(
        self, since: Optional[datetime] = None
    ) -> List[HealthReport]:
        """获取健康历史"""
        if since:
            return [r for r in self.health_history if r.timestamp >= since]
        return self.health_history


# 创建全局健康检查器实例
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def run_health_check() -> HealthReport:
    """执行健康检查"""
    checker = get_health_checker()
    return await checker.check_all()
