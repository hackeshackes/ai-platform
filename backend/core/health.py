"""
健康检查模块 - HealthChecker

提供系统各组件的健康状态检查
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class HealthChecker:
    """
    健康检查器
    
    支持自定义健康检查函数，可扩展
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化健康检查器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._checks: Dict[str, Callable] = {}
        self._check_results: Dict[str, HealthCheckResult] = {}
        self._initialized = False
        
        # 默认检查间隔（秒）
        self._check_interval = self.config.get('check_interval', 30)
        
        # 初始化默认检查
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """注册默认健康检查"""
        # 系统资源检查
        self.register_check("cpu", self._check_cpu)
        self.register_check("memory", self._check_memory)
        self.register_check("disk", self._check_disk)
        
        # 组件检查（可在子类中覆盖或外部注册）
        self.register_check("database", self._check_database)
        self.register_check("redis", self._check_redis)
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """
        注册健康检查函数
        
        Args:
            name: 检查名称
            check_func: 异步检查函数，返回HealthCheckResult
        """
        self._checks[name] = check_func
        logger.debug(f"已注册健康检查: {name}")
    
    async def initialize(self) -> None:
        """初始化健康检查器"""
        self._initialized = True
        logger.info("健康检查器初始化完成")
    
    async def shutdown(self) -> None:
        """关闭健康检查器"""
        self._check_results.clear()
        self._initialized = False
        logger.info("健康检查器已关闭")
    
    async def _check_cpu(self) -> HealthCheckResult:
        """检查CPU使用率"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            duration = psutil.cpu_times_percent().get(0, 0)
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"CPU使用率过高: {cpu_percent}%"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"CPU使用率较高: {cpu_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU使用率正常: {cpu_percent}%"
            
            return HealthCheckResult(
                name="cpu",
                status=status,
                message=message,
                details={"cpu_percent": cpu_percent}
            )
        except ImportError:
            # psutil 未安装时返回模拟数据
            return HealthCheckResult(
                name="cpu",
                status=HealthStatus.HEALTHY,
                message="CPU检查跳过（psutil未安装）",
                details={"cpu_percent": 25, "mock": True}
            )
        except Exception as e:
            return HealthCheckResult(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message="CPU检查失败",
                error=str(e)
            )
    
    async def _check_memory(self) -> HealthCheckResult:
        """检查内存使用率"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"内存使用率过高: {memory.percent}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"内存使用率较高: {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"内存使用率正常: {memory.percent}%"
            
            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                details={
                    "memory_percent": memory.percent,
                    "available_mb": memory.available // (1024 * 1024),
                    "total_mb": memory.total // (1024 * 1024)
                }
            )
        except ImportError:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.HEALTHY,
                message="内存检查跳过（psutil未安装）",
                details={"memory_percent": 45, "available_mb": 4096, "total_mb": 8192, "mock": True}
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="内存检查失败",
                error=str(e)
            )
    
    async def _check_disk(self) -> HealthCheckResult:
        """检查磁盘使用率"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            
            if disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"磁盘使用率过高: {disk.percent}%"
            elif disk.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"磁盘使用率较高: {disk.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"磁盘使用率正常: {disk.percent}%"
            
            return HealthCheckResult(
                name="disk",
                status=status,
                message=message,
                details={
                    "disk_percent": disk.percent,
                    "free_gb": disk.free // (1024 * 1024 * 1024),
                    "total_gb": disk.total // (1024 * 1024 * 1024)
                }
            )
        except ImportError:
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.HEALTHY,
                message="磁盘检查跳过（psutil未安装）",
                details={"disk_percent": 35, "free_gb": 200, "total_gb": 500, "mock": True}
            )
        except Exception as e:
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message="磁盘检查失败",
                error=str(e)
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """检查数据库连接（默认实现，可覆盖）"""
        # 默认返回healthy，实际使用时由外部注入
        return HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="数据库检查未配置",
            details={"configured": False}
        )
    
    async def _check_redis(self) -> HealthCheckResult:
        """检查Redis连接（默认实现，可覆盖）"""
        # 默认返回healthy，实际使用时由外部注入
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis检查未配置",
            details={"configured": False}
        )
    
    async def run_check(self, check_name: str) -> Optional[HealthCheckResult]:
        """
        运行单个健康检查
        
        Args:
            check_name: 检查名称
            
        Returns:
            HealthCheckResult或None
        """
        if check_name not in self._checks:
            logger.warning(f"未找到健康检查: {check_name}")
            return None
        
        check_func = self._checks[check_name]
        start_time = datetime.utcnow()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
        except Exception as e:
            result = HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNKNOWN,
                message=f"检查执行失败: {e}",
                error=str(e)
            )
        
        result.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self._check_results[check_name] = result
        
        return result
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """
        运行所有健康检查
        
        Returns:
            完整的健康检查结果
        """
        results = {}
        overall_status = HealthStatus.HEALTHY
        degraded_count = 0
        unhealthy_count = 0
        
        # 并行执行所有检查
        tasks = [self.run_check(name) for name in self._checks.keys()]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in check_results:
            if isinstance(result, HealthCheckResult):
                results[result.name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp.isoformat(),
                    "error": result.error
                }
                
                # 更新整体状态
                if result.status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    degraded_count += 1
                    overall_status = HealthStatus.DEGRADED
        
        # 计算响应时间
        total_duration = sum(r.duration_ms for r in self._check_results.values() if isinstance(r, HealthCheckResult))
        
        return {
            "status": overall_status.value,
            "overall_status": overall_status.value,
            "checks": results,
            "summary": {
                "total": len(self._checks),
                "healthy": len([r for r in self._check_results.values() if isinstance(r, HealthCheckResult) and r.status == HealthStatus.HEALTHY]),
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "total_duration_ms": total_duration,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取健康检查器状态
        
        Returns:
            状态信息
        """
        return {
            "initialized": self._initialized,
            "registered_checks": list(self._checks.keys()),
            "last_results": {
                name: {
                    "status": result.status.value,
                    "message": result.message
                }
                for name, result in self._check_results.items()
            }
        }
    
    def set_database_check(self, check_func: Callable) -> None:
        """
        设置数据库健康检查函数
        
        Args:
            check_func: 异步检查函数
        """
        self.register_check("database", check_func)
    
    def set_redis_check(self, check_func: Callable) -> None:
        """
        设置Redis健康检查函数
        
        Args:
            check_func: 异步检查函数
        """
        self.register_check("redis", check_func)
