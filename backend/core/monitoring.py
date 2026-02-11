"""
监控主模块 - MonitoringSystem

整合健康检查、性能指标、日志追踪和告警系统
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .health import HealthChecker
from .metrics import MetricsCollector
from .logger import StructuredLogger
from .alerts import AlertManager
from .tracing import TracingManager

logger = logging.getLogger(__name__)


class MonitoringSystem:
    """
    完整监控体系主类
    
    整合所有监控组件，提供统一的监控入口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化监控体系
        
        Args:
            config: 可选配置字典
        """
        self.config = config or {}
        self._initialized = False
        self._start_time = datetime.utcnow()
        
        # 初始化各组件
        self.health = HealthChecker(config=self.config.get('health'))
        self.metrics = MetricsCollector(config=self.config.get('metrics'))
        self.logger = StructuredLogger(config=self.config.get('logger'))
        self.alerts = AlertManager(config=self.config.get('alerts'))
        self.tracing = TracingManager(config=self.config.get('tracing'))
    
    async def initialize(self) -> None:
        """初始化所有监控组件"""
        if self._initialized:
            logger.warning("监控系统已初始化，跳过重复初始化")
            return
        
        try:
            # 初始化各组件
            await self.health.initialize()
            await self.metrics.initialize()
            await self.alerts.initialize()
            await self.tracing.initialize()
            
            self._initialized = True
            logger.info("监控系统初始化完成")
            
        except Exception as e:
            logger.error(f"监控系统初始化失败: {e}")
            raise
    
    async def shutdown(self) -> None:
        """关闭监控系统"""
        if not self._initialized:
            return
        
        try:
            await self.health.shutdown()
            await self.metrics.shutdown()
            await self.alerts.shutdown()
            await self.tracing.shutdown()
            
            self._initialized = False
            logger.info("监控系统已关闭")
            
        except Exception as e:
            logger.error(f"关闭监控系统时出错: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取监控体系整体状态
        
        Returns:
            状态信息字典
        """
        return {
            "initialized": self._initialized,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "components": {
                "health": self.health.get_status(),
                "metrics": self.metrics.get_status(),
                "alerts": self.alerts.get_status(),
                "tracing": self.tracing.get_status()
            }
        }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """
        执行完整健康检查
        
        Returns:
            健康检查结果
        """
        return await self.health.run_all_checks()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """
        收集所有性能指标
        
        Returns:
            性能指标字典
        """
        return await self.metrics.collect_all()
    
    async def check_alerts(self) -> list:
        """
        检查所有告警规则
        
        Returns:
            触发的告警列表
        """
        metrics = await self.collect_metrics()
        return await self.alerts.check_all(metrics)
    
    def get_uptime(self) -> float:
        """
        获取系统运行时间（秒）
        
        Returns:
            运行时间
        """
        return (datetime.utcnow() - self._start_time).total_seconds()


# 单例实例
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system(config: Optional[Dict[str, Any]] = None) -> MonitoringSystem:
    """
    获取监控体系单例
    
    Args:
        config: 可选配置
        
    Returns:
        MonitoringSystem实例
    """
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem(config)
    return _monitoring_system


async def init_monitoring(config: Optional[Dict[str, Any]] = None) -> MonitoringSystem:
    """
    初始化监控系统（异步）
    
    Args:
        config: 可选配置
        
    Returns:
        初始化的MonitoringSystem实例
    """
    system = get_monitoring_system(config)
    await system.initialize()
    return system


async def shutdown_monitoring() -> None:
    """关闭监控系统"""
    global _monitoring_system
    if _monitoring_system:
        await _monitoring_system.shutdown()
        _monitoring_system = None
