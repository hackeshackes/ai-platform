"""
配置模块 - Configuration
自愈系统的配置管理
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    health_check_interval: int = 30  # 秒
    max_restart_attempts: int = 3
    restart_delay: int = 5  # 秒
    dependency_timeout: int = 10  # 秒


@dataclass
class ResourceThresholds:
    """资源阈值配置"""
    cpu_warning: float = 80.0
    cpu_critical: float = 95.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    disk_warning: float = 80.0
    disk_critical: float = 90.0
    network_latency_warning: int = 100  # ms
    network_latency_critical: int = 500  # ms


@dataclass
class FixStrategy:
    """修复策略配置"""
    failure_type: str
    auto_fix: bool = True
    max_attempts: int = 3
    strategy: str = ""
    rollback_strategy: str = ""
    cooldown_period: int = 300  # 秒
    requires_approval: bool = False
    approvers: List[str] = field(default_factory=list)


@dataclass
class NotificationConfig:
    """通知配置"""
    enabled: bool = True
    channels: List[str] = field(default_factory=list)
    escalation_time: int = 300  # 秒
    include_logs: bool = True


class Config:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()
        self.services: Dict[str, ServiceConfig] = {}
        self.resource_thresholds = ResourceThresholds()
        self.fix_strategies: Dict[str, FixStrategy] = {}
        self.notification = NotificationConfig()
        self._load_config()
    
    def _default_config_path(self) -> str:
        """获取默认配置路径"""
        return os.environ.get(
            'SELF_HEALING_CONFIG',
            str(Path(__file__).parent / 'config.yaml')
        )
    
    def _load_config(self):
        """加载配置"""
        # 默认配置
        self._set_default_strategies()
        self._set_default_services()
    
    def _set_default_strategies(self):
        """设置默认修复策略"""
        strategies = {
            'service_down': FixStrategy(
                failure_type='service_down',
                auto_fix=True,
                max_attempts=3,
                strategy='restart_service',
                cooldown_period=60
            ),
            'memory_leak': FixStrategy(
                failure_type='memory_leak',
                auto_fix=True,
                max_attempts=2,
                strategy='clear_cache_restart',
                cooldown_period=300
            ),
            'cpu_high': FixStrategy(
                failure_type='cpu_high',
                auto_fix=True,
                max_attempts=3,
                strategy='scale_up_rate_limit',
                cooldown_period=180
            ),
            'network_jitter': FixStrategy(
                failure_type='network_jitter',
                auto_fix=True,
                max_attempts=2,
                strategy='switch_network_route',
                cooldown_period=120
            ),
            'disk_full': FixStrategy(
                failure_type='disk_full',
                auto_fix=True,
                max_attempts=2,
                strategy='cleanup_logs_expand_disk',
                cooldown_period=600
            ),
            'slow_query': FixStrategy(
                failure_type='slow_query',
                auto_fix=True,
                max_attempts=2,
                strategy='optimize_index_rate_limit',
                cooldown_period=300,
                requires_approval=True
            )
        }
        self.fix_strategies.update(strategies)
    
    def _set_default_services(self):
        """设置默认服务配置"""
        self.services = {
            'web_server': ServiceConfig(
                name='web_server',
                health_check_interval=30,
                max_restart_attempts=3
            ),
            'database': ServiceConfig(
                name='database',
                health_check_interval=60,
                max_restart_attempts=2
            ),
            'cache': ServiceConfig(
                name='cache',
                health_check_interval=30,
                max_restart_attempts=3
            ),
            'queue': ServiceConfig(
                name='queue',
                health_check_interval=30,
                max_restart_attempts=2
            )
        }
    
    def get_strategy(self, failure_type: str) -> Optional[FixStrategy]:
        """获取修复策略"""
        return self.fix_strategies.get(failure_type)
    
    def get_service(self, service_name: str) -> Optional[ServiceConfig]:
        """获取服务配置"""
        return self.services.get(service_name)
    
    def update_threshold(self, resource: str, warning: float, critical: float):
        """更新资源阈值"""
        if hasattr(self.resource_thresholds, resource):
            setattr(self.resource_thresholds, f'{resource}_warning', warning)
            setattr(self.resource_thresholds, f'{resource}_critical', critical)
    
    def add_service(self, service: ServiceConfig):
        """添加服务配置"""
        self.services[service.name] = service
    
    def add_strategy(self, strategy: FixStrategy):
        """添加修复策略"""
        self.fix_strategies[strategy.failure_type] = strategy


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config(config_path: Optional[str] = None):
    """重新加载配置"""
    global _config
    _config = Config(config_path)
    return _config
