"""
智能调度系统 - 配置模块
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json


class OptimizationStrategy(Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    BALANCED = "balanced"
    ENERGY = "energy"


class LoadBalancingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"


@dataclass
class SchedulerConfig:
    """调度器配置"""
    
    # 资源优化配置
    optimization_strategy: str = "balanced"
    cpu_threshold_high: float = 80.0
    cpu_threshold_low: float = 30.0
    memory_threshold_high: float = 85.0
    memory_threshold_low: float = 40.0
    gpu_allocation_strategy: str = "efficient"
    
    # 负载均衡配置
    load_balancing_algorithm: str = "adaptive"
    health_check_interval: int = 30
    session_timeout: int = 86400
    enable_health_check: bool = True
    
    # 自动伸缩配置
    default_cooldown: int = 300
    enable_predictive_scaling: bool = True
    predictive_look_ahead_minutes: int = 30
    min_instances: int = 1
    max_instances: int = 100
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    
    # 成本优化配置
    spot_discount: float = 0.7
    reserved_discount: float = 0.3
    idle_threshold: float = 0.2
    low_utilization_threshold: float = 0.3
    
    # 通用配置
    log_level: str = "INFO"
    metrics_collection_interval: int = 60
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SchedulerConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "optimization_strategy": self.optimization_strategy,
            "cpu_threshold_high": self.cpu_threshold_high,
            "cpu_threshold_low": self.cpu_threshold_low,
            "memory_threshold_high": self.memory_threshold_high,
            "memory_threshold_low": self.memory_threshold_low,
            "gpu_allocation_strategy": self.gpu_allocation_strategy,
            "load_balancing_algorithm": self.load_balancing_algorithm,
            "health_check_interval": self.health_check_interval,
            "session_timeout": self.session_timeout,
            "enable_health_check": self.enable_health_check,
            "default_cooldown": self.default_cooldown,
            "enable_predictive_scaling": self.enable_predictive_scaling,
            "predictive_look_ahead_minutes": self.predictive_look_ahead_minutes,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "spot_discount": self.spot_discount,
            "reserved_discount": self.reserved_discount,
            "idle_threshold": self.idle_threshold,
            "low_utilization_threshold": self.low_utilization_threshold,
            "log_level": self.log_level,
            "metrics_collection_interval": self.metrics_collection_interval
        }
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SchedulerConfig':
        """从JSON创建配置"""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2)


# 默认配置实例
DEFAULT_CONFIG = SchedulerConfig()


# 配置验证
def validate_config(config: SchedulerConfig) -> List[str]:
    """验证配置"""
    errors = []
    
    if config.cpu_threshold_high <= config.cpu_threshold_low:
        errors.append("CPU阈值设置错误: high必须大于low")
    
    if config.memory_threshold_high <= config.memory_threshold_low:
        errors.append("内存阈值设置错误: high必须大于low")
    
    if config.min_instances < 1:
        errors.append("最小实例数必须>=1")
    
    if config.max_instances < config.min_instances:
        errors.append("最大实例数必须>=最小实例数")
    
    if config.default_cooldown < 0:
        errors.append("冷却时间不能为负")
    
    if not (0 <= config.spot_discount <= 1):
        errors.append("竞价折扣必须在0-1之间")
    
    if not (0 <= config.reserved_discount <= 1):
        errors.append("预留折扣必须在0-1之间")
    
    return errors


# 预设配置
PRESETS = {
    "development": SchedulerConfig(
        optimization_strategy="balanced",
        load_balancing_algorithm="round_robin",
        enable_health_check=False,
        default_cooldown=60,
        min_instances=1,
        max_instances=10,
        log_level="DEBUG"
    ),
    
    "staging": SchedulerConfig(
        optimization_strategy="balanced",
        load_balancing_algorithm="adaptive",
        enable_health_check=True,
        default_cooldown=180,
        min_instances=2,
        max_instances=20,
        log_level="INFO"
    ),
    
    "production": SchedulerConfig(
        optimization_strategy="performance",
        load_balancing_algorithm="adaptive",
        enable_health_check=True,
        default_cooldown=300,
        enable_predictive_scaling=True,
        min_instances=3,
        max_instances=200,
        log_level="INFO"
    ),
    
    "cost_optimized": SchedulerConfig(
        optimization_strategy="cost",
        load_balancing_algorithm="least_connections",
        enable_health_check=True,
        default_cooldown=600,
        min_instances=1,
        max_instances=100,
        spot_discount=0.7,
        reserved_discount=0.4
    ),
    
    "high_performance": SchedulerConfig(
        optimization_strategy="performance",
        load_balancing_algorithm="adaptive",
        enable_health_check=True,
        default_cooldown=120,
        min_instances=5,
        max_instances=500,
        cpu_threshold_high=90.0,
        scale_up_threshold=70.0
    )
}


def get_preset(preset_name: str) -> Optional[SchedulerConfig]:
    """获取预设配置"""
    return PRESETS.get(preset_name)
