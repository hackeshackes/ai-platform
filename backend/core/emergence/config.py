"""
Emergence Engine Configuration - 涌现引擎配置
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class EmergenceConfig:
    """涌现引擎配置"""
    
    # 能力检测配置
    detection: Dict[str, Any] = field(default_factory=lambda: {
        'emergence_threshold': 0.7,
        'novelty_threshold': 0.5,
        'max_pattern_memory': 10000,
        'boundary_precision': 0.01,
        'min_occurrence': 3,
        'similarity_window': 100
    })
    
    # 自组织配置
    organization: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.01,
        'plasticity_scale': 1.0,
        'structural_plasticity_rate': 0.001,
        'max_neurons_per_layer': 100,
        'max_connections_per_neuron': 50,
        'pruning_threshold': 0.01,
        'emergence_window': 1000
    })
    
    # 创意生成配置
    creativity: Dict[str, Any] = field(default_factory=lambda: {
        'max_solutions': 10,
        'novelty_threshold': 0.6,
        'creativity_weight': 0.4,
        'analogy_depth': 3,
        'chain_length': 5,
        'temperature': 0.8,
        'max_iterations': 100
    })
    
    # 监控配置
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'max_events': 10000,
        'quality_threshold': 0.9,
        'safety_threshold': 0.8,
        'impact_window': 1000,
        'log_retention_days': 30,
        'auto_revert_enabled': True
    })
    
    # API配置
    api: Dict[str, Any] = field(default_factory=lambda: {
        'host': '0.0.0.0',
        'port': 8080,
        'debug': False
    })
    
    # 验收标准
    acceptance: Dict[str, Any] = field(default_factory=lambda: {
        'monthly_emergence_target': 1,
        'quality_threshold': 0.9,
        'safety_threshold': 0.95,
        'controllable': True,
        'reversible': True
    })
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmergenceConfig':
        """从字典创建配置"""
        return cls(
            detection=config_dict.get('detection', cls().detection),
            organization=config_dict.get('organization', cls().organization),
            creativity=config_dict.get('creativity', cls().creativity),
            monitoring=config_dict.get('monitoring', cls().monitoring),
            api=config_dict.get('api', cls().api),
            acceptance=config_dict.get('acceptance', cls().acceptance)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'detection': self.detection,
            'organization': self.organization,
            'creativity': self.creativity,
            'monitoring': self.monitoring,
            'api': self.api,
            'acceptance': self.acceptance
        }
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """获取组件配置"""
        config_map = {
            'detector': self.detection,
            'organizer': self.organization,
            'generator': self.creativity,
            'monitor': self.monitoring
        }
        return config_map.get(component, {})


# 默认配置实例
DEFAULT_CONFIG = EmergenceConfig()
