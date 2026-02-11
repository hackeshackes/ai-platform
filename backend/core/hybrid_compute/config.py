"""
混合计算配置
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HybridComputeConfig:
    """混合计算配置"""
    
    # 任务分解配置
    decomposition_strategy: str = "auto"
    quantum_threshold: float = 0.7
    
    # 编排器配置
    quantum_backend: str = "quantum_simulator"
    classical_backend: str = "python"
    max_workers: int = 4
    timeout: int = 3600
    fail_fast: bool = False
    
    # 资源管理配置
    load_balancing: str = "round_robin"
    cost_budget: float = 100.0
    
    # 量子电路配置
    default_qubits: int = 4
    default_shots: int = 1024
    default_depth: int = 10
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridComputeConfig':
        """从字典创建配置"""
        return cls(
            decomposition_strategy=config_dict.get('decomposition_strategy', cls.decomposition_strategy),
            quantum_threshold=config_dict.get('quantum_threshold', cls.quantum_threshold),
            quantum_backend=config_dict.get('quantum_backend', cls.quantum_backend),
            classical_backend=config_dict.get('classical_backend', cls.classical_backend),
            max_workers=config_dict.get('max_workers', cls.max_workers),
            timeout=config_dict.get('timeout', cls.timeout),
            fail_fast=config_dict.get('fail_fast', cls.fail_fast),
            load_balancing=config_dict.get('load_balancing', cls.load_balancing),
            cost_budget=config_dict.get('cost_budget', cls.cost_budget),
            default_qubits=config_dict.get('default_qubits', cls.default_qubits),
            default_shots=config_dict.get('default_shots', cls.default_shots),
            default_depth=config_dict.get('default_depth', cls.default_depth)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'decomposition_strategy': self.decomposition_strategy,
            'quantum_threshold': self.quantum_threshold,
            'quantum_backend': self.quantum_backend,
            'classical_backend': self.classical_backend,
            'max_workers': self.max_workers,
            'timeout': self.timeout,
            'fail_fast': self.fail_fast,
            'load_balancing': self.load_balancing,
            'cost_budget': self.cost_budget,
            'default_qubits': self.default_qubits,
            'default_shots': self.default_shots,
            'default_depth': self.default_depth
        }


# 默认配置
DEFAULT_CONFIG = HybridComputeConfig()


# 预设配置
PRESETS = {
    'development': HybridComputeConfig(
        quantum_backend='quantum_simulator',
        classical_backend='python',
        max_workers=2,
        timeout=600,
        cost_budget=10.0
    ),
    'production': HybridComputeConfig(
        quantum_backend='quantum_device',
        classical_backend='python',
        max_workers=8,
        timeout=3600,
        cost_budget=1000.0
    ),
    'research': HybridComputeConfig(
        quantum_backend='quantum_simulator',
        classical_backend='python',
        max_workers=16,
        timeout=7200,
        cost_budget=500.0
    )
}


def get_preset(preset_name: str) -> Optional[HybridComputeConfig]:
    """获取预设配置"""
    return PRESETS.get(preset_name)
