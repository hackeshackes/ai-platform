"""
Quantum Simulator Configuration - 配置管理
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class SimulatorConfig:
    """模拟器配置"""
    n_qubits: int = 50
    shots: int = 1024
    noise_model: Optional[str] = None
    approximation_level: float = 1.0
    use_sparse: bool = False
    precision: str = "complex128"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SimulatorConfig':
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_string: str) -> 'SimulatorConfig':
        config_dict = json.loads(json_string)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        return {
            'n_qubits': self.n_qubits,
            'shots': self.shots,
            'noise_model': self.noise_model,
            'approximation_level': self.approximation_level,
            'use_sparse': self.use_sparse,
            'precision': self.precision
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# 默认配置
DEFAULT_CONFIG = SimulatorConfig()

# 噪声配置
NOISE_CONFIGS = {
    'light': {'probability': 0.001},
    'moderate': {'probability': 0.01},
    'heavy': {'probability': 0.1}
}
