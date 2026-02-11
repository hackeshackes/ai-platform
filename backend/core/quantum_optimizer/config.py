"""
量子优化器配置模块
Quantum Optimizer Configuration Module
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class QuantumOptimizerConfig:
    """量子优化器配置类"""
    
    # 优化器设置
    optimizer: str = "cobyla"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    learning_rate: float = 0.01
    
    # QAOA设置
    qaoa_layers: int = 3
    qaoa_max_cut_iterations: int = 500
    
    # VQE设置
    vqe_max_iterations: int = 1000
    vqe_tolerance: float = 1e-8
    
    # 变分形式设置
    ansatz_type: str = "hardware_efficient"
    num_qubits: int = 4
    depth: int = 3
    
    # 优化器特定参数
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # 设备设置
    use_simulation: bool = True
    shots: int = 1024
    seed: int = 42
    
    # 性能设置
    enable_caching: bool = True
    parallel_execution: bool = False
    num_workers: int = 4
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantumOptimizerConfig":
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "optimizer": self.optimizer,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "learning_rate": self.learning_rate,
            "qaoa_layers": self.qaoa_layers,
            "qaoa_max_cut_iterations": self.qaoa_max_cut_iterations,
            "vqe_max_iterations": self.vqe_max_iterations,
            "vqe_tolerance": self.vqe_tolerance,
            "ansatz_type": self.ansatz_type,
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "optimizer_params": self.optimizer_params,
            "use_simulation": self.use_simulation,
            "shots": self.shots,
            "seed": self.seed,
            "enable_caching": self.enable_caching,
            "parallel_execution": self.parallel_execution,
            "num_workers": self.num_workers
        }
    
    @classmethod
    def default_qaoa(cls) -> "QuantumOptimizerConfig":
        """创建默认QAOA配置"""
        return cls(
            optimizer="cobyla",
            qaoa_layers=3,
            max_iterations=500,
            tolerance=1e-6
        )
    
    @classmethod
    def default_vqe(cls, num_qubits: int = 4) -> "QuantumOptimizerConfig":
        """创建默认VQE配置"""
        return cls(
            optimizer="spsa",
            num_qubits=num_qubits,
            depth=3,
            max_iterations=1000,
            tolerance=1e-8
        )


# 默认配置实例
DEFAULT_CONFIG = QuantumOptimizerConfig()

# 预定义配置
PRESETS = {
    "fast": QuantumOptimizerConfig(
        optimizer="cobyla",
        max_iterations=100,
        qaoa_layers=1,
        shots=256
    ),
    "accurate": QuantumOptimizerConfig(
        optimizer="spsa",
        max_iterations=2000,
        qaoa_layers=5,
        shots=8192,
        tolerance=1e-10
    ),
    "large_scale": QuantumOptimizerConfig(
        optimizer="cobyla",
        max_iterations=500,
        qaoa_layers=3,
        shots=1024,
        enable_caching=True,
        parallel_execution=True,
        num_workers=8
    )
}
