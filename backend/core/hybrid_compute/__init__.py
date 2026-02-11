"""
量子-经典混合计算平台

提供量子经典混合计算、任务分解自动化、混合编排等功能。
"""

__version__ = "1.0.0"

from .task_decomposer import TaskDecomposer
from .orchestrator import HybridOrchestrator
from .hybrid_circuits import HybridCircuit, QuantumSubCircuit, ClassicalControl
from .resource_manager import ResourceManager

__all__ = [
    'TaskDecomposer',
    'HybridOrchestrator', 
    'HybridCircuit',
    'QuantumSubCircuit',
    'ClassicalControl',
    'ResourceManager'
]
