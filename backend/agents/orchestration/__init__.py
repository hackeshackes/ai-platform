"""
Agent Orchestration - 多Agent编排模块
提供多Agent协同工作、工坊模式、任务调度和通信协议
"""

from .engine import OrchestrationEngine
from .workshop import WorkshopManager
from .communication import CommunicationManager

__all__ = [
    "OrchestrationEngine",
    "WorkshopManager", 
    "CommunicationManager"
]
