"""
Distributed Training Module - AI Platform v6

分布式训练模块，支持Ray集群和千卡级训练能力。
"""

from .ray import RayClusterManager, RayClient, RayClusterStatus
from .trainer import DistributedTrainer, TrainingConfig
from .scheduler import TaskScheduler, TrainingTask
from .monitor import ResourceMonitor, ClusterMetrics

__all__ = [
    "RayClusterManager",
    "RayClient", 
    "RayClusterStatus",
    "DistributedTrainer",
    "TrainingConfig",
    "TaskScheduler",
    "TrainingTask",
    "ResourceMonitor",
    "ClusterMetrics",
]
