"""
Distillation Package - v3.0 Core Feature

模型蒸馏包
"""
from .engine import (
    DistillationEngine,
    DistillationConfig,
    DistillationJob,
    DistillationStatus,
    DistillationStrategy,
    create_distillation_engine
)
from .losses import (
    DistillationLoss,
    KLDivergenceLoss,
    MSELoss,
    CosineEmbeddingLoss,
    AttentionBasedLoss,
    HiddenStateLoss,
    CombinedLoss,
    LossFactory
)

__all__ = [
    # Engine
    "DistillationEngine",
    "DistillationConfig",
    "DistillationJob",
    "DistillationStatus",
    "DistillationStrategy",
    "create_distillation_engine",
    # Losses
    "DistillationLoss",
    "KLDivergenceLoss",
    "MSELoss",
    "CosineEmbeddingLoss",
    "AttentionBasedLoss",
    "HiddenStateLoss",
    "CombinedLoss",
    "LossFactory",
]
