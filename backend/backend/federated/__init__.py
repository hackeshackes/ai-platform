"""
Privacy-Preserving Federated Learning Platform

A federated learning implementation with:
- FedAvg aggregation algorithm
- Differential privacy protection
- Secure communication support
- Complete error handling
"""

__version__ = "1.0.0"

from .fl_platform import FederatedLearningPlatform
from .privacy import PrivacyManager
from .aggregator import Aggregator
from .client import FLClient
from .models import (
    FLSession,
    FLConfig,
    LocalModel,
    GlobalModel,
    FLClientInfo,
    SessionStatus
)
from .storage import SessionStore

__all__ = [
    "FederatedLearningPlatform",
    "PrivacyManager",
    "Aggregator",
    "FLClient",
    "FLSession",
    "FLConfig",
    "LocalModel",
    "GlobalModel",
    "FLClientInfo",
    "SessionStatus",
    "SessionStore"
]
