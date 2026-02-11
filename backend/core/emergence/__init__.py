"""
Emergence Engine - 涌现能力引擎
自发产生新能力的AI系统
"""

__version__ = "1.0.0"
__author__ = "AI Platform Team"

from .capability_detector import CapabilityDetector
from .self_organization import SelfOrganization
from .creative_generator import CreativeGenerator
from .emergence_monitor import EmergenceMonitor
from .api import EmergenceAPI

__all__ = [
    'CapabilityDetector',
    'SelfOrganization', 
    'CreativeGenerator',
    'EmergenceMonitor',
    'EmergenceAPI'
]
