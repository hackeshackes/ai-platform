"""
深空探测系统 - Deep Space Exploration System
==============================================

火星/木星AI自动导航，SETI集成系统

核心模块:
- navigation.py - 深空导航
- planet_explorer.py - 行星探测
- seti_integration.py - SETI集成
- communication.py - 深空通信

作者: AI Platform Team
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Platform Team"

from .navigation import DeepSpaceNavigator, TrajectoryPlanner
from .planet_explorer import PlanetExplorer, TerrainAnalyzer
from .seti_integration import SETIAnalyzer, SignalProcessor
from .communication import DeepSpaceCommunicator, SignalEncoder

__all__ = [
    'DeepSpaceNavigator',
    'TrajectoryPlanner',
    'PlanetExplorer',
    'TerrainAnalyzer', 
    'SETIAnalyzer',
    'SignalProcessor',
    'DeepSpaceCommunicator',
    'SignalEncoder'
]
