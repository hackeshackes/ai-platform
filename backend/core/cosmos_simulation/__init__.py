"""
Cosmos Simulation Package - 宇宙模拟器

提供宇宙大爆炸、星系形成、恒星演化和宇宙学的完整模拟框架。
"""

from .big_bang import BigBang
from .galaxy_formation import GalaxyFormation
from .stellar_evolution import StellarEvolution
from .cosmology import Cosmology
from .cosmos_simulation import CosmosSimulation
from .api import CosmosAPI
from .config import SimulationConfig

__version__ = "1.0.0"
__author__ = "Cosmos Simulation Team"

__all__ = [
    "BigBang",
    "GalaxyFormation", 
    "StellarEvolution",
    "Cosmology",
    "CosmosSimulation",
    "CosmosAPI",
    "SimulationConfig",
]
