"""
Climate Model - 地球系统模拟器
分辨率: 1km
"""

__version__ = "1.0.0"
__author__ = "AI Platform Team"

from .atmosphere import AtmosphereModel
from .ocean import OceanModel
from .land import LandModel
from .climate_model import ClimateModel

__all__ = [
    'AtmosphereModel',
    'OceanModel', 
    'LandModel',
    'ClimateModel'
]
