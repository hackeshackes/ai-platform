"""
深空探测系统配置 - Configuration Module
========================================

系统配置参数，包括导航、探测、通信等各项参数

作者: AI Platform Team
版本: 1.0.0
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class Planet(Enum):
    """太阳系行星枚举"""
    MERCURY = "mercury"
    VENUS = "venus"
    EARTH = "earth"
    MARS = "mars"
    JUPITER = "jupiter"
    SATURN = "saturn"
    URANUS = "uranus"
    NEPTUNE = "neptune"
    PLUTO = "pluto"

@dataclass
class NavigationConfig:
    """导航系统配置"""
    # 轨道计算参数
    orbital_precision: float = 1e-6  # 轨道计算精度
    trajectory_steps: int = 10000     # 轨迹计算步数
    gravity_model: str = "jgm200"     # 重力模型
    
    # 路径规划参数
    max_route_time: float = 3600.0    # 最大路径规划时间(秒)
    fuel_efficiency_weight: float = 0.3  # 燃料效率权重
    time_efficiency_weight: float = 0.4  # 时间效率权重
    safety_margin: float = 0.1        # 安全边际
    
    # 障碍规避参数
    obstacle_detection_range: float = 1000000.0  # 障碍检测范围(km)
    avoidance_threshold: float = 0.05      # 规避阈值
    emergency_maneuvers: bool = True       # 紧急机动启用

@dataclass
class ExplorationConfig:
    """探测系统配置"""
    # 地形分析参数
    terrain_resolution: float = 0.1       # 地形分辨率(m)
    max_analysis_depth: float = 10000.0   # 最大分析深度(m)
    
    # 资源识别参数
    resource_detection_sensitivity: float = 0.8   # 资源检测灵敏度
    min_resource_confidence: float = 0.75         # 最小资源置信度
    
    # 着陆点选择参数
    landing_site_search_radius: float = 100.0     # 着陆点搜索半径(km)
    max_slope_angle: float = 15.0                  # 最大坡度角(度)
    min_landing_area: float = 10000.0              # 最小着陆面积(m²)
    rock_hazard_threshold: float = 0.3             # 岩石危害阈值
    
    # 样本采集参数
    sample_depth: float = 2.0          # 样本采集深度(m)
    sample_count: int = 10             # 样本数量
    contamination_prevention: bool = True  # 污染预防

@dataclass
class SETIConfig:
    """SETI系统配置"""
    # 信号处理参数
    frequency_range: tuple = (1.0, 10.0)  # 频率范围(GHz)
    bandwidth_resolution: float = 0.001    # 带宽分辨率(Hz)
    signal_processing_rate: int = 1000    # 信号处理速率(信号/秒)
    
    # 异常检测参数
    anomaly_sensitivity: float = 0.95    # 异常检测灵敏度
    false_positive_rate: float = 0.01     # 误报率
    
    # 模式识别参数
    pattern_recognition_model: str = "transformer"  # 模式识别模型
    min_pattern_confidence: float = 0.85            # 最小模式置信度
    
    # 文明评估参数
    civilization_detection_threshold: float = 0.9   # 文明检测阈值
    signal_classification_levels: int = 5           # 信号分类等级数

@dataclass
class CommunicationConfig:
    """通信系统配置"""
    # 深空通信参数
    base_frequency: float = 8.4e9      # 基础频率(Hz, X-band)
    ka_band_frequency: float = 32.0e9  # Ka-band频率(Hz)
    
    # 延迟补偿参数
    light_speed_correction: bool = True     # 光速校正
    adaptive_delay_buffer: float = 0.5      # 自适应延迟缓冲(秒)
    
    # 编码参数
    encoding_scheme: str = "ldpc"           # 编码方案
    code_rate: float = 0.9                  # 码率
    
    # 带宽优化参数
    max_bandwidth: float = 500.0e6         # 最大带宽(Hz)
    adaptive_modulation: bool = True       # 自适应调制
    min_snr_threshold: float = 10.0        # 最小信噪比(dB)
    
    # 错误纠正参数
    error_correction_scheme: str = "reed-solomon"  # 纠错方案
    interleaving_depth: int = 5                    # 交织深度

@dataclass
class SystemConfig:
    """系统总配置"""
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    seti: SETIConfig = field(default_factory=SETIConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    
    # 系统通用参数
    log_level: str = "INFO"
    max_concurrent_operations: int = 10
    data_retention_days: int = 365
    auto_save_interval: int = 300

def get_config() -> SystemConfig:
    """获取系统配置"""
    return SystemConfig()

def load_config(config_path: str) -> SystemConfig:
    """从文件加载配置"""
    import json
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # 解析配置数据并返回SystemConfig对象
    # 简化实现：返回默认配置
    return SystemConfig()

# 行星数据常量
PLANET_DATA: Dict[str, Dict] = {
    "mercury": {
        "name": "水星",
        "mass": 3.301e23,  # kg
        "radius": 2439.7,   # km
        "orbital_period": 88.0,  # days
        "surface_temp": (-173, 427),  # °C
        "gravity": 3.7,  # m/s²
        "atmosphere": False
    },
    "venus": {
        "name": "金星",
        "mass": 4.867e24,
        "radius": 6051.8,
        "orbital_period": 224.7,
        "surface_temp": (462, 462),
        "gravity": 8.87,
        "atmosphere": True,
        "atmosphere_pressure": 92
    },
    "earth": {
        "name": "地球",
        "mass": 5.972e24,
        "radius": 6371.0,
        "orbital_period": 365.25,
        "surface_temp": (-89, 58),
        "gravity": 9.8,
        "atmosphere": True,
        "atmosphere_pressure": 1
    },
    "mars": {
        "name": "火星",
        "mass": 6.417e23,
        "radius": 3389.5,
        "orbital_period": 687.0,
        "surface_temp": (-143, 35),
        "gravity": 3.71,
        "atmosphere": True,
        "atmosphere_pressure": 0.006
    },
    "jupiter": {
        "name": "木星",
        "mass": 1.898e27,
        "radius": 69911,
        "orbital_period": 4333,
        "surface_temp": (-145, -108),
        "gravity": 24.79,
        "atmosphere": True,
        "atmosphere_pressure": 1000
    },
    "saturn": {
        "name": "土星",
        "mass": 5.683e26,
        "radius": 58232,
        "orbital_period": 10759,
        "surface_temp": (-195, -178),
        "gravity": 10.44,
        "atmosphere": True
    },
    "uranus": {
        "name": "天王星",
        "mass": 8.681e25,
        "radius": 25362,
        "orbital_period": 30687,
        "surface_temp": (-224, -197),
        "gravity": 8.87,
        "atmosphere": True
    },
    "neptune": {
        "name": "海王星",
        "mass": 1.024e26,
        "radius": 24622,
        "orbital_period": 60190,
        "surface_temp": (-218, -201),
        "gravity": 11.15,
        "atmosphere": True
    }
}
