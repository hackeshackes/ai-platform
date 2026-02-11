"""
海洋模型 - OceanModel
负责模拟海洋物理和生物过程
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OceanState:
    """海洋状态"""
    sea_temperature: np.ndarray  # 海温 (K)
    salinity: np.ndarray          # 盐度 (PSU)
    u_velocity: np.ndarray        # 东向流速 (m/s)
    v_velocity: np.ndarray        # 北向流速 (m/s)
    sea_level: np.ndarray          # 海平面 (m)
    chlorophyll: np.ndarray       # 叶绿素浓度 (mg/m³)
    time: datetime


class OceanModel:
    """
    海洋层模拟器
    
    模拟以下过程:
    - 海温、盐度分布
    - 海流和潮汐
    - 海洋环流
    - 海洋生态系统
    """
    
    def __init__(self, resolution: str = "1km",
                 grid_size: int = 360,
                 lat_bands: int = 180,
                 depth_levels: int = 50):
        """
        初始化海洋模型
        
        Args:
            resolution: 空间分辨率
            grid_size: 经度方向网格数
            lat_bands: 纬度方向网格数
            depth_levels: 深度层数
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self.lat_bands = lat_bands
        self.depth_levels = depth_levels
        
        # 初始化状态数组
        self._initialize_arrays()
        
        # 物理参数
        self.g = 9.81
        self.rho = 1025  # 海水密度 (kg/m³)
        self.cw = 4000  # 比热容 (J/(kg·K))
        
    def _initialize_arrays(self):
        """初始化状态数组"""
        # 创建网格
        lons = np.linspace(0, 360, self.grid_size, endpoint=False)
        lats = np.linspace(-90, 90, self.lat_bands)
        
        self.lons, self.lats = np.meshgrid(lons, lats)
        
        # 初始海温 (K) - 表面温度
        self.sea_temperature = self._get_initial_temperature()
        
        # 初始盐度 (PSU)
        self.salinity = np.full((self.lat_bands, self.grid_size), 34.7)
        
        # 海流 (m/s)
        self.u_velocity = np.zeros((self.lat_bands, self.grid_size))
        self.v_velocity = np.zeros((self.lat_bands, self.grid_size))
        
        # 海平面 (m)
        self.sea_level = np.zeros((self.lat_bands, self.grid_size))
        
        # 叶绿素 (mg/m³)
        self.chlorophyll = np.random.uniform(0.1, 2.0, (self.lat_bands, self.grid_size))
        
        # 风应力 (初始化为0)
        self.wind_stress_u = np.zeros((self.lat_bands, self.grid_size))
        self.wind_stress_v = np.zeros((self.lat_bands, self.grid_size))
        
        self.time = datetime(1850, 1, 1)
        
    def _get_initial_temperature(self) -> np.ndarray:
        """
        计算初始海温分布
        
        Returns:
            海温数组 (K)
        """
        lat_rad = np.radians(self.lats)
        
        # 赤道热，极地冷
        t_mean = 300  # 赤道表面海温 (K)
        t_amp = 25     # 极地温差幅度
        
        t_lat = t_mean - t_amp * np.sin(lat_rad) ** 2
        
        # 添加随机变化
        return t_lat + np.random.normal(0, 2, (self.lat_bands, self.grid_size))
        
    def get_state(self) -> OceanState:
        """获取当前海洋状态"""
        return OceanState(
            sea_temperature=self.sea_temperature.copy(),
            salinity=self.salinity.copy(),
            u_velocity=self.u_velocity.copy(),
            v_velocity=self.v_velocity.copy(),
            sea_level=self.sea_level.copy(),
            chlorophyll=self.chlorophyll.copy(),
            time=self.time
        )
        
    def set_forcing(self, atmosphere_state):
        """
        设置大气强迫
        
        Args:
            atmosphere_state: 大气状态
        """
        self.atmosphere_temperature = atmosphere_state.temperature.copy()
        self.wind_stress_u = atmosphere_state.wind_u.copy()
        self.wind_stress_v = atmosphere_state.wind_v.copy()
        
    def step(self, dt: float = 86400, scenario: str = "RCP8.5"):
        """
        时间步进
        
        Args:
            dt: 时间步长 (秒)
            scenario: 气候情景
        """
        # 更新海温
        self._update_temperature(scenario)
        
        # 更新盐度
        self._update_salinity()
        
        # 更新海流
        self._update_currents()
        
        # 更新海平面
        self._update_sea_level()
        
        # 更新海洋生物
        self._update_biology()
        
        # 更新时间
        self.time = self.time.replace(year=self.time.year + 1)
        
    def _update_temperature(self, scenario: str):
        """更新海温"""
        # 大气强迫
        if hasattr(self, 'atmosphere_temperature'):
            t_air = self.atmosphere_temperature
        else:
            t_air = 288 + np.random.normal(0, 5)
            
        # 热交换系数
        gamma = 10 / 86400  # W/(m²·K) 转换为 m/s
        
        # 热传导
        heat_flux = gamma * (t_air - self.sea_temperature)
        
        # 温室效应强迫
        co2_rates = {"RCP2.6": 0, "RCP4.5": 0.01, "RCP6.0": 0.015, "RCP8.5": 0.02}
        warming = co2_rates.get(scenario, 0.01) * (datetime.now().year - 1850)
        
        # 更新海温
        self.sea_temperature += heat_flux / (self.rho * self.cw) * 86400 + warming
        
        # 极地冰盖效应
        lat_rad = np.radians(self.lats)
        polar_mask = np.abs(lat_rad) > np.radians(60)
        self.sea_temperature = np.where(polar_mask, 
                                        np.minimum(self.sea_temperature, 273.15),
                                        self.sea_temperature)
        
    def _update_salinity(self):
        """更新盐度"""
        # 简化: 极地盐度变化 (冰冻排斥盐分)
        lat_rad = np.radians(self.lats)
        
        # 极地盐度增加 (冰冻时盐分排出)
        polar_effect = np.where(np.abs(lat_rad) > np.radians(60), 0.5, 0)
        
        # 热带蒸发效应
        evaporation = np.where(np.abs(lat_rad) < np.radians(30), 0.2, 0)
        
        self.salinity += polar_effect - evaporation + np.random.normal(0, 0.01)
        self.salinity = np.clip(self.salinity, 30, 38)
        
    def _update_currents(self):
        """更新海流"""
        # 风生环流简化模型
        lat_rad = np.radians(self.lats)
        
        # 风应力旋度驱动
        wind_curl = np.gradient(self.wind_stress_u, axis=1) + \
                   np.gradient(self.wind_stress_v, axis=0)
        
        #  Sverdrup关系 (简化)
        beta = 2e-11
        stream_function = wind_curl / beta
        
        # 东向流速 (简化)
        self.u_velocity = -np.gradient(stream_function, axis=1) + \
                         np.random.normal(0, 0.1)
                         
        # 北向流速
        self.v_velocity = np.gradient(stream_function, axis=0) + \
                        np.random.normal(0, 0.1)
        
        # 边界限制
        self.u_velocity = np.clip(self.u_velocity, -2, 2)
        self.v_velocity = np.clip(self.v_velocity, -1, 1)
        
    def _update_sea_level(self):
        """更新海平面"""
        # 热膨胀效应
        t_anomaly = self.sea_temperature - 300  # 相对于工业革命前
        thermal_expansion = 0.1 * t_anomaly  # 简化系数
        
        # 热盐环流贡献
        steric_effect = thermal_expansion * 0.5
        
        # 更新海平面
        self.sea_level += steric_effect + np.random.normal(0, 0.001)
        
    def _update_biology(self):
        """更新海洋生态系统"""
        # 光照限制
        light_factor = np.clip(self.sea_temperature / 300, 0, 1)
        
        # 营养盐效应 (简化)
        nutrient_factor = np.clip(2 - self.chlorophyll, 0, 1)
        
        # 生长率
        growth_rate = 0.1 * light_factor * nutrient_factor
        
        #  mortality
        mortality = 0.05
        
        # 更新叶绿素
        self.chlorophyll += growth_rate - mortality
        self.chlorophyll = np.clip(self.chlorophyll, 0.01, 10)
        
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'mean_sea_temperature': float(np.mean(self.sea_temperature) - 273.15),
            'mean_salinity': float(np.mean(self.salinity)),
            'mean_u_current': float(np.mean(self.u_velocity)),
            'mean_v_current': float(np.mean(self.v_velocity)),
            'mean_sea_level': float(np.mean(self.sea_level)),
            'mean_chlorophyll': float(np.mean(self.chlorophyll)),
            'time': self.time.isoformat()
        }
