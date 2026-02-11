"""
大气模型 - AtmosphereModel
负责模拟大气层物理过程
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AtmosphereState:
    """大气状态"""
    temperature: np.ndarray  # 气温 (K)
    pressure: np.ndarray     # 气压 (Pa)
    humidity: np.ndarray      # 相对湿度 (%)
    wind_u: np.ndarray        # 东风分量 (m/s)
    wind_v: np.ndarray        # 北风分量 (m/s)
    cloud_cover: np.ndarray   # 云覆盖率 (%)
    precipitation: np.ndarray  # 降水量 (mm/day)
    co2_concentration: float  # CO2浓度 (ppm)
    time: datetime
    
    
class AtmosphereModel:
    """
    大气层模拟器
    
    模拟以下过程:
    - 气温、气压、湿度分布
    - 风场和气流
    - 云层形成和降水
    - 温室气体浓度
    """
    
    def __init__(self, resolution: str = "1km", 
                 grid_size: int = 360, 
                 lat_bands: int = 180):
        """
        初始化大气模型
        
        Args:
            resolution: 空间分辨率
            grid_size: 经度方向网格数
            lat_bands: 纬度方向网格数
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self.lat_bands = lat_bands
        
        # 初始化状态数组
        self._initialize_arrays()
        
        # 物理参数
        self.g = 9.81  # 重力加速度 (m/s²)
        self.R = 287.0  # 气体常数 (J/(kg·K))
        self.Lv = 2.501e6  # 蒸发潜热 (J/kg)
        
    def _initialize_arrays(self):
        """初始化状态数组"""
        # 创建网格
        lons = np.linspace(0, 360, self.grid_size, endpoint=False)
        lats = np.linspace(-90, 90, self.lat_bands)
        
        self.lons, self.lats = np.meshgrid(lons, lats)
        
        # 初始状态 - 气温 (K)
        self.temperature = self._get_initial_temperature()
        
        # 初始气压 (Pa) - 海平面气压
        self.pressure = np.full((self.lat_bands, self.grid_size), 101325.0)
        
        # 初始湿度 (%)
        self.humidity = np.random.uniform(40, 80, (self.lat_bands, self.grid_size))
        
        # 初始风场 (m/s)
        self.wind_u = np.zeros((self.lat_bands, self.grid_size))
        self.wind_v = np.zeros((self.lat_bands, self.grid_size))
        
        # 初始云覆盖率 (%)
        self.cloud_cover = np.random.uniform(20, 60, (self.lat_bands, self.grid_size))
        
        # 初始降水量 (mm/day)
        self.precipitation = np.random.uniform(0, 10, (self.lat_bands, self.grid_size))
        
        # CO2浓度 (ppm) - 工业革命前约280ppm
        self.co2_concentration = 280.0
        
        self.time = datetime(1850, 1, 1)
        
    def _get_initial_temperature(self) -> np.ndarray:
        """
        计算初始气温分布
        
        Returns:
            气温数组 (K)
        """
        # 基于纬度的平均气温
        lat_rad = np.radians(self.lats)
        t_mean = 288.15  # 全球平均气温 (K)
        t_amp = 33.0     # 极地与赤道温差幅度
        
        # 赤道最热，两极最冷
        t_lat = t_mean - t_amp * np.sin(lat_rad) ** 2
        
        # 添加随机变化
        t_variation = np.random.normal(0, 2, (self.lat_bands, self.grid_size))
        
        return t_lat + t_variation
        
    def set_co2_concentration(self, concentration: float):
        """
        设置CO2浓度
        
        Args:
            concentration: CO2浓度 (ppm)
        """
        self.co2_concentration = concentration
        
    def get_state(self) -> AtmosphereState:
        """获取当前大气状态"""
        return AtmosphereState(
            temperature=self.temperature.copy(),
            pressure=self.pressure.copy(),
            humidity=self.humidity.copy(),
            wind_u=self.wind_u.copy(),
            wind_v=self.wind_v.copy(),
            cloud_cover=self.cloud_cover.copy(),
            precipitation=self.precipitation.copy(),
            co2_concentration=self.co2_concentration,
            time=self.time
        )
        
    def step(self, dt: float = 86400, scenario: str = "RCP8.5"):
        """
        时间步进
        
        Args:
            dt: 时间步长 (秒)
            scenario: 排放情景
        """
        # 更新CO2浓度
        self._update_co2(scenario, dt)
        
        # 更新气温
        self._update_temperature()
        
        # 更新湿度
        self._update_humidity()
        
        # 更新风场
        self._update_wind()
        
        # 更新云层
        self._update_clouds()
        
        # 更新降水
        self._update_precipitation()
        
        # 更新时间
        self.time = self.time.replace(year=self.time.year + 1)
        
    def _update_co2(self, scenario: str, dt: float):
        """更新CO2浓度"""
        # 不同情景的CO2排放轨迹
        co2_rates = {
            "RCP2.6": -0.5,   # 下降
            "RCP4.5": 0.5,    # 缓慢增加
            "RCP6.0": 1.0,    # 中等增加
            "RCP8.5": 2.0     # 快速增加
        }
        
        rate = co2_rates.get(scenario, 1.0)
        
        # 添加年变化和随机波动
        annual_change = rate + np.random.normal(0, 0.1)
        self.co2_concentration = max(280, self.co2_concentration + annual_change)
        
    def _update_temperature(self):
        """更新气温分布"""
        # 简化的大气能量平衡
        # CO2强迫
        co2_forcing = 5.35 * np.log(self.co2_concentration / 280.0)  # W/m²
        
        # 纬向温度梯度
        lat_rad = np.radians(self.lats)
        lat_effect = -33 * np.sin(lat_rad) ** 2
        
        # 全球平均增温信号
        global_warming = (self.co2_concentration - 280) / 50  # 简化的增温估计
        
        # 更新气温
        self.temperature = 288.15 + lat_effect + global_warming + np.random.normal(0, 0.5)
        
    def _update_humidity(self):
        """更新湿度分布"""
        # 温度影响饱和水汽压
        saturation_vapor = 611.2 * np.exp(17.67 * (self.temperature - 273.15) / 
                                          (self.temperature - 29.65))  # Pa
        
        # 实际水汽压
        actual_vapor = self.humidity / 100 * saturation_vapor
        
        # 添加随机变化
        vapor_change = np.random.normal(0, 50, (self.lat_bands, self.grid_size))
        actual_vapor = np.clip(actual_vapor + vapor_change, 0, saturation_vapor)
        
        # 转换回相对湿度
        self.humidity = (actual_vapor / saturation_vapor) * 100
        self.humidity = np.clip(self.humidity, 0, 100)
        
    def _update_wind(self):
        """更新风场"""
        # 简化的纬向风 - 模拟信风
        lat_rad = np.radians(self.lats)
        
        # 东风带 (-u方向)
        u_trade = -8 * np.cos(2 * lat_rad)  # 信风
        u_mid = 10 * np.sin(lat_rad)        # 西风带
        
        # 组合
        self.wind_u = np.where(np.abs(lat_rad) < np.radians(30), 
                               u_trade, u_mid)
        
        # 添加随机扰动
        self.wind_u += np.random.normal(0, 2, (self.lat_bands, self.grid_size))
        
        # 北风分量 (较弱)
        self.wind_v = np.random.normal(0, 1, (self.lat_bands, self.grid_size))
        
    def _update_clouds(self):
        """更新云覆盖率"""
        # 温度影响云形成
        t_celsius = self.temperature - 273.15
        
        # 温暖地区更容易形成对流云
        cloud_baseline = 40
        t_effect = np.where(t_celsius > 20, 20, 0)
        t_effect = np.where(t_celsius < -20, -10, t_effect)
        
        # 湿度影响
        humidity_effect = (self.humidity - 50) * 0.5
        
        self.cloud_cover = np.clip(
            cloud_baseline + t_effect + humidity_effect + 
            np.random.normal(0, 10),
            0, 100
        )
        
    def _update_precipitation(self):
        """更新降水量"""
        # 对流降水 - 与温度和湿度相关
        t_effect = (self.temperature - 273.15) * 0.3
        
        # 对流有效位能简化估计
        cape = np.where(self.temperature > 290, 
                        (self.temperature - 290) * 10, 0)
        
        self.precipitation = np.clip(
            2 + t_effect + cape * 0.1 + 
            (self.humidity - 60) * 0.2 +
            np.random.normal(0, 1),
            0, 50
        )
        
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'mean_temperature': float(np.mean(self.temperature) - 273.15),  # °C
            'global_co2': self.co2_concentration,
            'mean_cloud_cover': float(np.mean(self.cloud_cover)),
            'mean_precipitation': float(np.mean(self.precipitation)),
            'time': self.time.isoformat()
        }
