"""
陆地模型 - LandModel
负责模拟陆地表面过程
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LandState:
    """陆地状态"""
    temperature: np.ndarray       # 地表温度 (K)
    soil_moisture: np.ndarray     # 土壤湿度 (m³/m³)
    snow_depth: np.ndarray        # 雪深 (m)
    vegetation: np.ndarray        # 植被指数 (0-1)
    glacier_mass: float           # 冰川质量 (Gt)
    permafrost_carbon: float      # 冻土碳储量 (GtC)
    river_discharge: np.ndarray   # 河流流量 (m³/s)
    lake_level: np.ndarray        # 湖泊水位 (m)
    land_use: np.ndarray          # 土地利用类型
    time: datetime


class LandModel:
    """
    陆地表面模拟器
    
    模拟以下过程:
    - 植被和土壤
    - 冰川和冻土
    - 河流和湖泊
    - 土地利用变化
    """
    
    def __init__(self, resolution: str = "1km",
                 grid_size: int = 360,
                 lat_bands: int = 180):
        """
        初始化陆地模型
        
        Args:
            resolution: 空间分辨率
            grid_size: 经度方向网格数
            lat_bands: 纬度方向网格数
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self.lat_bands = lat_bands
        
        # 初始化状态
        self._initialize_arrays()
        
        # 物理参数
        self.cw = 2000
        self.rho_s = 1500
        self.Lf = 3.34e5
        
    def _create_land_mask(self) -> np.ndarray:
        """创建陆地掩码"""
        lats_1d = np.linspace(-90, 90, self.lat_bands)
        land_mask = np.zeros((self.lat_bands, self.grid_size), dtype=bool)
        
        # 创建纬度掩码 (1D)
        lat_condition = {
            'north_america': (lats_1d > 20) & (lats_1d < 70),
            'europe': (lats_1d > 35) & (lats_1d < 70),
            'asia': (lats_1d > 10) & (lats_1d < 75),
            'africa': (lats_1d > -35) & (lats_1d < 35),
            'south_america': (lats_1d > -55) & (lats_1d < 10),
            'australia': (lats_1d > -45) & (lats_1d < -10),
            'arctic': lats_1d > 70,
            'antarctic': lats_1d < -70,
            'greenland': (lats_1d > 60) & (lats_1d < 83)
        }
        
        mid_lon = self.grid_size // 7
        
        # 应用掩码
        land_mask[lat_condition['north_america'], :mid_lon] = True
        land_mask[lat_condition['europe'], mid_lon:mid_lon*2] = True
        land_mask[lat_condition['asia'], mid_lon*2:mid_lon*4] = True
        land_mask[lat_condition['africa'], mid_lon*4:mid_lon*5] = True
        land_mask[lat_condition['south_america'], mid_lon*6:mid_lon*7] = True
        land_mask[lat_condition['australia'], mid_lon*7:] = True
        land_mask[lat_condition['antarctic'], :] = True
        land_mask[lat_condition['greenland'], :mid_lon//2] = True
        
        return land_mask
        
    def _initialize_arrays(self):
        """初始化状态数组"""
        lons = np.linspace(0, 360, self.grid_size, endpoint=False)
        lats = np.linspace(-90, 90, self.lat_bands)
        
        self.lons, self.lats = np.meshgrid(lons, lats)
        
        # 创建陆地掩码
        self.land_mask = self._create_land_mask()
        
        # 初始地表温度 (K)
        self.temperature = self._get_initial_temperature()
        
        self.soil_moisture = np.where(
            self.land_mask,
            np.random.uniform(0.2, 0.4, (self.lat_bands, self.grid_size)),
            0
        )
        
        self.snow_depth = np.where(
            self.land_mask & (np.abs(self.lats) > 50),
            np.random.uniform(0, 2, (self.lat_bands, self.grid_size)),
            0
        )
        
        self.vegetation = np.where(
            self.land_mask,
            np.random.uniform(0.3, 0.8, (self.lat_bands, self.grid_size)),
            0
        )
        
        self.glacier_mass = 1.5e5
        self.permafrost_carbon = 1400
        
        self.river_discharge = np.where(
            self.land_mask,
            np.random.uniform(100, 1000, (self.lat_bands, self.grid_size)),
            0
        )
        
        self.lake_level = np.zeros((self.lat_bands, self.grid_size))
        self.lake_level[:, 60:70] = 100
        
        self.land_use = self._get_initial_land_use()
        
        self.time = datetime(1850, 1, 1)
        
    def _get_initial_temperature(self) -> np.ndarray:
        """计算初始地表温度"""
        lat_rad = np.radians(self.lats)
        t_mean = 288
        t_amp = 40
        t_lat = t_mean - t_amp * np.sin(lat_rad) ** 2
        land_contrast = np.where(self.land_mask, 5, 0)
        return t_lat + land_contrast + np.random.normal(0, 3)
        
    def _get_initial_land_use(self) -> np.ndarray:
        """初始土地利用"""
        land_use = np.zeros((self.lat_bands, self.grid_size))
        land_use[:] = 0
        
        lat_rad = np.radians(self.lats)
        forest_mask = self.land_mask & (
            (np.abs(lat_rad) < np.radians(20)) | (np.abs(lat_rad) > np.radians(50))
        )
        land_use = np.where(forest_mask, 1, land_use)
        
        grass_mask = self.land_mask & (
            (np.abs(lat_rad) > np.radians(20)) & (np.abs(lat_rad) < np.radians(50))
        )
        land_use = np.where(grass_mask, 2, land_use)
        
        crop_mask = self.land_mask & (
            (np.abs(lat_rad) > np.radians(30)) & (np.abs(lat_rad) < np.radians(55))
        )
        land_use = np.where(crop_mask, 3, land_use)
        
        ice_mask = self.land_mask & (
            (np.abs(lat_rad) > np.radians(70)) | (self.lats > 45)
        )
        land_use = np.where(ice_mask, 5, land_use)
        
        return land_use
        
    def get_state(self) -> LandState:
        """获取当前陆地状态"""
        return LandState(
            temperature=self.temperature.copy(),
            soil_moisture=self.soil_moisture.copy(),
            snow_depth=self.snow_depth.copy(),
            vegetation=self.vegetation.copy(),
            glacier_mass=self.glacier_mass,
            permafrost_carbon=self.permafrost_carbon,
            river_discharge=self.river_discharge.copy(),
            lake_level=self.lake_level.copy(),
            land_use=self.land_use.copy(),
            time=self.time
        )
        
    def set_atmosphere_forcing(self, temperature: np.ndarray,
                                 precipitation: np.ndarray):
        """设置大气强迫"""
        self.atm_temperature = temperature
        self.precipitation = precipitation
        
    def step(self, dt: float = 86400, scenario: str = "RCP8.5"):
        """时间步进"""
        self._update_temperature(scenario)
        self._update_soil_moisture()
        self._update_snow()
        self._update_vegetation()
        self._update_glaciers(scenario)
        self._update_permafrost(scenario)
        self._update_rivers()
        self._update_land_use()
        self.time = self.time.replace(year=self.time.year + 1)
        
    def _update_temperature(self, scenario: str):
        """更新地表温度"""
        if hasattr(self, 'atm_temperature'):
            t_air = self.atm_temperature
        else:
            t_air = 288 + np.random.normal(0, 5)
            
        urban_heat = np.where(self.land_use == 4, 2, 0)
        albedo_cooling = np.where(self.land_use == 5, 3, 0)
        
        warming_rates = {"RCP2.6": 0.01, "RCP4.5": 0.02, "RCP6.0": 0.025, "RCP8.5": 0.04}
        trend = warming_rates.get(scenario, 0.02)
        years_since_1850 = self.time.year - 1850
        climate_change = trend * years_since_1850
        
        self.temperature = np.where(
            self.land_mask,
            t_air + urban_heat - albedo_cooling + climate_change + np.random.normal(0, 1),
            0
        )
        
    def _update_soil_moisture(self):
        """更新土壤湿度"""
        if hasattr(self, 'precipitation'):
            p = self.precipitation
        else:
            p = np.random.uniform(2, 5)
            
        evapotranspiration = np.where(
            self.land_mask,
            self.vegetation * 0.1 * (self.temperature - 273.15),
            0
        )
        evapotranspiration = np.clip(evapotranspiration, 0, 10)
        rainfall = p * 0.01
        
        self.soil_moisture += (rainfall - evapotranspiration) / 10
        self.soil_moisture = np.clip(self.soil_moisture, 0, 0.6)
        
    def _update_snow(self):
        """更新雪深"""
        if hasattr(self, 'precipitation'):
            precip = self.precipitation
        else:
            precip = 3
            
        snowfall = np.where(self.temperature < 273.15, precip * 0.01, 0)
        melt = np.where(self.temperature > 273.15, (self.temperature - 273.15) * 0.01, 0)
        
        self.snow_depth = np.where(
            self.land_mask,
            np.clip(self.snow_depth + snowfall - melt, 0, 10),
            0
        )
        
    def _update_vegetation(self):
        """更新植被"""
        # 温度和湿度限制
        t_opt = np.where((self.temperature > 280) & (self.temperature < 305), 1, 0)
        t_opt = np.where(self.temperature < 270, 0, t_opt)
        t_opt = np.where(self.temperature > 315, 0, t_opt)
        
        water_stress = np.clip(1 - self.soil_moisture / 0.3, 0, 1)
        
        growth_rate = 0.05 * t_opt * (1 - water_stress)
        mortality = 0.02
        
        self.vegetation = np.where(
            self.land_mask,
            np.clip(self.vegetation + growth_rate - mortality + np.random.normal(0, 0.01), 0, 1),
            0
        )
        
    def _update_glaciers(self, scenario: str):
        """更新冰川质量"""
        warming_rates = {"RCP2.6": -0.001, "RCP4.5": 0.001, "RCP6.0": 0.003, "RCP8.5": 0.005}
        rate = warming_rates.get(scenario, 0.002)
        
        # 质量平衡 (Gt/年)
        mass_balance = -1000 * rate * self.glacier_mass
        
        self.glacier_mass = max(0, self.glacier_mass + mass_balance)
        
    def _update_permafrost(self, scenario: str):
        """更新冻土碳储量"""
        warming = (self.time.year - 1850) * 0.02
        
        # 融化率
        melt_rate = 0.001 * warming
        
        self.permafrost_carbon -= melt_rate * self.permafrost_carbon
        self.permafrost_carbon = max(0, self.permafrost_carbon)
        
    def _update_rivers(self):
        """更新河流流量"""
        # 降水输入
        if hasattr(self, 'precipitation'):
            p = self.precipitation
        else:
            p = np.random.uniform(2, 5)
            
        # 融雪
        snowmelt = np.where(
            (self.temperature > 273.15) & (self.snow_depth > 0),
            (self.temperature - 273.15) * 10,
            0
        )
        
        self.river_discharge = np.where(
            self.land_mask,
            p * 50 + snowmelt + np.random.normal(0, 50),
            0
        )
        
    def _update_land_use(self):
        """更新土地利用"""
        # 城市化趋势
        urban_expansion = 0.001
        
        # 森林砍伐
        deforestation = 0.002
        
        self.land_use = np.where(
            (self.land_use == 1) & (np.random.random((self.lat_bands, self.grid_size)) < deforestation),
            3,  # 转为农田
            self.land_use
        )
        
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        land_temp = np.where(self.land_mask, self.temperature, np.nan)
        return {
            'mean_land_temperature': float(np.nanmean(land_temp) - 273.15),
            'mean_soil_moisture': float(np.nanmean(self.soil_moisture)),
            'mean_snow_depth': float(np.mean(self.snow_depth)),
            'mean_vegetation': float(np.nanmean(self.vegetation)),
            'glacier_mass': self.glacier_mass,
            'permafrost_carbon': self.permafrost_carbon,
            'time': self.time.isoformat()
        }
