"""
气候模型 - ClimateModel
地球系统模拟器主类
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

from .atmosphere import AtmosphereModel
from .ocean import OceanModel
from .land import LandModel
from . import config


class ClimateModel:
    """
    地球系统模拟器
    
    综合模拟大气、海洋、陆地相互作用:
    - 能量平衡
    - 碳循环
    - 气候反馈机制
    - 气候变化预测
    """
    
    def __init__(self, resolution: str = "1km",
                 grid_size: int = 360,
                 lat_bands: int = 180):
        """
        初始化气候模型
        
        Args:
            resolution: 空间分辨率
            grid_size: 经度方向网格数
            lat_bands: 纬度方向网格数
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self.lat_bands = lat_bands
        
        # 初始化各子系统
        self.atmosphere = AtmosphereModel(resolution, grid_size, lat_bands)
        self.ocean = OceanModel(resolution, grid_size, lat_bands)
        self.land = LandModel(resolution, grid_size, lat_bands)
        
        # 设置耦合
        self._couple_models()
        
        # 当前情景
        self.scenario = "RCP8.5"
        
        # 模拟状态
        self.is_running = False
        self.start_year = config.DEFAULT_START_YEAR
        self.end_year = config.DEFAULT_END_YEAR
        
        # 历史数据存储
        self.history = {
            'temperature': [],
            'co2': [],
            'sea_level': [],
            'precipitation': [],
            'years': []
        }
        
    def _couple_models(self):
        """耦合各子系统"""
        # 海洋接收大气强迫
        self.ocean.set_forcing(self.atmosphere.get_state())
        
        # 陆地接收大气强迫
        self.land.set_atmosphere_forcing(
            self.atmosphere.temperature.copy(),
            self.atmosphere.precipitation.copy()
        )
        
    def set_scenario(self, scenario: str):
        """
        设置气候情景
        
        Args:
            scenario: RCP2.6, RCP4.5, RCP6.0, RCP8.5
        """
        valid_scenarios = ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5"]
        if scenario not in valid_scenarios:
            raise ValueError(f"无效情景: {scenario}. 支持: {valid_scenarios}")
        self.scenario = scenario
        
    def run(self, start_year: int = 2020, 
            end_year: int = 2100,
            verbose: bool = True) -> Dict:
        """
        运行气候模拟
        
        Args:
            start_year: 起始年份
            end_year: 结束年份
            verbose: 是否输出进度
            
        Returns:
            模拟结果
        """
        self.start_year = start_year
        self.end_year = end_year
        self.is_running = True
        
        # 重置历史数据
        self.history = {
            'temperature': [],
            'co2': [],
            'sea_level': [],
            'precipitation': [],
            'years': []
        }
        
        # 同步时间
        current_time = datetime(start_year, 1, 1)
        self.atmosphere.time = current_time
        self.ocean.time = current_time
        self.land.time = current_time
        
        if verbose:
            print(f"开始气候模拟: {start_year} - {end_year}")
            print(f"情景: {self.scenario}")
            print("-" * 50)
            
        # 主循环
        year = start_year
        while year <= end_year:
            # 更新大气
            self.atmosphere.step(scenario=self.scenario)
            
            # 更新海洋
            self.ocean.set_forcing(self.atmosphere.get_state())
            self.ocean.step(scenario=self.scenario)
            
            # 更新陆地
            self.land.set_atmosphere_forcing(
                self.atmosphere.temperature.copy(),
                self.atmosphere.precipitation.copy()
            )
            self.land.step(scenario=self.scenario)
            
            # 更新碳循环
            self._update_carbon_cycle()
            
            # 记录历史数据
            stats = self._record_history(year)
            
            if verbose and year % 10 == 0:
                print(f"Year {year}: CO2={stats['co2']:.1f}ppm, "
                      f"Temp={stats['temperature']:.2f}°C, "
                      f"SeaLevel={stats['sea_level']:.3f}m")
                
            year += 1
            
        self.is_running = False
        
        return {
            'scenario': self.scenario,
            'start_year': start_year,
            'end_year': end_year,
            'history': self.history.copy(),
            'final_state': self.get_state()
        }
        
    def _update_carbon_cycle(self):
        """更新碳循环"""
        # 简化: 大气CO2增加导致海洋吸收和陆地碳释放
        co2 = self.atmosphere.co2_concentration
        
        # 海洋碳吸收 (约30%的人类排放被海洋吸收)
        ocean_uptake = (co2 - 280) * 0.3 * 0.01
        
        # 陆地碳排放 (约20%的人类排放被陆地释放,包括森林砍伐)
        land_release = (co2 - 280) * 0.2 * 0.01
        
        # 净大气CO2变化
        net_change = 2.0  # 简化: 假设持续排放
        
        self.atmosphere.co2_concentration += net_change
        
    def _record_history(self, year: int) -> Dict:
        """记录历史数据"""
        # 全球平均气温 (K -> °C)
        t_land = np.where(self.land.land_mask, self.land.temperature, np.nan)
        mean_temp = np.nanmean(t_land) - 273.15
        
        # CO2浓度
        co2 = self.atmosphere.co2_concentration
        
        # 海平面变化
        mean_sea_level = np.mean(self.ocean.sea_level)
        
        # 降水量
        mean_precip = np.mean(self.atmosphere.precipitation)
        
        # 记录
        self.history['temperature'].append(mean_temp)
        self.history['co2'].append(co2)
        self.history['sea_level'].append(mean_sea_level)
        self.history['precipitation'].append(mean_precip)
        self.history['years'].append(year)
        
        return {
            'temperature': mean_temp,
            'co2': co2,
            'sea_level': mean_sea_level,
            'precipitation': mean_precip
        }
        
    def get_prediction(self, variable: str = "temperature",
                       region: str = "global",
                       year: int = 2100) -> Dict:
        """
        获取气候预测
        
        Args:
            variable: 变量名 (temperature, precipitation, sea_level, co2)
            region: 区域 (global, northern, southern, tropical)
            year: 年份
            
        Returns:
            预测值
        """
        if not self.history['years']:
            return {'error': '没有模拟数据，请先运行 run()'}
            
        years = np.array(self.history['years'])
        mask = years <= year
        
        if variable == 'temperature':
            data = np.array(self.history['temperature'])[mask]
        elif variable == 'co2':
            data = np.array(self.history['co2'])[mask]
        elif variable == 'sea_level':
            data = np.array(self.history['sea_level'])[mask]
        elif variable == 'precipitation':
            data = np.array(self.history['precipitation'])[mask]
        else:
            return {'error': f'未知变量: {variable}'}
            
        return {
            'variable': variable,
            'region': region,
            'year': year,
            'value': float(data[-1]) if len(data) > 0 else None,
            'trend': float(np.polyfit(years[mask], data, 1)[0]) if len(data) > 1 else None,
            'mean': float(np.mean(data)),
            'std': float(np.std(data))
        }
        
    def get_state(self) -> Dict:
        """获取当前系统状态"""
        return {
            'atmosphere': self.atmosphere.get_statistics(),
            'ocean': self.ocean.get_statistics(),
            'land': self.land.get_statistics(),
            'scenario': self.scenario,
            'time': datetime.now().isoformat()
        }
        
    def get_feedback_analysis(self) -> Dict:
        """获取反馈机制分析"""
        # 简化反馈分析
        base_forcing = 5.35 * np.log(self.atmosphere.co2_concentration / 280)
        
        feedbacks = {
            'water_vapor': config.WATER_VAPOR_FEEDBACK,
            'cloud': config.CLOUD_FEEDBACK,
            'ice_albedo': config.ICE_ALBEDO_FEEDBACK,
            'lapse_rate': 1.0
        }
        
        total_feedback = sum(feedbacks.values())
        climate_sensitivity = base_forcing * (1 + total_feedback)
        
        return {
            'co2_forcing': base_forcing,
            'feedbacks': feedbacks,
            'total_feedback': total_feedback,
            'climate_sensitivity': climate_sensitivity,
            'equivalent_co2': self.atmosphere.co2_concentration * (1 + total_feedback)
        }
        
    def export_results(self, filepath: str):
        """导出模拟结果"""
        results = {
            'scenario': self.scenario,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'resolution': self.resolution,
            'history': {k: list(v) for k, v in self.history.items()},
            'final_state': self.get_state()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        return f"结果已保存到: {filepath}"
