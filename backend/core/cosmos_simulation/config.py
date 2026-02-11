"""
Configuration Module - 配置模块

提供宇宙模拟器的配置参数。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import json


@dataclass
class SimulationConfig:
    """模拟配置"""
    
    # 模拟基本参数
    simulation_name: str = "cosmos_v1"
    output_directory: str = "./output"
    
    # 宇宙学参数
    H0: float = 67.4  # km/s/Mpc
    Omega_m: float = 0.315
    Omega_b: float = 0.049
    Omega_Lambda: float = 0.685
    Omega_r: float = 8.4e-5
    n_s: float = 0.965
    sigma_8: float = 0.811
    
    # 模拟参数
    z_start: float = 1000.0  # 起始红移
    z_end: float = 0.0       # 结束红移
    time_step: str = "auto"  # 时间步长
    spatial_resolution: float = 1.0  # Mpc
    mass_resolution: float = 1e10    # M_sun
    
    # 输出配置
    output_format: str = "hdf5"
    save_frequency: int = 10
    verbose: bool = True
    
    # 物理模型选择
    dark_energy_model: str = "lcdm"
    dark_matter_model: str = "cdm"
    stellar_evolution_model: str = "pols"
    star_formation_law: str = "kennicutt"
    
    # 大爆炸参数
    primordial_spectral_index: float = 0.965
    tensor_to_scalar_ratio: float = 0.01
    reheating_temperature: float = 1e15  # K
    
    # 星系形成参数
    star_formation_efficiency: float = 0.01
    feedback_strength: float = 1.0
    merger_time_scale: float = 0.5  # Gyr
    cooling_model: str = "radiative"
    
    # 恒星演化参数
    imf_slope: float = -2.35
    binary_fraction: float = 0.1
    supernova_energy: float = 1e51  # erg
    neutron_star_kick: float = 500  # km/s
    
    # 数值参数
    n_bodies: int = 1000000
    force_accuracy: float = 1e-5
    softening_length: float = 0.01  # Mpc
    
    # 测试参数
    test_mode: bool = False
    n_test_steps: int = 10
    
    @classmethod
    def from_file(cls, file_path: str) -> "SimulationConfig":
        """从JSON文件加载配置"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()
    
    def to_file(self, file_path: str):
        """保存配置到JSON文件"""
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def high_resolution(cls) -> "SimulationConfig":
        """高分辨率配置"""
        return cls(
            spatial_resolution=0.1,
            mass_resolution=1e8,
            n_bodies=10000000,
            force_accuracy=1e-6,
        )
    
    @classmethod
    def test_config(cls) -> "SimulationConfig":
        """测试用配置"""
        return cls(
            simulation_name="cosmos_test",
            z_start=100,
            z_end=0,
            n_test_steps=10,
            test_mode=True,
            n_bodies=1000,
            verbose=False,
        )
    
    def get_cosmology_params(self) -> Dict:
        """获取宇宙学参数字典"""
        return {
            "H0": self.H0,
            "Omega_m": self.Omega_m,
            "Omega_b": self.Omega_b,
            "Omega_Lambda": self.Omega_Lambda,
            "Omega_r": self.Omega_r,
            "n_s": self.n_s,
            "sigma_8": self.sigma_8,
        }
    
    def get_time_step_seconds(self) -> float:
        """获取时间步长(秒)"""
        if self.time_step == "auto":
            # 自动计算: 1% Hubble time
            H0_inv = 9.778e9  # Hubble time in Gyr
            return 0.01 * H0_inv * 3.154e16
        else:
            # 解析时间字符串
            value = float(self.time_step[:-4])  # 去掉单位
            unit = self.time_step[-4:]
            
            multipliers = {
                "yr": 3.154e7,
                "Myr": 3.154e13,
                "Gyr": 3.154e16,
            }
            return value * multipliers.get(unit, 1.0)


# 默认配置实例
DEFAULT_CONFIG = SimulationConfig()

# 常用配置预设
PRESETS = {
    "planck": SimulationConfig(
        H0=67.4,
        Omega_m=0.315,
        Omega_b=0.049,
        Omega_Lambda=0.685,
        n_s=0.965,
        sigma_8=0.811,
    ),
    "wmap": SimulationConfig(
        H0=70.0,
        Omega_m=0.27,
        Omega_b=0.045,
        Omega_Lambda=0.73,
        n_s=0.97,
        sigma_8=0.79,
    ),
    "custom_low_z": SimulationConfig(
        z_start=100,
        z_end=0,
        spatial_resolution=0.5,
        n_bodies=100000,
    ),
}
