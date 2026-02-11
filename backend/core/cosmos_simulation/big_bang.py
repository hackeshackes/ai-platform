"""
Big Bang Module - 大爆炸模拟模块

实现宇宙起源、暴胀期、粒子产生和早期宇宙演化。
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class UniversePhase(Enum):
    """宇宙演化阶段"""
    PLANCK_EPOCH = "planck"           # 普朗克时代 (t < 10^-43 s)
    GRAND_UNIFICATION = "gut"         # 大统一时代 (10^-43 < t < 10^-36 s)
    INFLATION = "inflation"           # 暴胀期 (10^-36 < t < 10^-32 s)
    ELECTROWEAK = "electroweak"       # 电弱时代 (10^-32 < t < 10^-12 s)
    QUARK_HADRON = "quark_hadron"     # 夸克-强子时代 (10^-12 < t < 10^-6 s)
    NUCLEOSYNTHESIS = "big_bang_nucleosynthesis"  # 大爆炸核合成 (3min - 20min)
    RECOMBINATION = "recombination"   # 复合期 (380,000年)
    DARK_AGES = "dark_ages"           # 黑暗时代 (380,000 - 400Myr)
    REIONIZATION = "reionization"     # 再电离时代 (400Myr - 1Gyr)


@dataclass
class PhysicalConstants:
    """物理常数"""
    # 基础常数
    c: float = 299792458.0                    # 光速 (m/s)
    G: float = 6.67430e-11                    # 引力常数 (m^3/kg/s^2)
    h: float = 6.62607015e-34                 # 普朗克常数 (J/s)
    hbar: float = 1.054571817e-34             # 约化普朗克常数 (J/s)
    k_B: float = 1.380649e-23                 # 玻尔兹曼常数 (J/K)
    
    # 宇宙学参数
    T_cmb: float = 2.72548                    # CMB温度 (K)
    rho_critical: float = 9.47e-27            # 临界密度 (kg/m^3)
    H0: float = 67.4                          # 哈勃常数 (km/s/Mpc)
    
    # 粒子物理
    m_proton: float = 1.6726219e-27           # 质子质量 (kg)
    m_neutron: float = 1.674927498e-27        # 中子质量 (kg)
    m_electron: float = 9.1093837e-31         # 电子质量 (kg)
    m_photon: float = 0.0                     # 光子质量 (kg)
    
    # 宇宙学参数
    Omega_m: float = 0.315                    # 物质密度参数
    Omega_r: float = 8.4e-5                   # 辐射密度参数
    Omega_Lambda: float = 0.685               # 暗能量密度参数
    Omega_b: float = 0.049                    # 重子物质密度参数


@dataclass
class InitialConditions:
    """初始条件"""
    redshift: float = 1e12                    # 初始红移
    temperature: float = 1e28                 # 初始温度 (K)
    scale_factor: float = 1e-30               # 初始尺度因子
    density_fluctuations: float = 1e-5        # 密度扰动幅度
    primordial_power_spectrum_index: float = 0.965
    Hubble_parameter: float = None


class BigBang:
    """
    大爆炸模拟器
    
    实现从普朗克时代到复合期的宇宙早期演化。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化大爆炸模拟器
        
        Args:
            config: 配置参数
        """
        self.constants = PhysicalConstants()
        self.phase_history: List[Dict] = []
        
    def set_initial_conditions(self, z: float = 1e12) -> InitialConditions:
        """
        设置初始条件
        
        Args:
            z: 初始红移
            
        Returns:
            InitialConditions: 初始条件对象
        """
        # 计算初始温度
        T = self.constants.T_cmb * (1 + z)
        
        # 计算初始尺度因子
        a = 1 / (1 + z)
        
        # 计算初始哈勃参数 H = H0 * sqrt(Omega_m(1+z)^3 + Omega_r(1+z)^4 + Omega_Lambda)
        z_term = (1 + z)
        H = self.constants.H0 * 1000 / 3.086e19  # 转换为1/s
        H *= np.sqrt(
            self.constants.Omega_m * z_term**3 +
            self.constants.Omega_r * z_term**4 +
            self.constants.Omega_Lambda
        )
        
        self.initial_conditions = InitialConditions(
            redshift=z,
            temperature=T,
            scale_factor=a,
            density_fluctuations=1e-5,
            Hubble_parameter=H
        )
        
        return self.initial_conditions
    
    def get_temperature_at_redshift(self, z: float) -> float:
        """
        计算给定红移处的温度
        
        T(z) = T_cmb * (1 + z)
        
        Args:
            z: 红移
            
        Returns:
            float: 温度 (K)
        """
        return self.constants.T_cmb * (1 + z)
    
    def get_scale_factor(self, t: float) -> float:
        """
        计算给定时间的尺度因子
        
        使用辐射主导时期的近似: a(t) ~ t^0.5
        
        Args:
            t: 时间 (秒)
            
        Returns:
            float: 尺度因子
        """
        # 辐射主导时期: a ∝ t^0.5
        # 物质主导时期: a ∝ t^(2/3)
        # 暗能量主导时期: a ∝ exp(Ht)
        
        if t < 47000 * 3.154e7:  # 物质-辐射相等时间 ~47000年
            return (t / 3.154e7 / 1e10) ** 0.5  # 简化计算
        elif t < 9.8e9 * 3.154e7:  # 暗能量-物质相等时间 ~9.8Gyr
            return (t / 3.154e7 / 1e10) ** (2/3)
        else:
            return np.exp(t / (3.154e7 * 13.8e9))  # 简化暗能量模型
    
    def get_redshift_from_time(self, t: float) -> float:
        """
        从时间计算红移
        
        Args:
            t: 时间 (秒)
            
        Returns:
            float: 红移
        """
        a = self.get_scale_factor(t)
        return 1/a - 1
    
    def get_time_from_redshift(self, z: float) -> float:
        """
        从红移计算时间
        
        Args:
            z: 红移
            
        Returns:
            float: 时间 (秒)
        """
        a = 1 / (1 + z)
        
        # 使用简化的宇宙年龄计算
        if z > 3400:  # 辐射主导
            t = 2.0 / (3.0 * self.constants.H0 * 1e-19) * (1 + z)**(-3/2)
        else:  # 物质主导
            t = 2.0 / (3.0 * self.constants.H0 * 1e-19) * (1 + z)**(-3/2)
        
        return t
    
    def simulate_inflation(self, 
                           e_folds: float = 60,
                           inflation_field: float = 1e16) -> Dict:
        """
        模拟暴胀期
        
        Args:
            e_folds: 暴胀持续时间 (e-foldings)
            inflation_field: 暴胀场初始值 (GeV)
            
        Returns:
            Dict: 暴胀期模拟结果
        """
        # 暴胀结束时的能量尺度
        end_energy_scale = inflation_field / np.exp(e_folds)
        
        # 暴胀产生的原初密度扰动
        primordial_perturbations = {
            "amplitude": 2e-5,
            "spectral_index": 0.965,
            "tensor_to_scalar_ratio": 0.01,
            "scalar_spectral_index": 0.965,
        }
        
        # 暴胀后的温度扰动
        temperature_fluctuations = primordial_perturbations["amplitude"]
        
        result = {
            "phase": "inflation",
            "duration_e_folds": e_folds,
            "inflation_field_initial": inflation_field,
            "inflation_field_final": end_energy_scale,
            "primordial_perturbations": primordial_perturbations,
            "temperature_fluctuations": temperature_fluctuations,
            "reheating_temperature": 1e15,  # 重加热温度 (K)
        }
        
        self.phase_history.append(result)
        return result
    
    def simulate_nucleosynthesis(self, 
                                  z_start: float = 1e10,
                                  z_end: float = 1e8) -> Dict:
        """
        模拟大爆炸核合成 (BBN)
        
        Args:
            z_start: 起始红移
            z_end: 结束红移
            
        Returns:
            Dict: 核合成结果
        """
        # BBN时间: 3-20分钟
        # 温度: ~1e9 K
        
        # 轻元素丰度 (质量分数)
        abundances = {
            "H": 0.75,           # 氢-1
            "D": 2.5e-5,         # 氘
            "He3": 1.0e-5,       # 氦-3
            "He4": 0.25,         # 氦-4
            "Li7": 1.0e-10,      # 锂-7
            "Li6": 1.0e-14,      # 锂-6
            "B": 1.0e-10,        # 硼
        }
        
        result = {
            "phase": "big_bang_nucleosynthesis",
            "time_range": {
                "start": 180,      # 3分钟 (秒)
                "end": 1200,       # 20分钟 (秒)
            },
            "temperature_range": {
                "start": 1.0e9,    # 10亿 K
                "end": 3.0e8,      # 3亿 K
            },
            "abundances": abundances,
            "neutron_to_proton_ratio": 1/6,  # BBN结束时的n/p比
        }
        
        self.phase_history.append(result)
        return result
    
    def simulate_recombination(self, z: float = 1100) -> Dict:
        """
        模拟复合期
        
        Args:
            z: 复合期红移
            
        Returns:
            Dict: 复合期模拟结果
        """
        T = self.constants.T_cmb * (1 + z)
        
        result = {
            "phase": "recombination",
            "redshift": z,
            "temperature": T,
            "time": 3.8e5 * 3.154e7,  # 38万年 (秒)
            "scale_factor": 1 / (1 + z),
            "visibility": 1.0,       # 光子最后散射
            "compton_optical_depth": 0.001,  # 复合时的光学深度
            "baryon_to_photon_ratio": 6e-10,  # 重子-光子比
        }
        
        self.phase_history.append(result)
        return result
    
    def get_particle_content(self, z: float) -> Dict:
        """
        获取给定红移处的粒子内容
        
        Args:
            z: 红移
            
        Returns:
            Dict: 粒子内容
        """
        T = self.constants.T_cmb * (1 + z)
        
        particles = {
            "photons": True,
            "neutrinos": True,
            "electrons": True,
            "protons": True,
            "neutrons": True,
        }
        
        # 根据温度添加粒子
        if T > 1e12:  # 电弱对称性破缺温度
            particles["W_bosons"] = True
            particles["Z_bosons"] = True
            particles["Higgs"] = True
            
        if T > 1e15:  # 大统一温度
            particles["X_bosons"] = True
            particles["Y_bosons"] = True
            
        if T > 1e19:  # 普朗克温度
            particles["gravitons"] = True
        
        return {
            "temperature": T,
            "energy_density": self.calculate_energy_density(T),
            "particles": particles,
        }
    
    def calculate_energy_density(self, T: float) -> float:
        """
        计算给定温度的能量密度
        
        ρ = (π²/30) * g_* * T^4
        
        Args:
            T: 温度 (K)
            
        Returns:
            float: 能量密度 (J/m³)
        """
        # 有效自由度 (光子 + 3种中微子 + e+e- + ...)
        g_star = 3.36  # CMB时期的有效自由度
        
        # Stefan-Boltzmann常数
        sigma_sb = 5.670374419e-8
        
        return (np.pi**2 / 30) * g_star * sigma_sb * T**4
    
    def get_age_at_redshift(self, z: float) -> float:
        """
        计算给定红移处的宇宙年龄
        
        Args:
            z: 红移
            
        Returns:
            float: 年龄 (秒)
        """
        # 简化的宇宙年龄计算
        # t(z) = ∫₀^dz' / [H(z')(1+z')]
        
        H0 = self.constants.H0 * 1000 / 3.086e19  # H0 in 1/s
        Om = self.constants.Omega_m
        Ol = self.constants.Omega_Lambda
        
        # 简化的积分近似
        if z > 1:
            t = 2.0 / (3.0 * H0 * np.sqrt(Om)) * (1 + z)**(-3/2)
        else:
            t = 2.0 / (3.0 * H0 * np.sqrt(Ol)) * np.arcsinh(
                np.sqrt(Ol/Om) * (1 + z)**(-3/2)
            )
        
        return t
    
    def run_simulation(self, 
                       z_start: float = 1e12,
                       z_end: float = 0,
                       time_step: str = "auto") -> Dict:
        """
        运行完整的大爆炸模拟
        
        Args:
            z_start: 起始红移
            z_end: 结束红移
            time_step: 时间步长
            
        Returns:
            Dict: 完整模拟结果
        """
        self.set_initial_conditions(z_start)
        
        results = {
            "initial_conditions": self.initial_conditions.__dict__,
            "phases": [],
            "timeline": [],
        }
        
        # 模拟各个阶段
        if z_start > 1e50:
            results["phases"].append(self.simulate_inflation())
        
        if z_start > 1e9:
            results["phases"].append(self.simulate_nucleosynthesis())
        
        if z_start > 1100:
            results["phases"].append(self.simulate_recombination(1100))
        
        # 生成时间线
        z_values = np.logspace(np.log10(z_end), np.log10(z_start), 100)
        for z in z_values:
            t = self.get_age_at_redshift(z)
            T = self.get_temperature_at_redshift(z)
            
            results["timeline"].append({
                "redshift": z,
                "time": t,
                "time_Gyr": t / 3.154e16,
                "temperature": T,
            })
        
        return results
