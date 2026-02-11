"""
Cosmology Module - 宇宙学模块

实现暗能量、暗物质、宇宙微波背景和宇宙几何的模拟。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class DarkEnergyModel(Enum):
    """暗能量模型"""
    LAMBDA_CDM = "lcdm"           # Lambda-CDM
    QUINTESSENCE = "quintessence" # 精质场
    PHANTOM = "phantom"           # 幻影暗能量


class GeometryType(Enum):
    """宇宙几何类型"""
    FLAT = "flat"                 # 平直宇宙
    OPEN = "open"                 # 开宇宙
    CLOSED = "closed"             # 闭宇宙


@dataclass
class CosmologicalParameters:
    """宇宙学参数"""
    # 密度参数
    Omega_m: float = 0.315        # 物质密度参数
    Omega_b: float = 0.049        # 重子物质密度参数
    Omega_r: float = 8.4e-5       # 辐射密度参数
    Omega_Lambda: float = 0.685   # 暗能量密度参数
    Omega_k: float = 0.0          # 曲率密度参数
    
    # 哈勃参数
    H0: float = 67.4              # 哈勃常数 (km/s/Mpc)
    h: float = 0.674              # 简化哈勃参数
    
    # 扰动参数
    sigma_8: float = 0.811        # 功率谱振幅
    n_s: float = 0.965            # 标量谱指数
    
    # 暗能量参数
    w0: float = -1.0              # 暗能量状态方程参数
    wa: float = 0.0
    
    # 宇宙年龄
    age_Gyr: float = 13.8         # 宇宙年龄 (Gyr)
    
    # 曲率
    geometry: GeometryType = GeometryType.FLAT


@dataclass
class PowerSpectrum:
    """功率谱"""
    k: np.ndarray                 # 波数 (1/Mpc)
    P_k: np.ndarray               # 功率谱
    primordial_P_k: np.ndarray    # 原初功率谱
    transfer_function: np.ndarray # 转移函数


@dataclass
class CMBAnisotropy:
    """CMB各向异性"""
    ell: np.ndarray               # 多极矩
    C_ell_tt: np.ndarray          # TT功率谱
    C_ell_ee: np.ndarray          # EE功率谱
    C_ell_te: np.ndarray          # TE功率谱


class Cosmology:
    """
    宇宙学模拟器
    
    实现暗能量、暗物质、宇宙微波背景和宇宙几何的物理计算。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化宇宙学模拟器
        
        Args:
            config: 配置参数
        """
        self.params = CosmologicalParameters()
        
        if config:
            for key, value in config.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
        
        # 物理常数
        self.c = 299792.458  # 光速 (km/s)
        self.G = 4.517e-48  # G in (Mpc)^3/(M_sun s^2)
        self.rho_crit = self._compute_critical_density()
    
    def _compute_critical_density(self) -> float:
        """计算临界密度"""
        H = self.params.H0 / 3.086e19  # H in 1/s
        rho_c = 3 * H**2 / (8 * np.pi * self.G)
        return rho_c
    
    def compute_H(self, z: float) -> float:
        """
        计算哈勃参数 (红移处)
        """
        zp1 = 1 + z
        
        term_m = self.params.Omega_m * zp1**3
        term_r = self.params.Omega_r * zp1**4
        term_L = self.params.Omega_Lambda
        term_k = self.params.Omega_k * zp1**2
        
        H = self.params.H0 * np.sqrt(term_m + term_r + term_L + term_k)
        return H
    
    def compute_luminosity_distance(self, z: float) -> float:
        """
        计算光度距离 (Mpc)
        """
        from scipy import integrate
        
        def integrand(zp):
            return 1 / self.compute_H(zp)
        
        integral, _ = integrate.quad(integrand, 0, z)
        
        d_L = (1 + z) * integral * self.c / self.params.H0
        return d_L
    
    def compute_angular_diameter_distance(self, z: float) -> float:
        """计算角直径距离"""
        d_L = self.compute_luminosity_distance(z)
        d_A = d_L / (1 + z)**2
        return d_A
    
    def compute_lookback_time(self, z: float) -> float:
        """计算回溯时间 (Gyr)"""
        from scipy import integrate
        
        def integrand(zp):
            return 1 / ((1 + zp) * self.compute_H(zp))
        
        integral, _ = integrate.quad(integrand, 0, z)
        
        t = integral * 3.086e19 / (365.25 * 24 * 3600 * 1e9)
        return t
    
    def compute_age(self, z: float) -> float:
        """计算宇宙年龄 (Gyr)"""
        from scipy import integrate
        
        def integrand(zp):
            return 1 / ((1 + zp) * self.compute_H(zp))
        
        integral, _ = integrate.quad(integrand, z, 1000)
        
        t = integral * 3.086e19 / (365.25 * 24 * 3600 * 1e9)
        return t
    
    def compute_distance_modulus(self, z: float) -> float:
        """计算距离模数"""
        d_L = self.compute_luminosity_distance(z)
        mu = 5 * np.log10(d_L / 0.01)
        return mu
    
    def compute_linear_growth_factor(self, z: float) -> float:
        """计算线性增长因子 D(z)"""
        H = self.compute_H(z) / self.params.H0
        Omega_m_z = self.params.Omega_m * (1 + z)**3 / H**2
        
        if z > 1:
            D = 1 / (1 + z)
        else:
            D = 2.5 * Omega_m_z / (1.5 * Omega_m_z**0.5 - 1 + 
                np.sqrt(max(Omega_m_z, 0)) * np.arcsinh(np.sqrt(max((1 - Omega_m_z) / Omega_m_z, 1e-10))))
        
        return D
    
    def compute_power_spectrum(self, 
                                z: float = 0,
                                k_min: float = 1e-4,
                                k_max: float = 10,
                                n_k: int = 100) -> PowerSpectrum:
        """
        计算物质功率谱
        """
        k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
        
        # 原初功率谱
        A_s = 2.1e-9
        P_prim = A_s * (k / 0.05) ** (self.params.n_s - 1)
        
        # 转移函数
        def transfer_function(k_val):
            q = k_val / (self.params.Omega_m * self.params.h**2)
            q = max(q, 1e-10)
            T = np.log(1 + 2.34 * q) / (2.34 * q) * (1 + 3.89 * q + 
                (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4) ** (-0.25)
            return max(T, 1e-10)
        
        T_k = np.array([transfer_function(k_val) for k_val in k])
        D_z = self.compute_linear_growth_factor(z)
        P_k = P_prim * T_k**2 * D_z**2
        
        return PowerSpectrum(
            k=k,
            P_k=P_k,
            primordial_P_k=P_prim,
            transfer_function=T_k,
        )
    
    def compute_cmb_power_spectrum(self) -> CMBAnisotropy:
        """计算CMB功率谱"""
        ell = np.logspace(np.log10(2), np.log10(3000), 100)
        
        def C_ell_tt(l):
            if l < 30:
                return 1000 * (l / 10) ** (-0.5)
            elif l < 300:
                return 5000 * np.exp(-(l - 220)**2 / (2 * 50**2))
            elif l < 1000:
                return 2000 * np.exp(-(l - 550)**2 / (2 * 80**2))
            else:
                return 500 * np.exp(-(l - 800)**2 / (2 * 100**2))
        
        C_tt = np.array([C_ell_tt(l) for l in ell])
        C_ee = C_tt * 0.1
        C_te = C_tt * 0.05
        
        return CMBAnisotropy(
            ell=ell,
            C_ell_tt=C_tt,
            C_ell_ee=C_ee,
            C_ell_te=C_te,
        )
    
    def compute_density_parameter(self, z: float) -> Dict:
        """计算给定红移处的密度参数"""
        H = self.compute_H(z) / self.params.H0
        zp1 = 1 + z
        
        return {
            "z": z,
            "H": self.compute_H(z),
            "Omega_m_z": self.params.Omega_m * zp1**3 / H**2,
            "Omega_r_z": self.params.Omega_r * zp1**4 / H**2,
            "Omega_Lambda_z": self.params.Omega_Lambda / H**2,
            "Omega_k_z": self.params.Omega_k * zp1**2 / H**2,
        }
    
    def compute_clustering_statistics(self, z: float = 0) -> Dict:
        """计算聚类统计量"""
        P = self.compute_power_spectrum(z)
        
        R = 8 / self.params.h
        sigma_R = self._compute_sigma(R, P)
        
        r = np.logspace(-1, 2, 100)
        xi_r = self._compute_correlation_function(r, P)
        
        return {
            "z": z,
            "sigma_8": sigma_R,
            "power_spectrum": {"k": P.k.tolist(), "P_k": P.P_k.tolist()},
            "correlation_function": {"r": r.tolist(), "xi": xi_r.tolist()},
        }
    
    def _compute_sigma(self, R: float, P: PowerSpectrum) -> float:
        """计算质量方差"""
        def W(x):
            if x < 1e-10:
                return 1
            return 3 * (np.sin(x) - x * np.cos(x)) / x**3
        
        integrand = P.k**2 * P.P_k * W(P.k * R)**2
        sigma_sq = np.trapz(integrand, P.k) / (2 * np.pi**2)
        
        return np.sqrt(max(sigma_sq, 1e-10))
    
    def _compute_correlation_function(self, r: np.ndarray, P: PowerSpectrum) -> np.ndarray:
        """计算两点相关函数"""
        xi = np.zeros(len(r))
        
        for i, r_val in enumerate(r):
            integrand = P.k * P.P_k * np.sin(P.k * r_val) / r_val
            xi[i] = np.trapz(integrand, P.k) / (2 * np.pi**2)
        
        return xi
    
    def simulate_structure_formation(self,
                                      z_start: float = 100,
                                      z_end: float = 0,
                                      n_steps: int = 100) -> Dict:
        """模拟结构形成"""
        z_values = np.linspace(z_start, z_end, n_steps)
        
        history = []
        for z in z_values:
            D = self.compute_linear_growth_factor(z)
            sigma = self._compute_sigma(8 / self.params.h, 
                                        self.compute_power_spectrum(z))
            
            history.append({
                "redshift": z,
                "time": self.compute_age(z),
                "growth_factor": D,
                "sigma_8": sigma,
            })
        
        return {
            "history": history,
            "final_z": z_end,
            "initial_z": z_start,
        }
    
    def get_cosmological_parameters(self) -> Dict:
        """获取当前宇宙学参数"""
        return {
            "H0": self.params.H0,
            "Omega_m": self.params.Omega_m,
            "Omega_b": self.params.Omega_b,
            "Omega_Lambda": self.params.Omega_Lambda,
            "Omega_r": self.params.Omega_r,
            "Omega_k": self.params.Omega_k,
            "n_s": self.params.n_s,
            "sigma_8": self.params.sigma_8,
            "age": self.params.age_Gyr,
        }
    
    def set_cosmological_parameters(self, **kwargs):
        """设置宇宙学参数"""
        valid_params = ["H0", "Omega_m", "Omega_b", "Omega_Lambda", 
                        "Omega_r", "Omega_k", "n_s", "sigma_8", "w0", "wa"]
        
        for key, value in kwargs.items():
            if key in valid_params and hasattr(self.params, key):
                setattr(self.params, key, value)
        
        self.rho_crit = self._compute_critical_density()
