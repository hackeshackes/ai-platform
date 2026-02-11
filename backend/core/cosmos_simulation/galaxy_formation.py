"""
Galaxy Formation Module - 星系形成模块

实现暗物质晕、气体冷却、恒星形成和星系合并的模拟。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random


class HaloEnvironment(Enum):
    """暗物质晕环境"""
    FIELD = "field"           # 场环境
    GROUP = "group"           # 星系群
    CLUSTER = "cluster"       # 星系团
    FILAMENT = "filament"     # 宇宙网络纤维


class GalaxyType(Enum):
    """星系类型"""
    ELLIPTICAL = "elliptical"
    SPIRAL = "spiral"
    IRREGULAR = "irregular"
    DWARF = "dwarf"
    STARBURST = "starburst"
    ULTRALUMINOUS_INFRARED = "ulirg"


@dataclass
class DarkMatterHalo:
    """暗物质晕结构"""
    mass: float = 1e12              # 晕质量 (M_sun)
    radius: float = 200.0           # 晕半径 (kpc)
    concentration: float = 10.0     # 浓度参数
    spin_parameter: float = 0.04    # 自旋参数
    velocity_dispersion: float = 150.0  # 速度弥散 (km/s)
    environment: HaloEnvironment = HaloEnvironment.FIELD
    redshift: float = 0.0           # 形成时的红移
    formation_time: float = 5.0     # 形成时间 (Gyr)
    merge_history: List = field(default_factory=list)  # 合并历史


@dataclass
class GasComponent:
    """气体组件"""
    mass: float = 1e11              # 气体质量 (M_sun)
    temperature: float = 1e6        # 温度 (K)
    metallicity: float = 0.02       # 金属丰度 (Z_sun)
    cooling_rate: float = 0.1       # 冷却率
    angular_momentum: float = 1e10  # 角动量
    density_profile: str = "isothermal"  # 密度分布
    cooling_function: Dict = field(default_factory=dict)  # 冷却函数


@dataclass
class StarFormation:
    """恒星形成参数"""
    rate: float = 1.0               # 恒星形成率 (M_sun/yr)
    efficiency: float = 0.01        # 形成效率
    timescale: float = 100.0        # 时间尺度 (Myr)
    threshold_density: float = 100  # 阈值密度 (cm^-3)
    initial_mass_function: str = "kroupa"  # IMF
    stellar_population_age: float = 0  # 星族年龄
    total_stellar_mass: float = 0    # 总恒星质量


class GalaxyFormation:
    """
    星系形成模拟器
    
    实现从暗物质晕形成到完整星系演化的完整过程。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化星系形成模拟器
        
        Args:
            config: 配置参数
        """
        self.halos: List[DarkMatterHalo] = []
        self.galaxies: Dict[str, Dict] = {}
        self.config = config or {}
        
        # 宇宙学参数
        self.Omega_m = 0.315
        self.Omega_b = 0.049
        self.H0 = 67.4
        self.sigma_8 = 0.811
        
    def create_collapsed_halo(self, 
                               mass: float,
                               z: float = 0,
                               environment: str = "field") -> DarkMatterHalo:
        """
        创建暗物质晕
        
        Args:
            mass: 晕质量 (M_sun)
            z: 红移
            environment: 环境类型
            
        Returns:
            DarkMatterHalo: 暗物质晕对象
        """
        # 根据质量计算浓度参数
        concentration = 10 * (mass / 1e12) ** (-0.1) * (1 + z) ** (-0.5)
        concentration = max(5, min(20, concentration))
        
        # 计算晕半径 (virial radius)
        rho_crit = 9.47e-27 * self.H0**2 / (67.4**2)  # 临界密度
        delta_vir = 200  # virial overdensity
        
        r_vir = (3 * mass / (4 * np.pi * delta_vir * rho_crit)) ** (1/3)
        r_vir_kpc = r_vir / 3.086e19 / 3.086e3  # 转换为kpc
        
        # 速度弥散
        v_disp = 150 * (mass / 1e12) ** (1/3) * (1 + z) ** (1/2)
        
        halo = DarkMatterHalo(
            mass=mass,
            radius=r_vir_kpc,
            concentration=concentration,
            spin_parameter=0.03 + 0.02 * random.random(),
            velocity_dispersion=v_disp,
            environment=HaloEnvironment(environment),
            redshift=z,
            formation_time=10 - 3 * np.log10(1 + z),  # 简化的形成时间
        )
        
        self.halos.append(halo)
        return halo
    
    def simulate_gas_cooling(self, 
                              halo: DarkMatterHalo,
                              cooling_mode: str = "radiative") -> GasComponent:
        """
        模拟气体冷却
        
        Args:
            halo: 暗物质晕
            cooling_mode: 冷却模式
            
        Returns:
            GasComponent: 气体组件
        """
        # 初始气体温度 (virial temperature)
        T_vir = 35.9 * (halo.mass / 1e11) ** (2/3) * (1 + halo.redshift)  # K
        
        # 冷却率
        if cooling_mode == "radiative":
            cooling_time = 1e10 * (halo.mass / 1e12) ** 0.5  # 年
        else:
            cooling_time = 1e11
        
        # 气体质量 (假设baryon fraction)
        gas_mass = self.Omega_b / self.Omega_m * halo.mass
        
        gas = GasComponent(
            mass=gas_mass,
            temperature=T_vir,
            metallicity=0.001 * (1 + halo.redshift) ** 0.5,
            cooling_rate=1 / cooling_time,
            cooling_function=self._get_cooling_function(T_vir),
        )
        
        return gas
    
    def _get_cooling_function(self, T: float) -> Dict:
        """
        获取冷却函数
        
        Args:
            T: 温度 (K)
            
        Returns:
            Dict: 冷却函数参数
        """
        # 简化的冷却函数
        if T < 1e4:
            return {"mode": "recombination", "cooling_time": 1e9}
        elif T < 1e5:
            return {"mode": "collision", "cooling_time": 1e8}
        elif T < 1e7:
            return {"mode": "bremsstrahlung", "cooling_time": 1e7}
        else:
            return {"mode": " Compton", "cooling_time": 1e6}
    
    def simulate_galaxy_evolution(self, 
                                   galaxy_id: str,
                                   initial_mass: float,
                                   z_start: float = 10,
                                   z_end: float = 0,
                                   time_step: str = "1Gyr") -> Dict:
        """
        模拟单个星系的演化
        
        Args:
            galaxy_id: 星系ID
            initial_mass: 初始质量
            z_start: 起始红移
            z_end: 结束红移
            time_step: 时间步长
            
        Returns:
            Dict: 演化结果
        """
        # 创建初始暗物质晕
        halo = self.create_collapsed_halo(initial_mass, z_start)
        
        # 创建气体组件
        gas = self.simulate_gas_cooling(halo)
        
        # 初始化恒星形成
        sf = StarFormation(rate=1.0)
        
        # 演化时间线
        evolution = []
        z_values = np.linspace(z_start, z_end, 100)
        
        for z in z_values:
            t = self._get_time_from_redshift(z)
            age = t / 3.154e16  # Gyr
            
            # 更新恒星形成率
            sf.rate = self._compute_sfr(halo, gas, z)
            
            # 更新气体质量
            gas.mass -= sf.rate * 0.1  # 简化的气体消耗
            gas.mass = max(0, gas.mass)
            
            # 更新恒星质量
            sf.total_stellar_mass += sf.rate * 0.1
            sf.total_stellar_mass = min(sf.total_stellar_mass, halo.mass * 0.1)
            
            # 计算星系性质
            galaxy = self._compute_galaxy_properties(halo, gas, sf, z)
            
            evolution.append({
                "redshift": z,
                "time": t,
                "time_Gyr": age,
                "halo_mass": halo.mass,
                "gas_mass": gas.mass,
                "stellar_mass": sf.total_stellar_mass,
                "sfr": sf.rate,
                "metallicity": gas.metallicity,
                **galaxy,
            })
        
        self.galaxies[galaxy_id] = {
            "halo": halo,
            "gas": gas,
            "star_formation": sf,
            "evolution": evolution,
        }
        
        return {
            "galaxy_id": galaxy_id,
            "evolution": evolution,
            "final_state": evolution[-1] if evolution else None,
        }
    
    def _compute_sfr(self, 
                     halo: DarkMatterHalo, 
                     gas: GasComponent,
                     z: float) -> float:
        """
        计算恒星形成率 (Schmidt-Kennicutt定律)
        
        SFR ~ M_gas / t_dyn
        
        Args:
            halo: 暗物质晕
            gas: 气体
            z: 红移
            
        Returns:
            float: 恒星形成率 (M_sun/yr)
        """
        # 动力学时间
        t_dyn = 0.1 * (1 + z) ** (-1.5)  # Gyr
        
        # Schmidt-Kennicutt关系
        sfr = 0.1 * gas.mass / t_dyn  # M_sun/yr
        
        return sfr
    
    def _compute_galaxy_properties(self,
                                    halo: DarkMatterHalo,
                                    gas: GasComponent,
                                    sf: StarFormation,
                                    z: float) -> Dict:
        """
        计算星系性质
        
        Args:
            halo: 暗物质晕
            gas: 气体
            sf: 恒星形成
            z: 红移
            
        Returns:
            Dict: 星系性质
        """
        # 恒星质量比
        f_star = sf.total_stellar_mass / halo.mass if halo.mass > 0 else 0
        
        # 确定星系类型
        if f_star > 0.05:
            galaxy_type = GalaxyType.ELLIPTICAL
        elif f_star > 0.02:
            galaxy_type = GalaxyType.SPIRAL
        else:
            galaxy_type = GalaxyType.DWARF
        
        # 颜色 (简化的颜色-星等关系)
        if sf.rate > 10:
            color = {"u-g": 0.5, "g-r": 0.3}  # 蓝色
        elif sf.rate > 1:
            color = {"u-g": 0.8, "g-r": 0.5}  # 绿色
        else:
            color = {"u-g": 1.2, "g-r": 0.8}  # 红色
        
        # 光度
        M_star = sf.total_stellar_mass
        L = 1e10 * (M_star / 1e10) ** 1.0  # 简化的M-L关系
        
        return {
            "galaxy_type": galaxy_type.value,
            "stellar_mass": sf.total_stellar_mass,
            "star_formation_rate": sf.rate,
            "specific_sfr": sf.rate / sf.total_stellar_mass if sf.total_stellar_mass > 0 else 0,
            "gas_fraction": gas.mass / (halo.mass + gas.mass),
            "metallicity": gas.metallicity,
            "color": color,
            "luminosity": L,
        }
    
    def simulate_galaxy_merger(self,
                                galaxy1_id: str,
                                galaxy2_id: str,
                                merger_ratio: float = 1.0,
                                gas_fraction: float = 0.5) -> Dict:
        """
        模拟星系合并
        
        Args:
            galaxy1_id: 主星系ID
            galaxy2_id: 伴星系ID
            merger_ratio: 质量比 (minor/major)
            gas_fraction: 气体比例
            
        Returns:
            Dict: 合并结果
        """
        if galaxy1_id not in self.galaxies or galaxy2_id not in self.galaxies:
            raise ValueError(f"Galaxy not found: {galaxy1_id} or {galaxy2_id}")
        
        g1 = self.galaxies[galaxy1_id]
        g2 = self.galaxies[galaxy2_id]
        
        # 合并后的总质量
        total_halo_mass = g1["halo"].mass + g2["halo"].mass
        total_gas_mass = g1["gas"].mass * gas_fraction + g2["gas"].mass * gas_fraction
        total_stellar_mass = g1["star_formation"].total_stellar_mass + \
                            g2["star_formation"].total_stellar_mass
        
        # 恒星形成率激增 (starburst)
        sfr_peak = max(g1["star_formation"].rate, g2["star_formation"].rate) * merger_ratio
        
        # 创建合并后的星系
        merger_result = {
            "merger_ratio": merger_ratio,
            "status": "complete",
            "induced_starburst": {
                "peak_sfr": sfr_peak,
                "duration": 100,  # Myr
                "efficiency": 0.1 * merger_ratio,
            },
            "remnant": {
                "halo_mass": total_halo_mass,
                "gas_mass": total_gas_mass,
                "stellar_mass": total_stellar_mass,
                "morphology": "elliptical" if merger_ratio < 0.3 else "disturbed",
            },
            "stellar_feedback": {
                "energy": 1e59 * merger_ratio,  # erg
                "outflow_mass": total_gas_mass * 0.1,
            },
        }
        
        return merger_result
    
    def generate_population(self,
                            z: float = 0,
                            mass_range: Tuple[float, float] = (1e8, 1e14),
                            n_galaxies: int = 1000) -> List[Dict]:
        """
        生成星系群
        
        Args:
            z: 红移
            mass_range: 质量范围 (M_sun)
            n_galaxies: 星系数量
            
        Returns:
            List[Dict]: 星系列表
        """
        population = []
        
        # 质量函数 (Schechter函数)
        phi_star = 0.01  # Mpc^-3
        M_star = 1e10    # M_sun
        alpha = -1.1
        
        masses = 10 ** np.linspace(np.log10(mass_range[0]), 
                                   np.log10(mass_range[1]), 
                                   n_galaxies)
        
        for mass in masses:
            # 概率权重
            weight = (mass / M_star) ** alpha * np.exp(-mass / M_star)
            
            # 创建星系
            halo = self.create_collapsed_halo(mass, z)
            gas = self.simulate_gas_cooling(halo)
            
            # 计算恒星质量
            M_star_est = 0.05 * mass  # 简化的恒星质量比
            
            galaxy = {
                "id": f"galaxy_{len(population)}",
                "mass": mass,
                "stellar_mass": M_star_est,
                "type": self._determine_galaxy_type(mass, M_star_est),
                "sfr": self._estimate_sfr(M_star_est, z),
                "metallicity": 0.02 * (mass / 1e12) ** 0.2,
                "position": self._generate_position(),
                "velocity": self._generate_velocity(halo),
            }
            
            population.append(galaxy)
        
        return population
    
    def _determine_galaxy_type(self, halo_mass: float, stellar_mass: float) -> str:
        """确定星系类型"""
        f_star = stellar_mass / halo_mass
        
        if halo_mass < 1e9:
            return GalaxyType.DWARF.value
        elif f_star > 0.05:
            return GalaxyType.ELLIPTICAL.value
        elif f_star > 0.02:
            return GalaxyType.SPIRAL.value
        else:
            return GalaxyType.IRREGULAR.value
    
    def _estimate_sfr(self, stellar_mass: float, z: float) -> float:
        """估计恒星形成率"""
        # Main sequence关系
        sfr_main = 1.0 * (stellar_mass / 1e10) ** 0.7 * (1 + z) ** 2.5
        return sfr_main
    
    def _generate_position(self) -> List[float]:
        """生成位置"""
        return [random.uniform(-100, 100) for _ in range(3)]  # Mpc
    
    def _generate_velocity(self, halo: DarkMatterHalo) -> List[float]:
        """生成速度"""
        sigma = halo.velocity_dispersion
        return [random.gauss(0, sigma) for _ in range(3)]
    
    def _get_time_from_redshift(self, z: float) -> float:
        """从红移计算时间"""
        H0 = self.H0 * 1000 / 3.086e19
        return 2.0 / (3.0 * H0 * np.sqrt(self.Omega_m)) * (1 + z) ** (-3/2)
