"""
Cosmos Simulation - 宇宙模拟器主模块

提供从大爆炸到当前宇宙的完整演化模拟。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from big_bang import BigBang, InitialConditions
from galaxy_formation import GalaxyFormation
from stellar_evolution import StellarEvolution
from cosmology import Cosmology
from config import SimulationConfig


@dataclass
class EvolutionState:
    """演化状态"""
    redshift: float
    time: float                    # 时间 (Gyr)
    scale_factor: float
    age: float                     # 宇宙年龄 (Gyr)
    
    # 宇宙学状态
    hubble_parameter: float
    matter_density: float
    radiation_density: float
    dark_energy_density: float
    
    # 结构状态
    growth_factor: float
    power_spectrum_amplitude: float
    
    # 星系统计
    n_galaxies: int = 0
    total_stellar_mass: float = 0
    sfr_density: float = 0
    
    # 元数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvolutionHistory:
    """演化历史"""
    states: List[EvolutionState]
    initial_redshift: float
    final_redshift: float
    total_time: float
    
    def get_state(self, redshift: float) -> Optional[EvolutionState]:
        """获取特定红移的状态"""
        for state in self.states:
            if abs(state.redshift - redshift) < 0.01:
                return state
        return None
    
    def get_redshifts(self) -> List[float]:
        """获取所有红移值"""
        return [s.redshift for s in self.states]


class CosmosSimulation:
    """
    宇宙模拟器主类
    
    整合大爆炸、星系形成、恒星演化和宇宙学模块，
    提供完整的宇宙演化模拟。
    
    Example:
        >>> cosmos = CosmosSimulation()
        >>> cosmos.set_initial_conditions(z=1000)
        >>> evolution = cosmos.evolve(end_redshift=0, time_step="1Gyr")
        >>> state = evolution.get_state(redshift=0, scale="galaxy")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化宇宙模拟器
        
        Args:
            config: 配置参数字典
        """
        # 加载配置
        self.config = SimulationConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 初始化子模块
        self.big_bang = BigBang(self.config.__dict__)
        self.galaxy_formation = GalaxyFormation(self.config.__dict__)
        self.stellar_evolution = StellarEvolution(self.config.__dict__)
        self.cosmology = Cosmology(self.config.__dict__)
        
        # 状态变量
        self.initial_conditions: Optional[InitialConditions] = None
        self.evolution: Optional[EvolutionHistory] = None
        self._current_z = None
        
    def set_initial_conditions(self, 
                                z: float = 1000,
                                density_fluctuations: float = 1e-5,
                                **kwargs) -> InitialConditions:
        """
        设置模拟的初始条件
        
        Args:
            z: 初始红移 (默认: 1000, 复合期)
            density_fluctuations: 初始密度扰动幅度
            **kwargs: 其他参数
            
        Returns:
            InitialConditions: 初始条件对象
            
        Example:
            >>> cosmos.set_initial_conditions(z=1000)
            >>> cosmos.set_initial_conditions(z=100, density_fluctuations=1e-4)
        """
        self.initial_conditions = self.big_bang.set_initial_conditions(z)
        self._current_z = z
        
        # 同步宇宙学参数
        self.cosmology.set_cosmological_parameters(
            H0=self.config.H0,
            Omega_m=self.config.Omega_m,
            Omega_b=self.config.Omega_b,
            Omega_Lambda=self.config.Omega_Lambda,
        )
        
        return self.initial_conditions
    
    def evolve(self,
               end_redshift: float = 0,
               time_step: str = "1Gyr",
               save_states: bool = True) -> EvolutionHistory:
        """
        运行宇宙演化
        
        Args:
            end_redshift: 结束红移 (默认: 0, 当前宇宙)
            time_step: 时间步长 (如 "1Gyr", "100Myr", "auto")
            save_states: 是否保存中间状态
            
        Returns:
            EvolutionHistory: 演化历史对象
            
        Example:
            >>> evolution = cosmos.evolve(end_redshift=0, time_step="1Gyr")
            >>> evolution = cosmos.evolve(end_redshift=0, time_step="100Myr")
        """
        if self.initial_conditions is None:
            self.set_initial_conditions()
        
        z_start = self._current_z
        z_end = end_redshift
        
        # 解析时间步长
        dt = self._parse_time_step(time_step)
        
        # 生成红移序列
        z_values = self._generate_redshift_sequence(z_start, z_end, dt)
        
        # 计算总演化时间
        t_start = self.cosmology.compute_age(z_start)
        t_end = self.cosmology.compute_age(z_end)
        total_time = t_end - t_start
        
        # 演化状态列表
        states = []
        
        # 计算初始功率谱幅度
        P_initial = self.cosmology.compute_power_spectrum(z_start)
        P_amplitude_initial = np.max(P_initial.P_k) if len(P_initial.P_k) > 0 else 1e4
        
        for z in z_values:
            t = self.cosmology.compute_age(z)
            age = t
            
            # 计算宇宙学状态
            H = self.cosmology.compute_H(z)
            densities = self.cosmology.compute_density_parameter(z)
            
            # 计算增长因子
            D = self.cosmology.compute_linear_growth_factor(z)
            
            # 计算功率谱幅度
            P = self.cosmology.compute_power_spectrum(z)
            P_amplitude = np.max(P.P_k) / P_amplitude_initial if len(P.P_k) > 0 else 1.0
            
            # 简化的星系统计
            n_galaxies = int(1e6 * (1 + z) ** (-1.5))
            total_stellar_mass = 1e11 * (1 + z) ** (-1)
            sfr_density = 0.1 * (1 + z) ** 2.5
            
            state = EvolutionState(
                redshift=z,
                time=total_time - (t_end - t),
                scale_factor=1 / (1 + z),
                age=age,
                hubble_parameter=H,
                matter_density=densities["Omega_m_z"],
                radiation_density=densities["Omega_r_z"],
                dark_energy_density=densities["Omega_Lambda_z"],
                growth_factor=D,
                power_spectrum_amplitude=P_amplitude,
                n_galaxies=n_galaxies,
                total_stellar_mass=total_stellar_mass,
                sfr_density=sfr_density,
            )
            
            states.append(state)
        
        self.evolution = EvolutionHistory(
            states=states,
            initial_redshift=z_start,
            final_redshift=z_end,
            total_time=total_time,
        )
        
        self._current_z = z_end
        
        return self.evolution
    
    def get_state(self,
                  redshift: float = 0,
                  scale: str = "cosmic_web") -> Dict[str, Any]:
        """
        获取宇宙特定时刻的状态
        
        Args:
            redshift: 目标红移
            scale: 尺度 ("cosmic_web", "galaxy", "stellar", "black_hole")
            
        Returns:
            Dict: 宇宙状态
            
        Example:
            >>> state = cosmos.get_state(redshift=0, scale="galaxy")
            >>> print(f"Age: {state['age']:.2f} Gyr")
        """
        if self.evolution is None:
            self.evolve()
        
        # 获取最接近的状态
        state = self.evolution.get_state(redshift)
        
        if state is None:
            # 计算插值状态
            state = self._interpolate_state(redshift)
        
        # 根据尺度返回不同详细程度的信息
        result = {
            "redshift": state.redshift,
            "age": state.age,
            "scale_factor": state.scale_factor,
            "hubble_parameter": state.hubble_parameter,
            "densities": {
                "matter": state.matter_density,
                "radiation": state.radiation_density,
                "dark_energy": state.dark_energy_density,
            },
            "structure": {
                "growth_factor": state.growth_factor,
                "power_spectrum_amplitude": state.power_spectrum_amplitude,
            },
        }
        
        if scale in ["galaxy", "cosmic_web"]:
            result["galaxies"] = {
                "n_galaxies": state.n_galaxies,
                "total_stellar_mass": state.total_stellar_mass,
                "sfr_density": state.sfr_density,
            }
        
        if scale == "stellar":
            # 添加恒星演化信息
            result["stellar"] = {
                "mean_stellar_mass": 1.0,  # M_sun
                "metallicity": 0.02 * (1 + redshift) ** 0.5,
            }
        
        if scale == "black_hole":
            # 添加黑洞信息
            result["black_holes"] = {
                "n_supermassive": int(state.n_galaxies * 0.1),
                "total_mass": state.total_stellar_mass * 0.001,
            }
        
        return result
    
    def get_cosmological_parameters(self) -> Dict:
        """获取宇宙学参数"""
        return self.cosmology.get_cosmological_parameters()
    
    def compute_distance(self, z: float) -> Dict:
        """
        计算宇宙学距离
        
        Args:
            z: 红移
            
        Returns:
            Dict: 距离信息
        """
        return {
            "redshift": z,
            "luminosity_distance": self.cosmology.compute_luminosity_distance(z),
            "angular_diameter_distance": self.cosmology.compute_angular_diameter_distance(z),
            "lookback_time": self.cosmology.compute_lookback_time(z),
            "distance_modulus": self.cosmology.compute_distance_modulus(z),
        }
    
    def create_galaxy(self, 
                      mass: float,
                      z: float = None,
                      **kwargs) -> Dict:
        """
        创建单个星系
        
        Args:
            mass: 星系质量 (M_sun)
            z: 红移 (默认当前)
            **kwargs: 其他参数
            
        Returns:
            Dict: 星系信息
        """
        z = z if z is not None else self._current_z
        halo = self.galaxy_formation.create_collapsed_halo(mass, z, **kwargs)
        gas = self.galaxy_formation.simulate_gas_cooling(halo)
        
        return {
            "halo_mass": halo.mass,
            "halo_radius": halo.radius,
            "gas_mass": gas.mass,
            "gas_temperature": gas.temperature,
            "redshift": z,
        }
    
    def create_star(self,
                    mass: float,
                    metallicity: float = 0.02,
                    **kwargs) -> Dict:
        """
        创建单个恒星
        
        Args:
            mass: 恒星质量 (M_sun)
            metallicity: 金属丰度
            **kwargs: 其他参数
            
        Returns:
            Dict: 恒星信息
        """
        star = self.stellar_evolution.create_star(mass, metallicity, **kwargs)
        
        return {
            "mass": star.mass,
            "radius": star.radius,
            "luminosity": star.luminosity,
            "temperature": star.temperature,
            "lifetime": star.lifetime,
            "spectral_type": star.stellar_type.value,
        }
    
    def run_structural_analysis(self,
                                 z_start: float = None,
                                 z_end: float = None) -> Dict:
        """
        运行结构形成分析
        
        Args:
            z_start: 起始红移 (默认初始条件)
            z_end: 结束红移 (默认当前)
            
        Returns:
            Dict: 结构形成分析结果
        """
        z_start = z_start if z_start is not None else self._current_z
        z_end = z_end if z_end is not None else 0
        
        history = self.cosmology.simulate_structure_formation(z_start, z_end)
        
        return {
            "history": history["history"],
            "structure_amplitude": history["structure_amplitude"],
        }
    
    def generate_snapshot(self, 
                          z: float,
                          include_galaxies: bool = True,
                          include_stars: bool = True,
                          n_galaxies: int = 100,
                          n_stars: int = 1000) -> Dict:
        """
        生成宇宙快照
        
        Args:
            z: 红移
            include_galaxies: 是否包含星系
            include_stars: 是否包含恒星
            n_galaxies: 星系数量
            n_stars: 恒星数量
            
        Returns:
            Dict: 快照数据
        """
        # 获取基础状态
        state = self.get_state(z, scale="cosmic_web")
        
        snapshot = {
            "redshift": z,
            "timestamp": datetime.now().isoformat(),
            "cosmology": self.get_cosmological_parameters(),
            "state": state,
        }
        
        if include_galaxies:
            galaxies = self.galaxy_formation.generate_population(z, n_galaxies=n_galaxies)
            snapshot["galaxies"] = galaxies
        
        if include_stars:
            stellar_pop = self.stellar_evolution.generate_stellar_population(
                n_stars=n_stars,
                metallicity=0.02 * (1 + z) ** 0.5
            )
            snapshot["stars"] = stellar_pop
        
        return snapshot
    
    def _parse_time_step(self, time_step: str) -> float:
        """解析时间步长"""
        if time_step == "auto":
            # 自动: 1% Hubble time
            H0_inv = 9.778e9  # Gyr
            return 0.01 * H0_inv
        
        # 解析字符串
        try:
            value = float(time_step[:-3])
            unit = time_step[-3:]
            
            multipliers = {"Gyr": 1, "Myr": 1e-3, "kyr": 1e-6}
            multiplier = multipliers.get(unit, 1)
            
            return value * multiplier
        except:
            return 1.0  # 默认 1 Gyr
    
    def _generate_redshift_sequence(self,
                                     z_start: float,
                                     z_end: float,
                                     dt: float) -> List[float]:
        """生成红移序列"""
        # 计算总时间跨度
        t_start = self.cosmology.compute_age(z_start)
        t_end = self.cosmology.compute_age(z_end)
        total_time = t_end - t_start
        
        # 生成时间点
        n_steps = max(10, int(total_time / dt))
        times = np.linspace(t_start, t_end, n_steps)
        
        # 转换为红移
        z_values = []
        for t in times:
            # 简单的近似: z(t) ≈ (t_end/t)² - 1 (物质主导)
            if t > 0:
                z = (t_end / t) ** (2/3) - 1
                z_values.append(max(0, z))
        
        return z_values
    
    def _interpolate_state(self, z: float) -> EvolutionState:
        """插值计算状态"""
        if self.evolution is None:
            return EvolutionState(
                redshift=z,
                time=0,
                scale_factor=1/(1+z),
                age=self.cosmology.compute_age(z),
                hubble_parameter=self.cosmology.compute_H(z),
                matter_density=0.3,
                radiation_density=1e-4,
                dark_energy_density=0.7,
                growth_factor=1.0,
                power_spectrum_amplitude=1.0,
            )
        
        # 找到最近的两个状态
        states = self.evolution.states
        if len(states) < 2:
            return states[0] if states else EvolutionState(
                redshift=z, time=0, scale_factor=0, age=0,
                hubble_parameter=0, matter_density=0, 
                radiation_density=0, dark_energy_density=0,
                growth_factor=0, power_spectrum_amplitude=0,
            )
        
        # 线性插值
        z_vals = [s.redshift for s in states]
        
        for i in range(len(states) - 1):
            if z <= z_vals[i] and z >= z_vals[i+1]:
                # 找到区间
                z1, z2 = z_vals[i], z_vals[i+1]
                s1, s2 = states[i], states[i+1]
                
                f = (z - z2) / (z1 - z2) if z1 != z2 else 0
                
                return EvolutionState(
                    redshift=z,
                    time=s2.time + f * (s1.time - s2.time),
                    scale_factor=s2.scale_factor + f * (s1.scale_factor - s2.scale_factor),
                    age=s2.age + f * (s1.age - s2.age),
                    hubble_parameter=s2.hubble_parameter + f * (s1.hubble_parameter - s2.hubble_parameter),
                    matter_density=s2.matter_density + f * (s1.matter_density - s2.matter_density),
                    radiation_density=s2.radiation_density + f * (s1.radiation_density - s2.radiation_density),
                    dark_energy_density=s2.dark_energy_density + f * (s1.dark_energy_density - s2.dark_energy_density),
                    growth_factor=s2.growth_factor + f * (s1.growth_factor - s2.growth_factor),
                    power_spectrum_amplitude=s2.power_spectrum_amplitude + f * (s1.power_spectrum_amplitude - s2.power_spectrum_amplitude),
                    n_galaxies=int(s2.n_galaxies + f * (s1.n_galaxies - s2.n_galaxies)),
                    total_stellar_mass=s2.total_stellar_mass + f * (s1.total_stellar_mass - s2.total_stellar_mass),
                    sfr_density=s2.sfr_density + f * (s1.sfr_density - s2.sfr_density),
                )
        
        return states[-1]
