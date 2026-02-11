"""
Stellar Evolution Module - 恒星演化模块

实现主序星、巨星阶段、超新星和黑洞/中子星的演化模拟。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class StellarType(Enum):
    """恒星光谱类型"""
    O = "O"     # 蓝巨星
    B = "B"     # 蓝白星
    A = "A"     # 白星
    F = "F"     # 黄白星
    G = "G"     # 黄星 (太阳)
    K = "K"     # 橙星
    M = "M"     # 红矮星


class EvolutionaryStage(Enum):
    """演化阶段"""
    MAIN_SEQUENCE = "main_sequence"       # 主序星
    SUBGIANT = "subgiant"                 # 亚巨星
    RED_GIANT = "red_giant"               # 红巨星
    RED_SUPERGIANT = "red_supergiant"     # 红超巨星
    HELIUM_BURNING = "helium_burning"     # 氦燃烧
    PLANETARY_NEBULA = "planetary_nebula" # 行星状星云
    WHITE_DWARF = "white_dwarf"           # 白矮星
    NEUTRON_STAR = "neutron_star"         # 中子星
    BLACK_HOLE = "black_hole"             # 黑洞
    SUPERNOVA = "supernova"               # 超新星


class SupernovaType(Enum):
    """超新星类型"""
    TYPE_IA = "type_ia"       # Ia型 (白矮星爆炸)
    TYPE_II = "type_ii"       # II型 (核心塌缩)
    TYPE_IB = "type_ib"       # Ib型 (氦壳层剥离)
    TYPE_IC = "type_ic"       # Ic型 (氢氦壳层剥离)
    PAIR_INSTABILITY = "pair_instability"  # 对不稳定
    PULSAR_KICK = "pulsar_kick"  # 脉冲星kick


@dataclass
class StellarProperties:
    """恒星基本属性"""
    mass: float = 1.0                    # 质量 (M_sun)
    radius: float = 1.0                  # 半径 (R_sun)
    luminosity: float = 1.0              # 光度 (L_sun)
    temperature: float = 5778            # 表面温度 (K)
    metallicity: float = 0.02            # 金属丰度 (Z_sun)
    age: float = 0.0                     # 当前年龄 (Gyr)
    lifetime: float = 10.0               # 预期寿命 (Gyr)
    stellar_type: StellarType = StellarType.G
    evolutionary_stage: EvolutionaryStage = EvolutionaryStage.MAIN_SEQUENCE
    
    # 核心属性
    core_mass: float = 0.1               # 核心质量 (M_sun)
    core_temperature: float = 1.5e7       # 核心温度 (K)
    core_density: float = 100.0          # 核心密度 (g/cm³)
    
    # 元素丰度
    hydrogen_abundance: float = 0.74
    helium_abundance: float = 0.24
    metal_abundance: float = 0.02
    
    # 旋转
    rotation_rate: float = 1.0           # 相对太阳的旋转速度
    magnetic_field: float = 1.0          # 磁场强度 (B_sun)


@dataclass
class StellarEvolutionTrack:
    """恒星演化轨迹"""
    time: List[float]                    # 时间点 (Gyr)
    radius: List[float]                  # 半径演化 (R_sun)
    luminosity: List[float]              # 光度演化 (L_sun)
    temperature: List[float]             # 温度演化 (K)
    mass: List[float]                    # 质量演化 (M_sun)
    stage: List[str]                     # 演化阶段


@dataclass
class SupernovaExplosion:
    """超新星爆炸"""
    type: SupernovaType                  # 超新星类型
    ejecta_mass: float                   # 抛射质量 (M_sun)
    kinetic_energy: float                # 动能 (erg)
    nickel_mass: float                   # Ni-56质量 (M_sun)
    peak_luminosity: float               # 峰值光度 (L_sun)
    decline_rate: float                  # 光变曲线下降率
    remnant_mass: float                  # 残骸质量 (M_sun)
    remnant_type: EvolutionaryStage      # 残骸类型
    
    # 元素合成
    synthesized_elements: Dict[str, float] = field(default_factory=dict)


class StellarEvolution:
    """
    恒星演化模拟器
    
    实现从原恒星到致密残骸的完整演化过程。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化恒星演化模拟器
        
        Args:
            config: 配置参数
        """
        self.stars: Dict[str, StellarProperties] = {}
        self.tracks: Dict[str, StellarEvolutionTrack] = {}
        self.supernovae: List[SupernovaExplosion] = []
        self.config = config or {}
        
        # 太阳参数
        self.sun_properties = {
            "mass": 1.0,
            "radius": 1.0,
            "luminosity": 1.0,
            "temperature": 5778,
            "age": 4.6,  # Gyr
        }
    
    def create_star(self, 
                    mass: float,
                    metallicity: float = 0.02,
                    initial_composition: Optional[Dict] = None) -> StellarProperties:
        """
        创建恒星
        
        Args:
            mass: 初始质量 (M_sun)
            metallicity: 金属丰度 (Z_sun)
            initial_composition: 初始元素组成
            
        Returns:
            StellarProperties: 恒星属性
        """
        # 计算有效温度 (mass-luminosity关系)
        if mass < 0.5:
            L = 0.23 * mass ** 2.3
        elif mass < 2.0:
            L = mass ** 4.0
        else:
            L = 1.5 * mass ** 3.5
        
        # 计算半径 (mass-radius关系)
        if mass < 1.0:
            R = mass ** 0.8
        else:
            R = mass ** 0.7
        
        # 确定光谱类型
        spectral_type = self._determine_spectral_type(mass, L, R)
        
        # 计算寿命 (主序寿命)
        if mass < 1.0:
            lifetime = 10.0 * (mass / 1.0) ** (-2.5)
        else:
            lifetime = 10.0 * (mass / 1.0) ** (-2.5)
        
        # 初始元素丰度
        if initial_composition:
            H = initial_composition.get("H", 0.74 - 0.1 * metallicity)
            He = initial_composition.get("He", 0.24 + 0.05 * metallicity)
            metals = initial_composition.get("metals", metallicity)
        else:
            H = 0.74 - 0.5 * metallicity
            He = 0.24 + 0.4 * metallicity
            metals = metallicity
        
        star = StellarProperties(
            mass=mass,
            radius=R,
            luminosity=L,
            temperature=self._calculate_temperature(L, R),
            metallicity=metallicity,
            lifetime=lifetime,
            stellar_type=spectral_type,
            
            # 初始核心
            core_mass=0.1 * mass,
            core_temperature=1.5e7 * (mass / 1.0) ** 0.5,
            core_density=150.0 * (mass / 1.0),
            
            hydrogen_abundance=H,
            helium_abundance=He,
            metal_abundance=metals,
        )
        
        star_id = f"star_{len(self.stars)}"
        self.stars[star_id] = star
        
        return star
    
    def _determine_spectral_type(self, mass: float, L: float, R: float) -> StellarType:
        """确定光谱类型"""
        if mass > 16:
            return StellarType.O
        elif mass > 2.1:
            return StellarType.B
        elif mass > 1.4:
            return StellarType.A
        elif mass > 1.04:
            return StellarType.F
        elif mass > 0.8:
            return StellarType.G
        elif mass > 0.45:
            return StellarType.K
        else:
            return StellarType.M
    
    def _calculate_temperature(self, L: float, R: float) -> float:
        """计算有效温度 (Stefan-Boltzmann定律)"""
        # L = 4πR²σT⁴
        sigma = 5.67e-8
        L_sun = 3.828e26  # W
        R_sun = 6.96e8    # m
        
        T = (L * L_sun / (4 * np.pi * (R * R_sun)**2 * sigma)) ** 0.25
        return T
    
    def compute_main_sequence(self, 
                              star: StellarProperties,
                              time_steps: int = 100) -> StellarEvolutionTrack:
        """
        计算主序星演化
        
        Args:
            star: 恒星对象
            time_steps: 时间步数
            
        Returns:
            StellarEvolutionTrack: 演化轨迹
        """
        track = StellarEvolutionTrack(
            time=[],
            radius=[],
            luminosity=[],
            temperature=[],
            mass=[],
            stage=[],
        )
        
        dt = star.lifetime / time_steps
        
        for i in range(time_steps + 1):
            t = i * dt
            age = star.age + t
            
            # 主序星演化: 慢慢变亮变大
            f = age / star.lifetime
            
            # 质量损失 (太阳-like恒星质量损失很小)
            mass_loss = 1.0 - 1.4e-14 * star.mass * L ** 0.75 * t * 1e9
            
            # 半径和光度增加
            if star.mass < 1.0:
                R_factor = 1 + 0.4 * f
                L_factor = 1 + 0.4 * f
            else:
                R_factor = 1 + 0.8 * f
                L_factor = 1 + 1.5 * f
            
            # 更新恒星属性
            star.mass *= mass_loss
            star.radius *= R_factor
            star.luminosity *= L_factor
            star.temperature = self._calculate_temperature(star.luminosity, star.radius)
            
            # 核心演化
            star.core_mass = 0.1 * star.mass * (1 + f)
            star.core_temperature = 1.5e7 * (star.mass / 1.0) ** 0.5 * (1 + 0.5 * f)
            
            track.time.append(age)
            track.radius.append(star.radius)
            track.luminosity.append(star.luminosity)
            track.temperature.append(star.temperature)
            track.mass.append(star.mass)
            track.stage.append(EvolutionaryStage.MAIN_SEQUENCE.value)
        
        self.tracks[id(star)] = track
        return track
    
    def evolve_star(self, 
                    star_id: str,
                    end_stage: EvolutionaryStage = EvolutionaryStage.WHITE_DWARF,
                    time_step: float = 0.1) -> Dict:
        """
        演化单颗恒星
        
        Args:
            star_id: 恒星ID
            end_stage: 结束阶段
            time_step: 时间步长 (Gyr)
            
        Returns:
            Dict: 演化结果
        """
        if star_id not in self.stars:
            raise ValueError(f"Star not found: {star_id}")
        
        star = self.stars[star_id]
        
        # 创建演化轨迹
        track = StellarEvolutionTrack(
            time=[star.age],
            radius=[star.radius],
            luminosity=[star.luminosity],
            temperature=[star.temperature],
            mass=[star.mass],
            stage=[star.evolutionary_stage.value],
        )
        
        current_stage = star.evolutionary_stage
        supernova = None
        
        while current_stage != end_stage and star.age < star.lifetime:
            dt = min(time_step, star.lifetime - star.age)
            star.age += dt
            
            if current_stage == EvolutionaryStage.MAIN_SEQUENCE:
                # 主序星阶段
                track = self._evolve_main_sequence(star, dt, track)
                
                if star.age >= star.lifetime * 0.9:
                    current_stage = EvolutionaryStage.SUBGIANT
            
            elif current_stage == EvolutionaryStage.SUBGIANT:
                # 亚巨星分支
                track = self._evolve_subgiant(star, dt, track)
                current_stage = EvolutionaryStage.RED_GIANT
            
            elif current_stage == EvolutionaryStage.RED_GIANT:
                # 红巨星分支
                track = self._evolve_red_giant(star, dt, track)
                
                if star.mass > 0.5:
                    current_stage = EvolutionaryStage.HELIUM_BURNING
                else:
                    current_stage = EvolutionaryStage.PLANETARY_NEBULA
            
            elif current_stage == EvolutionaryStage.HELIUM_BURNING:
                # 氦燃烧
                track = self._evolve_helium_burning(star, dt, track)
                current_stage = EvolutionaryStage.RED_SUPERGIANT
            
            elif current_stage == EvolutionaryStage.RED_SUPERGIANT:
                # 红超巨星 (高质量恒星)
                track = self._evolve_red_supergiant(star, dt, track)
                
                if star.mass > 8:
                    supernova = self._explode_supernova(star)
                    current_stage = supernova.remnant_type
                else:
                    current_stage = EvolutionaryStage.PLANETARY_NEBULA
            
            elif current_stage == EvolutionaryStage.PLANETARY_NEBULA:
                # 行星状星云阶段
                track = self._evolve_planetary_nebula(star, dt, track)
                current_stage = EvolutionaryStage.WHITE_DWARF
            
            elif current_stage == EvolutionaryStage.WHITE_DWARF:
                # 白矮星冷却
                track = self._evolve_white_dwarf(star, dt, track)
                break
            
            elif current_stage == EvolutionaryStage.NEUTRON_STAR:
                # 中子星
                track = self._evolve_neutron_star(star, dt, track)
                break
            
            elif current_stage == EvolutionaryStage.BLACK_HOLE:
                # 黑洞
                track = self._evolve_black_hole(star, dt, track)
                break
            
            star.evolutionary_stage = current_stage
            
            track.time.append(star.age)
            track.radius.append(star.radius)
            track.luminosity.append(star.luminosity)
            track.temperature.append(star.temperature)
            track.mass.append(star.mass)
            track.stage.append(current_stage.value)
        
        self.tracks[star_id] = track
        
        return {
            "star_id": star_id,
            "initial_mass": track.mass[0] if track.mass else star.mass,
            "final_mass": star.mass,
            "final_stage": star.evolutionary_stage.value,
            "lifetime": star.lifetime,
            "track": track,
            "supernova": supernova.__dict__ if supernova else None,
        }
    
    def _evolve_main_sequence(self, 
                               star: StellarProperties, 
                               dt: float,
                               track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """主序星演化"""
        f = star.age / star.lifetime
        
        # 主序星缓慢演化
        star.radius *= 1 + 0.01 * dt
        star.luminosity *= 1 + 0.02 * dt
        star.temperature = self._calculate_temperature(star.luminosity, star.radius)
        
        # 核心质量增加
        star.core_mass = 0.1 * star.mass * (1 + 0.8 * f)
        
        return track
    
    def _evolve_subgiant(self, 
                         star: StellarProperties, 
                         dt: float,
                         track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """亚巨星演化"""
        # 亚巨星分支: 快速变亮
        star.radius *= 1 + 0.1 * dt
        star.luminosity *= 1 + 0.3 * dt
        star.temperature *= 0.99
        
        return track
    
    def _evolve_red_giant(self, 
                          star: StellarProperties, 
                          dt: float,
                          track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """红巨星演化"""
        # 红巨星: 巨大的半径和光度
        star.radius *= 1 + 0.2 * dt
        star.luminosity *= 1 + 0.5 * dt
        star.temperature *= 0.95
        
        # 核心收缩
        star.core_mass *= 1 + 0.05 * dt
        star.core_temperature *= 1 + 0.1 * dt
        
        return track
    
    def _evolve_helium_burning(self, 
                                star: StellarProperties, 
                                dt: float,
                                track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """氦燃烧阶段 (水平分支)"""
        # 氦燃烧: 相对稳定
        star.radius *= 1 + 0.02 * dt
        star.luminosity *= 1 + 0.05 * dt
        star.temperature *= 1.01
        
        return track
    
    def _evolve_red_supergiant(self, 
                                star: StellarProperties, 
                                dt: float,
                                track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """红超巨星演化"""
        # 红超巨星: 巨大的质量和半径
        star.radius *= 1 + 0.1 * dt
        star.luminosity *= 1 + 0.2 * dt
        star.temperature *= 0.98
        
        return track
    
    def _evolve_planetary_nebula(self, 
                                  star: StellarProperties, 
                                  dt: float,
                                  track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """行星状星云演化"""
        # 抛射包层
        star.mass = star.core_mass
        star.radius = 0.01  # 白矮星大小
        star.luminosity = 1000  # 热白矮星
        star.temperature = 100000  # 热的白矮星
        
        return track
    
    def _evolve_white_dwarf(self, 
                            star: StellarProperties, 
                            dt: float,
                            track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """白矮星冷却"""
        star.luminosity *= 0.9
        star.temperature *= 0.98
        
        return track
    
    def _evolve_neutron_star(self, 
                              star: StellarProperties, 
                              dt: float,
                              track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """中子星演化"""
        star.radius = 0.01  # ~10 km
        star.luminosity = 0.001
        
        return track
    
    def _evolve_black_hole(self, 
                           star: StellarProperties, 
                           dt: float,
                           track: StellarEvolutionTrack) -> StellarEvolutionTrack:
        """黑洞演化"""
        # 黑洞没有常规演化
        pass
    
    def _explode_supernova(self, star: StellarProperties) -> SupernovaExplosion:
        """
        超新星爆炸
        
        Args:
            star: 爆炸前的恒星
            
        Returns:
            SupernovaExplosion: 超新星爆炸参数
        """
        # 根据质量确定超新星类型
        if star.mass > 100:
            sn_type = SupernovaType.PAIR_INSTABILITY
        elif star.mass > 25:
            sn_type = SupernovaType.TYPE_II
        elif star.mass > 8:
            sn_type = SupernovaType.TYPE_II
        else:
            sn_type = SupernovaType.TYPE_IA
        
        # 计算抛射质量
        if sn_type == SupernovaType.TYPE_IA:
            ejecta_mass = star.mass  # 整个白矮星
            kinetic_energy = 1e51    # erg
            nickel_mass = 0.6
            remnant_type = EvolutionaryStage.BLACK_HOLE if star.mass > 1.4 else EvolutionaryStage.NEUTRON_STAR
            remnant_mass = 0
        else:
            ejecta_mass = star.mass - 1.4  # 除去核心
            kinetic_energy = 1e51 * (star.mass / 10)
            nickel_mass = 0.1 * (star.mass / 10)
            remnant_type = EvolutionaryStage.NEUTRON_STAR if star.mass < 25 else EvolutionaryStage.BLACK_HOLE
            remnant_mass = 1.4 if star.mass < 25 else star.mass - ejecta_mass
        
        # 元素合成
        elements = {
            "H": ejecta_mass * 0.1 if sn_type != SupernovaType.TYPE_IA else 0,
            "He": ejecta_mass * 0.2,
            "C": ejecta_mass * 0.01,
            "O": ejecta_mass * 0.05,
            "Si": ejecta_mass * 0.03,
            "Fe": ejecta_mass * 0.02,
            "Ni56": nickel_mass,
        }
        
        explosion = SupernovaExplosion(
            type=sn_type,
            ejecta_mass=ejecta_mass,
            kinetic_energy=kinetic_energy,
            nickel_mass=nickel_mass,
            peak_luminosity=1e9 * star.luminosity,
            decline_rate=0.01,
            remnant_mass=remnant_mass,
            remnant_type=remnant_type,
            synthesized_elements=elements,
        )
        
        self.supernovae.append(explosion)
        return explosion
    
    def generate_stellar_population(self,
                                     mass_range: Tuple[float, float] = (0.1, 100),
                                     metallicity: float = 0.02,
                                     n_stars: int = 1000,
                                     imf_slope: float = -2.35) -> Dict:
        """
        生成恒星群 (初始质量函数)
        
        Args:
            mass_range: 质量范围 (M_sun)
            metallicity: 金属丰度
            n_stars: 恒星数量
            imf_slope: IMF斜率 (Salpeter = -2.35)
            
        Returns:
            Dict: 恒星群统计
        """
        population = []
        
        # 使用IMF生成质量分布
        masses = self._generate_imf(mass_range, n_stars, imf_slope)
        
        for mass in masses:
            star = self.create_star(mass, metallicity)
            
            # 演化到当前年龄
            evolution = self.evolve_star(
                id(star),
                end_stage=EvolutionaryStage.MAIN_SEQUENCE,
                time_step=0.1
            )
            
            population.append({
                "id": id(star),
                "initial_mass": mass,
                "current_mass": star.mass,
                "current_radius": star.radius,
                "current_luminosity": star.luminosity,
                "current_temperature": star.temperature,
                "spectral_type": star.stellar_type.value,
                "evolutionary_stage": star.evolutionary_stage.value,
                "age": star.age,
                "lifetime": star.lifetime,
            })
        
        # 统计
        types = {}
        stages = {}
        
        for star in population:
            t = star["spectral_type"]
            s = star["evolutionary_stage"]
            types[t] = types.get(t, 0) + 1
            stages[s] = stages.get(s, 0) + 1
        
        return {
            "population": population,
            "total_stars": len(population),
            "spectral_type_distribution": types,
            "evolutionary_stage_distribution": stages,
            "mass_range": {"min": min(masses), "max": max(masses)},
            "mean_mass": np.mean(masses),
        }
    
    def _generate_imf(self,
                      mass_range: Tuple[float, float],
                      n: int,
                      alpha: float) -> np.ndarray:
        """
        生成初始质量函数 (IMF)
        
        dN/dM ~ M^(-alpha)
        
        Args:
            mass_range: 质量范围
            n: 数量
            alpha: IMF斜率
            
        Returns:
            np.ndarray: 质量数组
        """
        m_min, m_max = mass_range
        
        # 使用反变换采样
        r = np.random.random(n)
        
        # 积分IMF
        if alpha != 1:
            masses = (m_min ** (1 - alpha) + r * (m_max ** (1 - alpha) - m_min ** (1 - alpha))) ** (1 / (1 - alpha))
        else:
            masses = m_min * (m_max / m_min) ** r
        
        return masses
