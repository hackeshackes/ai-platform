"""
行星探测系统 - Planet Exploration Module
==========================================

地形分析、资源识别、着陆点选择和样本采集功能

作者: AI Platform Team
版本: 1.0.0
"""

import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import SystemConfig, PLANET_DATA, get_config

logger = logging.getLogger(__name__)

class TerrainType(Enum):
    """地形类型"""
    PLAIN = "plain"            # 平原
    MOUNTAIN = "mountain"       # 山地
    CRATER = "crater"           # 陨石坑
    RIDGE = "ridge"             # 山脊
    VALLEY = "valley"           # 山谷
    DUNE = "dune"               # 沙丘
    ROCKY = "rocky"             # 岩石区
    ICE = "ice"                 # 冰层
    LAVA = "lava"               # 熔岩
    REGOLITH = "regolith"       # 风化层
    CLIFF = "cliff"             # 悬崖
    CANYON = "canyon"           # 峡谷

class ResourceType(Enum):
    """资源类型"""
    WATER_ICE = "water_ice"         # 水冰
    IRON = "iron"                   # 铁
    ALUMINUM = "aluminum"           # 铝
    SILICON = "silicon"             # 硅
    OXYGEN = "oxygen"               # 氧
    HYDROGEN = "hydrogen"           # 氢
    CARBON = "carbon"               # 碳
    NITROGEN = "nitrogen"           # 氮
    RARE_EARTH = "rare_earth"        # 稀土元素
    PRECIOUS_METALS = "precious"    # 贵金属
    HELIUM_3 = "helium_3"            # 氦-3
    FUEL = "fuel"                   # 燃料

class LandingSiteQuality(Enum):
    """着陆点质量等级"""
    EXCELLENT = "excellent"     # 优秀
    GOOD = "good"              # 良好
    ACCEPTABLE = "acceptable"  # 可接受
    POOR = "poor"              # 较差
    UNSUITABLE = "unsuitable"  # 不适合

@dataclass
class TerrainData:
    """地形数据"""
    terrain_type: TerrainType
    elevation: float           # 海拔 (m)
    slope: float              # 坡度 (度)
    roughness: float          # 粗糙度 (0-1)
    rock_density: float        # 岩石密度 (个/m²)
    thermal_stability: float   # 热稳定性 (0-1)
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'terrain_type': self.terrain_type.value,
            'elevation': self.elevation,
            'slope': self.slope,
            'roughness': self.roughness,
            'rock_density': self.rock_density,
            'thermal_stability': self.thermal_stability,
            'description': self.description
        }

@dataclass
class ResourceDeposit:
    """资源矿床"""
    resource_type: ResourceType
    concentration: float       # 浓度 (0-1)
    depth: float              # 深度 (m)
    estimated_amount: float   # 估计储量 (tons)
    accessibility: float       # 可达性 (0-1)
    confidence: float          # 置信度 (0-1)
    location: Tuple[float, float]  # (纬度, 经度)
    
    def to_dict(self) -> Dict:
        return {
            'resource_type': self.resource_type.value,
            'concentration': self.concentration,
            'depth': self.depth,
            'estimated_amount': self.estimated_amount,
            'accessibility': self.accessibility,
            'confidence': self.confidence,
            'location': self.location
        }

@dataclass
class LandingSite:
    """着陆点"""
    name: str
    latitude: float            # 纬度
    longitude: float           # 经度
    elevation: float           # 海拔 (m)
    terrain: TerrainData
    quality: LandingSiteQuality
    safety_score: float        # 安全分数 (0-1)
    area: float               # 面积 (m²)
    distance_from_target: float  # 距目标距离 (km)
    recommended_approach: str   # 推荐进场方向
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'elevation': self.elevation,
            'terrain': self.terrain.to_dict(),
            'quality': self.quality.value,
            'safety_score': self.safety_score,
            'area': self.area,
            'distance_from_target': self.distance_from_target,
            'recommended_approach': self.recommended_approach
        }

@dataclass
class Sample:
    """采集样本"""
    sample_id: str
    location: Tuple[float, float]
    depth: float              # 采集深度 (m)
    mass: float               # 质量 (g)
    composition: Dict         # 成分分析
    collection_time: str       # 采集时间
    preservation_status: str   # 保存状态
    
    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'location': self.location,
            'depth': self.depth,
            'mass': self.mass,
            'composition': self.composition,
            'collection_time': self.collection_time,
            'preservation_status': self.preservation_status
        }


class TerrainAnalyzer:
    """地形分析器"""
    
    def __init__(self, planet: str, config: 'ExplorationConfig' = None):
        self.planet = planet.lower()
        self.config = config or get_config().exploration
        self.planet_data = PLANET_DATA.get(self.planet)
        
        if not self.planet_data:
            raise ValueError(f"未知行星: {planet}")
    
    def analyze_surface(
        self,
        resolution: float = None,
        region: Tuple[float, float, float, float] = None
    ) -> List[TerrainData]:
        """
        分析表面地形
        
        Args:
            resolution: 分析分辨率 (m)
            region: 分析区域 (lat_min, lon_min, lat_max, lon_max)
        
        Returns:
            List[TerrainData]: 地形数据列表
        """
        resolution = resolution or self.config.terrain_resolution
        
        # 生成模拟地形数据
        terrain_samples = []
        
        terrain_types = list(TerrainType)
        
        # 使用均匀分布
        for i in range(20):  # 生成20个采样点
            terrain_type = random.choice(terrain_types)
            
            terrain_data = TerrainData(
                terrain_type=terrain_type,
                elevation=self._generate_elevation(terrain_type),
                slope=self._generate_slope(terrain_type),
                roughness=random.uniform(0.1, 0.6),
                rock_density=self._generate_rock_density(terrain_type),
                thermal_stability=random.uniform(0.7, 0.95),
                description=self._describe_terrain(terrain_type)
            )
            terrain_samples.append(terrain_data)
        
        return terrain_samples
    
    def generate_elevation_map(
        self,
        width: int = 100,
        height: int = 100,
        base_elevation: float = 0
    ) -> List[List[float]]:
        """生成高程图"""
        elevation_map = []
        for y in range(height):
            row = []
            for x in range(width):
                # 简单的噪声生成
                noise = math.sin(x * 0.1) * math.cos(y * 0.1) * 100
                row.append(base_elevation + noise)
            elevation_map.append(row)
        return elevation_map
    
    def detect_hazards(
        self,
        terrain: List[TerrainData]
    ) -> List[Dict]:
        """检测危险区域"""
        hazards = []
        
        for i, td in enumerate(terrain):
            if td.slope > 30:
                hazards.append({
                    'type': 'steep_slope',
                    'location': i,
                    'severity': 'high' if td.slope > 45 else 'medium',
                    'description': f'坡度过陡: {td.slope:.1f}°'
                })
            if td.rock_density > 0.5:
                hazards.append({
                    'type': 'high_rock_density',
                    'location': i,
                    'severity': 'medium',
                    'description': f'岩石密度过高: {td.rock_density:.2f}'
                })
            if td.thermal_stability < 0.5:
                hazards.append({
                    'type': 'thermal_instability',
                    'location': i,
                    'severity': 'high',
                    'description': f'热稳定性差: {td.thermal_stability:.2f}'
                })
        
        return hazards
    
    def _generate_elevation(self, terrain_type: TerrainType) -> float:
        """生成海拔"""
        base_ranges = {
            TerrainType.PLAIN: (0, 500),
            TerrainType.MOUNTAIN: (2000, 8000),
            TerrainType.CRATER: (-500, 200),
            TerrainType.RIDGE: (1000, 4000),
            TerrainType.VALLEY: (-1000, 500),
            TerrainType.DUNE: (0, 300),
            TerrainType.ROCKY: (0, 2000),
            TerrainType.ICE: (-2000, 0),
            TerrainType.LAVA: (0, 500),
            TerrainType.REGOLITH: (0, 1000),
            TerrainType.CLIFF: (500, 3000)
        }
        range_ = base_ranges.get(terrain_type, (0, 1000))
        return random.uniform(*range_)
    
    def _generate_slope(self, terrain_type: TerrainType) -> float:
        """生成坡度"""
        if terrain_type in [TerrainType.PLAIN, TerrainType.DUNE]:
            return random.uniform(0, 5)
        elif terrain_type in [TerrainType.MOUNTAIN, TerrainType.CLIFF, TerrainType.RIDGE]:
            return random.uniform(20, 50)
        elif terrain_type == TerrainType.VALLEY:
            return random.uniform(5, 20)
        else:
            return random.uniform(5, 15)
    
    def _generate_rock_density(self, terrain_type: TerrainType) -> float:
        """生成岩石密度"""
        if terrain_type == TerrainType.ROCKY:
            return random.uniform(0.4, 0.8)
        elif terrain_type == TerrainType.CRATER:
            return random.uniform(0.2, 0.5)
        else:
            return random.uniform(0.05, 0.3)
    
    def _describe_terrain(self, terrain_type: TerrainType) -> str:
        """生成地形描述"""
        descriptions = {
            TerrainType.PLAIN: "平坦开阔的平原地区",
            TerrainType.MOUNTAIN: "崎岖的山地地形",
            TerrainType.CRATER: "布满陨石坑的区域",
            TerrainType.RIDGE: "线性的山脊地形",
            TerrainType.VALLEY: "低洼的山谷地带",
            TerrainType.DUNE: "风成沙丘区域",
            TerrainType.ROCKY: "岩石密布的区域",
            TerrainType.ICE: "覆盖冰层的区域",
            TerrainType.LAVA: "熔岩流地形",
            TerrainType.REGOLITH: "风化层覆盖区域",
            TerrainType.CLIFF: "陡峭的悬崖地形"
        }
        return descriptions.get(terrain_type, "未知地形")


class ResourceIdentifier:
    """资源识别器"""
    
    RESOURCE_INDICATORS = {
        ResourceType.WATER_ICE: ['h2o', 'ice_signature', 'hydrogen'],
        ResourceType.IRON: ['fe', 'magnetic_anomaly', 'metallic'],
        ResourceType.SILICON: ['si', 'silicate', 'sand'],
        ResourceType.ALUMINUM: ['al', 'light_metal'],
        ResourceType.OXYGEN: ['o2', 'oxide'],
        ResourceType.HELIUM_3: ['he3', 'rare_isotope'],
    }
    
    def __init__(self, planet: str, config: 'ExplorationConfig' = None):
        self.planet = planet.lower()
        self.config = config or get_config().exploration
    
    def scan_for_resources(
        self,
        area: Tuple[float, float, float, float] = None,
        sensitivity: float = None
    ) -> List[ResourceDeposit]:
        """扫描资源"""
        sensitivity = sensitivity or self.config.resource_detection_sensitivity
        
        resources = []
        resource_types = list(ResourceType)
        
        # 为每个可能的资源类型生成检测结果
        for resource in resource_types:
            if random.random() < sensitivity:
                deposit = ResourceDeposit(
                    resource_type=resource,
                    concentration=random.uniform(0.1, 0.9),
                    depth=random.uniform(0.5, 50.0),
                    estimated_amount=random.uniform(1000, 1000000),
                    accessibility=random.uniform(0.5, 1.0),
                    confidence=random.uniform(0.7, 0.95),
                    location=(random.uniform(-90, 90), random.uniform(-180, 180))
                )
                resources.append(deposit)
        
        return resources
    
    def estimate_resource_value(
        self,
        deposit: ResourceDeposit,
        market_prices: Dict = None
    ) -> float:
        """估算资源价值"""
        base_prices = {
            ResourceType.WATER_ICE: 500,       # $/ton
            ResourceType.IRON: 100,            # $/ton
            ResourceType.SILICON: 200,          # $/ton
            ResourceType.ALUMINUM: 1500,       # $/ton
            ResourceType.HELIUM_3: 5000000,    # $/ton
            ResourceType.RARE_EARTH: 500000,   # $/ton
            ResourceType.PRECIOUS_METALS: 50000000  # $/ton
        }
        
        base_price = base_prices.get(deposit.resource_type, 1000)
        value = deposit.estimated_amount * base_price * deposit.concentration
        
        return value


class LandingSiteSelector:
    """着陆点选择器"""
    
    def __init__(self, planet: str, config: 'ExplorationConfig' = None):
        self.planet = planet.lower()
        self.config = config or get_config().exploration
    
    def select_site(
        self,
        terrain: List[TerrainData],
        target_location: Tuple[float, float] = None,
        constraints: Dict = None
    ) -> List[LandingSite]:
        """
        选择最佳着陆点
        
        Args:
            terrain: 地形数据列表
            target_location: 目标位置 (lat, lon)
            constraints: 约束条件
        
        Returns:
            List[LandingSite]: 推荐的着陆点列表
        """
        constraints = constraints or {}
        target = target_location or (0, 0)
        
        candidates = []
        
        for i, td in enumerate(terrain):
            site = self._evaluate_site(td, i, target)
            if site:
                candidates.append(site)
        
        # 按质量排序
        candidates.sort(key=lambda x: x.safety_score, reverse=True)
        
        # 返回前5个候选点
        return candidates[:5]
    
    def _evaluate_site(
        self,
        terrain: TerrainData,
        index: int,
        target: Tuple[float, float]
    ) -> Optional[LandingSite]:
        """评估单个着陆点"""
        # 检查是否满足基本约束
        if terrain.slope > self.config.max_slope_angle:
            return None
        if terrain.rock_density > self.config.rock_hazard_threshold:
            return None
        
        # 计算安全分数
        safety_score = self._calculate_safety_score(terrain)
        
        # 确定质量等级
        if safety_score > 0.9:
            quality = LandingSiteQuality.EXCELLENT
        elif safety_score > 0.75:
            quality = LandingSiteQuality.GOOD
        elif safety_score > 0.5:
            quality = LandingSiteQuality.ACCEPTABLE
        elif safety_score > 0.3:
            quality = LandingSiteQuality.POOR
        else:
            return None
        
        # 计算面积
        area = self._estimate_area(terrain)
        
        # 估算到目标的距离
        distance = random.uniform(1, 50)
        
        return LandingSite(
            name=f"着陆区-{index+1:02d}",
            latitude=random.uniform(-30, 30),
            longitude=random.uniform(-30, 30),
            elevation=terrain.elevation,
            terrain=terrain,
            quality=quality,
            safety_score=safety_score,
            area=area,
            distance_from_target=distance,
            recommended_approach=self._recommend_approach(terrain)
        )
    
    def _calculate_safety_score(self, terrain: TerrainData) -> float:
        """计算安全分数"""
        score = 1.0
        
        # 坡度惩罚
        slope_penalty = min(terrain.slope / self.config.max_slope_angle, 1.0)
        score -= slope_penalty * 0.3
        
        # 岩石密度惩罚
        rock_penalty = terrain.rock_density / self.config.rock_hazard_threshold
        score -= rock_penalty * 0.25
        
        # 热稳定性加成
        score += terrain.thermal_stability * 0.2
        
        # 粗糙度惩罚
        score -= terrain.roughness * 0.15
        
        return max(0, min(1, score))
    
    def _estimate_area(self, terrain: TerrainData) -> float:
        """估算面积"""
        base_area = self.config.min_landing_area
        if terrain.terrain_type == TerrainType.PLAIN:
            return base_area * random.uniform(1, 3)
        else:
            return base_area * random.uniform(0.5, 1.5)
    
    def _recommend_approach(self, terrain: TerrainData) -> str:
        """推荐进场方向"""
        approaches = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
        if terrain.slope < 10:
            return f"任意方向进场（平坦地形）"
        else:
            return f"建议{approaches[random.randint(0, 7)]}方向进场"


class SampleCollector:
    """样本采集器"""
    
    def __init__(self, planet: str, config: 'ExplorationConfig' = None):
        self.planet = planet.lower()
        self.config = config or get_config().exploration
        self.collected_samples: List[Sample] = []
        self.sample_counter = 0
    
    def collect_sample(
        self,
        location: Tuple[float, float],
        depth: float = None,
        mass_target: float = None
    ) -> Sample:
        """采集样本"""
        depth = depth or self.config.sample_depth
        mass_target = mass_target or 100  # g
        
        # 生成样本ID
        self.sample_counter += 1
        sample_id = f"{self.planet.upper()}-{self.sample_counter:04d}"
        
        # 生成成分分析
        composition = self._analyze_composition()
        
        sample = Sample(
            sample_id=sample_id,
            location=location,
            depth=depth,
            mass=random.uniform(mass_target * 0.9, mass_target * 1.1),
            composition=composition,
            collection_time=self._get_timestamp(),
            preservation_status="sealed" if self.config.contamination_prevention else "exposed"
        )
        
        self.collected_samples.append(sample)
        
        logger.info(f"样本采集完成: {sample_id}")
        
        return sample
    
    def collect_samples_at_location(
        self,
        location: Tuple[float, float],
        count: int = None
    ) -> List[Sample]:
        """在指定位置采集多个样本"""
        count = count or self.config.sample_count
        samples = []
        
        for i in range(count):
            # 在位置附近稍微偏移
            offset_location = (
                location[0] + random.uniform(-0.001, 0.001),
                location[1] + random.uniform(-0.001, 0.001)
            )
            depth = self.config.sample_depth * random.uniform(0.5, 1.5)
            sample = self.collect_sample(offset_location, depth)
            samples.append(sample)
        
        return samples
    
    def _analyze_composition(self) -> Dict:
        """分析成分"""
        return {
            'silicon_dioxide': random.uniform(40, 60),
            'iron_oxide': random.uniform(5, 15),
            'aluminum_oxide': random.uniform(5, 10),
            'calcium_oxide': random.uniform(3, 8),
            'water_content': random.uniform(0, 5),
            'other': random.uniform(10, 20)
        }
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def get_collection_report(self) -> Dict:
        """获取采集报告"""
        return {
            'total_samples': len(self.collected_samples),
            'total_mass': sum(s.mass for s in self.collected_samples),
            'preserved_samples': sum(1 for s in self.collected_samples 
                                     if s.preservation_status == "sealed"),
            'samples': [s.to_dict() for s in self.collected_samples]
        }


class PlanetExplorer:
    """行星探测主类"""
    
    def __init__(self, planet: str, config: SystemConfig = None):
        self.planet = planet.lower()
        self.config = config or get_config()
        self.planet_data = PLANET_DATA.get(self.planet)
        
        # 初始化子系统
        self.terrain_analyzer = TerrainAnalyzer(planet, self.config.exploration)
        self.resource_identifier = ResourceIdentifier(planet, self.config.exploration)
        self.landing_selector = LandingSiteSelector(planet, self.config.exploration)
        self.sample_collector = SampleCollector(planet, self.config.exploration)
        
        logger.info(f"行星探测器初始化完成: {self.planet}")
    
    def analyze_surface(
        self,
        resolution: float = None,
        region: Tuple[float, float, float, float] = None
    ) -> Dict:
        """分析表面"""
        terrain_data = self.terrain_analyzer.analyze_surface(resolution, region)
        hazards = self.terrain_analyzer.detect_hazards(terrain_data)
        
        return {
            'planet': self.planet,
            'analysis_time': self._get_timestamp(),
            'terrain_samples': [t.to_dict() for t in terrain_data],
            'terrain_types_found': list(set(t.terrain_type.value for t in terrain_data)),
            'hazards': hazards,
            'average_elevation': sum(t.elevation for t in terrain_data) / len(terrain_data),
            'hazard_count': len(hazards)
        }
    
    def scan_resources(
        self,
        area: Tuple[float, float, float, float] = None
    ) -> Dict:
        """扫描资源"""
        deposits = self.resource_identifier.scan_for_resources(area)
        
        total_value = sum(
            self.resource_identifier.estimate_resource_value(d) 
            for d in deposits
        )
        
        return {
            'planet': self.planet,
            'scan_time': self._get_timestamp(),
            'resources_found': [d.to_dict() for d in deposits],
            'resource_types': list(set(d.resource_type.value for d in deposits)),
            'total_estimated_value': total_value,
            'deposit_count': len(deposits)
        }
    
    def select_site(
        self,
        terrain: List[TerrainData] = None,
        target_location: Tuple[float, float] = None
    ) -> Dict:
        """选择最佳着陆点"""
        if terrain is None:
            terrain = self.terrain_analyzer.analyze_surface()
        
        candidates = self.landing_selector.select_site(terrain, target_location)
        
        return {
            'planet': self.planet,
            'selection_time': self._get_timestamp(),
            'candidates': [c.to_dict() for c in candidates],
            'recommended_site': candidates[0].to_dict() if candidates else None,
            'total_candidates': len(candidates)
        }
    
    def collect_samples(
        self,
        location: Tuple[float, float],
        count: int = None,
        target_depth: float = None
    ) -> Dict:
        """采集样本"""
        samples = self.sample_collector.collect_samples_at_location(
            location, count
        )
        
        return {
            'planet': self.planet,
            'collection_time': self._get_timestamp(),
            'samples': [s.to_dict() for s in samples],
            'total_samples': len(samples),
            'report': self.sample_collector.get_collection_report()
        }
    
    def full_exploration(
        self,
        target_location: Tuple[float, float] = None
    ) -> Dict:
        """完整探索任务"""
        logger.info(f"开始对 {self.planet} 的完整探索任务")
        
        # 1. 地形分析
        terrain_result = self.analyze_surface()
        
        # 2. 资源扫描
        resource_result = self.scan_resources()
        
        # 3. 着陆点选择
        site_result = self.select_site(
            terrain=[TerrainData(**t) for t in terrain_result['terrain_samples'][:10]],
            target_location=target_location
        )
        
        return {
            'mission_summary': {
                'planet': self.planet,
                'mission_type': 'full_exploration',
                'status': 'completed'
            },
            'terrain_analysis': terrain_result,
            'resource_survey': resource_result,
            'landing_site_selection': site_result,
            'next_recommended_action': 'landing' if site_result['recommended_site'] else 'further_reconnaissance'
        }
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# 导入必要的配置
from .config import get_config, ExplorationConfig
