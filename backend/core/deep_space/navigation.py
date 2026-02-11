"""
深空导航系统 - Deep Space Navigation Module
============================================

轨道计算、路径规划、障碍规避和自主决策功能

作者: AI Platform Team
版本: 1.0.0
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import SystemConfig, PLANET_DATA, Planet

logger = logging.getLogger(__name__)

class RouteMethod(Enum):
    """航线规划方法"""
    OPTIMAL = "optimal"        # 最优（综合考虑时间和燃料）
    FASTEST = "fastest"        # 最快
    MOST_FUEL_EFFICIENT = "fuel"  # 最省燃料
    SHORTEST = "shortest"      # 最短距离
    SAFEST = "safest"          # 最安全

class ObstacleType(Enum):
    """障碍物类型"""
    ASTEROID = "asteroid"      # 小行星
    COMET = "comet"            # 彗星
    SPACE_DEBRIS = "debris"    # 太空碎片
    PLANET = "planet"          # 行星
    MOON = "moon"              # 卫星
    DUST_CLOUD = "dust"        # 尘埃云

@dataclass
class Position:
    """空间位置"""
    x: float  # X坐标 (AU)
    y: float  # Y坐标 (AU) 
    z: float  # Z坐标 (AU)
    
    def distance_to(self, other: 'Position') -> float:
        """计算到另一位置的距离"""
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )
    
    def to_vector(self) -> List[float]:
        """转换为向量"""
        return [self.x, self.y, self.z]

@dataclass 
class Velocity:
    """空间速度"""
    vx: float  # X方向速度 (AU/day)
    vy: float  # Y方向速度 (AU/day)
    vz: float  # Z方向速度 (AU/day)

@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: Position
    velocity: Velocity
    time: float  # 时间 (days from start)
    fuel_consumed: float  # 消耗燃料 (kg)
    status: str = "normal"  # 状态

@dataclass
class Obstacle:
    """障碍物"""
    name: str
    position: Position
    radius: float  # 半径 (km)
    type: ObstacleType
    velocity: Optional[Velocity] = None
    collision_probability: float = 0.0

@dataclass
class RoutePlan:
    """航线计划"""
    origin: str
    destination: str
    method: RouteMethod
    total_distance: float  # AU
    total_time: float     # days
    fuel_required: float  # kg
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    maneuvers: List[Dict] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    confidence: float = 0.0
    estimated_success_rate: float = 0.0


class OrbitalCalculator:
    """轨道计算器"""
    
    # 太阳系常数
    AU = 1.495978707e8  # 天文单位 (km)
    G = 6.67430e-11     # 万有引力常数 (m³/kg/s²)
    SUN_MASS = 1.989e30  # 太阳质量 (kg)
    MU = 1.32712440018e11  # 太阳引力参数 (km³/s²)
    
    @staticmethod
    def kepler_orbital_elements(
        semi_major_axis: float,
        eccentricity: float,
        inclination: float,
        longitude_ascending: float,
        argument_periapsis: float,
        true_anomaly: float
    ) -> Dict:
        """计算开普勒轨道参数"""
        return {
            'a': semi_major_axis,           # 半长轴 (AU)
            'e': eccentricity,               # 离心率
            'i': inclination,                # 倾角 (度)
            'Ω': longitude_ascending,        # 升交点经度 (度)
            'ω': argument_periapsis,         # 近地点幅角 (度)
            'ν': true_anomaly               # 真近点角 (度)
        }
    
    @staticmethod
    def orbital_period(semi_major_axis: float) -> float:
        """计算轨道周期（使用开普勒第三定律）"""
        # T² ∝ a³, T = 2π√(a³/μ)
        a_km = semi_major_axis * OrbitalCalculator.AU
        period = 2 * math.pi * math.sqrt(a_km**3 / OrbitalCalculator.MU)
        return period / 86400  # 转换为天
    
    @staticmethod
    def position_at_anomaly(
        orbital_elements: Dict,
        anomaly: float
    ) -> Position:
        """根据近点角计算位置"""
        a = orbital_elements['a']
        e = orbital_elements['e']
        i = math.radians(orbital_elements['i'])
        Ω = math.radians(orbital_elements['Ω'])
        ω = math.radians(orbital_elements['ω'])
        ν = math.radians(anomaly)
        
        # 轨道平面坐标
        r = a * (1 - e**2) / (1 + e * math.cos(ν))
        x_orb = r * math.cos(ν)
        y_orb = r * math.sin(ν)
        
        # 转换到黄道面
        cos_Ω, sin_Ω = math.cos(Ω), math.sin(Ω)
        cos_ω, sin_ω = math.cos(ω), math.sin(ω)
        cos_i, sin_i = math.cos(i), math.sin(i)
        
        x = (cos_Ω * cos_ω - sin_Ω * sin_ω * cos_i) * x_orb + \
            (-cos_Ω * sin_ω - sin_Ω * cos_ω * cos_i) * y_orb
        y = (sin_Ω * cos_ω + cos_Ω * sin_ω * cos_i) * x_orb + \
            (-sin_Ω * sin_ω + cos_Ω * cos_ω * cos_i) * y_orb
        z = (sin_ω * sin_i) * x_orb + (cos_ω * sin_i) * y_orb
        
        return Position(x/a, y/a, z/a)  # 返回AU单位
    
    @staticmethod
    def hohmann_transfer(
        r1: float,  # 起始轨道半径 (AU)
        r2: float   # 目标轨道半径 (AU)
    ) -> Dict:
        """霍曼转移轨道计算"""
        mu = OrbitalCalculator.MU / 1e9  # 转换为km³/s²
        
        # 速度增量
        v1 = math.sqrt(mu / (r1 * 1e9)) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
        v2 = math.sqrt(mu / (r2 * 1e9)) * (1 - math.sqrt(2 * r1 / (r1 + r2)))
        
        # 转移时间
        transfer_time = math.pi * math.sqrt((r1 + r2)**3 / (8 * mu)) / 86400
        
        return {
            'delta_v1': v1,  # km/s
            'delta_v2': v2,  # km/s
            'total_delta_v': abs(v1) + abs(v2),
            'transfer_time': transfer_time,  # days
            'semi_major_axis': (r1 + r2) / 2
        }
    
    @staticmethod
    def lambert_problem(
        r1: Position,
        r2: Position,
        dt: float,
        mu: float = 1.32712440018e11
    ) -> Velocity:
        """兰伯特问题求解 - 计算两点间的速度"""
        # 简化实现
        delta_r = r2.distance_to(r1)
        avg_velocity = delta_r * OrbitalCalculator.AU / (dt * 86400)
        
        return Velocity(
            vx=(r2.x - r1.x) / dt,
            vy=(r2.y - r1.y) / dt,
            vz=(r2.z - r1.z) / dt
        )


class ObstacleAvoider:
    """障碍规避器"""
    
    def __init__(self, config: 'NavigationConfig' = None):
        self.config = config or NavigationConfig()
        self.known_obstacles: List[Obstacle] = []
    
    def add_obstacle(self, obstacle: Obstacle):
        """添加障碍物"""
        self.known_obstacles.append(obstacle)
    
    def detect_collision(
        self,
        trajectory: List[TrajectoryPoint],
        spacecraft_radius: float = 1.0
    ) -> List[Dict]:
        """检测碰撞风险"""
        collisions = []
        
        for i, point in enumerate(trajectory):
            for obstacle in self.known_obstacles:
                distance = point.position.distance_to(obstacle.position)
                # 转换为km进行比较
                distance_km = distance * OrbitalCalculator.AU
                
                if distance_km < (obstacle.radius + spacecraft_radius * 1000):
                    collisions.append({
                        'time': point.time,
                        'obstacle': obstacle.name,
                        'distance': distance_km,
                        'severity': 'high' if distance_km < obstacle.radius else 'medium'
                    })
        
        return collisions
    
    def plan_avoidance_maneuver(
        self,
        collision: Dict,
        current_velocity: Velocity
    ) -> Dict:
        """规划规避机动"""
        # 简化实现：计算偏转速度增量
        collision_time = collision['time']
        obstacle_pos = None  # 需要从collision获取
        
        # 规避机动参数
        avoidance_delta_v = 0.1  # km/s
        avoidance_duration = 3600  # 秒
        
        return {
            'maneuver_type': 'avoidance',
            'delta_v': avoidance_delta_v,
            'duration': avoidance_duration,
            'timing': f"t-{avoidance_duration}s",
            'new_trajectory': []
        }
    
    def assess_route_safety(
        self,
        trajectory: List[TrajectoryPoint]
    ) -> float:
        """评估航线安全性"""
        collisions = self.detect_collision(trajectory)
        
        if len(collisions) == 0:
            return 1.0
        elif len(collisions) > 10:
            return 0.0
        else:
            return max(0.0, 1.0 - len(collisions) * 0.1)


class DeepSpaceNavigator:
    """深空导航器主类"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or get_config()
        self.navigation_config = self.config.navigation
        self.orbital_calculator = OrbitalCalculator()
        self.obstacle_avoider = ObstacleAvoider(self.navigation_config)
        self.current_position: Optional[Position] = None
        self.current_velocity: Optional[Velocity] = None
        
        logger.info("深空导航系统初始化完成")
    
    def set_position(self, position: Position, velocity: Velocity):
        """设置当前位置和速度"""
        self.current_position = position
        self.current_velocity = velocity
    
    def plan_route(
        self,
        origin: str,
        destination: str,
        method: str = "optimal",
        departure_time: float = 0.0
    ) -> RoutePlan:
        """
        规划深空航线
        
        Args:
            origin: 出发天体名称
            destination: 目标天体名称
            method: 规划方法 (optimal/fastest/fuel/shortest/safest)
            departure_time: 出发时间 (MJD)
        
        Returns:
            RoutePlan: 航线计划
        """
        method_enum = RouteMethod(method)
        
        # 获取行星轨道参数
        origin_data = PLANET_DATA.get(origin.lower())
        dest_data = PLANET_DATA.get(destination.lower())
        
        if not origin_data or not dest_data:
            raise ValueError(f"未知的天体: {origin} 或 {destination}")
        
        # 计算基本轨道参数
        r1 = origin_data['orbital_period'] / 365.25  # 转换为地球年
        r2 = dest_data['orbital_period'] / 365.25
        
        # 霍曼转移计算
        hohmann = OrbitalCalculator.hohmann_transfer(r1, r2)
        
        # 根据方法调整参数
        if method_enum == RouteMethod.FASTEST:
            # 考虑更快的转移选项（更高能量）
            total_distance = hohmann['transfer_time'] * 0.8
            fuel_factor = 1.5
        elif method_enum == RouteMethod.MOST_FUEL_EFFICIENT:
            # 最省燃料的选项
            total_distance = hohmann['transfer_time'] * 1.2
            fuel_factor = 0.8
        else:  # OPTIMAL
            total_distance = hohmann['transfer_time']
            fuel_factor = 1.0
        
        # 生成轨迹点
        trajectory = self._generate_trajectory(
            origin, destination, hohmann, total_distance
        )
        
        # 障碍检测和规避
        self._update_obstacles(origin, destination)
        collisions = self.obstacle_avoider.detect_collision(trajectory)
        
        # 计算燃料需求（简化模型）
        fuel_required = self._estimate_fuel(hohmann['total_delta_v'])
        
        # 评估安全性
        safety_score = self.obstacle_avoider.assess_route_safety(trajectory)
        
        # 计算成功率
        success_rate = 0.99 if safety_score > 0.9 else 0.95
        
        return RoutePlan(
            origin=origin,
            destination=destination,
            method=method_enum,
            total_distance=hohmann['semi_major_axis'],
            total_time=total_distance,
            fuel_required=fuel_required,
            trajectory=trajectory,
            maneuvers=self._generate_maneuvers(hohmann),
            obstacles=collisions,
            confidence=0.98,
            estimated_success_rate=success_rate
        )
    
    def _generate_trajectory(
        self,
        origin: str,
        destination: str,
        hohmann: Dict,
        total_time: float
    ) -> List[TrajectoryPoint]:
        """生成详细轨迹"""
        trajectory = []
        steps = self.navigation_config.trajectory_steps
        dt = total_time / steps
        
        for i in range(steps + 1):
            t = i * dt
            
            # 椭圆轨道参数
            a = hohmann['semi_major_axis']
            e = 0.3  # 简化转移轨道离心率
            ν = math.pi * i / steps  # 真近点角从0到π
            
            orbital_elements = OrbitalCalculator.kepler_orbital_elements(
                semi_major_axis=a,
                eccentricity=e,
                inclination=0,
                longitude_ascending=0,
                argument_periapsis=0,
                true_anomaly=ν
            )
            
            position = OrbitalCalculator.position_at_anomaly(orbital_elements, ν)
            
            trajectory.append(TrajectoryPoint(
                position=position,
                velocity=Velocity(0, 0, 0),  # 简化
                time=t,
                fuel_consumed=0
            ))
        
        return trajectory
    
    def _generate_maneuvers(self, hohmann: Dict) -> List[Dict]:
        """生成机动计划"""
        maneuvers = [
            {
                'name': '轨道插入点火',
                'time': 0,
                'delta_v': hohmann['delta_v1'],
                'duration': 600,
                'description': '从地球轨道出发，进入霍曼转移轨道'
            },
            {
                'name': '轨道捕获点火', 
                'time': hohmann['transfer_time'],
                'delta_v': hohmann['delta_v2'],
                'duration': 600,
                'description': '进入目标天体轨道'
            }
        ]
        return maneuvers
    
    def _update_obstacles(self, origin: str, destination: str):
        """更新已知障碍物"""
        # 添加一些典型的太阳系障碍物
        self.obstacle_avoider.add_obstacle(Obstacle(
            name="小行星带-谷神星",
            position=Position(2.77, 0.5, 0.1),
            radius=473,
            type=ObstacleType.ASTEROID
        ))
        
        self.obstacle_avoider.add_obstacle(Obstacle(
            name="小行星带-灶神星",
            position=Position(2.36, -0.8, 0.2),
            radius=262,
            type=ObstacleType.ASTEROID
        ))
    
    def _estimate_fuel(self, delta_v: float) -> float:
        """估算燃料需求（使用火箭方程）"""
        # Tsiolkovsky火箭方程简化估算
        isp = 450  # 秒 (典型的化学推进器)
        mass_ratio = math.exp(delta_v * 1000 / (isp * 9.81))
        dry_mass = 1000  # kg
        
        return mass_ratio * dry_mass  # kg
    
    def execute_autonomous_decision(
        self,
        situation: Dict,
        available_actions: List[Dict]
    ) -> Dict:
        """执行自主决策"""
        # 简化决策逻辑
        if situation.get('emergency'):
            # 紧急情况：选择最安全的选项
            safest_action = min(available_actions, 
                              key=lambda x: x.get('risk', 1.0))
            return {
                'action': safest_action['name'],
                'confidence': 0.95,
                'reasoning': '紧急避障'
            }
        else:
            # 正常情况：选择最优选项
            best_action = max(available_actions,
                             key=lambda x: x.get('efficiency', 0))
            return {
                'action': best_action['name'],
                'confidence': 0.90,
                'reasoning': '常规优化'
            }
    
    def getNavigationStatus(self) -> Dict:
        """获取导航状态"""
        return {
            'position': self.current_position.to_vector() if self.current_position else None,
            'velocity': [self.current_velocity.vx, self.current_velocity.vy, self.current_velocity.vz] if self.current_velocity else None,
            'known_obstacles': len(self.obstacle_avoider.known_obstacles),
            'system_ready': True
        }


class TrajectoryPlanner:
    """轨迹规划器"""
    
    def __init__(self, navigator: DeepSpaceNavigator = None):
        self.navigator = navigator or DeepSpaceNavigator()
    
    def optimize_trajectory(
        self,
        trajectory: List[TrajectoryPoint],
        objective: str = "time"
    ) -> List[TrajectoryPoint]:
        """优化轨迹"""
        # 简化实现：返回原始轨迹
        return trajectory
    
    def calculate_waypoints(
        self,
        origin: Position,
        destination: Position,
        num_waypoints: int = 5
    ) -> List[Position]:
        """计算航路点"""
        waypoints = []
        for i in range(num_waypoints + 2):
            t = i / (num_waypoints + 1)
            waypoints.append(Position(
                x=origin.x + t * (destination.x - origin.x),
                y=origin.y + t * (destination.y - origin.y),
                z=origin.z + t * (destination.z - origin.z)
            ))
        return waypoints[1:-1]


# 导入必要的配置
from .config import get_config, NavigationConfig
