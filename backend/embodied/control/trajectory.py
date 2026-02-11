"""
Trajectory - 轨迹规划

提供轨迹生成、插补和执行功能
"""
import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TrajectoryType(str, Enum):
    """轨迹类型"""
    LINEAR = "linear"  # 直线
    CIRCULAR = "circular"  # 圆弧
    JOINT = "joint"  # 关节空间
    B_SPLINE = "b_spline"  # B样条
    POLYNOMIAL = "polynomial"  # 多项式
    CUSTOM = "custom"


class InterpolationType(str, Enum):
    """插补类型"""
    TRAPEZOIDAL = "trapezoidal"  # 梯形
    S_CURVE = "s_curve"  # S曲线
    QUINTIC = "quintic"  # 五次多项式
    SPLINE = "spline"  # 样条插补


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: Dict[str, float]  # 位置 {x, y, z} 或 {joint1, joint2, ...}
    velocity: float = 0.0  # 速度
    acceleration: float = 0.0  # 加速度
    time: float = 0.0  # 时间戳
    orientation: Dict[str, float] = field(default_factory=dict)  # 姿态
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class Trajectory:
    """轨迹"""
    id: str = field(default_factory=lambda: f"traj_{uuid.uuid4().hex[:8]}")
    device_id: str = ""
    trajectory_type: TrajectoryType = TrajectoryType.LINEAR
    points: List[TrajectoryPoint] = field(default_factory=list)
    total_time: float = 0.0
    total_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CartesianPose:
    """笛卡尔空间位姿"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CartesianPose':
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            roll=data.get("roll", 0.0),
            pitch=data.get("pitch", 0.0),
            yaw=data.get("yaw", 0.0)
        )


class TrajectoryPlanner:
    """
    轨迹规划器
    
    功能:
    - 轨迹生成
    - 插值计算
    - 轨迹平滑
    - 时间最优规划
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        初始化轨迹规划器
        
        Args:
            simulation_mode: 模拟模式
        """
        self.simulation_mode = simulation_mode
        
        self._active_trajectories: Dict[str, Dict] = {}
        self._trajectory_history: List[Trajectory] = []
        self._execution_callbacks: List[Callable] = []
        self._lock = asyncio.Lock()
        
        # 配置参数
        self._default_max_velocity = 1.0
        self._default_max_acceleration = 2.0
        self._default_dt = 0.01  # 10ms采样周期
        
        logger.info("TrajectoryPlanner initialized")
    
    async def create_linear_trajectory(
        self,
        device_id: str,
        start_pose: Dict[str, float],
        end_pose: Dict[str, float],
        num_points: int = 100,
        max_velocity: float = None,
        interpolation: InterpolationType = InterpolationType.TRAPEZOIDAL
    ) -> Trajectory:
        """
        创建直线轨迹
        
        Args:
            device_id: 设备ID
            start_pose: 起始位置
            end_pose: 结束位置
            num_points: 轨迹点数
            max_velocity: 最大速度
            interpolation: 插补类型
            
        Returns:
            轨迹对象
        """
        points = []
        total_distance = 0.0
        
        # 计算距离
        if "x" in start_pose and "x" in end_pose:
            total_distance = math.sqrt(
                sum((end_pose[k] - start_pose[k]) ** 2 for k in ["x", "y", "z"])
            )
        else:
            total_distance = sum(abs(end_pose.get(k, 0) - start_pose.get(k, 0)) for k in start_pose)
        
        # 生成插值点
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # 位置插值
            position = {}
            for key in start_pose:
                position[key] = start_pose[key] + (end_pose.get(key, 0) - start_pose[key]) * t
            
            # 计算速度和加速度（基于插补类型）
            velocity, acceleration = self._compute_velocity_acceleration(
                t, max_velocity or self._default_max_velocity,
                interpolation
            )
            
            point = TrajectoryPoint(
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                time=t * (total_distance / (max_velocity or self._default_max_velocity))
            )
            points.append(point)
        
        # 总时间
        total_time = points[-1].time if points else 0.0
        
        trajectory = Trajectory(
            device_id=device_id,
            trajectory_type=TrajectoryType.LINEAR,
            points=points,
            total_time=total_time,
            total_distance=total_distance,
            metadata={
                "start_pose": start_pose,
                "end_pose": end_pose,
                "num_points": num_points,
                "interpolation": interpolation.value
            }
        )
        
        logger.info(f"Linear trajectory created: {trajectory.id} ({num_points} points)")
        return trajectory
    
    async def create_circular_trajectory(
        self,
        device_id: str,
        center: Dict[str, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        plane: str = "xy",
        num_points: int = 50,
        max_velocity: float = None
    ) -> Trajectory:
        """
        创建圆弧轨迹
        
        Args:
            device_id: 设备ID
            center: 圆心
            radius: 半径
            start_angle: 起始角度
            end_angle: 结束角度
            plane: 平面 (xy, yz, xz)
            num_points: 点数
            max_velocity: 最大速度
            
        Returns:
            轨迹对象
        """
        points = []
        angle_diff = end_angle - start_angle
        
        # 角度映射到平面坐标
        plane_axes = {"xy": ("x", "y"), "yz": ("y", "z"), "xz": ("x", "z")}
        axis1, axis2 = plane_axes.get(plane, ("x", "y"))
        
        for i in range(num_points):
            t = i / (num_points - 1)
            angle = start_angle + angle_diff * t
            
            position = dict(center)
            position[axis1] = center[axis1] + radius * math.cos(angle)
            position[axis2] = center[axis2] + radius * math.sin(angle)
            
            # 角速度
            velocity = (max_velocity or self._default_max_velocity) * (1 - t * 0.5)
            
            point = TrajectoryPoint(
                position=position,
                velocity=velocity,
                time=t * abs(angle_diff) * radius / (max_velocity or self._default_max_velocity)
            )
            points.append(point)
        
        # 总时间
        arc_length = abs(angle_diff) * radius
        total_time = arc_length / (max_velocity or self._default_max_velocity)
        
        trajectory = Trajectory(
            device_id=device_id,
            trajectory_type=TrajectoryType.CIRCULAR,
            points=points,
            total_time=total_time,
            total_distance=arc_length,
            metadata={
                "center": center,
                "radius": radius,
                "plane": plane
            }
        )
        
        logger.info(f"Circular trajectory created: {trajectory.id}")
        return trajectory
    
    async def create_joint_trajectory(
        self,
        device_id: str,
        start_joints: Dict[str, float],
        end_joints: Dict[str, float],
        num_points: int = 100,
        max_velocity: float = None
    ) -> Trajectory:
        """
        创建关节空间轨迹
        
        Args:
            device_id: 设备ID
            start_joints: 起始关节角度
            end_joints: 目标关节角度
            num_points: 点数
            max_velocity: 最大速度
            
        Returns:
            轨迹对象
        """
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # 平滑过渡（余弦插值）
            smooth_t = (1 - math.cos(t * math.pi)) / 2
            
            position = {}
            for key in start_joints:
                position[key] = start_joints[key] + (end_joints.get(key, 0) - start_joints[key]) * smooth_t
            
            velocity = (max_velocity or self._default_max_velocity) * math.sin(t * math.pi)
            
            point = TrajectoryPoint(
                position=position,
                velocity=velocity,
                time=t * 2.0  # 假设总时间2秒
            )
            points.append(point)
        
        trajectory = Trajectory(
            device_id=device_id,
            trajectory_type=TrajectoryType.JOINT,
            points=points,
            total_time=2.0,
            total_distance=0.0,
            metadata={"joints": list(start_joints.keys())}
        )
        
        logger.info(f"Joint trajectory created: {trajectory.id}")
        return trajectory
    
    async def concatenate_trajectories(
        self,
        trajectories: List[Trajectory]
    ) -> Trajectory:
        """
        连接轨迹
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            连接后的轨迹
        """
        if not trajectories:
            raise ValueError("No trajectories to concatenate")
        
        # 合并点
        all_points = []
        time_offset = 0.0
        
        for traj in trajectories:
            for point in traj.points:
                new_point = TrajectoryPoint(
                    position=point.position,
                    velocity=point.velocity,
                    acceleration=point.acceleration,
                    time=point.time + time_offset,
                    orientation=point.orientation,
                    metadata=point.metadata
                )
                all_points.append(new_point)
            
            time_offset += traj.total_time
        
        total_time = all_points[-1].time if all_points else 0.0
        
        return Trajectory(
            device_id=trajectories[0].device_id,
            trajectory_type=TrajectoryType.CUSTOM,
            points=all_points,
            total_time=total_time,
            metadata={"concatenated": len(trajectories)}
        )
    
    def get_point_at_time(
        self,
        trajectory: Trajectory,
        time: float
    ) -> Optional[TrajectoryPoint]:
        """
        获取指定时间的轨迹点
        
        Args:
            trajectory: 轨迹
            time: 时间
            
        Returns:
            轨迹点
        """
        if not trajectory.points:
            return None
        
        # 边界检查
        time = max(0, min(time, trajectory.total_time))
        
        # 查找对应的点
        for point in trajectory.points:
            if point.time >= time:
                return point
        
        return trajectory.points[-1]
    
    def get_pose_at_time(
        self,
        trajectory: Trajectory,
        time: float
    ) -> Dict[str, float]:
        """
        获取指定时间的位置
        
        Args:
            trajectory: 轨迹
            time: 时间
            
        Returns:
            位置字典
        """
        point = self.get_point_at_time(trajectory, time)
        return point.position if point else {}
    
    def resample_trajectory(
        self,
        trajectory: Trajectory,
        new_dt: float
    ) -> Trajectory:
        """
        重采样轨迹
        
        Args:
            trajectory: 原始轨迹
            new_dt: 新的采样周期
            
        Returns:
            重采样后的轨迹
        """
        num_points = max(2, int(trajectory.total_time / new_dt) + 1)
        
        new_points = []
        for i in range(num_points):
            t = i * new_dt
            point = self.get_point_at_time(trajectory, t)
            if point:
                new_points.append(point)
        
        trajectory.points = new_points
        trajectory.total_time = new_points[-1].time if new_points else 0.0
        
        return trajectory
    
    def smooth_trajectory(
        self,
        trajectory: Trajectory,
        window_size: int = 5
    ) -> Trajectory:
        """
        平滑轨迹（移动平均）
        
        Args:
            trajectory: 轨迹
            window_size: 窗口大小
            
        Returns:
            平滑后的轨迹
        """
        if len(trajectory.points) < window_size:
            return trajectory
        
        for i in range(len(trajectory.points)):
            start = max(0, i - window_size // 2)
            end = min(len(trajectory.points), i + window_size // 2 + 1)
            window = trajectory.points[start:end]
            
            # 对位置取平均
            if window and "x" in window[0].position:
                avg_x = sum(p.position["x"] for p in window) / len(window)
                avg_y = sum(p.position["y"] for p in window) / len(window)
                avg_z = sum(p.position["z"] for p in window) / len(window)
                trajectory.points[i].position = {"x": avg_x, "y": avg_y, "z": avg_z}
        
        return trajectory
    
    def _compute_velocity_acceleration(
        self,
        t: float,
        max_velocity: float,
        interpolation: InterpolationType
    ) -> Tuple[float, float]:
        """计算速度和加速度"""
        if interpolation == InterpolationType.TRAPEZOIDAL:
            # 梯形速度曲线
            if t < 0.25:
                velocity = 2 * max_velocity * t
                acceleration = 2 * max_velocity
            elif t < 0.75:
                velocity = max_velocity
                acceleration = 0
            else:
                velocity = 2 * max_velocity * (1 - t)
                acceleration = -2 * max_velocity
        elif interpolation == InterpolationType.S_CURVE:
            # S曲线
            if t < 0.125:
                velocity = 4 * max_velocity * t * t
                acceleration = 8 * max_velocity * t
            elif t < 0.375:
                velocity = max_velocity * (4 * t - 0.5)
                acceleration = 4 * max_velocity
            elif t < 0.625:
                velocity = max_velocity
                acceleration = 0
            elif t < 0.875:
                velocity = max_velocity * (3.5 - 4 * t)
                acceleration = -4 * max_velocity
            else:
                velocity = 4 * max_velocity * (1 - t) * (1 - t)
                acceleration = -8 * max_velocity * (1 - t)
        else:
            # 默认
            velocity = max_velocity
            acceleration = 0
        
        return velocity, acceleration
    
    def export_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """导出轨迹为字典"""
        return {
            "id": trajectory.id,
            "device_id": trajectory.device_id,
            "trajectory_type": trajectory.trajectory_type.value,
            "total_time": trajectory.total_time,
            "total_distance": trajectory.total_distance,
            "points": [
                {
                    "position": p.position,
                    "velocity": p.velocity,
                    "acceleration": p.acceleration,
                    "time": p.time
                }
                for p in trajectory.points
            ],
            "metadata": trajectory.metadata
        }
    
    def import_trajectory(self, data: Dict[str, Any]) -> Trajectory:
        """从字典导入轨迹"""
        points = [
            TrajectoryPoint(
                position=p["position"],
                velocity=p.get("velocity", 0),
                acceleration=p.get("acceleration", 0),
                time=p["time"]
            )
            for p in data.get("points", [])
        ]
        
        return Trajectory(
            id=data.get("id", f"traj_{uuid.uuid4().hex[:8]}"),
            device_id=data.get("device_id", ""),
            trajectory_type=TrajectoryType(data.get("trajectory_type", "linear")),
            points=points,
            total_time=data.get("total_time", 0),
            total_distance=data.get("total_distance", 0),
            metadata=data.get("metadata", {})
        )
