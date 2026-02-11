"""
Motion Control - 运动控制引擎

提供机器人运动控制功能
"""
import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class MotionType(str, Enum):
    """运动类型"""
    JOINT = "joint"  # 关节空间
    CARTESIAN = "cartesian"  # 笛卡尔空间
    LINEAR = "linear"  # 直线
    CIRCULAR = "circular"  # 圆弧
    P2P = "point_to_point"  # 点到点


class MotionState(str, Enum):
    """运动状态"""
    IDLE = "idle"
    MOVING = "moving"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MotionCommand:
    """运动命令"""
    device_id: str
    motion_type: MotionType
    target: Dict[str, float]  # 目标位置/姿态
    velocity: float = 0.5
    acceleration: float = 0.5
    blocking: bool = False
    priority: int = 0


@dataclass
class MotionProfile:
    """运动参数配置"""
    max_velocity: float = 1.0
    max_acceleration: float = 1.0
    max_jerk: float = 10.0  # 加加速度限制
    velocity_ratio: float = 1.0
    acceleration_ratio: float = 1.0


class MotionController:
    """
    运动控制器
    
    功能:
    - 运动命令队列管理
    - 运动规划与执行
    - 运动状态监控
    - 轨迹插补
    """
    
    def __init__(
        self,
        simulation_mode: bool = True,
        default_profile: MotionProfile = None
    ):
        """
        初始化运动控制器
        
        Args:
            simulation_mode: 模拟模式
            default_profile: 默认运动参数
        """
        self.simulation_mode = simulation_mode
        self.default_profile = default_profile or MotionProfile()
        
        self._command_queues: Dict[str, asyncio.Queue] = {}
        self._motion_states: Dict[str, MotionState] = {}
        self._motion_profiles: Dict[str, MotionProfile] = {}
        self._active_motions: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
        
        logger.info("MotionController initialized")
    
    async def submit_motion(
        self,
        device_id: str,
        command: MotionCommand
    ) -> str:
        """
        提交运动命令
        
        Args:
            device_id: 设备ID
            command: 运动命令
            
        Returns:
            运动ID
        """
        import uuid
        motion_id = f"motion_{uuid.uuid4().hex[:8]}"
        
        async with self._lock:
            # 初始化设备队列
            if device_id not in self._command_queues:
                self._command_queues[device_id] = asyncio.Queue()
                self._motion_states[device_id] = MotionState.IDLE
            
            # 设置运动状态
            self._motion_states[device_id] = MotionState.MOVING
            
            # 存储运动信息
            self._active_motions[motion_id] = {
                "device_id": device_id,
                "command": command,
                "start_time": datetime.utcnow().isoformat(),
                "progress": 0.0
            }
            
            # 放入队列
            await self._command_queues[device_id].put((motion_id, command))
            
            logger.info(f"Motion submitted: {motion_id} for {device_id}")
            
            return motion_id
    
    async def execute_motion(
        self,
        device_id: str,
        command: MotionCommand
    ) -> Dict[str, Any]:
        """
        执行单个运动
        
        Args:
            device_id: 设备ID
            command: 运动命令
            
        Returns:
            执行结果
        """
        try:
            if command.blocking:
                # 阻塞执行
                await self._perform_motion(device_id, command)
                return {"success": True, "status": "completed"}
            else:
                # 非阻塞执行
                asyncio.create_task(self._perform_motion(device_id, command))
                return {"success": True, "status": "started"}
                
        except Exception as e:
            logger.error(f"Motion execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _perform_motion(
        self,
        device_id: str,
        command: MotionCommand
    ):
        """执行运动（内部方法）"""
        profile = self._motion_profiles.get(device_id, self.default_profile)
        
        # 根据运动类型执行
        if command.motion_type == MotionType.JOINT:
            await self._joint_motion(device_id, command, profile)
        elif command.motion_type == MotionType.CARTESIAN:
            await self._cartesian_motion(device_id, command, profile)
        elif command.motion_type == MotionType.LINEAR:
            await self._linear_motion(device_id, command, profile)
        else:
            # 默认P2P运动
            await self._joint_motion(device_id, command, profile)
        
        # 更新状态
        async with self._lock:
            self._motion_states[device_id] = MotionState.COMPLETED
    
    async def _joint_motion(
        self,
        device_id: str,
        command: MotionCommand,
        profile: MotionProfile
    ):
        """关节空间运动"""
        target = command.target
        velocity = command.velocity * profile.max_velocity * profile.velocity_ratio
        
        # 模拟运动过程
        steps = 20
        for i in range(steps):
            if not await self._check_motion_active(device_id):
                return
            
            progress = (i + 1) / steps
            await asyncio.sleep(velocity * 0.1)
        
        logger.debug(f"Joint motion completed: {device_id} -> {target}")
    
    async def _cartesian_motion(
        self,
        device_id: str,
        command: MotionCommand,
        profile: MotionProfile
    ):
        """笛卡尔空间运动"""
        target = command.target
        velocity = command.velocity * profile.max_velocity * profile.velocity_ratio
        
        steps = 30
        for i in range(steps):
            if not await self._check_motion_active(device_id):
                return
            
            await asyncio.sleep(velocity * 0.05)
        
        logger.debug(f"Cartesian motion completed: {device_id}")
    
    async def _linear_motion(
        self,
        device_id: str,
        command: MotionCommand,
        profile: MotionProfile
    ):
        """直线插补运动"""
        target = command.target
        velocity = command.velocity * profile.max_velocity * profile.velocity_ratio
        
        # 生成插值点
        num_points = 50
        
        for i in range(num_points):
            if not await self._check_motion_active(device_id):
                return
            
            await asyncio.sleep(velocity * 0.02)
        
        logger.debug(f"Linear motion completed: {device_id}")
    
    async def _check_motion_active(self, device_id: str) -> bool:
        """检查运动是否还在进行"""
        state = self._motion_states.get(device_id, MotionState.IDLE)
        return state in [MotionState.MOVING, MotionState.PAUSED]
    
    async def pause_motion(self, device_id: str) -> bool:
        """
        暂停运动
        
        Args:
            device_id: 设备ID
            
        Returns:
            是否成功
        """
        if device_id in self._motion_states:
            self._motion_states[device_id] = MotionState.PAUSED
            logger.info(f"Motion paused: {device_id}")
            return True
        return False
    
    async def resume_motion(self, device_id: str) -> bool:
        """
        恢复运动
        
        Args:
            device_id: 设备ID
            
        Returns:
            是否成功
        """
        if device_id in self._motion_states:
            self._motion_states[device_id] = MotionState.MOVING
            logger.info(f"Motion resumed: {device_id}")
            return True
        return False
    
    async def stop_motion(self, device_id: str) -> bool:
        """
        停止运动
        
        Args:
            device_id: 设备ID
            
        Returns:
            是否成功
        """
        if device_id in self._motion_states:
            self._motion_states[device_id] = MotionState.IDLE
            
            # 清空队列
            if device_id in self._command_queues:
                queue = self._command_queues[device_id]
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            logger.info(f"Motion stopped: {device_id}")
            return True
        return False
    
    async def get_motion_state(self, device_id: str) -> Dict[str, Any]:
        """
        获取运动状态
        
        Args:
            device_id: 设备ID
            
        Returns:
            状态信息
        """
        state = self._motion_states.get(device_id, MotionState.IDLE)
        
        # 统计队列中的运动
        queue_size = 0
        if device_id in self._command_queues:
            queue_size = self._command_queues[device_id].qsize()
        
        return {
            "device_id": device_id,
            "state": state.value,
            "pending_motions": queue_size,
            "profile": self._motion_profiles.get(device_id, self.default_profile).__dict__
        }
    
    def set_motion_profile(self, device_id: str, profile: MotionProfile):
        """
        设置运动参数
        
        Args:
            device_id: 设备ID
            profile: 运动参数
        """
        self._motion_profiles[device_id] = profile
        logger.info(f"Motion profile set: {device_id}")
    
    async def get_all_states(self) -> Dict[str, Dict]:
        """获取所有设备运动状态"""
        return {
            device_id: await self.get_motion_state(device_id)
            for device_id in self._motion_states
        }
