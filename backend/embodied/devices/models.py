"""
Device Models - 设备模型定义

定义设备、控制命令等核心数据模型
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """设备类型枚举"""
    ROBOT = "ROBOT"
    SENSOR = "SENSOR"
    IOT = "IOT"
    CAMERA = "CAMERA"
    ACTUATOR = "ACTUATOR"
    CONTROLLER = "CONTROLLER"


class DeviceStatus(str, Enum):
    """设备状态枚举"""
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    BUSY = "BUSY"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"


class CommunicationProtocol(str, Enum):
    """通信协议枚举"""
    MQTT = "MQTT"
    REST = "REST"
    WEBSOCKET = "WS"
    ROS = "ROS"
    OPCUA = "OPCUA"
    MODBUS = "MODBUS"


class Device(BaseModel):
    """设备模型"""
    id: str = Field(..., description="设备唯一标识")
    name: str = Field(..., description="设备名称")
    type: DeviceType = Field(..., description="设备类型")
    protocol: CommunicationProtocol = Field(..., description="通信协议")
    status: DeviceStatus = Field(default=DeviceStatus.OFFLINE, description="设备状态")
    capabilities: List[str] = Field(default_factory=list, description="设备能力列表")
    config: Dict[str, Any] = Field(default_factory=dict, description="设备配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="设备元数据")
    last_seen: Optional[str] = Field(None, description="最后活动时间")


class DeviceCommand(BaseModel):
    """设备控制命令"""
    device_id: str = Field(..., description="目标设备ID")
    action: str = Field(..., description="操作动作")
    params: Dict[str, Any] = Field(default_factory=dict, description="操作参数")
    timeout: float = Field(default=10.0, description="超时时间(秒)")
    priority: int = Field(default=0, description="命令优先级")
    correlation_id: Optional[str] = Field(None, description="关联ID")


class DeviceResponse(BaseModel):
    """设备操作响应"""
    success: bool = Field(..., description="是否成功")
    device_id: str = Field(..., description="设备ID")
    message: str = Field(default="", description="响应消息")
    data: Dict[str, Any] = Field(default_factory=dict, description="返回数据")
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())


class DeviceRegistration(BaseModel):
    """设备注册请求"""
    name: str = Field(..., description="设备名称")
    type: DeviceType = Field(..., description="设备类型")
    protocol: CommunicationProtocol = Field(..., description="通信协议")
    capabilities: List[str] = Field(default_factory=list, description="设备能力")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class MotionCommand(BaseModel):
    """运动控制命令"""
    device_id: str = Field(..., description="设备ID")
    motion_type: str = Field(..., description="运动类型")
    position: Dict[str, float] = Field(default_factory=dict, description="目标位置")
    velocity: float = Field(default=0.0, description="速度")
    acceleration: float = Field(default=0.0, description="加速度")
    duration: float = Field(default=1.0, description="持续时间")
    continuous: bool = Field(default=False, description="是否连续运动")


class SensorReading(BaseModel):
    """传感器读数"""
    device_id: str = Field(..., description="设备ID")
    sensor_type: str = Field(..., description="传感器类型")
    value: Any = Field(..., description="读数值")
    unit: Optional[str] = Field(None, description="单位")
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())
    quality: int = Field(default=192, description="数据质量")


class TrajectoryPoint(BaseModel):
    """轨迹点"""
    position: Dict[str, float] = Field(..., description="位置")
    orientation: Dict[str, float] = Field(default_factory=dict, description="姿态")
    velocity: float = Field(default=0.0, description="速度")
    timestamp: Optional[float] = Field(None, description="时间戳")


class Trajectory(BaseModel):
    """轨迹"""
    id: str = Field(..., description="轨迹ID")
    device_id: str = Field(..., description="设备ID")
    points: List[TrajectoryPoint] = Field(default_factory=list, description="轨迹点")
    total_time: float = Field(default=0.0, description="总时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class DeviceCapability(str, Enum):
    """设备能力枚举"""
    MOVE = "move"
    TURN = "turn"
    GRASP = "grasp"
    RELEASE = "release"
    SENSE = "sense"
    STREAM = "stream"
    CALIBRATE = "calibrate"
    DIAGNOSE = "diagnose"
