"""
Embodied AI Integration Layer

具身AI集成层 - 机器人/IoT设备集成
"""
from embodied.devices.device_manager import DeviceManager
from embodied.control.motion_control import MotionController
from embodied.control.sensor_processor import SensorProcessor

__all__ = [
    "DeviceManager",
    "MotionController",
    "SensorProcessor",
]
