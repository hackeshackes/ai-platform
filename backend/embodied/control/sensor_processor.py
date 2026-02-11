"""
Sensor Processor - 传感器数据处理

提供传感器数据采集、滤波、融合和分析功能
"""
import asyncio
import json
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """传感器类型"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    GPS = "gps"
    CAMERA = "camera"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    ENCODER = "encoder"
    FORCE = "force"
    TORQUE = "torque"
    PROXIMITY = "proximity"


@dataclass
class SensorReading:
    """传感器读数"""
    device_id: str
    sensor_type: SensorType
    value: Any
    unit: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    quality: int = 192  # MQTT QoS: good
    metadata: Dict = field(default_factory=dict)


@dataclass
class SensorConfig:
    """传感器配置"""
    sample_rate: float = 10.0  # 采样率 Hz
    filter_window: int = 5  # 滤波窗口大小
    outlier_threshold: float = 3.0  # 异常值阈值
    calibration_offset: float = 0.0  # 校准偏移
    calibration_scale: float = 1.0  # 校准比例


class SensorProcessor:
    """
    传感器处理器
    
    功能:
    - 数据采集与缓冲
    - 低通滤波
    - 异常值检测
    - 数据融合
    - 统计分析
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        初始化传感器处理器
        
        Args:
            simulation_mode: 模拟模式
        """
        self.simulation_mode = simulation_mode
        
        self._sensors: Dict[str, SensorConfig] = {}
        self._data_buffers: Dict[str, deque] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = asyncio.Lock()
        
        logger.info("SensorProcessor initialized")
    
    async def start(self):
        """启动处理器"""
        self._running = True
        logger.info("SensorProcessor started")
    
    async def stop(self):
        """停止处理器"""
        self._running = False
        
        # 取消所有任务
        for task in self._processing_tasks.values():
            task.cancel()
        
        self._processing_tasks.clear()
        logger.info("SensorProcessor stopped")
    
    async def register_sensor(
        self,
        device_id: str,
        sensor_type: SensorType,
        config: SensorConfig = None
    ) -> bool:
        """
        注册传感器
        
        Args:
            device_id: 设备ID
            sensor_type: 传感器类型
            config: 配置
            
        Returns:
            是否成功
        """
        async with self._lock:
            sensor_key = f"{device_id}_{sensor_type.value}"
            
            self._sensors[sensor_key] = config or SensorConfig()
            self._data_buffers[sensor_key] = deque(maxlen=100)  # 最多保留100个数据点
            
            logger.info(f"Sensor registered: {sensor_key}")
            return True
    
    async def unregister_sensor(self, device_id: str, sensor_type: SensorType) -> bool:
        """注销传感器"""
        sensor_key = f"{device_id}_{sensor_type.value}"
        
        if sensor_key in self._sensors:
            del self._sensors[sensor_key]
            del self._data_buffers[sensor_key]
            logger.info(f"Sensor unregistered: {sensor_key}")
            return True
        return False
    
    async def process_reading(
        self,
        reading: SensorReading
    ) -> SensorReading:
        """
        处理传感器读数
        
        Args:
            reading: 原始读数
            
        Returns:
            处理后的读数
        """
        sensor_key = f"{reading.device_id}_{reading.sensor_type.value}"
        
        # 应用校准
        config = self._sensors.get(sensor_key)
        if config:
            reading = self._calibrate_reading(reading, config)
            
            # 异常值检测
            if self._is_outlier(reading, config):
                reading.quality = 0  # 标记为坏数据
                logger.debug(f"Outlier detected: {reading.value}")
            
            # 滤波处理
            if config.filter_window > 1:
                reading.value = self._filter_value(sensor_key, reading.value, config)
        
        # 存入缓冲区
        if sensor_key in self._data_buffers:
            self._data_buffers[sensor_key].append(reading.value)
        
        # 通知订阅者
        await self._notify_subscribers(sensor_key, reading)
        
        return reading
    
    def _calibrate_reading(
        self,
        reading: SensorReading,
        config: SensorConfig
    ) -> SensorReading:
        """应用校准"""
        if isinstance(reading.value, (int, float)):
            reading.value = (reading.value + config.calibration_offset) * config.calibration_scale
        return reading
    
    def _is_outlier(
        self,
        reading: SensorReading,
        config: SensorConfig
    ) -> bool:
        """异常值检测（基于Z-score）"""
        if not isinstance(reading.value, (int, float)):
            return False
        
        sensor_key = f"{reading.device_id}_{reading.sensor_type.value}"
        buffer = self._data_buffers.get(sensor_key, deque())
        
        if len(buffer) < config.filter_window:
            return False
        
        values = list(buffer)
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        
        if std == 0:
            return False
        
        z_score = abs(reading.value - mean) / std
        return z_score > config.outlier_threshold
    
    def _filter_value(
        self,
        sensor_key: str,
        value: float,
        config: SensorConfig
    ) -> float:
        """简单移动平均滤波"""
        buffer = self._data_buffers.get(sensor_key, deque())
        values = list(buffer)
        values.append(value)
        
        # 取最近N个值的平均
        window = values[-config.filter_window:]
        return sum(window) / len(window)
    
    async def subscribe(
        self,
        device_id: str,
        sensor_type: SensorType,
        callback: Callable[[SensorReading], None]
    ):
        """订阅传感器数据"""
        sensor_key = f"{device_id}_{sensor_type.value}"
        
        if sensor_key not in self._subscribers:
            self._subscribers[sensor_key] = []
        
        self._subscribers[sensor_key].append(callback)
    
    async def unsubscribe(
        self,
        device_id: str,
        sensor_type: SensorType,
        callback: Callable = None
    ):
        """取消订阅"""
        sensor_key = f"{device_id}_{sensor_type.value}"
        
        if sensor_key in self._subscribers:
            if callback:
                self._subscribers[sensor_key].remove(callback)
            else:
                self._subscribers[sensor_key].clear()
    
    async def _notify_subscribers(self, sensor_key: str, reading: SensorReading):
        """通知订阅者"""
        subscribers = self._subscribers.get(sensor_key, [])
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(reading)
                else:
                    callback(reading)
            except Exception as e:
                logger.error(f"Subscribe callback error: {e}")
    
    async def get_reading(
        self,
        device_id: str,
        sensor_type: SensorType
    ) -> Optional[SensorReading]:
        """
        获取最新读数
        
        Args:
            device_id: 设备ID
            sensor_type: 传感器类型
            
        Returns:
            最新读数
        """
        sensor_key = f"{device_id}_{sensor_type.value}"
        buffer = self._data_buffers.get(sensor_key)
        
        if buffer and len(buffer) > 0:
            return SensorReading(
                device_id=device_id,
                sensor_type=sensor_type,
                value=buffer[-1],
                timestamp=datetime.utcnow().isoformat()
            )
        return None
    
    async def get_statistics(
        self,
        device_id: str,
        sensor_type: SensorType
    ) -> Dict[str, Any]:
        """
        获取统计数据
        
        Args:
            device_id: 设备ID
            sensor_type: 传感器类型
            
        Returns:
            统计数据
        """
        sensor_key = f"{device_id}_{sensor_type.value}"
        buffer = list(self._data_buffers.get(sensor_key, deque()))
        
        if not buffer:
            return {"count": 0}
        
        return {
            "count": len(buffer),
            "mean": sum(buffer) / len(buffer),
            "min": min(buffer),
            "max": max(buffer),
            "std": math.sqrt(sum((v - sum(buffer) / len(buffer)) ** 2 for v in buffer) / len(buffer))
            if len(buffer) > 1 else 0
        }
    
    async def generate_simulated_reading(
        self,
        device_id: str,
        sensor_type: SensorType
    ) -> SensorReading:
        """
        生成模拟传感器读数
        
        Args:
            device_id: 设备ID
            sensor_type: 传感器类型
            
        Returns:
            模拟读数
        """
        import random
        
        # 根据传感器类型生成不同的模拟数据
        base_values = {
            SensorType.TEMPERATURE: (25.0, 2.0),  # 平均值, 标准差
            SensorType.HUMIDITY: (50.0, 5.0),
            SensorType.PRESSURE: (1013.25, 1.0),
            SensorType.ACCELEROMETER: (0.0, 0.1),  # m/s^2
            SensorType.GYROSCOPE: (0.0, 0.01),  # rad/s
            SensorType.PROXIMITY: (0.5, 0.1),  # 米
        }
        
        mean, std = base_values.get(sensor_type, (0.0, 1.0))
        value = random.gauss(mean, std)
        
        unit_map = {
            SensorType.TEMPERATURE: "°C",
            SensorType.HUMIDITY: "%",
            SensorType.PRESSURE: "hPa",
            SensorType.ACCELEROMETER: "m/s²",
            SensorType.GYROSCOPE: "rad/s",
            SensorType.PROXIMITY: "m",
        }
        
        return SensorReading(
            device_id=device_id,
            sensor_type=sensor_type,
            value=round(value, 3),
            unit=unit_map.get(sensor_type)
        )
    
    def get_sensor_list(self) -> List[Dict[str, Any]]:
        """获取已注册的传感器列表"""
        return [
            {"key": key, "config": config.__dict__}
            for key, config in self._sensors.items()
        ]
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """获取缓冲区统计"""
        return {key: len(buffer) for key, buffer in self._data_buffers.items()}
