"""
Device Manager - 设备管理器

负责设备注册、发现、状态管理和命令分发
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

from embodied.devices.models import (
    Device,
    DeviceType,
    DeviceStatus,
    CommunicationProtocol,
    DeviceRegistration,
    DeviceCommand,
    DeviceResponse,
)


class DeviceManager:
    """
    设备管理器
    
    核心功能:
    - 设备注册与注销
    - 设备状态管理
    - 设备发现与心跳
    - 命令分发
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        初始化设备管理器
        
        Args:
            simulation_mode: 是否使用模拟模式
        """
        self.simulation_mode = simulation_mode
        self.devices: Dict[str, Device] = {}
        self.device_protocols: Dict[str, Any] = {}
        self.command_queues: Dict[str, asyncio.Queue] = {}
        self.status_subscribers: Dict[str, List[callable]] = {}
        self._lock = asyncio.Lock()
        
        # 模拟模式下的虚拟设备
        self._simulated_devices: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"DeviceManager initialized in {'simulation' if simulation_mode else 'real'} mode")
    
    async def register_device(self, registration: DeviceRegistration) -> Device:
        """
        注册新设备
        
        Args:
            registration: 设备注册信息
            
        Returns:
            注册后的设备对象
        """
        async with self._lock:
            # 生成设备ID
            device_id = f"device_{uuid.uuid4().hex[:12]}"
            
            # 创建设备对象
            device = Device(
                id=device_id,
                name=registration.name,
                type=registration.type,
                protocol=registration.protocol,
                status=DeviceStatus.ONLINE,
                capabilities=registration.capabilities,
                config=registration.config,
                metadata=registration.metadata,
                last_seen=datetime.utcnow().isoformat()
            )
            
            # 初始化设备协议
            await self._init_device_protocol(device)
            
            # 初始化命令队列
            self.command_queues[device_id] = asyncio.Queue()
            
            # 存储设备
            self.devices[device_id] = device
            
            # 初始化模拟设备
            if self.simulation_mode:
                self._init_simulated_device(device)
            
            logger.info(f"Device registered: {device_id} - {device.name}")
            
            # 通知状态变化
            await self._notify_status_change(device)
            
            return device
    
    async def unregister_device(self, device_id: str) -> bool:
        """
        注销设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            是否成功注销
        """
        async with self._lock:
            if device_id not in self.devices:
                logger.warning(f"Device not found: {device_id}")
                return False
            
            # 清理协议连接
            if device_id in self.device_protocols:
                await self._cleanup_protocol(device_id)
            
            # 清理命令队列
            if device_id in self.command_queues:
                del self.command_queues[device_id]
            
            # 清理模拟设备
            if device_id in self._simulated_devices:
                del self._simulated_devices[device_id]
            
            # 移除设备
            del self.devices[device_id]
            
            logger.info(f"Device unregistered: {device_id}")
            return True
    
    async def get_device(self, device_id: str) -> Optional[Device]:
        """
        获取设备信息
        
        Args:
            device_id: 设备ID
            
        Returns:
            设备对象，不存在则返回None
        """
        return self.devices.get(device_id)
    
    async def list_devices(
        self,
        device_type: Optional[DeviceType] = None,
        status: Optional[DeviceStatus] = None,
        protocol: Optional[CommunicationProtocol] = None
    ) -> List[Device]:
        """
        列出设备
        
        支持按类型、状态、协议过滤
        
        Args:
            device_type: 设备类型过滤
            status: 设备状态过滤
            protocol: 通信协议过滤
            
        Returns:
            符合条件的设备列表
        """
        devices = list(self.devices.values())
        
        if device_type:
            devices = [d for d in devices if d.type == device_type]
        
        if status:
            devices = [d for d in devices if d.status == status]
        
        if protocol:
            devices = [d for d in devices if d.protocol == protocol]
        
        return devices
    
    async def update_device_status(
        self,
        device_id: str,
        status: DeviceStatus
    ) -> bool:
        """
        更新设备状态
        
        Args:
            device_id: 设备ID
            status: 新状态
            
        Returns:
            是否成功更新
        """
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        device.status = status
        device.last_seen = datetime.utcnow().isoformat()
        
        await self._notify_status_change(device)
        
        logger.info(f"Device status updated: {device_id} -> {status}")
        return True
    
    async def send_command(self, command: DeviceCommand) -> DeviceResponse:
        """
        发送设备命令
        
        Args:
            command: 设备命令
            
        Returns:
            命令执行响应
        """
        device = self.devices.get(command.device_id)
        
        if not device:
            return DeviceResponse(
                success=False,
                device_id=command.device_id,
                message="Device not found"
            )
        
        if device.status == DeviceStatus.OFFLINE:
            return DeviceResponse(
                success=False,
                device_id=command.device_id,
                message="Device is offline"
            )
        
        try:
            # 在模拟模式下执行模拟命令
            if self.simulation_mode:
                result = await self._execute_simulated_command(command)
            else:
                # 通过协议发送命令
                protocol = self.device_protocols.get(command.device_id)
                if protocol:
                    result = await protocol.execute_command(command)
                else:
                    # 无协议连接时的模拟执行
                    result = await self._execute_simulated_command(command)
            
            return DeviceResponse(
                success=True,
                device_id=command.device_id,
                message="Command executed successfully",
                data=result
            )
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return DeviceResponse(
                success=False,
                device_id=command.device_id,
                message=f"Command failed: {str(e)}"
            )
    
    async def heartbeat(self, device_id: str) -> bool:
        """
        设备心跳
        
        Args:
            device_id: 设备ID
            
        Returns:
            是否收到心跳
        """
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        device.last_seen = datetime.utcnow().isoformat()
        
        if device.status == DeviceStatus.OFFLINE:
            device.status = DeviceStatus.ONLINE
        
        return True
    
    async def get_device_capabilities(self, device_id: str) -> List[str]:
        """
        获取设备能力列表
        
        Args:
            device_id: 设备ID
            
        Returns:
            能力列表
        """
        device = self.devices.get(device_id)
        if not device:
            return []
        return device.capabilities
    
    async def validate_command(
        self,
        device_id: str,
        action: str,
        params: Dict[str, Any]
    ) -> bool:
        """
        验证命令是否有效
        
        Args:
            device_id: 设备ID
            action: 操作动作
            params: 操作参数
            
        Returns:
            命令是否有效
        """
        device = self.devices.get(device_id)
        if not device:
            return False
        
        # 检查设备是否在线
        if device.status == DeviceStatus.OFFLINE:
            return False
        
        # 检查设备是否支持该动作
        if action not in device.capabilities:
            return False
        
        # 验证参数
        # TODO: 实现更详细的参数验证
        
        return True
    
    async def _init_device_protocol(self, device: Device):
        """初始化设备通信协议"""
        protocol_map = {
            CommunicationProtocol.MQTT: "MQTTProtocol",
            CommunicationProtocol.REST: "RESTProtocol",
            CommunicationProtocol.WEBSOCKET: "WebSocketProtocol",
            CommunicationProtocol.ROS: "ROSProtocol",
        }
        
        protocol_name = protocol_map.get(device.protocol)
        if protocol_name:
            # 延迟导入以避免循环依赖
            from embodied.devices.protocol.factory import get_protocol
            self.device_protocols[device.id] = get_protocol(protocol_name, device)
            logger.info(f"Protocol initialized: {device.id} -> {protocol_name}")
    
    async def _cleanup_protocol(self, device_id: str):
        """清理协议连接"""
        protocol = self.device_protocols.pop(device_id, None)
        if protocol and hasattr(protocol, 'close'):
            await protocol.close()
        logger.info(f"Protocol cleaned up: {device_id}")
    
    def _init_simulated_device(self, device: Device):
        """初始化模拟设备"""
        self._simulated_devices[device.id] = {
            "state": {},
            "history": []
        }
        logger.info(f"Simulated device initialized: {device.id}")
    
    async def _execute_simulated_command(
        self,
        command: DeviceCommand
    ) -> Dict[str, Any]:
        """执行模拟命令"""
        device_id = command.device_id
        action = command.action
        params = command.params
        
        sim_device = self._simulated_devices.get(device_id, {})
        result = {"action": action, "params": params}
        
        # 根据动作类型返回模拟结果
        if action in ["turn_on", "enable"]:
            result["result"] = "activated"
            result["state"] = "on"
        elif action in ["turn_off", "disable"]:
            result["result"] = "deactivated"
            result["state"] = "off"
        elif action == "move":
            result["result"] = "moved"
            result["target_position"] = params.get("position", {})
        elif action == "get_status":
            result["result"] = "status retrieved"
            result["state"] = sim_device.get("state", {})
        elif action == "sense":
            result["result"] = "sensor reading"
            result["value"] = self._generate_simulated_sensor_reading()
        elif action == "set_value":
            result["result"] = "value set"
            result["value"] = params.get("value")
        else:
            result["result"] = "command executed"
            result["acknowledged"] = True
        
        # 记录到历史
        if "history" in sim_device:
            sim_device["history"].append({
                "command": action,
                "params": params,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return result
    
    def _generate_simulated_sensor_reading(self) -> Dict[str, Any]:
        """生成模拟传感器读数"""
        import random
        return {
            "temperature": round(random.uniform(20.0, 30.0), 1),
            "humidity": round(random.uniform(30.0, 70.0), 1),
            "pressure": round(random.uniform(1000.0, 1020.0), 1),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _notify_status_change(self, device: Device):
        """通知状态变化"""
        subscribers = self.status_subscribers.get(device.id, [])
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(device)
                else:
                    callback(device)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def subscribe_status(self, device_id: str, callback: callable):
        """订阅设备状态变化"""
        if device_id not in self.status_subscribers:
            self.status_subscribers[device_id] = []
        self.status_subscribers[device_id].append(callback)
    
    def unsubscribe_status(self, device_id: str, callback: callable):
        """取消订阅"""
        if device_id in self.status_subscribers:
            try:
                self.status_subscribers[device_id].remove(callback)
            except ValueError:
                pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取设备管理器统计信息"""
        devices_by_type = {}
        devices_by_status = {}
        
        for device in self.devices.values():
            # 按类型统计
            type_key = device.type.value
            devices_by_type[type_key] = devices_by_type.get(type_key, 0) + 1
            
            # 按状态统计
            status_key = device.status.value
            devices_by_status[status_key] = devices_by_status.get(status_key, 0) + 1
        
        return {
            "total_devices": len(self.devices),
            "by_type": devices_by_type,
            "by_status": devices_by_status,
            "simulation_mode": self.simulation_mode
        }
