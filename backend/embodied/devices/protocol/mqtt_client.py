"""
MQTT Client - MQTT协议客户端

提供MQTT设备通信功能
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MQTTQoS(int, Enum):
    """MQTT QoS等级"""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class MQTTClient:
    """
    MQTT客户端
    
    用于与MQTT broker通信，支持发布/订阅消息
    """
    
    def __init__(
        self,
        client_id: str = None,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: str = None,
        password: str = None,
        use_tls: bool = False,
        simulation_mode: bool = True
    ):
        """
        初始化MQTT客户端
        
        Args:
            client_id: 客户端ID
            broker_host: broker主机地址
            broker_port: broker端口
            username: 用户名
            password: 密码
            use_tls: 是否使用TLS
            simulation_mode: 是否使用模拟模式
        """
        self.client_id = client_id or f"embodied_ai_{datetime.utcnow().timestamp()}"
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.simulation_mode = simulation_mode
        
        self.connected = False
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._message_buffer: asyncio.Queue = asyncio.Queue()
        self._client = None
        
        logger.info(f"MQTTClient initialized: {self.broker_host}:{self.broker_port}")
    
    async def connect(self) -> bool:
        """
        连接到MQTT broker
        
        Returns:
            是否连接成功
        """
        if self.simulation_mode:
            # 模拟连接
            self.connected = True
            logger.info(f"MQTT simulated connection established: {self.client_id}")
            return True
        
        try:
            # 实际MQTT连接逻辑（需要aiomqtt库）
            import aiomqtt
            self._client = aiomqtt.Client(
                client_id=self.client_id,
                clean_session=True,
                protocol=5  # MQTT 5.0
            )
            
            if self.username and self.password:
                self._client.username_pw_set(self.username, self.password)
            
            if self.use_tls:
                self._client.tls_set()
            
            await asyncio.to_thread(self._client.connect, self.broker_host, self.broker_port)
            self._client.loop_start()
            
            self.connected = True
            logger.info(f"MQTT connection established: {self.broker_host}:{self.broker_port}")
            return True
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        
        self.connected = False
        logger.info("MQTT connection closed")
    
    async def subscribe(
        self,
        topic: str,
        qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE,
        callback: Callable[[str, bytes], None] = None
    ) -> bool:
        """
        订阅主题
        
        Args:
            topic: MQTT主题
            qos: QoS等级
            callback: 消息回调函数
            
        Returns:
            是否订阅成功
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        try:
            if self.simulation_mode:
                self._subscriptions[topic] = {
                    "qos": qos,
                    "callback": callback
                }
                logger.info(f"MQTT subscribed (simulated): {topic}")
                return True
            
            qos_int = int(qos)
            self._client.message_callback_add(topic, callback)
            self._client.subscribe(topic, qos_int)
            
            logger.info(f"MQTT subscribed: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"MQTT subscribe failed: {e}")
            return False
    
    async def unsubscribe(self, topic: str) -> bool:
        """
        取消订阅
        
        Args:
            topic: MQTT主题
            
        Returns:
            是否取消成功
        """
        if not self.connected:
            return False
        
        try:
            if topic in self._subscriptions:
                del self._subscriptions[topic]
            
            if self._client:
                self._client.unsubscribe(topic)
            
            logger.info(f"MQTT unsubscribed: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"MQTT unsubscribe failed: {e}")
            return False
    
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE,
        retain: bool = False
    ) -> bool:
        """
        发布消息
        
        Args:
            topic: MQTT主题
            payload: 消息负载
            qos: QoS等级
            retain: 是否保留消息
            
        Returns:
            是否发布成功
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        try:
            message = json.dumps(payload, default=str)
            
            if self.simulation_mode:
                # 模拟发布
                logger.debug(f"MQTT published (simulated): {topic} -> {payload}")
                return True
            
            if self._client:
                info = self._client.publish(
                    topic,
                    message,
                    qos=int(qos),
                    retain=retain
                )
                info.wait_for_publish(timeout=5)
            
            logger.debug(f"MQTT published: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"MQTT publish failed: {e}")
            return False
    
    async def get_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        获取消息（从缓冲区）
        
        Args:
            timeout: 超时时间
            
        Returns:
            消息内容
        """
        try:
            return await asyncio.wait_for(
                self._message_buffer.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    async def publish_device_status(
        self,
        device_id: str,
        status: str,
        **extra_data
    ) -> bool:
        """
        发布设备状态
        
        Args:
            device_id: 设备ID
            status: 状态
            **extra_data: 附加数据
            
        Returns:
            是否发布成功
        """
        payload = {
            "device_id": device_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            **extra_data
        }
        
        topic = f"embodied/devices/{device_id}/status"
        return await self.publish(topic, payload)
    
    async def publish_sensor_reading(
        self,
        device_id: str,
        sensor_type: str,
        value: Any,
        unit: str = None
    ) -> bool:
        """
        发布传感器读数
        
        Args:
            device_id: 设备ID
            sensor_type: 传感器类型
            value: 读数值
            unit: 单位
            
        Returns:
            是否发布成功
        """
        payload = {
            "device_id": device_id,
            "sensor_type": sensor_type,
            "value": value,
            "unit": unit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        topic = f"embodied/devices/{device_id}/telemetry"
        return await self.publish(topic, payload)
    
    async def send_command(
        self,
        device_id: str,
        command: str,
        params: Dict[str, Any] = None
    ) -> bool:
        """
        发送命令到设备
        
        Args:
            device_id: 目标设备ID
            command: 命令
            params: 命令参数
            
        Returns:
            是否发送成功
        """
        payload = {
            "command": command,
            "params": params or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        topic = f"embodied/devices/{device_id}/commands"
        return await self.publish(topic, payload)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        获取连接状态
        
        Returns:
            状态信息
        """
        return {
            "connected": self.connected,
            "broker": f"{self.broker_host}:{self.broker_port}",
            "client_id": self.client_id,
            "subscriptions_count": len(self._subscriptions),
            "simulation_mode": self.simulation_mode
        }
