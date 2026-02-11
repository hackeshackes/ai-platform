"""
WebSocket - WebSocket协议通信

提供WebSocket实时通信功能
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class WSMessageType(str, Enum):
    """WebSocket消息类型"""
    TEXT = "text"
    BINARY = "binary"
    PING = "ping"
    PONG = "pong"
    CLOSE = "close"
    ERROR = "error"


class WebSocketClient:
    """
    WebSocket客户端
    
    用于WebSocket实时双向通信
    """
    
    def __init__(
        self,
        device_id: str,
        url: str = None,
        simulation_mode: bool = True,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
        heartbeat_interval: float = 30.0
    ):
        """
        初始化WebSocket客户端
        
        Args:
            device_id: 设备ID
            url: WebSocket服务器URL
            simulation_mode: 模拟模式
            auto_reconnect: 自动重连
            reconnect_delay: 重连延迟
            heartbeat_interval: 心跳间隔
        """
        self.device_id = device_id
        self.url = url
        self.simulation_mode = simulation_mode
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        
        self.connected = False
        self._websocket = None
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._receive_queue: asyncio.Queue = asyncio.Queue()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 模拟数据
        self._simulated_messages: List[Dict] = []
        
        logger.info(f"WebSocketClient initialized: {device_id}")
    
    async def connect(self, url: str = None) -> bool:
        """
        连接到WebSocket服务器
        
        Args:
            url: WebSocket URL，覆盖初始化时的URL
            
        Returns:
            是否连接成功
        """
        self.url = url or self.url
        
        if not self.url:
            logger.warning("No WebSocket URL provided")
            return False
        
        if self.simulation_mode:
            self.connected = True
            self._running = True
            logger.info(f"WebSocket simulated connection: {self.url}")
            
            # 启动心跳
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            return True
        
        try:
            import websockets
            
            self._websocket = await websockets.connect(
                self.url,
                ping_interval=self.heartbeat_interval,
                ping_timeout=10
            )
            
            self.connected = True
            self._running = True
            
            # 启动接收循环
            asyncio.create_task(self._receive_loop())
            
            # 启动心跳
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info(f"WebSocket connected: {self.url}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """断开连接"""
        self._running = False
        
        # 取消任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        
        # 关闭连接
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        self.connected = False
        logger.info("WebSocket disconnected")
    
    async def send(self, message: Dict[str, Any], msg_type: WSMessageType = WSMessageType.TEXT) -> bool:
        """
        发送消息
        
        Args:
            message: 消息内容
            msg_type: 消息类型
            
        Returns:
            是否发送成功
        """
        if not self.connected:
            logger.warning("WebSocket not connected")
            return False
        
        try:
            if self.simulation_mode:
                self._simulated_messages.append({
                    "sent": True,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                })
                logger.debug(f"WebSocket simulated send: {message}")
                return True
            
            if msg_type == WSMessageType.TEXT:
                await self._websocket.send(json.dumps(message))
            elif msg_type == WSMessageType.BINARY:
                await self._websocket.send(json.dumps(message).encode())
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}")
            return False
    
    async def receive(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        接收消息
        
        Args:
            timeout: 超时时间
            
        Returns:
            消息内容
        """
        if timeout:
            try:
                return await asyncio.wait_for(
                    self._receive_queue.get(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return None
        else:
            return await self._receive_queue.get()
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        订阅消息事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type not in self._message_handlers:
            self._message_handlers[event_type] = []
        self._message_handlers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable = None):
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            callback: 回调函数，不传则取消所有
        """
        if event_type in self._message_handlers:
            if callback:
                try:
                    self._message_handlers[event_type].remove(callback)
                except ValueError:
                    pass
            else:
                self._message_handlers[event_type].clear()
    
    async def send_command(
        self,
        command: str,
        params: Dict[str, Any] = None,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        发送命令并等待响应
        
        Args:
            command: 命令名称
            params: 命令参数
            timeout: 响应超时时间
            
        Returns:
            响应数据
        """
        message = {
            "device_id": self.device_id,
            "command": command,
            "params": params or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send(message)
        
        # 等待响应
        try:
            response = await asyncio.wait_for(
                self.receive(timeout),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            return {"error": "timeout", "command": command}
    
    async def stream_telemetry(
        self,
        callback: Callable[[Dict], None],
        max_messages: int = None
    ) -> asyncio.Task:
        """
        流式接收遥测数据
        
        Args:
            callback: 数据回调
            max_messages: 最大消息数
            
        Returns:
            订阅任务
        """
        async def stream_loop():
            count = 0
            while self._running and (max_messages is None or count < max_messages):
                try:
                    message = await self.receive(timeout=60.0)
                    if message:
                        callback(message)
                        count += 1
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    break
        
        task = asyncio.create_task(stream_loop())
        return task
    
    async def _receive_loop(self):
        """接收消息循环"""
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    
                    # 添加到队列
                    await self._receive_queue.put(data)
                    
                    # 分发到处理器
                    await self._dispatch_message(data)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
                    
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            self.connected = False
            
            # 自动重连
            if self.auto_reconnect and self._running:
                self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _dispatch_message(self, message: Dict[str, Any]):
        """分发消息到处理器"""
        msg_type = message.get("type", "data")
        
        handlers = self._message_handlers.get(msg_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if self.connected:
                    await self.send({"type": "ping"}, WSMessageType.PING)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _reconnect(self):
        """重连逻辑"""
        await asyncio.sleep(self.reconnect_delay)
        
        if self._running and not self.connected:
            logger.info("Attempting to reconnect...")
            success = await self.connect()
            
            if success:
                logger.info("Reconnected successfully")
            elif self.auto_reconnect:
                asyncio.create_task(self._reconnect())
    
    async def send_device_status(self, status: str, **extra):
        """
        发送设备状态
        
        Args:
            status: 状态
            **extra: 附加信息
        """
        message = {
            "type": "status",
            "device_id": self.device_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            **extra
        }
        await self.send(message)
    
    async def send_sensor_reading(self, sensor_type: str, value: Any, unit: str = None):
        """
        发送传感器读数
        
        Args:
            sensor_type: 传感器类型
            value: 读数值
            unit: 单位
        """
        message = {
            "type": "telemetry",
            "device_id": self.device_id,
            "sensor_type": sensor_type,
            "value": value,
            "unit": unit,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send(message)
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取连接状态
        
        Returns:
            状态信息
        """
        return {
            "connected": self.connected,
            "url": self.url,
            "device_id": self.device_id,
            "simulation_mode": self.simulation_mode,
            "auto_reconnect": self.auto_reconnect,
            "handlers_count": sum(len(h) for h in self._message_handlers.values()),
            "queued_messages": self._receive_queue.qsize()
        }
