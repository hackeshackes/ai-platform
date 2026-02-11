"""
ROS Bridge - ROS机器人操作系统桥接

提供ROS通信接口，支持话题发布/订阅和服务调用
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ROSMsgType(str, Enum):
    """ROS消息类型"""
    TWIST = "geometry_msgs/Twist"
    POINT = "geometry_msgs/Point"
    POSE = "geometry_msgs/Pose"
    JOINT_STATE = "sensor_msgs/JointState"
    IMAGE = "sensor_msgs/Image"
    LASER_SCAN = "sensor_msgs/LaserScan"


@dataclass
class Pose:
    """位姿"""
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


class ROSBridge:
    """
    ROS桥接器
    
    用于与ROS系统通信，支持:
    - 发布者/订阅者管理
    - 服务客户端
    - 行动客户端
    """
    
    def __init__(
        self,
        node_name: str = "embodied_ai_bridge",
        master_uri: str = "ros://localhost:11311",
        simulation_mode: bool = True
    ):
        """
        初始化ROS桥接器
        
        Args:
            node_name: ROS节点名称
            master_uri: ROS master URI
            simulation_mode: 模拟模式
        """
        self.node_name = node_name
        self.master_uri = master_uri
        self.simulation_mode = simulation_mode
        
        self.connected = False
        self._node = None
        self._publishers: Dict[str, Dict] = {}
        self._subscribers: Dict[str, Dict] = {}
        self._service_clients: Dict[str, Dict] = {}
        self._action_clients: Dict[str, Dict] = {}
        
        # 模拟数据
        self._simulated_topics: Dict[str, List[Dict]] = {}
        self._joint_positions: Dict[str, float] = {}
        
        logger.info(f"ROSBridge initialized: {node_name}")
    
    async def connect(self, master_uri: str = None) -> bool:
        """
        连接到ROS master
        
        Args:
            master_uri: ROS master URI
            
        Returns:
            是否连接成功
        """
        self.master_uri = master_uri or self.master_uri
        
        if self.simulation_mode:
            self.connected = True
            logger.info(f"ROS simulated connection: {self.master_uri}")
            return True
        
        try:
            # 使用roslibpy进行ROS通信
            import roslibpy
            
            self._node = roslibpy.Node(self.node_name, master=self.master_uri)
            self._node.init()
            
            self.connected = True
            logger.info(f"ROS connected: {self.node_name}")
            return True
            
        except Exception as e:
            logger.error(f"ROS connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """断开连接"""
        # 清理发布者
        for topic in list(self._publishers.keys()):
            await self.unregister_publisher(topic)
        
        # 清理订阅者
        for topic in list(self._subscribers.keys()):
            await self.unregister_subscriber(topic)
        
        # 关闭节点
        if self._node:
            self._node.terminate()
            self._node = None
        
        self.connected = False
        logger.info("ROS disconnected")
    
    def create_publisher(
        self,
        topic: str,
        msg_type: ROSMsgType = ROSMsgType.TWIST
    ) -> str:
        """
        创建发布者
        
        Args:
            topic: 话题名称
            msg_type: 消息类型
            
        Returns:
            发布者ID
        """
        publisher_id = f"pub_{topic.replace('/', '_')}"
        
        self._publishers[topic] = {
            "id": publisher_id,
            "msg_type": msg_type,
            "publisher": None
        }
        
        logger.info(f"ROS publisher created: {topic}")
        return publisher_id
    
    def register_publisher(self, topic: str):
        """注册发布者到ROS"""
        if topic not in self._publishers:
            self.create_publisher(topic)
        
        pub_info = self._publishers[topic]
        
        if self.simulation_mode:
            self._simulated_topics[topic] = []
            return
        
        if self._node:
            import roslibpy
            pub_info["publisher"] = roslibpy.Topic(
                self._node,
                topic,
                pub_info["msg_type"],
                latch=True
            )
            pub_info["publisher"].advertise()
    
    async def unregister_publisher(self, topic: str):
        """注销发布者"""
        if topic in self._publishers:
            pub_info = self._publishers[topic]
            
            if pub_info["publisher"]:
                pub_info["publisher"].unadvertise()
                pub_info["publisher"] = None
            
            del self._publishers[topic]
            logger.info(f"ROS publisher unregistered: {topic}")
    
    def create_subscriber(
        self,
        topic: str,
        msg_type: ROSMsgType,
        callback: Callable[[Dict], None]
    ) -> str:
        """
        创建订阅者
        
        Args:
            topic: 话题名称
            msg_type: 消息类型
            callback: 回调函数
            
        Returns:
            订阅者ID
        """
        subscriber_id = f"sub_{topic.replace('/', '_')}"
        
        self._subscribers[topic] = {
            "id": subscriber_id,
            "msg_type": msg_type,
            "callback": callback,
            "subscriber": None
        }
        
        logger.info(f"ROS subscriber created: {topic}")
        return subscriber_id
    
    def register_subscriber(self, topic: str):
        """注册订阅者"""
        if topic not in self._subscribers:
            logger.warning(f"Subscriber not found: {topic}")
            return
        
        sub_info = self._subscribers[topic]
        
        if self.simulation_mode:
            # 初始化模拟话题
            if topic not in self._simulated_topics:
                self._simulated_topics[topic] = []
            return
        
        if self._node:
            import roslibpy
            
            def wrapped_callback(message):
                # 在线程池中执行回调
                asyncio.create_task(self._run_callback(sub_info["callback"], message))
            
            sub_info["subscriber"] = roslibpy.Topic(
                self._node,
                topic,
                sub_info["msg_type"]
            )
            sub_info["subscriber"].subscribe(wrapped_callback)
    
    async def unregister_subscriber(self, topic: str):
        """注销订阅者"""
        if topic in self._subscribers:
            sub_info = self._subscribers[topic]
            
            if sub_info["subscriber"]:
                sub_info["subscriber"].unsubscribe()
                sub_info["subscriber"] = None
            
            del self._subscribers[topic]
            logger.info(f"ROS subscriber unregistered: {topic}")
    
    async def _run_callback(self, callback, message):
        """执行回调"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.error(f"ROS subscriber callback error: {e}")
    
    def create_service_client(
        self,
        service_name: str,
        service_type: str
    ) -> str:
        """
        创建服务客户端
        
        Args:
            service_name: 服务名称
            service_type: 服务类型
            
        Returns:
            客户端ID
        """
        client_id = f"svc_{service_name.replace('/', '_')}"
        
        self._service_clients[service_name] = {
            "id": client_id,
            "type": service_type,
            "client": None
        }
        
        return client_id
    
    def register_service_client(self, service_name: str):
        """注册服务客户端"""
        if service_name not in self._service_clients:
            logger.warning(f"Service client not found: {service_name}")
            return
        
        svc_info = self._service_clients[service_name]
        
        if self.simulation_mode:
            return
        
        if self._node:
            import roslibpy
            svc_info["client"] = roslibpy.Service(
                self._node,
                service_name,
                svc_info["type"]
            )
    
    async def call_service(
        self,
        service_name: str,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        调用服务
        
        Args:
            service_name: 服务名称
            args: 服务参数
            
        Returns:
            服务响应
        """
        if not self.connected:
            return {"success": False, "error": "Not connected"}
        
        if self.simulation_mode:
            logger.debug(f"ROS service call (simulated): {service_name}")
            return {"success": True, "result": {"simulated": True}}
        
        if service_name not in self._service_clients:
            return {"success": False, "error": "Service not found"}
        
        svc_info = self._service_clients[service_name]
        
        if svc_info["client"]:
            import roslibpy
            
            request = roslibpy.ServiceRequest(args)
            response = svc_info["client"].call(request)
            
            return {"success": True, "result": dict(response)}
        
        return {"success": False, "error": "Client not ready"}
    
    async def publish(self, topic: str, message: Dict[str, Any]):
        """
        发布消息
        
        Args:
            topic: 话题名称
            message: 消息内容
        """
        if not self.connected:
            return
        
        if topic not in self._publishers:
            logger.warning(f"Publisher not found: {topic}")
            return
        
        pub_info = self._publishers[topic]
        
        if self.simulation_mode:
            # 模拟发布
            self._simulated_topics[topic].append({
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            return
        
        if pub_info["publisher"]:
            if isinstance(message, dict):
                import roslibpy
                pub_info["publisher"].publish(roslibpy.Message(message))
    
    async def simulate_message(self, topic: str, message: Dict[str, Any]):
        """
        模拟发布消息（用于测试）
        
        Args:
            topic: 话题名称
            message: 消息内容
        """
        self._simulated_topics[topic].append({
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # 触发订阅者回调
        if topic in self._subscribers:
            sub_info = self._subscribers[topic]
            await self._run_callback(sub_info["callback"], message)
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取状态
        
        Returns:
            状态信息
        """
        return {
            "connected": self.connected,
            "node_name": self.node_name,
            "master_uri": self.master_uri,
            "publishers_count": len(self._publishers),
            "subscribers_count": len(self._subscribers),
            "services_count": len(self._service_clients),
            "simulation_mode": self.simulation_mode
        }
