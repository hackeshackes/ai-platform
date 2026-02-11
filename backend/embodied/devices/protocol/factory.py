"""
Protocol Factory - 协议工厂

根据协议类型创建对应的协议适配器
"""
from typing import Any, Dict


def get_protocol(protocol_type: str, device) -> Any:
    """
    获取协议适配器
    
    Args:
        protocol_type: 协议类型
        device: 设备对象
        
    Returns:
        协议适配器实例
    """
    protocols: Dict[str, Any] = {
        "MQTTProtocol": lambda d: MQTTProtocol(d),
        "RESTProtocol": lambda d: RESTProtocol(d),
        "WebSocketProtocol": lambda d: WebSocketProtocol(d),
        "ROSProtocol": lambda d: ROSProtocol(d),
        "OPCUAProtocol": lambda d: OPCUAProtocol(d),
        "ModbusProtocol": lambda d: ModbusProtocol(d),
    }
    
    protocol_class = protocols.get(protocol_type)
    if protocol_class:
        return protocol_class(device)
    else:
        # 默认返回模拟协议
        return SimulatedProtocol(device)


class SimulatedProtocol:
    """模拟协议 - 用于测试"""
    
    def __init__(self, device):
        self.device = device
        self.connected = False
    
    async def connect(self) -> bool:
        """模拟连接"""
        self.connected = True
        return True
    
    async def disconnect(self):
        """模拟断开连接"""
        self.connected = False
    
    async def send_command(self, command: dict) -> dict:
        """模拟发送命令"""
        return {"status": "ok", "result": "simulated"}
    
    async def receive(self) -> dict:
        """模拟接收数据"""
        return {"data": "simulated"}
    
    async def execute_command(self, command) -> dict:
        """执行命令"""
        return {"success": True, "response": "simulated"}


class MQTTProtocol:
    """MQTT协议适配器"""
    
    def __init__(self, device):
        self.device = device
        self.client = None
        self.connected = False
        self._config = device.config or {}
    
    async def connect(self, broker: str = None, port: int = None, **kwargs) -> bool:
        """连接MQTT broker"""
        # 模拟连接
        self.connected = True
        return True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def subscribe(self, topic: str):
        """订阅主题"""
        pass
    
    async def publish(self, topic: str, payload: dict):
        """发布消息"""
        pass
    
    async def execute_command(self, command) -> dict:
        """执行MQTT命令"""
        return {"status": "published", "topic": f"device/{self.device.id}/command"}


class RESTProtocol:
    """REST协议适配器"""
    
    def __init__(self, device):
        self.device = device
        self.base_url = ""
        self._config = device.config or {}
    
    async def connect(self, base_url: str = None, **kwargs) -> bool:
        """建立REST连接"""
        if base_url:
            self.base_url = base_url
        self._config["connected"] = True
        return True
    
    async def disconnect(self):
        """断开连接"""
        self._config["connected"] = False
    
    async def get(self, endpoint: str, params: dict = None) -> dict:
        """GET请求"""
        return {"method": "GET", "endpoint": endpoint}
    
    async def post(self, endpoint: str, data: dict = None) -> dict:
        """POST请求"""
        return {"method": "POST", "endpoint": endpoint, "data": data}
    
    async def put(self, endpoint: str, data: dict = None) -> dict:
        """PUT请求"""
        return {"method": "PUT", "endpoint": endpoint, "data": data}
    
    async def delete(self, endpoint: str) -> dict:
        """DELETE请求"""
        return {"method": "DELETE", "endpoint": endpoint}
    
    async def execute_command(self, command) -> dict:
        """执行REST命令"""
        action = command.action
        params = command.params
        
        endpoint = f"/{action}"
        method_map = {
            "turn_on": ("POST", {"state": "on"}),
            "turn_off": ("POST", {"state": "off"}),
            "get_status": ("GET", None),
            "set_value": ("PUT", params),
        }
        
        method, data = method_map.get(action, ("POST", params))
        
        if method == "GET":
            result = await self.get(endpoint, params)
        elif method == "PUT":
            result = await self.put(endpoint, data)
        else:
            result = await self.post(endpoint, data)
        
        return result


class WebSocketProtocol:
    """WebSocket协议适配器"""
    
    def __init__(self, device):
        self.device = device
        self.ws = None
        self.connected = False
        self._config = device.config or {}
        self._message_handlers: list = []
    
    async def connect(self, url: str = None, **kwargs) -> bool:
        """建立WebSocket连接"""
        self.connected = True
        return True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
        self._message_handlers.clear()
    
    async def send(self, message: dict):
        """发送消息"""
        pass
    
    async def receive(self) -> dict:
        """接收消息"""
        return {"type": "message"}
    
    async def subscribe(self, channel: str):
        """订阅频道"""
        pass
    
    async def execute_command(self, command) -> dict:
        """执行WebSocket命令"""
        return {"status": "sent", "command": command.action}


class ROSProtocol:
    """ROS协议适配器"""
    
    def __init__(self, device):
        self.device = device
        self.node = None
        self.connected = False
        self._config = device.config or {}
        self._subscribers: dict = {}
        self._publishers: dict = {}
    
    async def connect(self, master_uri: str = None, **kwargs) -> bool:
        """连接ROS master"""
        self.connected = True
        return True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    def create_publisher(self, topic: str, msg_type: type):
        """创建发布者"""
        self._publishers[topic] = {"type": msg_type}
    
    def create_subscriber(self, topic: str, msg_type: type, callback):
        """创建订阅者"""
        self._subscribers[topic] = {"type": msg_type, "callback": callback}
    
    async def publish(self, topic: str, message):
        """发布消息"""
        pass
    
    async def execute_command(self, command) -> dict:
        """执行ROS命令"""
        action = command.action
        params = command.params
        
        if action == "move_joint":
            return {"status": "joint_moved", "target": params}
        elif action == "move_pose":
            return {"status": "pose_set", "pose": params}
        else:
            return {"status": "executed", "action": action}


class OPCUAProtocol:
    """OPCUA协议适配器"""
    
    def __init__(self, device):
        self.device = device
        self.client = None
        self.connected = False
        self._config = device.config or {}
    
    async def connect(self, endpoint: str = None, **kwargs) -> bool:
        """连接OPCUA服务器"""
        self.connected = True
        return True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def read_node(self, node_id: str) -> any:
        """读取节点值"""
        return None
    
    async def write_node(self, node_id: str, value: any) -> bool:
        """写入节点值"""
        return True
    
    async def execute_command(self, command) -> dict:
        """执行OPCUA命令"""
        return {"status": "ok"}


class ModbusProtocol:
    """Modbus协议适配器"""
    
    def __init__(self, device):
        self.device = device
        self.client = None
        self.connected = False
        self._config = device.config or {}
    
    async def connect(self, host: str = None, port: int = None, **kwargs) -> bool:
        """连接Modbus设备"""
        self.connected = True
        return True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def read_coils(self, address: int, count: int) -> list:
        """读取线圈"""
        return [False] * count
    
    async def read_holding_registers(self, address: int, count: int) -> list:
        """读取保持寄存器"""
        return [0] * count
    
    async def write_coil(self, address: int, value: bool) -> bool:
        """写入线圈"""
        return True
    
    async def write_register(self, address: int, value: int) -> bool:
        """写入寄存器"""
        return True
    
    async def execute_command(self, command) -> dict:
        """执行Modbus命令"""
        action = command.action
        params = command.params
        
        if action == "read_coils":
            return {"values": await self.read_coils(
                params.get("address", 0),
                params.get("count", 1)
            )}
        elif action == "write_coil":
            return {"status": await self.write_coil(
                params.get("address", 0),
                params.get("value", False)
            )}
        else:
            return {"status": "executed"}
