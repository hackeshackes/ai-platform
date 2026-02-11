"""
Agent Communication - Agent通信协议
提供Agent间的消息传递、事件通信和协议转换
"""

from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    EVENT = "event"
    BROADCAST = "broadcast"
    ERROR = "error"


class CommunicationProtocol(Enum):
    """通信协议枚举"""
    DIRECT = "direct"
    PUBLISH_SUBSCRIBE = "pubsub"
    REQUEST_REPLY = "request_reply"
    STREAM = "stream"


@dataclass
class Message:
    """消息数据模型"""
    id: str
    type: MessageType
    sender_id: str
    receiver_id: str
    content: Dict[str, Any]
    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    
    @classmethod
    def create(cls, sender_id: str, receiver_id: str, 
               content: Dict[str, Any],
               msg_type: MessageType = MessageType.REQUEST,
               protocol: CommunicationProtocol = CommunicationProtocol.DIRECT,
               metadata: Dict[str, Any] = None,
               correlation_id: str = None) -> "Message":
        """创建消息"""
        return cls(
            id=str(uuid.uuid4()),
            type=msg_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            protocol=protocol,
            correlation_id=correlation_id or str(uuid.uuid4()),
            metadata=metadata or {},
            timestamp=datetime.now()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "protocol": self.protocol.value,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建"""
        data["type"] = MessageType(data["type"])
        data["protocol"] = CommunicationProtocol(data["protocol"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class AgentEndpoint:
    """Agent端点"""
    agent_id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    protocols: List[CommunicationProtocol] = field(
        default_factory=lambda: [CommunicationProtocol.DIRECT]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "protocols": [p.value for p in self.protocols],
            "metadata": self.metadata,
            "last_seen": self.last_seen.isoformat()
        }


class MessageHandler(ABC):
    """消息处理器抽象基类"""
    
    @abstractmethod
    async def handle(self, message: Message) -> Optional[Message]:
        """处理消息"""
        pass


class DirectHandler(MessageHandler):
    """直接消息处理器"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    async def start(self):
        """启动处理器"""
        self._running = True
        
    async def stop(self):
        """停止处理器"""
        self._running = False
        
    async def handle(self, message: Message) -> Optional[Message]:
        """处理消息"""
        if message.receiver_id == self.agent_id:
            await self.inbox.put(message)
            return None
        return message
    
    async def receive(self, timeout: float = None) -> Message:
        """接收消息"""
        return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
    
    async def send(self, message: Message):
        """发送消息"""
        await communication_manager.deliver(message)


class PubSubHandler(MessageHandler):
    """发布/订阅消息处理器"""
    
    def __init__(self):
        self.topics: Dict[str, set] = {}
        self.subscriptions: Dict[str, set] = {}
        self._running = False
    
    async def start(self):
        self._running = True
        
    async def stop(self):
        self._running = False
    
    async def publish(self, topic: str, message: Message):
        """发布消息到主题"""
        if topic not in self.topics:
            self.topics[topic] = set()
        
        for subscriber in self.topics.get(topic, set()):
            await communication_manager.deliver_to_agent(subscriber, message)
        
        logger.info(f"Published message to topic {topic}")
    
    async def subscribe(self, agent_id: str, topic: str):
        """订阅主题"""
        if topic not in self.topics:
            self.topics[topic] = set()
        self.topics[topic].add(agent_id)
        
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = set()
        self.subscriptions[agent_id].add(topic)
        
        logger.info(f"Agent {agent_id} subscribed to topic {topic}")
    
    async def unsubscribe(self, agent_id: str, topic: str):
        """取消订阅"""
        if topic in self.topics:
            self.topics[topic].discard(agent_id)
        if agent_id in self.subscriptions:
            self.subscriptions[agent_id].discard(topic)
    
    async def handle(self, message: Message) -> Optional[Message]:
        """处理消息 - 用于路由"""
        return message


class RequestReplyHandler(MessageHandler):
    """请求/回复处理器"""
    
    def __init__(self):
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._running = False
    
    async def start(self):
        self._running = True
        
    async def stop(self):
        self._running = False
    
    async def request(self, message: Message, timeout: float = 30.0) -> Message:
        """发送请求并等待回复"""
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[message.correlation_id] = future
        
        await communication_manager.deliver(message)
        
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            del self.pending_requests[message.correlation_id]
            raise TimeoutError(f"Request timeout: {message.correlation_id}")
    
    async def reply(self, original_message: Message, content: Dict[str, Any]):
        """回复请求"""
        reply = Message.create(
            sender_id=original_message.receiver_id,
            receiver_id=original_message.sender_id,
            content=content,
            msg_type=MessageType.RESPONSE,
            correlation_id=original_message.correlation_id
        )
        
        future = self.pending_requests.pop(original_message.correlation_id, None)
        if future and not future.done():
            future.set_result(reply)
        
        await communication_manager.deliver(reply)
    
    async def handle(self, message: Message) -> Optional[Message]:
        """处理消息"""
        if message.type == MessageType.REQUEST:
            # 保存请求以供处理
            logger.info(f"Request received: {message.id}")
        elif message.type == MessageType.RESPONSE:
            # 唤醒等待的请求
            future = self.pending_requests.get(message.correlation_id)
            if future and not future.done():
                future.set_result(message)
        
        return message


class CommunicationManager:
    """
    Agent通信管理器
    
    核心功能：
    - 消息路由与传递
    - Agent注册与发现
    - 多种通信协议支持
    - 消息序列化与反序列化
    - 通信监控
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.agents: Dict[str, AgentEndpoint] = {}
        self.handlers: Dict[str, MessageHandler] = {}
        self.message_history: List[Message] = []
        self.direct_handler = DirectHandler("manager")
        self.pubsub_handler = PubSubHandler()
        self.request_reply_handler = RequestReplyHandler()
        self.event_callbacks: List[Callable] = None
        self._running = False
        
        self._initialized = True
    
    async def start(self):
        """启动通信管理器"""
        self._running = True
        await self.direct_handler.start()
        await self.pubsub_handler.start()
        await self.request_reply_handler.start()
        logger.info("Communication Manager started")
    
    async def stop(self):
        """停止通信管理器"""
        self._running = False
        await self.direct_handler.stop()
        await self.pubsub_handler.stop()
        await self.request_reply_handler.stop()
        logger.info("Communication Manager stopped")
    
    def register_agent(self, agent_id: str, name: str,
                       capabilities: List[str] = None,
                       protocols: List[CommunicationProtocol] = None,
                       metadata: Dict[str, Any] = None) -> AgentEndpoint:
        """注册Agent"""
        endpoint = AgentEndpoint(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities or [],
            protocols=protocols or [CommunicationProtocol.DIRECT],
            metadata=metadata or {}
        )
        
        self.agents[agent_id] = endpoint
        
        # 创建处理器
        handler = DirectHandler(agent_id)
        self.handlers[agent_id] = handler
        
        self._emit_event("agent_registered", endpoint)
        logger.info(f"Registered agent: {agent_id}")
        
        return endpoint
    
    def unregister_agent(self, agent_id: str) -> bool:
        """注销Agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.handlers:
                del self.handlers[agent_id]
            self._emit_event("agent_unregistered", {"agent_id": agent_id})
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentEndpoint]:
        """获取Agent信息"""
        return self.agents.get(agent_id)
    
    def list_agents(self, capability: str = None) -> List[Dict[str, Any]]:
        """列出Agent"""
        agents = list(self.agents.values())
        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        return [a.to_dict() for a in agents]
    
    async def deliver(self, message: Message) -> bool:
        """传递消息到接收者"""
        self.message_history.append(message)
        
        if message.receiver_id in self.handlers:
            handler = self.handlers[message.receiver_id]
            await handler.handle(message)
            self._emit_event("message_delivered", message.to_dict())
            return True
        
        self._emit_event("message_failed", {
            "message_id": message.id,
            "reason": "Receiver not found"
        })
        return False
    
    async def deliver_to_agent(self, agent_id: str, message: Message):
        """传递消息到指定Agent"""
        message.receiver_id = agent_id
        await self.deliver(message)
    
    async def send_direct(self, sender_id: str, receiver_id: str,
                          content: Dict[str, Any],
                          metadata: Dict[str, Any] = None) -> Message:
        """发送直接消息"""
        message = Message.create(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            msg_type=MessageType.REQUEST,
            protocol=CommunicationProtocol.DIRECT,
            metadata=metadata
        )
        
        await self.deliver(message)
        return message
    
    async def send_request(self, sender_id: str, receiver_id: str,
                           content: Dict[str, Any],
                           timeout: float = 30.0) -> Message:
        """发送请求并等待回复"""
        message = Message.create(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            msg_type=MessageType.REQUEST,
            protocol=CommunicationProtocol.REQUEST_REPLY
        )
        
        return await self.request_reply_handler.request(message, timeout)
    
    async def send_reply(self, original_message: Message, content: Dict[str, Any]):
        """发送回复"""
        await self.request_reply_handler.reply(original_message, content)
    
    async def publish(self, topic: str, sender_id: str,
                      content: Dict[str, Any],
                      metadata: Dict[str, Any] = None):
        """发布消息到主题"""
        message = Message.create(
            sender_id=sender_id,
            receiver_id="",
            content=content,
            msg_type=MessageType.BROADCAST,
            protocol=CommunicationProtocol.PUBLISH_SUBSCRIBE,
            metadata={**(metadata or {}), "topic": topic}
        )
        
        await self.pubsub_handler.publish(topic, message)
    
    async def subscribe(self, agent_id: str, topic: str):
        """订阅主题"""
        await self.pubsub_handler.subscribe(agent_id, topic)
    
    async def unsubscribe(self, agent_id: str, topic: str):
        """取消订阅"""
        await self.pubsub_handler.unsubscribe(agent_id, topic)
    
    async def broadcast(self, sender_id: str, content: Dict[str, Any],
                       exclude: List[str] = None):
        """广播消息到所有Agent"""
        for agent_id in self.agents:
            if exclude and agent_id in exclude:
                continue
            await self.send_direct(sender_id, agent_id, content)
    
    def get_message_history(self, agent_id: str = None,
                           msg_type: MessageType = None) -> List[Dict[str, Any]]:
        """获取消息历史"""
        messages = self.message_history
        
        if agent_id:
            messages = [m for m in messages 
                       if m.sender_id == agent_id or m.receiver_id == agent_id]
        
        if msg_type:
            messages = [m for m in messages if m.type == msg_type]
        
        return [m.to_dict() for m in messages]
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        total = len(self.message_history)
        by_type = {}
        for msg in self.message_history:
            t = msg.type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total_messages": total,
            "messages_by_type": by_type,
            "registered_agents": len(self.agents),
            "active_handlers": len(self.handlers)
        }
    
    def _emit_event(self, event_type: str, data: Any):
        """触发事件"""
        if self.event_callbacks:
            for callback in self.event_callbacks:
                try:
                    callback(event_type, data)
                except Exception as e:
                    logger.error(f"Communication event callback error: {e}")
    
    def on_event(self, callback: Callable):
        """注册事件回调"""
        self.event_callbacks.append(callback)


# 全局通信管理器实例
communication_manager = CommunicationManager()
