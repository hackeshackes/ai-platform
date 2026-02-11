"""
Agent间通信模块
提供Agent之间的消息传递、事件通知和通信协议
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
from enum import Enum
import logging

from .models import Message, SessionStatus

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """消息类型"""
    TASK = "task"                   # 任务消息
    RESULT = "result"               # 结果消息
    EVENT = "event"                 # 事件消息
    CONTROL = "control"             # 控制消息
    QUERY = "query"                 # 查询消息
    RESPONSE = "response"           # 响应消息
    ERROR = "error"                 # 错误消息
    HEARTBEAT = "heartbeat"         # 心跳消息
    SYNC = "sync"                   # 同步消息


class CommunicationChannel:
    """通信通道"""
    
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self.subscribers: Set[str] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    async def start(self) -> None:
        """启动通道"""
        self._running = True
        logger.info(f"Communication channel {self.channel_id} started")
    
    async def stop(self) -> None:
        """停止通道"""
        self._running = False
        logger.info(f"Communication channel {self.channel_id} stopped")
    
    async def publish(self, message: Message) -> None:
        """发布消息"""
        await self.message_queue.put(message)
        logger.debug(f"Message published to channel {self.channel_id}: {message.message_id}")
    
    async def subscribe(self, agent_id: str) -> asyncio.Queue:
        """订阅消息"""
        self.subscribers.add(agent_id)
        return self.message_queue
    
    def unsubscribe(self, agent_id: str) -> None:
        """取消订阅"""
        self.subscribers.discard(agent_id)


class MessageRouter:
    """消息路由器"""
    
    def __init__(self):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.direct_messages: Dict[str, asyncio.Queue] = {}
        self.broadcast_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    async def create_channel(self, channel_id: str) -> CommunicationChannel:
        """创建通信通道"""
        async with self._lock:
            if channel_id not in self.channels:
                self.channels[channel_id] = CommunicationChannel(channel_id)
                await self.channels[channel_id].start()
            return self.channels[channel_id]
    
    async def delete_channel(self, channel_id: str) -> None:
        """删除通信通道"""
        async with self._lock:
            if channel_id in self.channels:
                await self.channels[channel_id].stop()
                del self.channels[channel_id]
    
    async def send_direct(self, message: Message) -> None:
        """发送直接消息"""
        queue = self.direct_messages.get(message.receiver_id)
        if queue:
            await queue.put(message)
        else:
            logger.warning(f"No message queue for agent {message.receiver_id}")
    
    async def register_agent(self, agent_id: str) -> asyncio.Queue:
        """注册Agent"""
        queue = asyncio.Queue()
        self.direct_messages[agent_id] = queue
        logger.info(f"Agent {agent_id} registered for messaging")
        return queue
    
    async def unregister_agent(self, agent_id: str) -> None:
        """注销Agent"""
        if agent_id in self.direct_messages:
            del self.direct_messages[agent_id]
            logger.info(f"Agent {agent_id} unregistered")
    
    def add_broadcast_handler(self, handler: Callable) -> None:
        """添加广播处理器"""
        self.broadcast_handlers.append(handler)
    
    async def broadcast(self, message: Message) -> None:
        """广播消息"""
        for handler in self.broadcast_handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Broadcast handler error: {e}")


class CommunicationManager:
    """通信管理器"""
    
    def __init__(self):
        self.router = MessageRouter()
        self.session_channels: Dict[str, str] = {}
        self.message_history: List[Message] = []
        self.max_history = 1000
    
    async def initialize(self) -> None:
        """初始化"""
        await self.router.create_channel("global")
        logger.info("Communication Manager initialized")
    
    async def shutdown(self) -> None:
        """关闭"""
        for channel_id in list(self.router.channels.keys()):
            await self.router.delete_channel(channel_id)
        logger.info("Communication Manager shutdown")
    
    async def create_session_channel(self, session_id: str) -> str:
        """为会话创建专用通道"""
        channel_id = f"session_{session_id}"
        await self.router.create_channel(channel_id)
        self.session_channels[session_id] = channel_id
        return channel_id
    
    async def send_message(
        self,
        session_id: str,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        correlation_id: Optional[str] = None
    ) -> Message:
        """发送消息"""
        message = Message(
            session_id=session_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id
        )
        
        await self.router.send_direct(message)
        
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
        
        logger.debug(f"Message sent: {message.message_id}")
        return message
    
    async def broadcast_to_session(
        self,
        session_id: str,
        sender_id: str,
        message_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """向会话中所有Agent广播消息"""
        channel_id = self.session_channels.get(session_id)
        if channel_id and channel_id in self.router.channels:
            channel = self.router.channels[channel_id]
            message = Message(
                session_id=session_id,
                sender_id=sender_id,
                receiver_id="all",
                message_type=message_type,
                payload=payload
            )
            await channel.publish(message)
    
    async def subscribe_session(
        self,
        session_id: str,
        agent_id: str
    ) -> asyncio.Queue:
        """订阅会话消息"""
        channel_id = self.session_channels.get(session_id)
        if channel_id and channel_id in self.router.channels:
            return await self.router.channels[channel_id].subscribe(agent_id)
        return asyncio.Queue()
    
    async def register_agent(self, agent_id: str) -> asyncio.Queue:
        """注册Agent"""
        return await self.router.register_agent(agent_id)
    
    async def receive_message(self, agent_id: str, timeout: float = 10.0) -> Optional[Message]:
        """接收消息（带超时）"""
        queue = self.router.direct_messages.get(agent_id)
        if queue:
            try:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
        return None
    
    async def send_task(
        self,
        session_id: str,
        task_id: str,
        sender_id: str,
        receiver_id: str,
        task_data: Dict[str, Any]
    ) -> Message:
        """发送任务消息"""
        return await self.send_message(
            session_id=session_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.TASK.value,
            payload={
                "task_id": task_id,
                "data": task_data,
                "action": "execute"
            }
        )
    
    async def send_result(
        self,
        session_id: str,
        task_id: str,
        sender_id: str,
        receiver_id: str,
        result_data: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ) -> Message:
        """发送结果消息"""
        return await self.send_message(
            session_id=session_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.RESULT.value,
            payload={
                "task_id": task_id,
                "success": success,
                "result": result_data,
                "error": error
            },
            correlation_id=task_id
        )
    
    async def send_event(
        self,
        session_id: str,
        sender_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """发送事件"""
        await self.broadcast_to_session(
            session_id=session_id,
            sender_id=sender_id,
            message_type=MessageType.EVENT.value,
            payload={
                "event_type": event_type,
                "data": event_data
            }
        )
    
    def get_message_history(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        message_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Message]:
        """获取消息历史"""
        history = self.message_history
        
        if session_id:
            history = [m for m in history if m.session_id == session_id]
        if agent_id:
            history = [m for m in history if m.sender_id == agent_id or m.receiver_id == agent_id]
        if message_type:
            history = [m for m in history if m.message_type == message_type]
        
        return history[-limit:]
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        return {
            "active_channels": len(self.router.channels),
            "registered_agents": len(self.router.direct_messages),
            "message_count": len(self.message_history),
            "session_channels": len(self.session_channels)
        }


# 全局通信管理器实例
_communication_manager: Optional[CommunicationManager] = None


def get_communication_manager() -> CommunicationManager:
    """获取全局通信管理器"""
    global _communication_manager
    if _communication_manager is None:
        _communication_manager = CommunicationManager()
    return _communication_manager


async def init_communication() -> CommunicationManager:
    """初始化通信管理器"""
    manager = get_communication_manager()
    await manager.initialize()
    return manager


async def shutdown_communication() -> None:
    """关闭通信管理器"""
    global _communication_manager
    if _communication_manager:
        await _communication_manager.shutdown()
        _communication_manager = None
