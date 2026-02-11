"""
WebSocket Handler - Real-time Inference

WebSocket处理器，支持实时双向通信
"""
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionState(Enum):
    """连接状态"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    state: ConnectionState = ConnectionState.CONNECTING
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionManager:
    """
    WebSocket连接管理器
    
    管理所有WebSocket连接，支持广播和特定客户端消息
    """
    
    def __init__(self):
        self.active_connections: Dict[str, ClientInfo] = {}
        self.client_subscriptions: Dict[str, set] = defaultdict(set)  # job_id -> client_ids
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        建立WebSocket连接
        
        Args:
            websocket: FastAPI WebSocket对象
            client_id: 客户端ID (可选，自动生成)
            user_id: 用户ID
            metadata: 附加元数据
            
        Returns:
            客户端ID
        """
        await websocket.accept()
        
        if client_id is None:
            client_id = str(uuid.uuid4())[:8]
        
        client_info = ClientInfo(
            client_id=client_id,
            websocket=websocket,
            user_id=user_id,
            state=ConnectionState.CONNECTED,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.active_connections[client_id] = client_info
        
        # 启动连接监听
        asyncio.create_task(self._handle_connection(client_id))
        
        return client_id
    
    async def disconnect(self, client_id: str):
        """断开连接"""
        async with self._lock:
            if client_id in self.active_connections:
                client_info = self.active_connections[client_id]
                client_info.state = ConnectionState.DISCONNECTED
                del self.active_connections[client_id]
        
        # 清理订阅
        for job_id in list(self.client_subscriptions.keys()):
            self.client_subscriptions[job_id].discard(client_id)
            if not self.client_subscriptions[job_id]:
                del self.client_subscriptions[job_id]
    
    async def send_personal_message(self, client_id: str, message: Dict[str, Any]):
        """发送消息给特定客户端"""
        if client_id in self.active_connections:
            client_info = self.active_connections[client_id]
            try:
                await client_info.websocket.send_json(message)
            except Exception:
                await self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[set] = None):
        """广播消息给所有连接"""
        exclude = exclude or set()
        disconnected = []
        
        for client_id, client_info in self.active_connections.items():
            if client_id in exclude:
                continue
            
            try:
                await client_info.websocket.send_json(message)
            except Exception:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            await self.disconnect(client_id)
    
    async def send_to_subscribers(self, job_id: str, message: Dict[str, Any]):
        """发送消息给订阅特定job的所有客户端"""
        subscribers = self.client_subscriptions.get(job_id, set())
        await self.broadcast(message, exclude=set(self.active_connections.keys()) - subscribers)
    
    def subscribe(self, client_id: str, job_id: str):
        """客户端订阅job"""
        self.client_subscriptions[job_id].add(client_id)
    
    def unsubscribe(self, client_id: str, job_id: str):
        """取消订阅"""
        if job_id in self.client_subscriptions:
            self.client_subscriptions[job_id].discard(client_id)
    
    async def _handle_connection(self, client_id: str):
        """处理连接消息"""
        try:
            client_info = self.active_connections.get(client_id)
            if client_info is None:
                return
            
            while client_info.state == ConnectionState.CONNECTED:
                try:
                    data = await asyncio.wait_for(
                        client_info.websocket.receive_text(),
                        timeout=300.0  # 5分钟超时
                    )
                    await self._handle_message(client_id, data)
                except asyncio.TimeoutError:
                    # 发送心跳
                    await self.send_personal_message(
                        client_id,
                        {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()}
                    )
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await self.disconnect(client_id)
    
    async def _handle_message(self, client_id: str, data: str):
        """处理客户端消息"""
        try:
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "subscribe":
                job_id = message.get("job_id")
                if job_id:
                    self.subscribe(client_id, job_id)
                    await self.send_personal_message(
                        client_id,
                        {"type": "subscribed", "job_id": job_id}
                    )
            
            elif msg_type == "unsubscribe":
                job_id = message.get("job_id")
                if job_id:
                    self.unsubscribe(client_id, job_id)
                    await self.send_personal_message(
                        client_id,
                        {"type": "unsubscribed", "job_id": job_id}
                    )
            
            elif msg_type == "ping":
                await self.send_personal_message(
                    client_id,
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                )
            
            elif msg_type == "inference_request":
                # 处理推理请求
                await self._handle_inference_request(client_id, message)
            
        except json.JSONDecodeError:
            await self.send_personal_message(
                client_id,
                {"type": "error", "message": "Invalid JSON format"}
            )
    
    async def _handle_inference_request(self, client_id: str, message: Dict[str, Any]):
        """处理推理请求"""
        # 这里应该集成到实际的推理服务
        await self.send_personal_message(
            client_id,
            {
                "type": "inference_response",
                "status": "received",
                "request_id": message.get("request_id"),
                "job_id": str(uuid.uuid4())[:8]
            }
        )
    
    def get_connection_info(self, client_id: str) -> Optional[ClientInfo]:
        """获取连接信息"""
        return self.active_connections.get(client_id)
    
    def get_active_connections_count(self) -> int:
        """获取活跃连接数"""
        return len(self.active_connections)


# 全局连接管理器实例
connection_manager = ConnectionManager()
