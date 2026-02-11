"""
SSE Support - Server-Sent Events

SSE支持，用于向客户端推送实时更新
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# 尝试导入sse_starlette，如果不可用则使用备用实现
try:
    from sse_starlette.sse import EventSourceResponse
    SSE_STARLETTE_AVAILABLE = True
except ImportError:
    SSE_STARLETTE_AVAILABLE = False
    EventSourceResponse = None

from fastapi import Request, APIRouter
from starlette.responses import StreamingResponse


class SSEEventType(Enum):
    """SSE事件类型"""
    START = "start"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    COMPLETE = "complete"
    ERROR = "error"
    PROGRESS = "progress"


@dataclass
class SSEEvent:
    """SSE事件"""
    event_type: SSEEventType
    data: Dict[str, Any]
    event_id: Optional[str] = None
    retry: Optional[int] = None  # 重连时间（毫秒）
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SSEManager:
    """
    SSE管理器
    
    管理Server-Sent Events连接和事件推送
    """
    
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.stream_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def create_stream(
        self,
        stream_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建SSE流"""
        if stream_id is None:
            stream_id = str(uuid.uuid4())[:12]
        
        async with self._lock:
            self.active_streams[stream_id] = asyncio.Queue()
            self.stream_metadata[stream_id] = metadata or {}
        
        return stream_id
    
    async def close_stream(self, stream_id: str):
        """关闭SSE流"""
        async with self._lock:
            if stream_id in self.active_streams:
                await self.active_streams[stream_id].put(None)
                del self.active_streams[stream_id]
            
            if stream_id in self.stream_metadata:
                del self.stream_metadata[stream_id]
    
    async def send_event(
        self,
        stream_id: str,
        event_type: SSEEventType,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
        retry: Optional[int] = None
    ):
        """发送SSE事件"""
        if stream_id not in self.active_streams:
            return
        
        event = SSEEvent(
            event_type=event_type,
            data=data,
            event_id=event_id or str(uuid.uuid4())[:8],
            retry=retry
        )
        
        await self.active_streams[stream_id].put(event)
    
    async def send_data(
        self,
        stream_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """发送数据事件"""
        await self.send_event(
            stream_id,
            SSEEventType.DATA,
            {"content": content, **(metadata or {})}
        )
    
    async def send_progress(
        self,
        stream_id: str,
        progress: float,
        message: Optional[str] = None
    ):
        """发送进度事件"""
        await self.send_event(
            stream_id,
            SSEEventType.PROGRESS,
            {"progress": progress, "message": message}
        )
    
    async def send_complete(self, stream_id: str, summary: Optional[Dict[str, Any]] = None):
        """发送完成事件"""
        await self.send_event(
            stream_id,
            SSEEventType.COMPLETE,
            summary or {}
        )
    
    async def send_error(self, stream_id: str, error_message: str):
        """发送错误事件"""
        await self.send_event(
            stream_id,
            SSEEventType.ERROR,
            {"error": error_message},
            retry=5000
        )
    
    def is_active(self, stream_id: str) -> bool:
        """检查流是否活跃"""
        return stream_id in self.active_streams
    
    def get_metadata(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流元数据"""
        return self.stream_metadata.get(stream_id)
    
    async def event_generator(
        self,
        stream_id: str,
        heartbeat_interval: float = 15.0
    ) -> AsyncGenerator[str, None]:
        """
        生成SSE事件数据
        
        Args:
            stream_id: 流ID
            heartbeat_interval: 心跳间隔（秒）
            
        Yields:
            SSE格式的事件字符串
        """
        queue = self.active_streams.get(stream_id)
        if queue is None:
            return
        
        heartbeat_task = None
        
        try:
            # 启动心跳
            async def send_heartbeat():
                while True:
                    await asyncio.sleep(heartbeat_interval)
                    event = SSEEvent(
                        event_type=SSEEventType.HEARTBEAT,
                        data={"timestamp": datetime.utcnow().isoformat()}
                    )
                    await queue.put(event)
            
            heartbeat_task = asyncio.create_task(send_heartbeat())
            
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    
                    if event is None:
                        # 流结束
                        break
                    
                    # 格式化SSE事件
                    sse_data = self._format_sse_event(event)
                    yield sse_data
                    
                    if event.event_type == SSEEventType.COMPLETE:
                        break
                        
                except asyncio.TimeoutError:
                    # 发送心跳
                    heartbeat = f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"
                    yield heartbeat
        
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
    
    def _format_sse_event(self, event: SSEEvent) -> str:
        """格式化SSE事件"""
        lines = []
        
        if event.event_id:
            lines.append(f"id: {event.event_id}")
        
        lines.append(f"event: {event.event_type.value}")
        
        data = {
            "type": event.event_type.value,
            "data": event.data,
            "timestamp": event.timestamp.isoformat()
        }
        lines.append(f"data: {json.dumps(data)}")
        
        if event.retry:
            lines.append(f"retry: {event.retry}")
        
        return "\n".join(lines) + "\n\n"


def create_sse_response(
    stream_id: str,
    manager: SSEManager,
    heartbeat_interval: float = 15.0
) -> StreamingResponse:
    """
    创建SSE响应
    
    Args:
        stream_id: 流ID
        manager: SSE管理器
        heartbeat_interval: 心跳间隔
        
    Returns:
        StreamingResponse
    """
    async def event_stream():
        async for event_data in manager.event_generator(stream_id, heartbeat_interval):
            yield event_data
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        }
    )


class StreamingSSEAdapter:
    """
    流式输出适配器
    
    将其他流式输出转换为SSE格式
    """
    
    def __init__(self, sse_manager: SSEManager):
        self.sse_manager = sse_manager
    
    async def stream_from_queue(
        self,
        stream_id: str,
        queue: asyncio.Queue
    ):
        """从队列流式输出"""
        try:
            # 发送开始事件
            await self.sse_manager.send_event(
                stream_id,
                SSEEventType.START,
                {"stream_id": stream_id}
            )
            
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    
                    if data is None:
                        break
                    
                    await self.sse_manager.send_data(stream_id, data)
                    
                except asyncio.TimeoutError:
                    await self.sse_manager.send_event(
                        stream_id,
                        SSEEventType.HEARTBEAT,
                        {"timestamp": datetime.utcnow().isoformat()}
                    )
            
            # 发送完成事件
            await self.sse_manager.send_complete(stream_id)
            
        except Exception as e:
            await self.sse_manager.send_error(stream_id, str(e))
        finally:
            await self.sse_manager.close_stream(stream_id)


# 全局SSE管理器实例
sse_manager = SSEManager()
