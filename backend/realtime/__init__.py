"""
Real-time Module - v4.0

实时推理与流式输出模块
"""

from .websocket import connection_manager, ConnectionManager
from .streaming import streaming_service, StreamingService, StreamManager

# SSE支持（可选依赖）
try:
    from .sse import sse_manager, SSEManager
    __all__ = [
        "connection_manager",
        "ConnectionManager",
        "streaming_service",
        "StreamingService",
        "StreamManager",
        "sse_manager",
        "SSEManager"
    ]
except ImportError:
    __all__ = [
        "connection_manager",
        "ConnectionManager",
        "streaming_service",
        "StreamingService",
        "StreamManager"
    ]
