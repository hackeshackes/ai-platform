"""
Real-time API Endpoints - v4.0

实时推理与流式输出API端点
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel, Field

from realtime.websocket import connection_manager
from realtime.streaming import streaming_service, StreamState

# SSE支持（可选依赖）
try:
    from realtime.sse import sse_manager, SSEEventType
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False


router = APIRouter()


# ==================== 请求/响应模型 ====================

class InferenceRequest(BaseModel):
    """推理请求"""
    prompt: str = Field(..., description="输入提示词")
    model: str = Field(default="default", description="模型名称")
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=32768, description="最大输出token")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    stream: bool = Field(default=True, description="是否流式输出")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="附加元数据")


class InferenceResponse(BaseModel):
    """推理响应"""
    job_id: str
    status: str
    created_at: str


class TokenCountRequest(BaseModel):
    """Token计数请求"""
    prompt: str = Field(..., description="输入文本")
    completion: Optional[str] = Field(default=None, description="输出文本")
    model: str = Field(..., description="模型名称")


class TokenCountResponse(BaseModel):
    """Token计数响应"""
    job_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


class StreamStatusResponse(BaseModel):
    """流状态响应"""
    job_id: str
    state: str
    chunks_count: int
    total_tokens: int
    created_at: str
    completed_at: Optional[str] = None
    cost: float


# ==================== WebSocket端点 ====================

@router.websocket("/connect")
async def websocket_connect(websocket: WebSocket):
    """
    WebSocket连接端点
    
    建立实时双向通信连接
    """
    client_id = await connection_manager.connect(websocket)
    
    # 发送连接确认
    await connection_manager.send_personal_message(
        client_id,
        {
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "WebSocket连接已建立"
        }
    )


# ==================== 流式输出端点 ====================

@router.post("/inference", response_model=InferenceResponse)
async def create_inference(request: InferenceRequest):
    """
    创建流式推理任务
    
    启动一个新的流式推理任务，返回job_id
    """
    job_id = await streaming_service.start_streaming(
        prompt=request.prompt,
        model=request.model,
        metadata={
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            **(request.metadata or {})
        }
    )
    
    return InferenceResponse(
        job_id=job_id,
        status="started",
        created_at=datetime.utcnow().isoformat()
    )


@router.get("/stream/{job_id}")
async def get_stream(job_id: str):
    """
    获取流式输出
    
    通过SSE方式获取流式推理结果
    """
    if not streaming_service.stream_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    
    return streaming_service.stream_manager.stream_generator(job_id)


@router.get("/stream/{job_id}/status", response_model=StreamStatusResponse)
async def get_stream_status(job_id: str):
    """
    获取流状态
    
    查询流式任务的当前状态
    """
    job = streaming_service.stream_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    stats = streaming_service.stream_manager.get_job_statistics(job_id)
    
    return StreamStatusResponse(
        job_id=job_id,
        state=stats["state"],
        chunks_count=stats["chunks_count"],
        total_tokens=stats["total_tokens"],
        created_at=stats["created_at"],
        completed_at=stats["completed_at"],
        cost=stats["cost"]
    )


# ==================== Token计数端点 ====================

@router.post("/token", response_model=TokenCountResponse)
async def count_tokens(request: TokenCountRequest, job_id: Optional[str] = Query(None)):
    """
    实时Token计数
    
    计算输入输出token数并估算成本
    """
    if job_id is None:
        job_id = f"token_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    # 计数tokens
    token_info = await streaming_service.token_counter.count_tokens(
        job_id=job_id,
        prompt=request.prompt,
        completion=request.completion or "",
        model=request.model
    )
    
    # 计算成本
    estimated_cost = streaming_service.streaming_service.calculate_cost(
        model=request.model,
        prompt_tokens=token_info["prompt_tokens"],
        completion_tokens=token_info["completion_tokens"]
    )
    
    return TokenCountResponse(
        job_id=job_id,
        prompt_tokens=token_info["prompt_tokens"],
        completion_tokens=token_info["completion_tokens"],
        total_tokens=token_info["total_tokens"],
        estimated_cost=estimated_cost
    )


# ==================== SSE端点（可选） ====================

if SSE_AVAILABLE:

    @router.get("/sse/{stream_id}")
    async def sse_stream(stream_id: str):
        """
        SSE流式输出
        
        通过Server-Sent Events获取实时更新
        """
        from realtime.sse import create_sse_response
        
        if not sse_manager.is_active(stream_id):
            # 创建新流
            await sse_manager.create_stream(
                stream_id=stream_id,
                metadata={"created_at": datetime.utcnow().isoformat()}
            )
        
        return create_sse_response(stream_id, sse_manager)

    @router.post("/sse/{stream_id}/send")
    async def sse_send_event(
        stream_id: str,
        event_type: str,
        data: Dict[str, Any]
    ):
        """
        发送SSE事件
        
        向指定SSE流发送事件
        """
        if not sse_manager.is_active(stream_id):
            raise HTTPException(status_code=404, detail="Stream not found")
        
        try:
            event_enum = SSEEventType(event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid event type")
        
        await sse_manager.send_event(stream_id, event_enum, data)
        
        return {"status": "sent", "stream_id": stream_id}

    @router.delete("/sse/{stream_id}")
    async def sse_close_stream(stream_id: str):
        """关闭SSE流"""
        await sse_manager.close_stream(stream_id)
        return {"status": "closed", "stream_id": stream_id}

    SSE_STATS_KEY = "sse"
else:

    @router.get("/sse/{stream_id}")
    async def sse_stream(stream_id: str):
        """SSE不可用"""
        raise HTTPException(status_code=503, detail="SSE not available. Install sse-starlette.")

    @router.post("/sse/{stream_id}/send")
    async def sse_send_event(stream_id: str, event_type: str, data: Dict[str, Any]):
        raise HTTPException(status_code=503, detail="SSE not available")

    @router.delete("/sse/{stream_id}")
    async def sse_close_stream(stream_id: str):
        raise HTTPException(status_code=503, detail="SSE not available")

    SSE_STATS_KEY = None


# ==================== 统计端点 ====================

@router.get("/stats")
async def get_realtime_stats():
    """
    获取实时统计
    
    返回当前实时服务的统计信息
    """
    stats = {
        "websocket": {
            "active_connections": connection_manager.get_active_connections_count()
        },
        "streaming": {
            "active_jobs": streaming_service.stream_manager.get_active_jobs_count()
        }
    }
    
    if SSE_AVAILABLE and SSE_STATS_KEY:
        stats[SSE_STATS_KEY] = {
            "active_streams": len(sse_manager.active_streams)
        }
    
    return stats


@router.get("/health")
async def realtime_health():
    """实时服务健康检查"""
    return {
        "status": "healthy",
        "service": "realtime",
        "timestamp": datetime.utcnow().isoformat()
    }
