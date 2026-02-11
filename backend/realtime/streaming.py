"""
Streaming Output - Real-time Inference

流式输出支持，实现实时推理结果推送
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager


class StreamState(Enum):
    """流状态"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StreamChunk:
    """流数据块"""
    chunk_id: str
    job_id: str
    sequence: int
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False
    token_info: Optional[Dict[str, Any]] = None


@dataclass
class StreamJob:
    """流任务"""
    job_id: str
    state: StreamState = StreamState.IDLE
    chunks: List[StreamChunk] = field(default_factory=list)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0


class StreamManager:
    """
    流式输出管理器
    
    管理流式任务的生命周期和输出
    """
    
    def __init__(self):
        self.active_streams: Dict[str, StreamJob] = {}
        self.stream_queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
    
    async def create_job(
        self,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建流任务"""
        if job_id is None:
            job_id = str(uuid.uuid4())[:12]
        
        async with self._lock:
            self.active_streams[job_id] = StreamJob(
                job_id=job_id,
                state=StreamState.STARTING,
                metadata=metadata or {}
            )
            self.stream_queues[job_id] = asyncio.Queue()
        
        return job_id
    
    async def start_job(self, job_id: str):
        """启动流任务"""
        async with self._lock:
            if job_id in self.active_streams:
                self.active_streams[job_id].state = StreamState.RUNNING
    
    async def add_chunk(
        self,
        job_id: str,
        content: str,
        sequence: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_info: Optional[Dict[str, Any]] = None
    ):
        """添加流数据块"""
        async with self._lock:
            if job_id not in self.active_streams:
                return
            
            job = self.active_streams[job_id]
            if sequence is None:
                sequence = len(job.chunks)
            
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4())[:8],
                job_id=job_id,
                sequence=sequence,
                content=content,
                metadata=metadata or {},
                token_info=token_info
            )
            
            job.chunks.append(chunk)
            
            # 更新token计数
            if token_info:
                job.prompt_tokens = token_info.get("prompt_tokens", 0)
                job.completion_tokens = token_info.get("completion_tokens", 0)
                job.total_tokens = job.prompt_tokens + job.completion_tokens
            
            # 发送到队列
            if job_id in self.stream_queues:
                await self.stream_queues[job_id].put(chunk)
    
    async def complete_job(self, job_id: str, final_metadata: Optional[Dict[str, Any]] = None):
        """完成流任务"""
        async with self._lock:
            if job_id not in self.active_streams:
                return
            
            job = self.active_streams[job_id]
            job.state = StreamState.COMPLETED
            job.completed_at = datetime.utcnow()
            if final_metadata:
                job.metadata.update(final_metadata)
            
            # 发送结束标记
            if job_id in self.stream_queues:
                end_chunk = StreamChunk(
                    chunk_id="end",
                    job_id=job_id,
                    sequence=len(job.chunks),
                    content="",
                    is_final=True
                )
                await self.stream_queues[job_id].put(end_chunk)
                # 清理队列
                del self.stream_queues[job_id]
    
    async def error_job(self, job_id: str, error_message: str):
        """流任务错误"""
        async with self._lock:
            if job_id not in self.active_streams:
                return
            
            job = self.active_streams[job_id]
            job.state = StreamState.ERROR
            job.metadata["error"] = error_message
            
            if job_id in self.stream_queues:
                error_chunk = StreamChunk(
                    chunk_id="error",
                    job_id=job_id,
                    sequence=len(job.chunks),
                    content=error_message,
                    is_final=True
                )
                await self.stream_queues[job_id].put(error_chunk)
                del self.stream_queues[job_id]
    
    def get_job(self, job_id: str) -> Optional[StreamJob]:
        """获取任务信息"""
        return self.active_streams.get(job_id)
    
    async def stream_generator(
        self,
        job_id: str,
        chunk_timeout: float = 30.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        生成流数据
        
        Args:
            job_id: 任务ID
            chunk_timeout: 块超时时间
            
        Yields:
            流数据字典
        """
        queue = self.stream_queues.get(job_id)
        if queue is None:
            return
        
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=chunk_timeout)
                    
                    yield {
                        "job_id": chunk.job_id,
                        "chunk_id": chunk.chunk_id,
                        "sequence": chunk.sequence,
                        "content": chunk.content,
                        "timestamp": chunk.timestamp.isoformat(),
                        "metadata": chunk.metadata,
                        "is_final": chunk.is_final,
                        "token_info": chunk.token_info
                    }
                    
                    if chunk.is_final:
                        break
                        
                except asyncio.TimeoutError:
                    # 发送心跳
                    yield {
                        "type": "heartbeat",
                        "job_id": job_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        except asyncio.CancelledError:
            # 清理
            if job_id in self.stream_queues:
                del self.stream_queues[job_id]
    
    def get_active_jobs_count(self) -> int:
        """获取活跃任务数"""
        return len([
            job for job in self.active_streams.values()
            if job.state in (StreamState.RUNNING, StreamState.STARTING)
        ])
    
    def get_job_statistics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务统计"""
        job = self.active_streams.get(job_id)
        if job is None:
            return None
        
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "chunks_count": len(job.chunks),
            "total_tokens": job.total_tokens,
            "prompt_tokens": job.prompt_tokens,
            "completion_tokens": job.completion_tokens,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "cost": job.cost,
            "duration_seconds": (
                (job.completed_at - job.created_at).total_seconds()
                if job.completed_at else None
            )
        }


class StreamingService:
    """
    流式服务
    
    提供高-level流式输出功能
    """
    
    def __init__(self):
        self.stream_manager = StreamManager()
        self.token_counter = TokenCounter()
    
    async def start_streaming(
        self,
        prompt: str,
        model: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """开始流式推理"""
        job_id = await self.stream_manager.create_job(metadata={
            "prompt": prompt,
            "model": model,
            **(metadata or {})
        })
        
        # 模拟开始流式输出
        asyncio.create_task(self._simulate_streaming(job_id, prompt, model))
        
        return job_id
    
    async def _simulate_streaming(
        self,
        job_id: str,
        prompt: str,
        model: str
    ):
        """模拟流式输出（实际应调用LLM提供商）"""
        await self.stream_manager.start_job(job_id)
        
        # 模拟tokens
        words = prompt.split()
        for i, word in enumerate(words[:10]):  # 简化模拟
            content = f"{word} "
            token_info = {
                "prompt_tokens": len(words),
                "completion_tokens": i + 1,
                "total_tokens": len(words) + i + 1
            }
            
            await self.stream_manager.add_chunk(
                job_id=job_id,
                content=content,
                token_info=token_info
            )
            
            await asyncio.sleep(0.1)  # 模拟延迟
        
        # 完成
        await self.stream_manager.complete_job(job_id)
    
    async def get_stream(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """获取流"""
        async for chunk in self.stream_manager.stream_generator(job_id):
            yield chunk
    
    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """计算成本"""
        rates = {
            "gpt-4": (0.03, 0.06),
            "gpt-3.5-turbo": (0.001, 0.002),
            "claude-3-opus": (0.015, 0.075),
            "default": (0.01, 0.03)
        }
        
        input_rate, output_rate = rates.get(model, rates["default"])
        return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1000


class TokenCounter:
    """
    Token计数器
    
    实时追踪token使用量和成本
    """
    
    def __init__(self):
        self.usage_records: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def count_tokens(
        self,
        job_id: str,
        prompt: str,
        completion: str,
        model: str
    ) -> Dict[str, Any]:
        """计数tokens"""
        # 简化计算（实际应使用tokenizer）
        prompt_tokens = len(prompt.split())
        completion_tokens = len(completion.split())
        total_tokens = prompt_tokens + completion_tokens
        
        async with self._lock:
            self.usage_records[job_id] = {
                "job_id": job_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": model,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    
    def get_usage(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取使用记录"""
        return self.usage_records.get(job_id)
    
    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        """获取所有使用记录"""
        return self.usage_records.copy()


# 全局服务实例
streaming_service = StreamingService()
