"""
Realtime Data Handler - AI Platform v5

实时数据处理模块 - 处理训练过程中的实时数据流
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import copy


class TrainingStatus(Enum):
    """训练状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingMetrics:
    """单步训练指标"""
    step: int
    epoch: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Loss
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    
    # Learning Rate
    learning_rate: Optional[float] = None
    
    # GPU
    gpu_utilization: Optional[float] = None
    gpu_memory: Optional[float] = None
    gpu_temperature: Optional[float] = None
    
    # Metrics
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    # Progress
    progress_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory": self.gpu_memory,
            "gpu_temperature": self.gpu_temperature,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "progress_percent": self.progress_percent,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingMetrics":
        """从字典创建"""
        return cls(
            step=data.get("step", 0),
            epoch=data.get("epoch", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            train_loss=data.get("train_loss"),
            val_loss=data.get("val_loss"),
            learning_rate=data.get("learning_rate"),
            gpu_utilization=data.get("gpu_utilization"),
            gpu_memory=data.get("gpu_memory"),
            gpu_temperature=data.get("gpu_temperature"),
            accuracy=data.get("accuracy"),
            f1=data.get("f1"),
            precision=data.get("precision"),
            recall=data.get("recall"),
            progress_percent=data.get("progress_percent"),
        )


@dataclass
class TrainingJob:
    """训练作业"""
    job_id: str
    name: str
    model_name: str
    status: TrainingStatus = TrainingStatus.PENDING
    total_epochs: int = 1
    total_steps: int = 0
    current_epoch: int = 0
    current_step: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # 数据存储
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    
    # 配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "model_name": self.model_name,
            "status": self.status.value,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metrics_count": len(self.metrics_history),
            "config": self.config,
        }
    
    def get_loss_data(self) -> Dict[str, List[float]]:
        """获取Loss数据"""
        train_loss = [m.train_loss for m in self.metrics_history if m.train_loss is not None]
        val_loss = [m.val_loss for m in self.metrics_history if m.val_loss is not None]
        return {"train": train_loss, "val": val_loss}
    
    def get_gpu_data(self) -> Dict[str, List[float]]:
        """获取GPU数据"""
        return {
            "utilization": [m.gpu_utilization for m in self.metrics_history if m.gpu_utilization is not None],
            "memory": [m.gpu_memory for m in self.metrics_history if m.gpu_memory is not None],
            "temperature": [m.gpu_temperature for m in self.metrics_history if m.gpu_temperature is not None],
        }
    
    def get_lr_data(self) -> List[float]:
        """获取学习率数据"""
        return [m.learning_rate for m in self.metrics_history if m.learning_rate is not None]
    
    def get_metrics_data(self) -> Dict[str, List[float]]:
        """获取评估指标数据"""
        return {
            "accuracy": [m.accuracy for m in self.metrics_history if m.accuracy is not None],
            "f1": [m.f1 for m in self.metrics_history if m.f1 is not None],
            "precision": [m.precision for m in self.metrics_history if m.precision is not None],
            "recall": [m.recall for m in self.metrics_history if m.recall is not None],
        }
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """获取最新指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None


class TrainingDataStore:
    """
    训练数据存储
    
    管理所有训练作业的数据存储
    """
    
    def __init__(self, max_history_per_job: int = 10000):
        """
        初始化数据存储
        
        Args:
            max_history_per_job: 每个作业最大历史记录数
        """
        self._jobs: Dict[str, TrainingJob] = {}
        self._max_history = max_history_per_job
        self._lock = asyncio.Lock()
    
    async def create_job(
        self,
        job_id: Optional[str] = None,
        name: str = "Training Job",
        model_name: str = "unknown",
        total_epochs: int = 10,
        total_steps: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingJob:
        """
        创建新训练作业
        
        Args:
            job_id: 作业ID (可选，自动生成)
            name: 作业名称
            model_name: 模型名称
            total_epochs: 总轮次
            total_steps: 总步数
            config: 额外配置
        
        Returns:
            创建的作业
        """
        if job_id is None:
            job_id = str(uuid.uuid4())[:8]
        
        job = TrainingJob(
            job_id=job_id,
            name=name,
            model_name=model_name,
            total_epochs=total_epochs,
            total_steps=total_steps,
            config=config or {},
        )
        
        async with self._lock:
            self._jobs[job_id] = job
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """获取作业"""
        return self._jobs.get(job_id)
    
    async def list_jobs(self, status: Optional[TrainingStatus] = None) -> List[TrainingJob]:
        """
        列出所有作业
        
        Args:
            status: 按状态筛选
        
        Returns:
            作业列表
        """
        async with self._lock:
            jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def add_metrics(self, job_id: str, metrics: TrainingMetrics) -> bool:
        """
        添加训练指标
        
        Args:
            job_id: 作业ID
            metrics: 指标数据
        
        Returns:
            是否成功
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            
            job.metrics_history.append(metrics)
            
            # 限制历史记录数量
            if len(job.metrics_history) > self._max_history:
                job.metrics_history = job.metrics_history[-self._max_history:]
        
        return True
    
    async def update_status(self, job_id: str, status: TrainingStatus) -> bool:
        """
        更新作业状态
        
        Args:
            job_id: 作业ID
            status: 新状态
        
        Returns:
            是否成功
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            
            old_status = job.status
            job.status = status
            
            # 状态变更时更新时间戳
            if status == TrainingStatus.RUNNING and old_status != TrainingStatus.RUNNING:
                job.started_at = datetime.now().isoformat()
            elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                job.completed_at = datetime.now().isoformat()
        
        return True
    
    async def delete_job(self, job_id: str) -> bool:
        """删除作业"""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
        return False
    
    async def get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        获取作业完整数据
        
        Args:
            job_id: 作业ID
        
        Returns:
            作业数据字典
        """
        job = await self.get_job(job_id)
        if job is None:
            return None
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "model_name": job.model_name,
            "status": job.status.value,
            "total_epochs": job.total_epochs,
            "total_steps": job.total_steps,
            "current_epoch": job.current_epoch,
            "current_step": job.current_step,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "loss": job.get_loss_data(),
            "gpu": job.get_gpu_data(),
            "learning_rate": job.get_lr_data(),
            "metrics": job.get_metrics_data(),
            "config": job.config,
        }
    
    async def get_loss_history(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取Loss历史"""
        job = await self.get_job(job_id)
        if job is None:
            return None
        return {
            "job_id": job_id,
            "train_loss": job.get_loss_data()["train"],
            "val_loss": job.get_loss_data()["val"],
            "steps": list(range(len(job.metrics_history))),
        }
    
    async def get_gpu_history(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取GPU历史"""
        job = await self.get_job(job_id)
        if job is None:
            return None
        return {
            "job_id": job_id,
            "utilization": job.get_gpu_data()["utilization"],
            "memory": job.get_gpu_data()["memory"],
            "temperature": job.get_gpu_data()["temperature"],
            "timestamps": [m.timestamp for m in job.metrics_history],
        }
    
    async def get_metrics_history(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取评估指标历史"""
        job = await self.get_job(job_id)
        if job is None:
            return None
        return {
            "job_id": job_id,
            "metrics": job.get_metrics_data(),
            "epochs": list(range(len(job.metrics_history))),
        }


class RealtimeDataHandler:
    """
    实时数据处理器
    
    处理SSE流式推送和事件分发
    """
    
    def __init__(self, data_store: Optional[TrainingDataStore] = None):
        """
        初始化实时数据处理器
        
        Args:
            data_store: 数据存储实例
        """
        self.data_store = data_store or get_training_store()
        self._subscribers: Dict[str, Set[str]] = defaultdict(set)  # job_id -> set of session_ids
        self._sse_queues: Dict[str, asyncio.Queue] = {}  # session_id -> queue
    
    async def subscribe(self, job_id: str, session_id: str) -> asyncio.Queue:
        """
        订阅作业更新
        
        Args:
            job_id: 作业ID
            session_id: 会话ID
        
        Returns:
            消息队列
        """
        queue = asyncio.Queue()
        self._subscribers[job_id].add(session_id)
        self._sse_queues[session_id] = queue
        return queue
    
    async def unsubscribe(self, job_id: str, session_id: str):
        """
        取消订阅
        
        Args:
            job_id: 作业ID
            session_id: 会话ID
        """
        self._subscribers[job_id].discard(session_id)
        if session_id in self._sse_queues:
            del self._sse_queues[session_id]
    
    async def broadcast(self, job_id: str, event_type: str, data: Dict[str, Any]):
        """
        广播消息给所有订阅者
        
        Args:
            job_id: 作业ID
            event_type: 事件类型
            data: 消息数据
        """
        message = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        
        for session_id in self._subscribers.get(job_id, set()):
            queue = self._sse_queues.get(session_id)
            if queue:
                try:
                    await queue.put(message)
                except Exception:
                    pass
    
    async def publish_metrics(self, job_id: str, metrics: TrainingMetrics):
        """
        发布新指标
        
        Args:
            job_id: 作业ID
            metrics: 指标数据
        """
        # 存储数据
        await self.data_store.add_metrics(job_id, metrics)
        
        # 广播更新
        await self.broadcast(job_id, "metrics", metrics.to_dict())
    
    async def publish_status_change(self, job_id: str, old_status: TrainingStatus, new_status: TrainingStatus):
        """
        发布状态变更
        
        Args:
            job_id: 作业ID
            old_status: 旧状态
            new_status: 新状态
        """
        await self.broadcast(job_id, "status", {
            "job_id": job_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
        })
    
    async def publish_epoch_complete(self, job_id: str, epoch: int, metrics: Dict[str, Any]):
        """
        发布轮次完成
        
        Args:
            job_id: 作业ID
            epoch: 轮次
            metrics: 汇总指标
        """
        await self.broadcast(job_id, "epoch_complete", {
            "job_id": job_id,
            "epoch": epoch,
            "metrics": metrics,
        })
    
    async def get_active_subscribers(self, job_id: str) -> int:
        """获取活跃订阅者数"""
        return len(self._subscribers.get(job_id, set()))


class SSEPublisher:
    """
    SSE (Server-Sent Events) 发布器
    
    用于生成SSE格式的响应
    """
    
    @staticmethod
    def format_sse(
        event: str,
        data: Dict[str, Any],
        id: Optional[str] = None,
    ) -> str:
        """
        格式化SSE消息
        
        Args:
            event: 事件类型
            data: 数据字典
            id: 消息ID
        
        Returns:
            SSE格式字符串
        """
        lines = []
        
        if id:
            lines.append(f"id: {id}")
        
        lines.append(f"event: {event}")
        lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
        lines.append("")  # 空行结束
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def format_keepalive() -> str:
        """生成心跳消息"""
        return ": keepalive\n\n"
    
    @staticmethod
    async def stream_generator(
        queue: asyncio.Queue,
        job_id: str,
        timeout: float = 30.0,
    ) -> str:
        """
        生成SSE流
        
        Args:
            queue: 消息队列
            job_id: 作业ID
            timeout: 超时时间
        
        Yields:
            SSE格式字符串
        """
        message_id = 0
        
        while True:
            try:
                # 等待消息，带超时
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
                message_id += 1
                
                yield SSEPublisher.format_sse(
                    event=message["event"],
                    data=message["data"],
                    id=str(message_id),
                )
            
            except asyncio.TimeoutError:
                # 发送心跳
                yield SSEPublisher.format_keepalive()
            
            except asyncio.CancelledError:
                break
            
            except Exception:
                yield SSEPublisher.format_sse(
                    event="error",
                    data={"message": "Stream error"},
                )
                break


# 全局单例
_training_store_instance: Optional[TrainingDataStore] = None
_realtime_handler_instance: Optional[RealtimeDataHandler] = None


def get_training_store(max_history_per_job: int = 10000) -> TrainingDataStore:
    """获取训练数据存储单例"""
    global _training_store_instance
    if _training_store_instance is None:
        _training_store_instance = TrainingDataStore(max_history_per_job)
    return _training_store_instance


def get_realtime_handler() -> RealtimeDataHandler:
    """获取实时数据处理器单例"""
    global _realtime_handler_instance
    if _realtime_handler_instance is None:
        _realtime_handler_instance = RealtimeDataHandler(get_training_store())
    return _realtime_handler_instance


# ============ 便捷函数 ============

async def create_demo_job(job_id: str = "demo-001") -> TrainingJob:
    """
    创建演示用训练作业
    
    Args:
        job_id: 作业ID
    
    Returns:
        训练作业
    """
    store = get_training_store()
    
    job = await store.create_job(
        job_id=job_id,
        name="Demo Training Job",
        model_name="bert-base",
        total_epochs=10,
        total_steps=1000,
        config={"batch_size": 32, "learning_rate": 2e-5},
    )
    
    await store.update_status(job_id, TrainingStatus.RUNNING)
    
    # 生成模拟数据
    import random
    
    for epoch in range(10):
        for step in range(100):
            metrics = TrainingMetrics(
                step=epoch * 100 + step,
                epoch=epoch,
                train_loss=2.0 * (0.95 ** (epoch * 100 + step / 100)) + random.uniform(0, 0.1),
                val_loss=2.2 * (0.95 ** (epoch * 100 + step / 100)) + random.uniform(0, 0.1),
                learning_rate=2e-5 * (0.95 ** (epoch * 100 + step / 100)),
                gpu_utilization=random.uniform(60, 95),
                gpu_memory=random.uniform(4000, 8000),
                gpu_temperature=random.uniform(65, 80),
                accuracy=min(1.0, 0.5 + 0.05 * epoch + random.uniform(0, 0.02)),
                f1=min(1.0, 0.45 + 0.05 * epoch + random.uniform(0, 0.02)),
                precision=min(1.0, 0.48 + 0.05 * epoch + random.uniform(0, 0.02)),
                recall=min(1.0, 0.47 + 0.05 * epoch + random.uniform(0, 0.02)),
                progress_percent=(epoch * 100 + step + 1) / 1000 * 100,
            )
            await store.add_metrics(job_id, metrics)
    
    await store.update_status(job_id, TrainingStatus.COMPLETED)
    
    return job
