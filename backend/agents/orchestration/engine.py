"""
Orchestration Engine - Agent编排引擎
负责任务分解、调度、执行和监控
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """任务数据模型"""
    id: str
    name: str
    description: str
    agent_type: str
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "input_data": self.input_data,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "priority": self.priority.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


@dataclass
class CollaborationSession:
    """协作会话数据模型"""
    id: str
    name: str
    description: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "agents": self.agents,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class OrchestrationEngine:
    """
    Agent编排引擎
    
    核心功能：
    - 任务分解与调度
    - 多Agent协作管理
    - 执行流程控制
    - 状态监控
    """
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.agent_registry: Dict[str, Callable] = {}
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.event_callbacks: List[Callable] = []
        self._running = False
        
    async def start(self):
        """启动编排引擎"""
        self._running = True
        asyncio.create_task(self._process_execution_queue())
        logger.info("Orchestration Engine started")
        
    async def stop(self):
        """停止编排引擎"""
        self._running = False
        logger.info("Orchestration Engine stopped")
        
    def register_agent(self, agent_type: str, agent_func: Callable):
        """注册Agent类型"""
        self.agent_registry[agent_type] = agent_func
        logger.info(f"Registered agent type: {agent_type}")
        
    def create_session(self, name: str, description: str = "", 
                       metadata: Dict[str, Any] = None) -> CollaborationSession:
        """创建协作会话"""
        session = CollaborationSession(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            metadata=metadata or {}
        )
        self.sessions[session.id] = session
        self._emit_event("session_created", session)
        logger.info(f"Created collaboration session: {session.id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """获取协作会话"""
        return self.sessions.get(session_id)
    
    def list_sessions(self, status: str = None) -> List[Dict[str, Any]]:
        """列出协作会话"""
        sessions = list(self.sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return [s.to_dict() for s in sessions]
    
    def add_task(self, session_id: str, task: Task) -> Task:
        """添加任务到会话"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.tasks[task.id] = task
        session.updated_at = datetime.now()
        self._emit_event("task_added", task)
        logger.info(f"Added task {task.id} to session {session_id}")
        return task
    
    def create_task(self, session_id: str, name: str, agent_type: str,
                    input_data: Dict[str, Any], description: str = "",
                    dependencies: List[str] = None,
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    metadata: Dict[str, Any] = None) -> Task:
        """创建并添加任务"""
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            agent_type=agent_type,
            input_data=input_data,
            dependencies=dependencies or [],
            priority=priority,
            metadata=metadata or {}
        )
        return self.add_task(session_id, task)
    
    async def schedule_tasks(self, session_id: str) -> List[Task]:
        """调度会话中的任务"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # 拓扑排序基于依赖关系
        scheduled = []
        pending_tasks = {k: v for k, v in session.tasks.items() 
                        if v.status == TaskStatus.PENDING}
        
        while pending_tasks:
            # 找到没有未完成依赖的任务
            ready_tasks = [
                t for t in pending_tasks.values()
                if all(d in scheduled or 
                       session.tasks.get(d, Task("", "", "", "", {})).status == TaskStatus.COMPLETED
                       for d in t.dependencies)
            ]
            
            if not ready_tasks:
                raise ValueError("Circular dependency detected")
            
            # 按优先级排序
            ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            for task in ready_tasks:
                task.status = TaskStatus.SCHEDULED
                await self.execution_queue.put(task)
                scheduled.append(task.id)
                pending_tasks.pop(task.id)
        
        self._emit_event("tasks_scheduled", {"session_id": session_id, "count": len(scheduled)})
        return [self.sessions[session_id].tasks[tid] for tid in scheduled]
    
    async def execute_task(self, task: Task, session: CollaborationSession) -> Task:
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        self._emit_event("task_started", task)
        logger.info(f"Executing task: {task.id} - {task.name}")
        
        try:
            agent_func = self.agent_registry.get(task.agent_type)
            if not agent_func:
                # 默认处理：简单返回输入
                task.result = {"output": task.input_data, "agent": task.agent_type}
            else:
                task.result = await agent_func(task.input_data, task.metadata)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self._emit_event("task_completed", task)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            self._emit_event("task_failed", task)
            logger.error(f"Task {task.id} failed: {e}")
        
        return task
    
    async def execute_session(self, session_id: str, sync: bool = True) -> CollaborationSession:
        """执行整个会话"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.status = "running"
        
        # 调度任务
        await self.schedule_tasks(session_id)
        
        if sync:
            # 同步执行：等待所有任务完成
            while any(t.status == TaskStatus.SCHEDULED or t.status == TaskStatus.RUNNING 
                     for t in session.tasks.values()):
                task = await self.execution_queue.get()
                await self.execute_task(task, session)
                self.execution_queue.task_done()
        else:
            # 异步执行：在后台运行
            asyncio.create_task(self._execute_all_tasks(session))
        
        session.updated_at = datetime.now()
        self._emit_event("session_executed", session)
        return session
    
    async def _execute_all_tasks(self, session: CollaborationSession):
        """执行所有任务（后台）"""
        while not self.execution_queue.empty():
            task = await self.execution_queue.get()
            await self.execute_task(task, session)
            self.execution_queue.task_done()
    
    async def _process_execution_queue(self):
        """处理执行队列"""
        while self._running:
            task = await self.execution_queue.get()
            # 任务执行逻辑
            self.execution_queue.task_done()
    
    def cancel_task(self, session_id: str, task_id: str) -> bool:
        """取消任务"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        task = session.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        self._emit_event("task_cancelled", task)
        return True
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        tasks = list(session.tasks.values())
        return {
            "session_id": session_id,
            "name": session.name,
            "status": session.status,
            "total_tasks": len(tasks),
            "pending_tasks": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "running_tasks": sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
            "completed_tasks": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "progress": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED) / len(tasks) * 100 if tasks else 0
        }
    
    def monitor_sessions(self) -> List[Dict[str, Any]]:
        """监控所有会话状态"""
        return [self.get_session_status(sid) for sid in self.sessions]
    
    def _emit_event(self, event_type: str, data: Any):
        """触发事件"""
        for callback in self.event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def on_event(self, callback: Callable):
        """注册事件回调"""
        self.event_callbacks.append(callback)


# 全局编排引擎实例
orchestration_engine = OrchestrationEngine()
