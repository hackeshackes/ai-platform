"""
协作管理器 - Agent Collaboration Manager
支持多Agent协同工作、任务分解与自动执行
"""

import uuid
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import asyncio


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


class SessionStatus(Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass
class Task:
    """任务数据类"""
    id: str
    name: str
    description: str
    agent_type: str
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error
        }


@dataclass
class CollaborationSession:
    """协作会话数据类"""
    id: str
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks],
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }


class AgentRegistry:
    """Agent注册表 - 管理可用的Agent类型"""
    
    def __init__(self):
        self._agents: Dict[str, Callable] = {}
        self._agent_configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, agent_type: str, agent_factory: Callable, config: Dict[str, Any] = None):
        """注册Agent类型"""
        self._agents[agent_type] = agent_factory
        if config:
            self._agent_configs[agent_type] = config
    
    def get_agent(self, agent_type: str, **kwargs) -> Any:
        """获取Agent实例"""
        if agent_type not in self._agents:
            raise ValueError(f"Agent type '{agent_type}' not registered")
        return self._agents[agent_type](**kwargs)
    
    def list_types(self) -> List[str]:
        """列出所有注册的Agent类型"""
        return list(self._agents.keys())
    
    def get_config(self, agent_type: str) -> Dict[str, Any]:
        """获取Agent配置"""
        return self._agent_configs.get(agent_type, {})


class CollaborationManager:
    """协作管理器主类"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.registry = AgentRegistry()
        self._setup_default_agents()
    
    def _setup_default_agents(self):
        """设置默认Agent类型"""
        # 默认Agent将在实际使用时动态加载
        pass
    
    def create_session(self, name: str, description: str = "", metadata: Dict[str, Any] = None) -> CollaborationSession:
        """创建新的协作会话"""
        session_id = str(uuid.uuid4())
        session = CollaborationSession(
            id=session_id,
            name=name,
            description=description,
            metadata=metadata or {}
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def list_sessions(self, status: SessionStatus = None) -> List[CollaborationSession]:
        """列出所有会话"""
        if status:
            return [s for s in self.sessions.values() if s.status == status]
        return list(self.sessions.values())
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def add_task(self, session_id: str, task: Task) -> bool:
        """向会话添加任务"""
        session = self.sessions.get(session_id)
        if session:
            session.tasks.append(task)
            session.updated_at = datetime.now().isoformat()
            return True
        return False
    
    def create_task(
        self,
        session_id: str,
        name: str,
        description: str,
        agent_type: str,
        input_data: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> Optional[Task]:
        """创建并添加任务"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            description=description,
            agent_type=agent_type,
            input_data=input_data or {},
            dependencies=dependencies or []
        )
        session.tasks.append(task)
        session.updated_at = datetime.now().isoformat()
        return task
    
    def get_task(self, session_id: str, task_id: str) -> Optional[Task]:
        """获取任务"""
        session = self.sessions.get(session_id)
        if session:
            for task in session.tasks:
                if task.id == task_id:
                    return task
        return None
    
    def update_task_status(self, session_id: str, task_id: str, status: TaskStatus, 
                          output_data: Dict[str, Any] = None, error: str = None) -> bool:
        """更新任务状态"""
        task = self.get_task(session_id, task_id)
        if task:
            task.status = status
            task.updated_at = datetime.now().isoformat()
            if output_data:
                task.output_data = output_data
            if error:
                task.error = error
            return True
        return False
    
    def execute_task(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """执行单个任务"""
        session = self.sessions.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        task = self.get_task(session_id, task_id)
        if not task:
            return {"success": False, "error": "Task not found"}
        
        # 检查依赖
        for dep_id in task.dependencies:
            dep_task = self.get_task(session_id, dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                return {"success": False, "error": f"Dependency {dep_id} not completed"}
        
        # 更新状态为进行中
        self.update_task_status(session_id, task_id, TaskStatus.IN_PROGRESS)
        
        try:
            # 获取Agent并执行
            agent = self.registry.get_agent(task.agent_type)
            
            # 如果是异步函数
            if asyncio.iscoroutinefunction(agent):
                result = asyncio.run(agent(task.input_data))
            else:
                result = agent(task.input_data)
            
            self.update_task_status(session_id, task_id, TaskStatus.COMPLETED, output_data=result)
            return {"success": True, "result": result}
            
        except Exception as e:
            self.update_task_status(session_id, task_id, TaskStatus.FAILED, error=str(e))
            return {"success": False, "error": str(e)}
    
    async def execute_session_async(self, session_id: str) -> Dict[str, Any]:
        """异步执行会话中的所有任务"""
        session = self.sessions.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        results = []
        for task in session.tasks:
            # 按依赖关系排序执行
            result = await self._execute_task_async(session_id, task.id)
            results.append(result)
        
        return {"success": True, "results": results}
    
    async def _execute_task_async(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """异步执行单个任务"""
        task = self.get_task(session_id, task_id)
        if not task:
            return {"success": False, "error": "Task not found"}
        
        # 等待依赖完成
        for dep_id in task.dependencies:
            dep_result = await self._wait_for_task(session_id, dep_id)
            if not dep_result.get("success", False):
                return {"success": False, "error": f"Dependency {dep_id} failed"}
        
        # 执行任务
        self.update_task_status(session_id, task_id, TaskStatus.IN_PROGRESS)
        
        try:
            agent = self.registry.get_agent(task.agent_type)
            if asyncio.iscoroutinefunction(agent):
                result = await agent(task.input_data)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, agent, task.input_data)
            
            self.update_task_status(session_id, task_id, TaskStatus.COMPLETED, output_data=result)
            return {"success": True, "result": result}
            
        except Exception as e:
            self.update_task_status(session_id, task_id, TaskStatus.FAILED, error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _wait_for_task(self, session_id: str, task_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """等待任务完成"""
        task = self.get_task(session_id, task_id)
        if not task:
            return {"success": False, "error": "Task not found"}
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return {"success": task.status == TaskStatus.COMPLETED, 
                    "result": task.output_data, 
                    "error": task.error}
        
        # 轮询等待
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            await asyncio.sleep(0.5)
            task = self.get_task(session_id, task_id)
            if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return {"success": task.status == TaskStatus.COMPLETED,
                        "result": task.output_data,
                        "error": task.error}
        
        return {"success": False, "error": "Timeout waiting for dependency"}
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """导出会话为JSON"""
        session = self.sessions.get(session_id)
        if session:
            return session.to_dict()
        return None
    
    def import_session(self, session_data: Dict[str, Any]) -> CollaborationSession:
        """从JSON导入会话"""
        session_id = session_data.get("id", str(uuid.uuid4()))
        tasks = []
        for t_data in session_data.get("tasks", []):
            status = TaskStatus(t_data.get("status", "pending"))
            task = Task(
                id=t_data["id"],
                name=t_data["name"],
                description=t_data["description"],
                agent_type=t_data["agent_type"],
                status=status,
                input_data=t_data.get("input_data", {}),
                output_data=t_data.get("output_data"),
                dependencies=t_data.get("dependencies", []),
                created_at=t_data.get("created_at", datetime.now().isoformat()),
                updated_at=t_data.get("updated_at", datetime.now().isoformat()),
                error=t_data.get("error")
            )
            tasks.append(task)
        
        session = CollaborationSession(
            id=session_id,
            name=session_data["name"],
            description=session_data.get("description", ""),
            tasks=tasks,
            status=SessionStatus(session_data.get("status", "active")),
            created_at=session_data.get("created_at", datetime.now().isoformat()),
            updated_at=session_data.get("updated_at", datetime.now().isoformat()),
            metadata=session_data.get("metadata", {})
        )
        self.sessions[session_id] = session
        return session


# 全局协作管理器实例
collab_manager = CollaborationManager()
