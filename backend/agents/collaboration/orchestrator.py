"""
协作编排器
协调多Agent的协作工作流
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import logging

from .models import (
    CollaborationSession, CollaborationMode, SessionStatus,
    AgentInfo, TaskInput, TaskOutput, WorkflowDefinition,
    WorkflowExecutionResult, SessionProgress, AgentRole
)
from .communication import CommunicationManager, get_communication_manager
from .workflow import WorkflowEngine, get_workflow_engine, WorkflowExecutor
from .task_decomposer import TaskDecomposer, create_task_decomposer
from .consensus import ConsensusManager, create_consensus_manager

logger = logging.getLogger(__name__)


class OrchestrationSession:
    """编排会话"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.active_executors: Dict[str, WorkflowExecutor] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(
        self,
        name: str,
        description: str = "",
        mode: CollaborationMode = CollaborationMode.SEQUENTIAL,
        workflow: Optional[WorkflowDefinition] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CollaborationSession:
        """创建协作会话"""
        session = CollaborationSession(
            name=name,
            description=description,
            mode=mode,
            workflow=workflow,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.sessions[session.session_id] = session
            self.session_locks[session.session_id] = asyncio.Lock()
        
        logger.info(f"Session created: {session.session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    async def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[CollaborationSession]:
        """列出会话"""
        sessions = list(self.sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        return sessions[-limit:]
    
    async def add_agent(
        self,
        session_id: str,
        agent_id: str,
        role: AgentRole = AgentRole.WORKER,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None
    ) -> bool:
        """添加Agent到会话"""
        async with self.session_locks.get(session_id, asyncio.Lock()):
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            agent = AgentInfo(
                agent_id=agent_id,
                role=role,
                name=name or f"Agent_{agent_id}",
                capabilities=capabilities or [],
                status="joined"
            )
            
            session.add_agent(agent)
            logger.info(f"Agent {agent_id} joined session {session_id}")
            return True
    
    async def add_task(
        self,
        session_id: str,
        name: str,
        description: str = "",
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        assigned_agent: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> Optional[TaskInput]:
        """添加任务到会话"""
        async with self.session_locks.get(session_id, asyncio.Lock()):
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            task = TaskInput(
                name=name,
                description=description,
                payload=payload or {},
                priority=priority,
                assigned_agent=assigned_agent,
                dependencies=dependencies or []
            )
            
            session.add_task(task)
            logger.info(f"Task {task.task_id} added to session {session_id}")
            return task
    
    async def remove_agent(self, session_id: str, agent_id: str) -> bool:
        """从会话移除Agent"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.agents = [a for a in session.agents if a.agent_id != agent_id]
        return True
    
    async def update_session_status(
        self,
        session_id: str,
        status: SessionStatus
    ) -> bool:
        """更新会话状态"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.status = status
        
        if status == SessionStatus.ACTIVE:
            session.start_time = datetime.utcnow()
        elif status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED]:
            session.end_time = datetime.utcnow()
        
        return True


class CollaborationOrchestrator:
    """协作编排器"""
    
    def __init__(self):
        self.session_manager = OrchestrationSession()
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.communication: Optional[CommunicationManager] = None
        self.decomposer = create_task_decomposer()
        self.consensus_manager = create_consensus_manager()
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化编排器"""
        if self._initialized:
            return
        
        self.workflow_engine = get_workflow_engine()
        await self.workflow_engine.initialize()
        
        self.communication = get_communication_manager()
        await self.communication.initialize()
        
        self._initialized = True
        logger.info("Collaboration Orchestrator initialized")
    
    async def shutdown(self) -> None:
        """关闭编排器"""
        if self.workflow_engine:
            self._initialized = False
        logger.info("Collaboration Orchestrator shutdown")
    
    async def create_collaboration_session(
        self,
        name: str,
        mode: str = "sequential",
        description: str = "",
        agent_ids: Optional[List[str]] = None,
        workflow_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建协作会话"""
        if not self._initialized:
            await self.initialize()
        
        # 解析协作模式
        try:
            collab_mode = CollaborationMode(mode)
        except ValueError:
            collab_mode = CollaborationMode.SEQUENTIAL
        
        # 创建会话
        session = await self.session_manager.create_session(
            name=name,
            description=description,
            mode=collab_mode
        )
        
        # 创建工作流
        if workflow_config:
            workflow = await self.workflow_engine.create_workflow(
                name=f"{name}_workflow",
                description=workflow_config.get("description", ""),
                mode=collab_mode,
                agent_ids=agent_ids
            )
            session.workflow = workflow
        
        # 添加Agent
        if agent_ids:
            for agent_id in agent_ids:
                await self.session_manager.add_agent(
                    session_id=session.session_id,
                    agent_id=agent_id,
                    name=f"Agent_{agent_id}"
                )
        
        # 创建通信通道
        await self.communication.create_session_channel(session.session_id)
        
        return {
            "success": True,
            "session_id": session.session_id,
            "name": session.name,
            "mode": session.mode.value,
            "status": session.status.value
        }
    
    async def join_session(
        self,
        session_id: str,
        agent_id: str,
        role: str = "worker",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Agent加入会话"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # 解析角色
        try:
            agent_role = AgentRole(role)
        except ValueError:
            agent_role = AgentRole.WORKER
        
        # 添加Agent
        success = await self.session_manager.add_agent(
            session_id=session_id,
            agent_id=agent_id,
            role=agent_role,
            capabilities=metadata.get("capabilities", []) if metadata else []
        )
        
        if success:
            # 注册到通信系统
            await self.communication.register_agent(agent_id)
            
            # 通知其他Agent
            await self.communication.send_event(
                session_id=session_id,
                sender_id=agent_id,
                event_type="agent_joined",
                event_data={"agent_id": agent_id, "role": role}
            )
        
        return {
            "success": success,
            "session_id": session_id,
            "agent_id": agent_id
        }
    
    async def assign_task(
        self,
        session_id: str,
        task_name: str,
        description: str = "",
        payload: Optional[Dict[str, Any]] = None,
        assigned_agent: Optional[str] = None,
        priority: int = 0,
        dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """分配任务"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        task = await self.session_manager.add_task(
            session_id=session_id,
            name=task_name,
            description=description,
            payload=payload,
            assigned_agent=assigned_agent,
            priority=priority,
            dependencies=dependencies or []
        )
        
        if task:
            # 发送任务消息
            if assigned_agent:
                await self.communication.send_task(
                    session_id=session_id,
                    task_id=task.task_id,
                    sender_id="orchestrator",
                    receiver_id=assigned_agent,
                    task_data={
                        "task_name": task_name,
                        "description": description,
                        "payload": payload or {}
                    }
                )
        
        return {
            "success": task is not None,
            "task_id": task.task_id if task else None,
            "session_id": session_id
        }
    
    async def execute_collaboration(
        self,
        session_id: str,
        task_inputs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """执行协作"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # 更新状态
        await self.session_manager.update_session_status(
            session_id, SessionStatus.EXECUTING
        )
        
        # 准备任务列表
        tasks = []
        
        if task_inputs:
            for task_data in task_inputs:
                task = await self.session_manager.add_task(
                    session_id=session_id,
                    name=task_data.get("name", "Task"),
                    description=task_data.get("description", ""),
                    payload=task_data.get("payload", {}),
                    priority=task_data.get("priority", 0),
                    assigned_agent=task_data.get("assigned_agent"),
                    dependencies=task_data.get("dependencies", [])
                )
                tasks.append(task)
        else:
            tasks = session.tasks
        
        # 创建Agent映射
        agent_map = {agent.agent_id: agent for agent in session.agents}
        
        # 创建工作流
        workflow = session.workflow or await self.workflow_engine.create_workflow(
            name=f"{session.name}_workflow",
            description=session.description,
            mode=session.mode,
            agent_ids=[a.agent_id for a in session.agents]
        )
        
        try:
            # 执行工作流
            executor = await self.workflow_engine.execute(workflow, tasks, agent_map)
            self.session_manager.active_executors[session_id] = executor
            
            # 等待完成
            while executor.state.value not in ["completed", "failed", "cancelled"]:
                await asyncio.sleep(0.5)
            
            # 获取结果
            progress = executor.get_progress()
            result = WorkflowExecutionResult(
                session_id=session_id,
                success=executor.state == "completed",
                task_results=executor.execution_history,
                total_execution_time_ms=sum(
                    t.get("execution_time_ms", 0)
                    for t in executor.execution_history
                )
            )
            
            # 更新会话状态
            final_status = SessionStatus.COMPLETED if result.success else SessionStatus.FAILED
            await self.session_manager.update_session_status(session_id, final_status)
            
            return {
                "success": result.success,
                "session_id": session_id,
                "result": result.dict()
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            await self.session_manager.update_session_status(
                session_id, SessionStatus.FAILED
            )
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_session_result(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """获取会话结果"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            return None
        
        executor = self.session_manager.active_executors.get(session_id)
        progress = session.get_progress()
        
        return {
            "session_id": session_id,
            "name": session.name,
            "status": session.status.value,
            "progress": progress.dict(),
            "task_count": len(session.tasks),
            "completed_task_count": len([
                t for t in session.task_outputs.values()
                if t.status.value == "completed"
            ]),
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "execution_history": executor.execution_history if executor else []
        }
    
    async def get_session_progress(
        self,
        session_id: str
    ) -> Optional[SessionProgress]:
        """获取会话进度"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            return None
        
        return session.get_progress()
    
    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """列出活跃会话"""
        sessions = await self.session_manager.list_sessions(
            status=SessionStatus.ACTIVE, limit=20
        )
        
        return [
            {
                "session_id": s.session_id,
                "name": s.name,
                "mode": s.mode.value,
                "status": s.status.value,
                "agent_count": len(s.agents),
                "task_count": len(s.tasks)
            }
            for s in sessions
        ]
    
    async def cancel_session(self, session_id: str) -> bool:
        """取消会话"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            return False
        
        executor = self.session_manager.active_executors.get(session_id)
        if executor:
            await executor.cancel()
        
        await self.session_manager.update_session_status(
            session_id, SessionStatus.CANCELLED
        )
        
        return True


# 全局编排器实例
_orchestrator: Optional[CollaborationOrchestrator] = None


def get_orchestrator() -> CollaborationOrchestrator:
    """获取全局编排器"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CollaborationOrchestrator()
    return _orchestrator


async def init_orchestrator() -> CollaborationOrchestrator:
    """初始化编排器"""
    orchestrator = get_orchestrator()
    await orchestrator.initialize()
    return orchestrator


async def shutdown_orchestrator() -> None:
    """关闭编排器"""
    global _orchestrator
    if _orchestrator:
        await _orchestrator.shutdown()
        _orchestrator = None
