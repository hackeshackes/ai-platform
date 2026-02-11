"""
Agent协作网络API端点 - FastAPI版本
提供多Agent协作的REST API接口
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents/collaboration", tags=["Agent Collaboration"])


# ============== 请求/响应模型 ==============

class CreateSessionRequest(BaseModel):
    """创建协作会话请求"""
    name: str = Field(..., min_length=1, max_length=200, description="会话名称")
    description: str = Field(default="", description="会话描述")
    mode: str = Field(default="sequential", description="协作模式")
    agents: List[str] = Field(default_factory=list, description="Agent ID列表")
    workflow: Optional[Dict[str, Any]] = Field(default=None, description="工作流配置")


class CreateSessionResponse(BaseModel):
    """创建协作会话响应"""
    success: bool
    session_id: str
    name: str
    mode: str
    status: str


class JoinSessionRequest(BaseModel):
    """加入会话请求"""
    agent_id: str = Field(..., description="Agent ID")
    role: str = Field(default="worker", description="角色")
    name: Optional[str] = Field(default=None, description="Agent名称")
    capabilities: List[str] = Field(default_factory=list, description="能力列表")


class AssignTaskRequest(BaseModel):
    """分配任务请求"""
    name: str = Field(..., min_length=1, max_length=200, description="任务名称")
    description: str = Field(default="", description="任务描述")
    payload: Dict[str, Any] = Field(default_factory=dict, description="任务数据")
    assigned_agent: Optional[str] = Field(default=None, description="分配的Agent ID")
    priority: int = Field(default=0, description="优先级")
    dependencies: List[str] = Field(default_factory=list, description="依赖任务ID")


class ExecuteCollaborationRequest(BaseModel):
    """执行协作请求"""
    tasks: Optional[List[Dict[str, Any]]] = Field(default=None, description="任务列表")


class TaskDecomposeRequest(BaseModel):
    """任务分解请求"""
    name: str = Field(..., description="任务名称")
    description: str = Field(default="", description="任务描述")
    strategy: str = Field(default="hierarchical", description="分解策略")


# ============== 协作会话API ==============

@router.post("/session", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_collaboration_session(request: CreateSessionRequest):
    """
    创建协作会话
    
    创建一个新的多Agent协作会话。
    """
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        result = await orchestrator.create_collaboration_session(
            name=request.name,
            description=request.description,
            mode=request.mode,
            agent_ids=request.agents,
            workflow_config=request.workflow
        )
        
        return CreateSessionResponse(
            success=result.get("success", False),
            session_id=result.get("session_id", ""),
            name=result.get("name", request.name),
            mode=result.get("mode", request.mode),
            status=result.get("status", "created")
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sessions")
async def list_sessions(
    status: Optional[str] = None,
    limit: int = 50
):
    """
    列出协作会话
    
    返回所有协作会话，或根据状态过滤。
    """
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        from agents.collaboration.models import SessionStatus
        
        status_filter = None
        if status:
            try:
                status_filter = SessionStatus(status)
            except ValueError:
                pass
        
        sessions = await orchestrator.session_manager.list_sessions(
            status=status_filter, limit=limit
        )
        
        return {
            "success": True,
            "count": len(sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "name": s.name,
                    "mode": s.mode.value,
                    "status": s.status.value,
                    "agent_count": len(s.agents),
                    "task_count": len(s.tasks),
                    "created_at": s.created_at.isoformat()
                }
                for s in sessions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """获取会话详情"""
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        session = await orchestrator.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            "success": True,
            "session": {
                "session_id": session.session_id,
                "name": session.name,
                "description": session.description,
                "mode": session.mode.value,
                "status": session.status.value,
                "agents": [
                    {
                        "agent_id": a.agent_id,
                        "name": a.name,
                        "role": a.role.value,
                        "status": a.status
                    }
                    for a in session.agents
                ],
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "name": t.name,
                        "status": "pending",
                        "dependencies": t.dependencies
                    }
                    for t in session.tasks
                ],
                "created_at": session.created_at.isoformat(),
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "end_time": session.end_time.isoformat() if session.end_time else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除/取消会话"""
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        success = await orchestrator.cancel_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            "success": True,
            "message": f"Session {session_id} cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============== Agent管理API ==============

@router.post("/session/{session_id}/join")
async def join_session(session_id: str, request: JoinSessionRequest):
    """
    Agent加入会话
    
    将Agent添加到协作会话中。
    """
    try:
        from agents.collaboration import get_orchestrator
        from agents.collaboration.models import AgentRole
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        session = await orchestrator.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        result = await orchestrator.join_session(
            session_id=session_id,
            agent_id=request.agent_id,
            role=request.role,
            metadata={
                "name": request.name,
                "capabilities": request.capabilities
            }
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to join session")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/session/{session_id}/leave")
async def leave_session(session_id: str, agent_id: str):
    """
    Agent离开会话
    
    从协作会话中移除Agent。
    """
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        success = await orchestrator.session_manager.remove_agent(session_id, agent_id)
        
        return {
            "success": success,
            "message": f"Agent {agent_id} left session"
        }
        
    except Exception as e:
        logger.error(f"Error leaving session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============== 任务管理API ==============

@router.post("/session/{session_id}/task")
async def assign_task(session_id: str, request: AssignTaskRequest):
    """
    分配任务
    
    向会话中的Agent分配任务。
    """
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        session = await orchestrator.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        result = await orchestrator.assign_task(
            session_id=session_id,
            task_name=request.name,
            description=request.description,
            payload=request.payload,
            assigned_agent=request.assigned_agent,
            priority=request.priority
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to assign task"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/session/{session_id}/tasks")
async def list_tasks(session_id: str):
    """列出会话任务"""
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        session = await orchestrator.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            "success": True,
            "count": len(session.tasks),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "description": t.description,
                    "status": "waiting" if t.dependencies else "pending",
                    "assigned_agent": t.assigned_agent,
                    "priority": t.priority,
                    "dependencies": t.dependencies
                }
                for t in session.tasks
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============== 执行API ==============

@router.post("/session/{session_id}/execute")
async def execute_collaboration(
    session_id: str,
    request: Optional[ExecuteCollaborationRequest] = None,
    background_tasks: BackgroundTasks = None
):
    """
    执行协作
    
    在协作会话中执行任务。
    """
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        session = await orchestrator.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # 准备任务输入
        task_inputs = None
        if request and request.tasks:
            task_inputs = request.tasks
        
        # 异步执行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: asyncio.run(orchestrator.execute_collaboration(session_id, task_inputs))
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing collaboration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/session/{session_id}/cancel")
async def cancel_execution(session_id: str):
    """取消执行"""
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        success = await orchestrator.cancel_session(session_id)
        
        return {
            "success": success,
            "message": "Execution cancelled" if success else "Failed to cancel"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============== 结果和进度API ==============

@router.get("/session/{session_id}/result")
async def get_result(session_id: str):
    """获取执行结果"""
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        result = await orchestrator.get_session_result(session_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            "success": True,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/session/{session_id}/progress")
async def get_progress(session_id: str):
    """获取执行进度"""
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        progress = await orchestrator.get_session_progress(session_id)
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            "success": True,
            "progress": {
                "session_id": progress.session_id,
                "total_tasks": progress.total_tasks,
                "completed_tasks": progress.completed_tasks,
                "failed_tasks": progress.failed_tasks,
                "blocked_tasks": progress.blocked_tasks,
                "progress_percentage": progress.progress_percentage,
                "current_phase": progress.current_phase,
                "last_updated": progress.last_updated.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============== 工具API ==============

@router.post("/decompose")
async def decompose_task(request: TaskDecomposeRequest):
    """
    分解任务
    
    将复杂任务分解为可执行的子任务。
    """
    try:
        from agents.collaboration import create_task_decomposer
        from agents.collaboration.models import TaskInput
        from agents.collaboration.task_decomposer import DecompositionStrategy
        
        decomposer = create_task_decomposer()
        
        task = TaskInput(
            name=request.name,
            description=request.description,
            payload={},
            priority=0,
            dependencies=[]
        )
        
        try:
            strategy = DecompositionStrategy(request.strategy)
        except ValueError:
            strategy = DecompositionStrategy.HIERARCHICAL
        
        subtasks = await decomposer.decompose(task, strategy=strategy)
        
        return {
            "success": True,
            "task_id": task.task_id,
            "subtasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "description": t.description,
                    "dependencies": t.dependencies
                }
                for t in subtasks
            ]
        }
        
    except Exception as e:
        logger.error(f"Error decomposing task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/active-sessions")
async def list_active_sessions():
    """列出活跃会话"""
    try:
        from agents.collaboration import get_orchestrator
        
        orchestrator = get_orchestrator()
        if not orchestrator._initialized:
            await orchestrator.initialize()
        
        sessions = await orchestrator.list_active_sessions()
        
        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Error listing active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# 导入asyncio用于异步执行
import asyncio
