"""
Agent Orchestration API - Agent编排API端点
提供多Agent协同工作平台的REST API接口
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import logging

from agents.orchestration.engine import (
    orchestration_engine,
    TaskPriority,
    TaskStatus
)
from agents.orchestration.workshop import (
    WorkshopManager,
    WorkshopStatus,
    ParticipantRole
)
from agents.orchestration.communication import (
    communication_manager,
    MessageType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents/orchestration", tags=["Agent Orchestration"])

# 初始化管理器
workshop_manager = WorkshopManager(orchestration_engine)


# ============== 请求/响应模型 ==============

class CreateSessionRequest(BaseModel):
    """创建协作会话请求"""
    name: str = Field(..., min_length=1, max_length=200, description="会话名称")
    description: str = Field(default="", description="会话描述")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class CreateSessionResponse(BaseModel):
    """创建协作会话响应"""
    success: bool
    session: Dict[str, Any]
    message: str


class CreateTaskRequest(BaseModel):
    """创建任务请求"""
    name: str = Field(..., min_length=1, max_length=200, description="任务名称")
    description: str = Field(default="", description="任务描述")
    agent_type: str = Field(..., description="Agent类型标识")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    dependencies: List[str] = Field(default_factory=list, description="依赖任务ID列表")
    priority: int = Field(default=2, ge=1, le=4, description="优先级(1-4)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ExecuteTaskRequest(BaseModel):
    """执行任务请求"""
    task_id: str = Field(..., description="任务ID")
    sync: bool = Field(default=True, description="是否同步执行")


class ExecuteSessionRequest(BaseModel):
    """执行会话请求"""
    sync: bool = Field(default=True, description="是否同步执行")


class CancelTaskRequest(BaseModel):
    """取消任务请求"""
    task_id: str = Field(..., description="任务ID")


class CreateWorkshopRequest(BaseModel):
    """创建工坊请求"""
    name: str = Field(..., min_length=1, max_length=200, description="工坊名称")
    host_id: str = Field(..., description="主持人ID")
    description: str = Field(default="", description="工坊描述")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class AddStepRequest(BaseModel):
    """添加步骤请求"""
    name: str = Field(..., description="步骤名称")
    agent_type: str = Field(..., description="Agent类型")
    instruction: str = Field(..., description="指令")
    description: str = Field(default="", description="描述")
    order: int = Field(default=None, description="顺序")


class AddParticipantRequest(BaseModel):
    """添加参与者请求"""
    name: str = Field(..., description="参与者名称")
    agent_types: List[str] = Field(default_factory=list, description="Agent类型列表")
    role: str = Field(default="contributor", description="角色")


class WorkshopControlRequest(BaseModel):
    """工坊控制请求"""
    action: str = Field(..., description="控制动作(start/stop/pause/resume)")


class SendMessageRequest(BaseModel):
    """发送消息请求"""
    receiver_id: str = Field(..., description="接收者ID")
    content: Dict[str, Any] = Field(..., description="消息内容")
    msg_type: str = Field(default="request", description="消息类型")


class PublishTopicRequest(BaseModel):
    """发布主题请求"""
    topic: str = Field(..., description="主题")
    content: Dict[str, Any] = Field(..., description="内容")


# ============== 协作会话API ==============

@router.post("/create", response_model=CreateSessionResponse)
async def create_collaboration_session(request: CreateSessionRequest):
    """
    创建协作会话
    
    创建一个新的多Agent协作会话，用于管理多个Agent的协同工作。
    """
    try:
        session = orchestration_engine.create_session(
            name=request.name,
            description=request.description,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "session": session.to_dict(),
            "message": f"Session '{request.name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Create session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(status: str = None):
    """
    获取协作会话列表
    
    查询所有协作会话，支持按状态过滤。
    """
    try:
        sessions = orchestration_engine.list_sessions(status=status)
        return {
            "success": True,
            "sessions": sessions,
            "total": len(sessions)
        }
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    获取会话详情
    """
    try:
        session = orchestration_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session": session.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/tasks")
async def add_task(session_id: str, request: CreateTaskRequest):
    """
    添加任务到会话
    """
    try:
        task = orchestration_engine.create_task(
            session_id=session_id,
            name=request.name,
            agent_type=request.agent_type,
            input_data=request.input_data,
            description=request.description,
            dependencies=request.dependencies,
            priority=TaskPriority(request.priority),
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "task": task.to_dict(),
            "message": f"Task '{request.name}' added successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Add task error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/execute")
async def execute_session(session_id: str, request: ExecuteSessionRequest):
    """
    执行协作会话
    
    执行会话中的所有任务，支持同步/异步模式。
    """
    try:
        session = await orchestration_engine.execute_session(
            session_id=session_id,
            sync=request.sync
        )
        
        return {
            "success": True,
            "session": session.to_dict(),
            "message": "Session execution started"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Execute session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/tasks/{task_id}/cancel")
async def cancel_task(session_id: str, request: CancelTaskRequest):
    """
    取消任务
    """
    try:
        success = orchestration_engine.cancel_task(session_id, request.task_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel task")
        
        return {
            "success": True,
            "message": "Task cancelled successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel task error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """
    获取会话状态
    """
    try:
        status = orchestration_engine.get_session_status(session_id)
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 工坊模式API ==============

@router.post("/workshop")
async def create_workshop(request: CreateWorkshopRequest):
    """
    创建工坊
    
    创建一个新的Agent工坊会话，用于交互式协作。
    """
    try:
        workshop = workshop_manager.create_workshop(
            name=request.name,
            host_id=request.host_id,
            description=request.description,
            configuration=request.configuration,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "workshop": workshop.to_dict(),
            "message": f"Workshop '{request.name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Create workshop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workshops")
async def list_workshops(status: str = None):
    """
    获取工坊列表
    """
    try:
        workshops = workshop_manager.list_workshops(status=status)
        return {
            "success": True,
            "workshops": workshops,
            "total": len(workshops)
        }
    except Exception as e:
        logger.error(f"List workshops error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workshops/{workshop_id}")
async def get_workshop(workshop_id: str):
    """
    获取工坊详情
    """
    try:
        workshop = workshop_manager.get_workshop(workshop_id)
        if not workshop:
            raise HTTPException(status_code=404, detail="Workshop not found")
        
        return {
            "success": True,
            "workshop": workshop.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get workshop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workshops/{workshop_id}/steps")
async def add_workshop_step(workshop_id: str, request: AddStepRequest):
    """
    添加工坊步骤
    """
    try:
        step = workshop_manager.add_step(
            workshop_id=workshop_id,
            name=request.name,
            agent_type=request.agent_type,
            instruction=request.instruction,
            description=request.description,
            order=request.order
        )
        
        return {
            "success": True,
            "step": step.to_dict(),
            "message": f"Step '{request.name}' added successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Add step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workshops/{workshop_id}/participants")
async def add_workshop_participant(workshop_id: str, request: AddParticipantRequest):
    """
    添加工坊参与者
    """
    try:
        role = ParticipantRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    try:
        participant = workshop_manager.add_participant(
            workshop_id=workshop_id,
            name=request.name,
            agent_types=request.agent_types,
            role=role
        )
        
        return {
            "success": True,
            "participant": participant.to_dict(),
            "message": f"Participant '{request.name}' joined successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Add participant error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workshops/{workshop_id}/control")
async def control_workshop(workshop_id: str, request: WorkshopControlRequest):
    """
    控制工坊
    
    启动、暂停、恢复或停止工坊。
    """
    try:
        workshop = workshop_manager.get_workshop(workshop_id)
        if not workshop:
            raise HTTPException(status_code=404, detail="Workshop not found")
        
        action = request.action.lower()
        
        if action == "start":
            workshop = await workshop_manager.start_workshop(workshop_id)
            message = "Workshop started"
        elif action == "stop":
            workshop = await workshop_manager.stop_workshop(workshop_id)
            message = "Workshop stopped"
        elif action == "pause":
            workshop = await workshop_manager.pause_workshop(workshop_id)
            message = "Workshop paused"
        elif action == "resume":
            workshop = await workshop_manager.resume_workshop(workshop_id)
            message = "Workshop resumed"
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        return {
            "success": True,
            "workshop": workshop.to_dict(),
            "message": message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Control workshop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workshops/{workshop_id}/status")
async def get_workshop_status(workshop_id: str):
    """
    获取工坊状态
    """
    try:
        status = workshop_manager.get_workshop_status(workshop_id)
        if not status:
            raise HTTPException(status_code=404, detail="Workshop not found")
        
        return {
            "success": True,
            "status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get workshop status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workshops/{workshop_id}/logs")
async def get_workshop_logs(workshop_id: str):
    """
    获取工坊日志
    """
    try:
        logs = workshop_manager.get_workshop_logs(workshop_id)
        return {
            "success": True,
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Get workshop logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 协作监控API ==============

@router.get("/monitor")
async def monitor_collaboration():
    """
    协作监控
    
    获取所有协作会话和工坊的监控信息。
    """
    try:
        sessions = orchestration_engine.monitor_sessions()
        workshops = workshop_manager.monitor_workshops()
        
        return {
            "success": True,
            "monitor": {
                "sessions": sessions,
                "workshops": workshops,
                "active_sessions": len([s for s in sessions if s["status"] == "running"]),
                "active_workshops": len([w for w in workshops if w["status"] == "running"])
            }
        }
    except Exception as e:
        logger.error(f"Monitor error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/active")
async def get_active_collaborations():
    """
    获取活跃协作
    """
    try:
        active_sessions = orchestration_engine.list_sessions(status="running")
        active_workshops = workshop_manager.get_active_workshops()
        
        return {
            "success": True,
            "active_sessions": active_sessions,
            "active_workshops": active_workshops
        }
    except Exception as e:
        logger.error(f"Get active collaborations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Agent通信API ==============

@router.post("/message/send")
async def send_message(request: SendMessageRequest, sender_id: str = "api"):
    """
    发送Agent消息
    """
    try:
        msg_type = MessageType(request.msg_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid message type")
    
    try:
        message = await communication_manager.send_direct(
            sender_id=sender_id,
            receiver_id=request.receiver_id,
            content=request.content
        )
        
        return {
            "success": True,
            "message": message.to_dict(),
            "message_id": message.id
        }
    except Exception as e:
        logger.error(f"Send message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/publish")
async def publish_topic(request: PublishTopicRequest, sender_id: str = "api"):
    """
    发布主题消息
    """
    try:
        await communication_manager.publish(
            topic=request.topic,
            sender_id=sender_id,
            content=request.content
        )
        
        return {
            "success": True,
            "message": f"Message published to topic '{request.topic}'"
        }
    except Exception as e:
        logger.error(f"Publish topic error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/subscribe")
async def subscribe_topic(topic: str, agent_id: str):
    """
    订阅主题
    """
    try:
        await communication_manager.subscribe(agent_id=agent_id, topic=topic)
        
        return {
            "success": True,
            "message": f"Subscribed to topic '{topic}'"
        }
    except Exception as e:
        logger.error(f"Subscribe topic error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents(capability: str = None):
    """
    列出Agent
    """
    try:
        agents = communication_manager.list_agents(capability=capability)
        return {
            "success": True,
            "agents": agents,
            "total": len(agents)
        }
    except Exception as e:
        logger.error(f"List agents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communication/stats")
async def get_communication_stats():
    """
    获取通信统计
    """
    try:
        stats = communication_manager.get_communication_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Get communication stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communication/history")
async def get_message_history(agent_id: str = None, msg_type: str = None):
    """
    获取消息历史
    """
    try:
        message_type = None
        if msg_type:
            message_type = MessageType(msg_type)
        
        history = communication_manager.get_message_history(
            agent_id=agent_id,
            msg_type=message_type
        )
        
        return {
            "success": True,
            "history": history,
            "total": len(history)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid message type")
    except Exception as e:
        logger.error(f"Get message history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
