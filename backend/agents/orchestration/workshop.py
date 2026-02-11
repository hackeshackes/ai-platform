"""
Workshop Manager - Agent工坊模式
提供交互式Agent工作坊环境，支持实时协作和调试
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class WorkshopStatus(Enum):
    """工坊状态枚举"""
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class ParticipantRole(Enum):
    """参与者角色枚举"""
    HOST = "host"
    CONTRIBUTOR = "contributor"
    OBSERVER = "observer"
    REVIEWER = "reviewer"


@dataclass
class WorkshopParticipant:
    """工坊参与者"""
    id: str
    name: str
    role: ParticipantRole
    agent_types: List[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "agent_types": self.agent_types,
            "joined_at": self.joined_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class WorkshopSession:
    """工坊会话"""
    id: str
    name: str
    description: str
    host_id: str
    participants: Dict[str, WorkshopParticipant] = field(default_factory=dict)
    status: WorkshopStatus = WorkshopStatus.IDLE
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "host_id": self.host_id,
            "participants": {k: v.to_dict() for k, v in self.participants.items()},
            "status": self.status.value,
            "configuration": self.configuration,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "logs": self.logs,
            "metadata": self.metadata
        }


@dataclass
class WorkshopStep:
    """工坊步骤"""
    id: str
    name: str
    description: str
    agent_type: str
    instruction: str
    order: int
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "instruction": self.instruction,
            "order": self.order,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at.isoformat()
        }


class WorkshopManager:
    """
    Agent工坊管理器
    
    核心功能：
    - 工坊会话管理
    - 参与者协调
    - 步骤编排与执行
    - 实时协作支持
    - 调试与监控
    """
    
    def __init__(self, orchestration_engine):
        self.engine = orchestration_engine
        self.workshops: Dict[str, WorkshopSession] = {}
        self.steps: Dict[str, List[WorkshopStep]] = {}
        self.event_callbacks: List[Callable] = []
        self._active_workshops = set()
        
    def create_workshop(self, name: str, host_id: str, description: str = "",
                        configuration: Dict[str, Any] = None,
                        metadata: Dict[str, Any] = None) -> WorkshopSession:
        """创建工坊"""
        workshop = WorkshopSession(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            host_id=host_id,
            configuration=configuration or {},
            metadata=metadata or {}
        )
        
        # 添加主机为参与者
        host = WorkshopParticipant(
            id=host_id,
            name="Host",
            role=ParticipantRole.HOST,
            agent_types=["coordinator"]
        )
        workshop.participants[host_id] = host
        
        self.workshops[workshop.id] = workshop
        self.steps[workshop.id] = []
        
        self._log_event(workshop.id, "workshop_created", 
                        {"name": name, "host_id": host_id})
        logger.info(f"Created workshop: {workshop.id}")
        
        return workshop
    
    def get_workshop(self, workshop_id: str) -> Optional[WorkshopSession]:
        """获取工坊"""
        return self.workshops.get(workshop_id)
    
    def list_workshops(self, status: str = None) -> List[Dict[str, Any]]:
        """列出工坊"""
        workshops = list(self.workshops.values())
        if status:
            workshops = [w for w in workshops if w.status.value == status]
        return [w.to_dict() for w in workshops]
    
    def add_participant(self, workshop_id: str, name: str, 
                        agent_types: List[str] = None,
                        role: ParticipantRole = ParticipantRole.CONTRIBUTOR,
                        metadata: Dict[str, Any] = None) -> WorkshopParticipant:
        """添加参与者"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            raise ValueError(f"Workshop not found: {workshop_id}")
        
        participant = WorkshopParticipant(
            id=str(uuid.uuid4()),
            name=name,
            role=role,
            agent_types=agent_types or [],
            metadata=metadata or {}
        )
        
        workshop.participants[participant.id] = participant
        
        self._log_event(workshop_id, "participant_joined", 
                       {"participant_id": participant.id, "name": name})
        
        return participant
    
    def remove_participant(self, workshop_id: str, participant_id: str) -> bool:
        """移除参与者"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            return False
        
        if participant_id in workshop.participants:
            del workshop.participants[participant_id]
            self._log_event(workshop_id, "participant_left", 
                           {"participant_id": participant_id})
            return True
        
        return False
    
    def add_step(self, workshop_id: str, name: str, agent_type: str,
                 instruction: str, description: str = "",
                 order: int = None) -> WorkshopStep:
        """添加工坊步骤"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            raise ValueError(f"Workshop not found: {workshop_id}")
        
        step = WorkshopStep(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            agent_type=agent_type,
            instruction=instruction,
            order=order or len(self.steps[workshop_id]) + 1
        )
        
        self.steps[workshop_id].append(step)
        
        self._log_event(workshop_id, "step_added", 
                       {"step_id": step.id, "name": name})
        
        return step
    
    def configure_workshop(self, workshop_id: str, 
                          configuration: Dict[str, Any]) -> WorkshopSession:
        """配置工坊"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            raise ValueError(f"Workshop not found: {workshop_id}")
        
        workshop.configuration.update(configuration)
        return workshop
    
    async def start_workshop(self, workshop_id: str) -> WorkshopSession:
        """启动工坊"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            raise ValueError(f"Workshop not found: {workshop_id}")
        
        if workshop.status != WorkshopStatus.IDLE:
            raise ValueError(f"Workshop cannot start from status: {workshop.status.value}")
        
        workshop.status = WorkshopStatus.PREPARING
        workshop.started_at = datetime.now()
        
        self._active_workshops.add(workshop_id)
        
        # 创建编排会话
        session = self.engine.create_session(
            name=f"Workshop-{workshop.name}",
            description=workshop.description,
            metadata={"workshop_id": workshop_id}
        )
        
        # 将步骤转换为任务
        for step in sorted(self.steps[workshop_id], key=lambda s: s.order):
            self.engine.create_task(
                session_id=session.id,
                name=step.name,
                agent_type=step.agent_type,
                input_data={"instruction": step.instruction},
                description=step.description,
                metadata={"step_id": step.id}
            )
        
        workshop.status = WorkshopStatus.RUNNING
        
        self._log_event(workshop_id, "workshop_started", 
                        {"session_id": session.id})
        
        # 在后台执行
        asyncio.create_task(self._run_workshop(workshop_id, session.id))
        
        return workshop
    
    async def _run_workshop(self, workshop_id: str, session_id: str):
        """执行工坊"""
        try:
            await self.engine.execute_session(session_id, sync=True)
            
            # 更新步骤状态
            for step in self.steps[workshop_id]:
                task_status = self.engine.get_session_status(session_id)
                # 根据任务完成情况更新步骤状态
            
            workshop = self.workshops[workshop_id]
            workshop.status = WorkshopStatus.COMPLETED
            workshop.completed_at = datetime.now()
            
            self._log_event(workshop_id, "workshop_completed", {})
            
        except Exception as e:
            workshop = self.workshops[workshop_id]
            workshop.status = WorkshopStatus.ERROR
            self._log_event(workshop_id, "workshop_error", {"error": str(e)})
            logger.error(f"Workshop {workshop_id} error: {e}")
        finally:
            self._active_workshops.discard(workshop_id)
    
    async def pause_workshop(self, workshop_id: str) -> WorkshopSession:
        """暂停工坊"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            raise ValueError(f"Workshop not found: {workshop_id}")
        
        if workshop.status != WorkshopStatus.RUNNING:
            raise ValueError(f"Workshop cannot pause from status: {workshop.status.value}")
        
        workshop.status = WorkshopStatus.PAUSED
        self._log_event(workshop_id, "workshop_paused", {})
        
        return workshop
    
    async def resume_workshop(self, workshop_id: str) -> WorkshopSession:
        """恢复工坊"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            raise ValueError(f"Workshop not found: {workshop_id}")
        
        if workshop.status != WorkshopStatus.PAUSED:
            raise ValueError(f"Workshop cannot resume from status: {workshop.status.value}")
        
        workshop.status = WorkshopStatus.RUNNING
        self._log_event(workshop_id, "workshop_resumed", {})
        
        return workshop
    
    async def stop_workshop(self, workshop_id: str) -> WorkshopSession:
        """停止工坊"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            raise ValueError(f"Workshop not found: {workshop_id}")
        
        workshop.status = WorkshopStatus.COMPLETED
        workshop.completed_at = datetime.now()
        self._active_workshops.discard(workshop_id)
        
        self._log_event(workshop_id, "workshop_stopped", {})
        
        return workshop
    
    def get_workshop_status(self, workshop_id: str) -> Dict[str, Any]:
        """获取工坊状态"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            return None
        
        steps = self.steps.get(workshop_id, [])
        return {
            "workshop_id": workshop_id,
            "name": workshop.name,
            "status": workshop.status.value,
            "participants_count": len(workshop.participants),
            "steps_count": len(steps),
            "completed_steps": sum(1 for s in steps if s.status == "completed"),
            "progress": sum(1 for s in steps if s.status == "completed") / len(steps) * 100 if steps else 0,
            "started_at": workshop.started_at.isoformat() if workshop.started_at else None,
            "duration": (datetime.now() - workshop.started_at).total_seconds() if workshop.started_at else 0
        }
    
    def get_workshop_logs(self, workshop_id: str) -> List[Dict[str, Any]]:
        """获取工坊日志"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            return []
        
        return workshop.logs
    
    def add_log(self, workshop_id: str, event_type: str, 
                data: Dict[str, Any]):
        """添加日志"""
        self._log_event(workshop_id, event_type, data)
    
    def _log_event(self, workshop_id: str, event_type: str, 
                   data: Dict[str, Any]):
        """记录事件"""
        workshop = self.workshops.get(workshop_id)
        if not workshop:
            return
        
        log = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        workshop.logs.append(log)
        
        # 触发回调
        for callback in self.event_callbacks:
            try:
                callback(workshop_id, event_type, data)
            except Exception as e:
                logger.error(f"Workshop event callback error: {e}")
    
    def on_event(self, callback: Callable):
        """注册事件回调"""
        self.event_callbacks.append(callback)
    
    def get_active_workshops(self) -> List[Dict[str, Any]]:
        """获取活跃工坊"""
        return [self.get_workshop_status(wid) for wid in self._active_workshops]
    
    def monitor_workshops(self) -> List[Dict[str, Any]]:
        """监控所有工坊"""
        return [self.get_workshop_status(wid) for wid in self.workshops]


# 默认Agent处理器
async def default_agent_handler(input_data: Dict[str, Any], 
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
    """默认Agent处理器"""
    return {
        "output": input_data.get("instruction", ""),
        "processed": True,
        "timestamp": datetime.now().isoformat()
    }
