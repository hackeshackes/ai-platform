"""
协作模型定义
提供Agent协作网络的核心数据模型
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class CollaborationMode(str, Enum):
    """协作模式枚举"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"           # 并行执行
    HIERARCHICAL = "hierarchical"   # 层级协作
    CONSENSUS = "consensus"         # 共识决策


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"             # 待执行
    IN_PROGRESS = "in_progress"     # 执行中
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"               # 失败
    BLOCKED = "blocked"              # 阻塞
    WAITING = "waiting"              # 等待前置任务


class SessionStatus(str, Enum):
    """协作会话状态枚举"""
    CREATED = "created"             # 创建
    ACTIVE = "active"               # 活跃
    EXECUTING = "executing"         # 执行中
    COMPLETED = "completed"         # 完成
    FAILED = "failed"               # 失败
    CANCELLED = "cancelled"         # 取消


class AgentRole(str, Enum):
    """Agent角色枚举"""
    COORDINATOR = "coordinator"     # 协调者
    WORKER = "worker"               # 工作节点
    SUPERVISOR = "supervisor"       # 监督者
    REVIEWER = "reviewer"           # 审核者


class AgentInfo(BaseModel):
    """Agent信息"""
    agent_id: str = Field(..., description="Agent唯一标识")
    role: AgentRole = Field(default=AgentRole.WORKER, description="Agent角色")
    name: str = Field(..., description="Agent名称")
    capabilities: List[str] = Field(default_factory=list, description="能力列表")
    status: str = Field(default="idle", description="状态")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class TaskInput(BaseModel):
    """任务输入"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务ID")
    name: str = Field(..., description="任务名称")
    description: str = Field(default="", description="任务描述")
    payload: Dict[str, Any] = Field(default_factory=dict, description="任务数据")
    priority: int = Field(default=0, description="优先级")
    assigned_agent: Optional[str] = Field(default=None, description="分配的Agent")
    dependencies: List[str] = Field(default_factory=list, description="依赖任务ID列表")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class TaskOutput(BaseModel):
    """任务输出"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="状态")
    result: Optional[Dict[str, Any]] = Field(default=None, description="执行结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    execution_time_ms: int = Field(default=0, description="执行时间(ms)")
    agent_id: Optional[str] = Field(default=None, description="执行Agent ID")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class TaskDefinition(BaseModel):
    """任务定义"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务ID")
    name: str = Field(..., description="任务名称")
    description: str = Field(default="", description="任务描述")
    source: str = Field(..., description="源Agent ID")
    target: str = Field(..., description="目标Agent ID")
    condition: str = Field(default="", description="触发条件")
    payload_template: Dict[str, Any] = Field(default_factory=dict, description="数据模板")
    timeout_ms: int = Field(default=300000, description="超时时间(ms)")
    retry_count: int = Field(default=3, description="重试次数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class WorkflowDefinition(BaseModel):
    """工作流定义"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="工作流ID")
    name: str = Field(..., description="工作流名称")
    description: str = Field(default="", description="工作流描述")
    mode: CollaborationMode = Field(default=CollaborationMode.SEQUENTIAL, description="协作模式")
    agents: List[AgentInfo] = Field(default_factory=list, description="参与的Agent列表")
    tasks: List[TaskDefinition] = Field(default_factory=list, description="任务列表")
    start_agent: str = Field(default="", description="起始Agent")
    end_agents: List[str] = Field(default_factory=list, description="结束Agent列表")
    global_inputs: Dict[str, Any] = Field(default_factory=dict, description="全局输入")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class Message(BaseModel):
    """Agent间消息"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="消息ID")
    session_id: str = Field(..., description="会话ID")
    sender_id: str = Field(..., description="发送者ID")
    receiver_id: str = Field(..., description="接收者ID")
    message_type: str = Field(default="task", description="消息类型")
    payload: Dict[str, Any] = Field(default_factory=dict, description="消息内容")
    priority: int = Field(default=0, description="优先级")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    correlation_id: Optional[str] = Field(default=None, description="关联ID")


class ConsensusProposal(BaseModel):
    """共识提案"""
    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="提案ID")
    session_id: str = Field(..., description="会话ID")
    proposer_id: str = Field(..., description="提案者ID")
    proposal_type: str = Field(..., description="提案类型")
    content: Dict[str, Any] = Field(default_factory=dict, description="提案内容")
    votes: Dict[str, bool] = Field(default_factory=dict, description="投票")
    threshold: float = Field(default=0.5, description="通过阈值")
    status: str = Field(default="pending", description="状态")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow(), description="过期时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class SessionProgress(BaseModel):
    """会话进度"""
    session_id: str = Field(..., description="会话ID")
    total_tasks: int = Field(default=0, description="总任务数")
    completed_tasks: int = Field(default=0, description="完成的任务数")
    failed_tasks: int = Field(default=0, description="失败的任务数")
    blocked_tasks: int = Field(default=0, description="阻塞的任务数")
    current_phase: str = Field(default="", description="当前阶段")
    progress_percentage: float = Field(default=0.0, description="进度百分比")
    estimated_time_remaining_ms: int = Field(default=0, description="预计剩余时间(ms)")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最后更新")


class CollaborationSession(BaseModel):
    """协作会话"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="会话ID")
    name: str = Field(..., description="会话名称")
    description: str = Field(default="", description="会话描述")
    mode: CollaborationMode = Field(default=CollaborationMode.SEQUENTIAL, description="协作模式")
    status: SessionStatus = Field(default=SessionStatus.CREATED, description="会话状态")
    workflow: Optional[WorkflowDefinition] = Field(default=None, description="工作流定义")
    agents: List[AgentInfo] = Field(default_factory=list, description="参与的Agent列表")
    tasks: List[TaskInput] = Field(default_factory=list, description="任务队列")
    task_outputs: Dict[str, TaskOutput] = Field(default_factory=dict, description="任务输出")
    messages: List[Message] = Field(default_factory=list, description="消息历史")
    global_context: Dict[str, Any] = Field(default_factory=dict, description="全局上下文")
    start_time: Optional[datetime] = Field(default=None, description="开始时间")
    end_time: Optional[datetime] = Field(default=None, description="结束时间")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def add_agent(self, agent: AgentInfo) -> None:
        """添加Agent"""
        self.agents.append(agent)
    
    def add_task(self, task: TaskInput) -> None:
        """添加任务"""
        self.tasks.append(task)
    
    def update_task_output(self, output: TaskOutput) -> None:
        """更新任务输出"""
        self.task_outputs[output.task_id] = output
    
    def get_progress(self) -> SessionProgress:
        """获取进度"""
        total = len(self.tasks)
        completed = sum(1 for t in self.task_outputs.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.task_outputs.values() if t.status == TaskStatus.FAILED)
        blocked = sum(1 for t in self.task_outputs.values() if t.status == TaskStatus.BLOCKED)
        
        percentage = (completed / total * 100) if total > 0 else 0.0
        
        return SessionProgress(
            session_id=self.session_id,
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            blocked_tasks=blocked,
            progress_percentage=percentage
        )


class WorkflowExecutionResult(BaseModel):
    """工作流执行结果"""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="执行ID")
    session_id: str = Field(..., description="会话ID")
    success: bool = Field(..., description="是否成功")
    final_outputs: Dict[str, Any] = Field(default_factory=dict, description="最终输出")
    task_results: Dict[str, TaskOutput] = Field(default_factory=dict, description="任务结果")
    total_execution_time_ms: int = Field(default=0, description="总执行时间")
    agent_contributions: Dict[str, int] = Field(default_factory=dict, description="Agent贡献统计")
    errors: List[str] = Field(default_factory=list, description="错误列表")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="完成时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
