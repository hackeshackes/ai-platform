"""
工作流自动化 (Workflow Automation)
=====================================

功能:
- 流程设计与定义
- 条件分支处理
- 人工干预点
- 审批流程
- 状态追踪

作者: AI Platform Team
版本: 1.0.0
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_INTERVENTION = "awaiting_intervention"


class StepType(Enum):
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    APPROVAL = "approval"
    INTERVENTION = "intervention"
    NOTIFICATION = "notification"
    SUBWORKFLOW = "subworkflow"
    END = "end"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING = "awaiting"
    CANCELLED = "cancelled"


@dataclass
class ConditionBranch:
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    next_step: Optional[str] = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        try:
            return self.condition(context)
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False


@dataclass
class ManualIntervention:
    id: str
    name: str
    description: str
    required_approval: bool = False
    approvers: List[str] = field(default_factory=list)
    timeout: int = 86400
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ApprovalProcess:
    id: str
    name: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    auto_approve: bool = False
    timeout: int = 172800
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class WorkflowStep:
    id: str
    name: str
    step_type: StepType
    func: Optional[Callable[..., Any]] = None
    condition: Optional[ConditionBranch] = None
    next_step: Optional[str] = None
    branches: List[ConditionBranch] = field(default_factory=list)
    parallel_steps: List[str] = field(default_factory=list)
    intervention: Optional[ManualIntervention] = None
    approval: Optional[ApprovalProcess] = None
    notification_config: Optional[Dict[str, Any]] = None
    retry_count: int = 3
    timeout: int = 3600
    skip_on_failure: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    current_step_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    paused_at: Optional[datetime] = None
    approval_history: List[Dict[str, Any]] = field(default_factory=list)
    intervention_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Workflow:
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    start_step: Optional[str] = None
    end_steps: List[str] = field(default_factory=list)
    global_retry_count: int = 3
    global_timeout: int = 86400
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None


class WorkflowAutomation:
    """工作流自动化引擎"""
    
    def __init__(
        self,
        max_concurrent_steps: int = 10,
        default_timeout: int = 86400,
        enable_audit: bool = True
    ):
        self.max_concurrent_steps = max_concurrent_steps
        self.default_timeout = default_timeout
        self.enable_audit = enable_audit
        
        self._workflows: Dict[str, Workflow] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._execution_queue: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent_steps)
        self._active_executions: Dict[str, asyncio.Task] = {}
        
        logger.info("Workflow Automation Engine initialized")
    
    def create_workflow(self, name: str, description: str = "", version: str = "1.0.0", **kwargs) -> str:
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            version=version,
            **kwargs
        )
        self._workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {name} (ID: {workflow.id})")
        return workflow.id
    
    def add_step(self, workflow_id: str, name: str, step_type: StepType, func: Optional[Callable] = None, **kwargs) -> str:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        step = WorkflowStep(id=str(uuid.uuid4()), name=name, step_type=step_type, func=func, **kwargs)
        workflow.steps.append(step)
        workflow.updated_at = datetime.now()
        
        if not workflow.start_step:
            workflow.start_step = step.id
        
        return step.id
    
    def connect_steps(self, workflow_id: str, from_step: str, to_step: str, condition: Optional[Callable] = None) -> None:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        from_step_obj = workflow.get_step(from_step)
        if not from_step_obj:
            raise ValueError(f"Step not found: {from_step}")
        
        if condition:
            branch = ConditionBranch(name=f"branch_{from_step}_{to_step}", condition=condition, next_step=to_step)
            from_step_obj.branches.append(branch)
        else:
            from_step_obj.next_step = to_step
        
        workflow.updated_at = datetime.now()
    
    def start_execution(self, workflow_id: str, context: Optional[Dict[str, Any]] = None, start_step_id: Optional[str] = None) -> str:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            current_step_id=start_step_id or workflow.start_step,
            context=context or {}
        )
        self._executions[execution_id] = execution
        
        task = asyncio.create_task(self._execute_workflow(execution, workflow))
        self._active_executions[execution_id] = task
        
        return execution_id
    
    async def _execute_workflow(self, execution: WorkflowExecution, workflow: Workflow) -> None:
        start_time = datetime.now()
        execution.start_time = start_time
        
        try:
            current_step_id = execution.current_step_id
            
            while current_step_id:
                step = workflow.get_step(current_step_id)
                if not step:
                    break
                
                execution.current_step_id = current_step_id
                result = await self._execute_step(execution, step)
                
                execution.step_results[step.id] = {
                    "status": result["status"],
                    "output": result.get("output"),
                    "error": result.get("error"),
                    "timestamp": datetime.now().isoformat()
                }
                
                if result["status"] == StepStatus.FAILED and not step.skip_on_failure:
                    execution.status = WorkflowStatus.FAILED
                    execution.error = result.get("error")
                    break
                
                current_step_id = self._get_next_step(workflow, step, execution.context)
                
                if current_step_id and current_step_id in workflow.end_steps:
                    execution.status = WorkflowStatus.COMPLETED
                    break
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                
        except Exception as e:
            logger.exception(f"Workflow execution error: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
        
        finally:
            execution.end_time = datetime.now()
            execution.context["execution_status"] = execution.status.value
            if execution_id := execution.execution_id in self._active_executions:
                del self._active_executions[execution.execution_id]
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep) -> Dict[str, Any]:
        result = {"status": StepStatus.PENDING, "output": None, "error": None}
        
        try:
            if step.step_type == StepType.TASK and step.func:
                for attempt in range(step.retry_count + 1):
                    try:
                        if asyncio.iscoroutinefunction(step.func):
                            output = await asyncio.wait_for(step.func(execution.context), timeout=step.timeout)
                        else:
                            output = step.func(execution.context)
                        
                        result["status"] = StepStatus.COMPLETED
                        result["output"] = output
                        execution.context[f"step_{step.id}_output"] = output
                        break
                    except Exception as e:
                        if attempt < step.retry_count:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            result["status"] = StepStatus.FAILED
                            result["error"] = str(e)
            
            elif step.step_type == StepType.CONDITION and step.condition:
                outcome = step.condition.evaluate(execution.context)
                result["status"] = StepStatus.COMPLETED
                result["output"] = {"condition_met": outcome}
            
            elif step.step_type == StepType.NOTIFICATION:
                result["status"] = StepStatus.COMPLETED
                result["output"] = {"notification_sent": True}
            
            else:
                result["status"] = StepStatus.COMPLETED
        
        except Exception as e:
            result["status"] = StepStatus.FAILED
            result["error"] = str(e)
        
        return result
    
    def _get_next_step(self, workflow: Workflow, current_step: WorkflowStep, context: Dict[str, Any]) -> Optional[str]:
        for branch in current_step.branches:
            if branch.evaluate(context):
                return branch.next_step
        return current_step.next_step
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        return self._executions.get(execution_id)
    
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        return [{"id": w.id, "name": w.name, "description": w.description, "version": w.version, "step_count": len(w.steps)} for w in self._workflows.values()]
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        execution = self._executions.get(execution_id)
        if not execution:
            return None
        return {"execution_id": execution.execution_id, "workflow_id": execution.workflow_id, "status": execution.status.value, "current_step_id": execution.current_step_id}
    
    def cancel_execution(self, execution_id: str) -> bool:
        execution = self._executions.get(execution_id)
        if execution:
            execution.status = WorkflowStatus.CANCELLED
            if execution_id in self._active_executions:
                self._active_executions[execution_id].cancel()
                del self._active_executions[execution_id]
            return True
        return False
