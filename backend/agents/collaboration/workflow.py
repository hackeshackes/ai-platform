"""
工作流引擎
管理Agent协作工作流的执行和调度
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import logging

from .models import (
    WorkflowDefinition, TaskDefinition, TaskInput, TaskOutput,
    TaskStatus, CollaborationMode, SessionProgress, AgentInfo
)
from .task_decomposer import TaskDecomposer, DecompositionStrategy
from .communication import CommunicationManager, get_communication_manager
from .consensus import ConsensusManager, create_consensus_manager

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """工作流状态"""
    IDLE = "idle"                   # 空闲
    INITIALIZED = "initialized"     # 已初始化
    RUNNING = "running"             # 运行中
    PAUSED = "paused"               # 暂停
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"               # 失败
    CANCELLED = "cancelled"          # 取消


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self):
        self.state = WorkflowState.IDLE
        self.current_task: Optional[TaskInput] = None
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.blocked_tasks: Set[str] = set()
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.execution_history: List[Dict] = []
        self.callbacks: Dict[str, List[Callable]] = {}
        self._running = False
    
    async def initialize(
        self,
        workflow: WorkflowDefinition,
        tasks: List[TaskInput],
        agent_map: Dict[str, AgentInfo]
    ) -> None:
        """初始化工作流"""
        self.workflow = workflow
        self.tasks = tasks
        self.agent_map = agent_map
        self.state = WorkflowState.INITIALIZED
        
        # 初始化任务队列
        for task in tasks:
            if not task.dependencies:
                await self.task_queue.put(task)
        
        logger.info(f"Workflow initialized with {len(tasks)} tasks")
    
    async def start(self) -> None:
        """开始执行"""
        if self.state != WorkflowState.INITIALIZED:
            raise RuntimeError("Workflow not initialized")
        
        self.state = WorkflowState.RUNNING
        self._running = True
        
        # 启动执行循环
        await self._execution_loop()
    
    async def _execution_loop(self) -> None:
        """执行循环"""
        while self._running and not self.task_queue.empty():
            try:
                # 获取任务
                task = await self.task_queue.get()
                self.current_task = task
                
                # 执行任务
                result = await self._execute_task(task)
                
                # 处理结果
                await self._handle_task_result(task, result)
                
            except Exception as e:
                logger.error(f"Task execution error: {e}")
                self.state = WorkflowState.FAILED
                break
        
        # 检查是否完成
        if self._running:
            if all(t.task_id in self.completed_tasks for t in self.tasks):
                self.state = WorkflowState.COMPLETED
                await self._notify_callbacks("on_complete", None)
            elif self.failed_tasks:
                self.state = WorkflowState.FAILED
    
    async def _execute_task(self, task: TaskInput) -> TaskOutput:
        """执行单个任务"""
        start_time = datetime.utcnow()
        
        try:
            # 模拟任务执行（实际会调用Agent）
            agent_id = task.assigned_agent or "default_agent"
            
            # 查找Agent
            agent = self.agent_map.get(agent_id)
            
            # 执行任务逻辑
            result = await self._run_task_logic(task, agent)
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return TaskOutput(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time_ms=execution_time,
                agent_id=agent_id
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            return TaskOutput(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
    
    async def _run_task_logic(
        self,
        task: TaskInput,
        agent: Optional[AgentInfo]
    ) -> Dict[str, Any]:
        """运行任务逻辑"""
        # 实际实现会根据任务类型调用不同的Agent处理
        await asyncio.sleep(0.1)  # 模拟处理时间
        
        return {
            "task_id": task.task_id,
            "output": f"Result for {task.name}",
            "processed_by": agent.agent_id if agent else "unknown"
        }
    
    async def _handle_task_result(
        self,
        task: TaskInput,
        result: TaskOutput
    ) -> None:
        """处理任务结果"""
        # 记录历史
        self.execution_history.append({
            "task_id": task.task_id,
            "result": result.status.value,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if result.status == TaskStatus.COMPLETED:
            self.completed_tasks.add(task.task_id)
            await self._notify_callbacks("on_task_complete", task, result)
            
            # 查找依赖此任务的后续任务
            await self._unlock_dependent_tasks(task.task_id)
            
        else:
            self.failed_tasks.add(task.task_id)
            await self._notify_callbacks("on_task_failed", task, result)
    
    async def _unlock_dependent_tasks(self, completed_task_id: str) -> None:
        """解锁依赖任务"""
        for task in self.tasks:
            if completed_task_id in task.dependencies:
                # 检查是否所有依赖都已完成
                all_deps_done = all(
                    dep_id in self.completed_tasks
                    for dep_id in task.dependencies
                )
                
                if all_deps_done:
                    await self.task_queue.put(task)
    
    async def pause(self) -> None:
        """暂停工作流"""
        self.state = WorkflowState.PAUSED
        self._running = False
    
    async def resume(self) -> None:
        """恢复工作流"""
        if self.state == WorkflowState.PAUSED:
            self._running = True
            self.state = WorkflowState.RUNNING
            await self._execution_loop()
    
    async def cancel(self) -> None:
        """取消工作流"""
        self.state = WorkflowState.CANCELLED
        self._running = False
    
    def add_callback(
        self,
        event: str,
        callback: Callable
    ) -> None:
        """添加回调"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    async def _notify_callbacks(
        self,
        event: str,
        *args
    ) -> None:
        """通知回调"""
        for callback in self.callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_progress(self) -> SessionProgress:
        """获取进度"""
        total = len(self.tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        blocked = len(self.blocked_tasks)
        
        percentage = (completed / total * 100) if total > 0 else 0.0
        
        return SessionProgress(
            session_id=self.workflow.workflow_id if self.workflow else "",
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            blocked_tasks=blocked,
            progress_percentage=percentage,
            current_phase=self.state.value
        )


class SequentialExecutor(WorkflowExecutor):
    """顺序执行器"""
    
    async def _execution_loop(self) -> None:
        """顺序执行循环"""
        for task in self.tasks:
            if not self._running:
                break
            
            self.current_task = task
            result = await self._execute_task(task)
            await self._handle_task_result(task, result)
            
            if result.status == TaskStatus.FAILED and self._running:
                self.state = WorkflowState.FAILED
                break
        
        if self._running:
            self.state = WorkflowState.COMPLETED


class ParallelExecutor(WorkflowExecutor):
    """并行执行器"""
    
    async def _execution_loop(self) -> None:
        """并行执行循环"""
        # 创建所有任务的并发执行
        tasks_to_run = []
        
        for task in self.tasks:
            if not task.dependencies:
                tasks_to_run.append(self._execute_with_deps(task))
        
        # 等待所有任务完成
        if tasks_to_run:
            await asyncio.gather(*tasks_to_run, return_exceptions=True)
        
        # 检查是否全部完成
        if self._running:
            if all(t.task_id in self.completed_tasks for t in self.tasks):
                self.state = WorkflowState.COMPLETED
            else:
                self.state = WorkflowState.FAILED
    
    async def _execute_with_deps(self, task: TaskInput) -> None:
        """执行任务（带依赖检查）"""
        # 等待依赖完成
        for dep_id in task.dependencies:
            while dep_id not in self.completed_tasks and self._running:
                await asyncio.sleep(0.1)
        
        if self._running:
            result = await self._execute_task(task)
            await self._handle_task_result(task, result)


class HierarchicalExecutor(WorkflowExecutor):
    """层级执行器"""
    
    def __init__(self):
        super().__init__()
        self.supervisor_queue: asyncio.Queue = asyncio.Queue()
        self.worker_queue: asyncio.Queue = asyncio.Queue()
        self.supervisor: Optional[AgentInfo] = None
    
    async def initialize(
        self,
        workflow: WorkflowDefinition,
        tasks: List[TaskInput],
        agent_map: Dict[str, AgentInfo]
    ) -> None:
        """初始化层级执行器"""
        await super().initialize(workflow, tasks, agent_map)
        
        # 识别监督者
        for agent in agent_map.values():
            if agent.role.value == "supervisor":
                self.supervisor = agent
                break
        
        # 分类任务
        for task in tasks:
            if task.metadata.get("phase") == "planning":
                await self.supervisor_queue.put(task)
            else:
                await self.worker_queue.put(task)
    
    async def _execution_loop(self) -> None:
        """层级执行循环"""
        # 首先监督者规划
        while not self.supervisor_queue.empty() and self._running:
            supervisor_task = await self.supervisor_queue.get()
            result = await self._execute_task(supervisor_task)
            await self._handle_task_result(supervisor_task, result)
            
            if result.status == TaskStatus.FAILED:
                break
        
        # 然后工作者执行
        if self._running:
            await self._run_worker_pool()
        
        # 最后监督者审核
        if self._running:
            await self._supervisor_review()
    
    async def _run_worker_pool(self) -> None:
        """工作者池并行执行"""
        worker_tasks = []
        
        while not self.worker_queue.empty() and self._running:
            task = await self.worker_queue.get()
            worker_tasks.append(self._execute_task(task))
            
            # 限制并行数
            if len(worker_tasks) >= 5:
                completed, worker_tasks = await asyncio.wait(
                    worker_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in completed:
                    await self._handle_task_result(
                        completed_task.task_id,  # 需要修改
                        completed_task
                    )
        
        # 等待剩余任务
        if worker_tasks:
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, TaskOutput):
                    await self._handle_task_result(result.task_id, result)
    
    async def _supervisor_review(self) -> None:
        """监督者审核阶段"""
        # 检查所有工作者任务
        review_task = TaskInput(
            name="Supervisor Review",
            description="监督者审核所有任务结果",
            payload={"review_type": "final"}
        )
        
        result = await self._execute_task(review_task)
        await self._handle_task_result(review_task, result)


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowExecutor] = {}
        self.completed_workflows: Dict[str, WorkflowExecutor] = {}
        self.decomposer = TaskDecomposer()
        self.consensus_manager = create_consensus_manager()
        self.communication: Optional[CommunicationManager] = None
    
    async def initialize(self) -> None:
        """初始化引擎"""
        self.communication = get_communication_manager()
        logger.info("Workflow Engine initialized")
    
    async def create_workflow(
        self,
        name: str,
        description: str = "",
        mode: CollaborationMode = CollaborationMode.SEQUENTIAL,
        tasks: Optional[List[TaskInput]] = None,
        agent_ids: Optional[List[str]] = None
    ) -> WorkflowDefinition:
        """创建工作流"""
        workflow = WorkflowDefinition(
            name=name,
            description=description,
            mode=mode,
            agents=[
                AgentInfo(agent_id=aid, name=f"Agent_{aid}")
                for aid in (agent_ids or [])
            ]
        )
        
        if tasks:
            workflow.tasks = [
                TaskDefinition(
                    task_id=t.task_id,
                    name=t.name,
                    description=t.description,
                    source="",
                    target=t.assigned_agent or "",
                    condition="",
                    payload_template=t.payload
                )
                for t in tasks
            ]
        
        return workflow
    
    async def execute(
        self,
        workflow: WorkflowDefinition,
        tasks: List[TaskInput],
        agent_map: Dict[str, AgentInfo]
    ) -> WorkflowExecutor:
        """执行工作流"""
        # 选择执行器
        if workflow.mode == CollaborationMode.SEQUENTIAL:
            executor = SequentialExecutor()
        elif workflow.mode == CollaborationMode.PARALLEL:
            executor = ParallelExecutor()
        elif workflow.mode == CollaborationMode.HIERARCHICAL:
            executor = HierarchicalExecutor()
        else:
            executor = WorkflowExecutor()
        
        # 初始化并启动
        await executor.initialize(workflow, tasks, agent_map)
        await executor.start()
        
        workflow_id = workflow.workflow_id
        self.active_workflows[workflow_id] = executor
        
        return executor
    
    async def get_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        executor = self.active_workflows.get(workflow_id)
        if not executor:
            executor = self.completed_workflows.get(workflow_id)
        
        if not executor:
            return None
        
        progress = executor.get_progress()
        
        return {
            "workflow_id": workflow_id,
            "state": executor.state.value,
            "progress": progress.dict(),
            "completed_tasks": list(executor.completed_tasks),
            "failed_tasks": list(executor.failed_tasks),
            "execution_history": executor.execution_history
        }


# 引擎工厂
_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """获取工作流引擎"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine


async def init_workflow_engine() -> WorkflowEngine:
    """初始化工作流引擎"""
    engine = get_workflow_engine()
    await engine.initialize()
    return engine
