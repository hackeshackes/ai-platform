"""
流水线引擎 (Pipeline Engine)
=============================

功能:
- 任务编排与依赖管理
- 并行执行支持
- 失败自动重试
- 实时状态监控
- 结果收集与汇总

作者: AI Platform Team
版本: 1.0.0
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class PipelineStatus(Enum):
    """流水线状态枚举"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    task_name: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    output: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "output": self.output,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


@dataclass
class Task:
    """
    任务定义
    
    Attributes:
        name: 任务名称
        func: 可执行函数
        dependencies: 依赖任务列表（任务名称）
        retry_count: 重试次数
        timeout: 超时时间（秒）
        parallel: 是否并行执行
        skip_if: 条件跳过函数
        fallback: 失败时回退函数
    """
    name: str
    func: Callable[..., Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300
    parallel: bool = False
    skip_if: Optional[Callable[..., bool]] = None
    fallback: Optional[Callable[..., Any]] = None
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())
        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None


@dataclass
class PipelineResult:
    """流水线执行结果"""
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    task_results: List[TaskResult] = field(default_factory=list)
    overall_output: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "task_count": len(self.task_results),
            "success_count": sum(1 for r in self.task_results if r.status == TaskStatus.SUCCESS),
            "failed_count": sum(1 for r in self.task_results if r.status == TaskStatus.FAILED),
            "task_results": [r.to_dict() for r in self.task_results],
            "overall_output": self.overall_output,
            "error": self.error
        }


class PipelineEngine:
    """
    流水线引擎
    
    支持:
    - 任务依赖图管理
    - 并行/串行执行
    - 自动重试机制
    - 实时状态回调
    - 结果汇总分析
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        default_timeout: int = 300,
        default_retries: int = 3,
        progress_callback: Optional[Callable[[str, TaskStatus, float], None]] = None
    ):
        """
        初始化流水线引擎
        
        Args:
            max_concurrent_tasks: 最大并发任务数
            default_timeout: 默认超时时间
            default_retries: 默认重试次数
            progress_callback: 进度回调函数 (task_name, status, progress)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.default_retries = default_retries
        self.progress_callback = progress_callback
        
        self._tasks: Dict[str, Task] = {}
        self._task_graph: Dict[str, Set[str]] = {}  # task_id -> dependencies
        self._reverse_graph: Dict[str, Set[str]] = {}  # task_id -> dependent_tasks
        
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        self._running_tasks: Set[str] = set()
        self._completed_tasks: Set[str] = set()
        self._failed_tasks: Set[str] = set()
        
        self._execution_history: List[PipelineResult] = []
    
    def add_task(self, task: Task) -> str:
        """
        添加任务到流水线
        
        Args:
            task: 任务对象
            
        Returns:
            任务ID
        """
        self._tasks[task.id] = task
        self._task_graph[task.id] = set(task.dependencies)
        
        # 更新反向图
        for dep_name in task.dependencies:
            for tid, t in self._tasks.items():
                if t.name == dep_name:
                    self._reverse_graph.setdefault(tid, set()).add(task.id)
                    break
        
        return task.id
    
    def add_tasks(self, tasks: List[Task]) -> List[str]:
        """批量添加任务"""
        return [self.add_task(t) for t in tasks]
    
    def set_dependencies(self, task_name: str, dependencies: List[str]) -> None:
        """设置任务的依赖关系"""
        for task in self._tasks.values():
            if task.name == task_name:
                task.dependencies = dependencies
                self._task_graph[task.id] = set(dependencies)
                break
    
    def _get_ready_tasks(self) -> List[Task]:
        """获取可以执行的任务（所有依赖都已完成）"""
        ready = []
        for task_id, task in self._tasks.items():
            if task_id in self._completed_tasks or task_id in self._failed_tasks:
                continue
            
            deps = self._task_graph.get(task_id, set())
            if deps.issubset(self._completed_tasks):
                ready.append(task)
        return ready
    
    def _get_completed_count(self) -> float:
        """获取完成进度 (0.0 - 1.0)"""
        total = len(self._tasks)
        if total == 0:
            return 1.0
        return len(self._completed_tasks) / total
    
    def _notify_progress(self, task: Task, status: TaskStatus) -> None:
        """通知进度"""
        if self.progress_callback:
            progress = self._get_completed_count()
            self.progress_callback(task.name, status, progress)
    
    async def _execute_task(self, task: Task) -> TaskResult:
        """
        执行单个任务
        
        Args:
            task: 任务对象
            
        Returns:
            任务执行结果
        """
        result = TaskResult(
            task_id=task.id,
            task_name=task.name,
            status=TaskStatus.RUNNING,
            start_time=datetime.now(),
            retry_count=0,
            max_retries=task.retry_count
        )
        
        self._running_tasks.add(task.id)
        task.status = TaskStatus.RUNNING
        
        try:
            # 检查跳过条件
            if task.skip_if:
                skip_result = await task.skip_if()
                if skip_result:
                    result.status = TaskStatus.SKIPPED
                    result.end_time = datetime.now()
                    result.duration = (result.end_time - result.start_time).total_seconds()
                    self._completed_tasks.add(task.id)
                    self._running_tasks.discard(task.id)
                    task.status = TaskStatus.SKIPPED
                    self._notify_progress(task, TaskStatus.SKIPPED)
                    return result
            
            # 执行任务
            for attempt in range(task.retry_count + 1):
                try:
                    if asyncio.iscoroutinefunction(task.func):
                        output = await asyncio.wait_for(
                            task.func(),
                            timeout=task.timeout
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        output = await asyncio.wait_for(
                            loop.run_in_executor(
                                self._executor,
                                lambda: task.func()
                            ),
                            timeout=task.timeout
                        )
                    
                    result.status = TaskStatus.SUCCESS
                    result.output = output
                    result.retry_count = attempt
                    break
                    
                except Exception as e:
                    result.error = str(e)
                    if attempt < task.retry_count:
                        result.status = TaskStatus.RETRYING
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                    else:
                        result.status = TaskStatus.FAILED
                        logger.error(f"Task {task.name} failed after {attempt + 1} attempts: {e}")
                        
                        # 执行回退函数
                        if task.fallback:
                            try:
                                result.output = await task.fallback()
                                result.status = TaskStatus.SUCCESS
                                logger.info(f"Fallback executed for task {task.name}")
                            except Exception as fallback_e:
                                logger.error(f"Fallback failed for task {task.name}: {fallback_e}")
            
        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error = f"Task timeout after {task.timeout}s"
            logger.error(f"Task {task.name} timed out")
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            logger.exception(f"Task {task.name} execution error")
        
        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration = (result.end_time - result.start_time).total_seconds()
            
            self._running_tasks.discard(task.id)
            
            if result.status == TaskStatus.SUCCESS:
                self._completed_tasks.add(task.id)
                task.status = TaskStatus.SUCCESS
            elif result.status == TaskStatus.FAILED:
                self._failed_tasks.add(task.id)
                task.status = TaskStatus.FAILED
            else:
                self._completed_tasks.add(task.id)
            
            task.result = result
            self._notify_progress(task, result.status)
        
        return result
    
    async def run(
        self,
        tasks: List[Task],
        pipeline_name: str = "DefaultPipeline",
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        执行流水线
        
        Args:
            tasks: 任务列表
            pipeline_name: 流水线名称
            context: 上下文数据（可在任务间共享）
            
        Returns:
            流水线执行结果
        """
        # 初始化
        self._tasks.clear()
        self._task_graph.clear()
        self._reverse_graph.clear()
        self._running_tasks.clear()
        self._completed_tasks.clear()
        self._failed_tasks.clear()
        
        # 添加任务
        self.add_tasks(tasks)
        
        start_time = datetime.now()
        pipeline_id = str(uuid.uuid4())
        
        logger.info(f"Starting pipeline '{pipeline_name}' with {len(tasks)} tasks")
        
        # 初始化状态
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
        
        task_results: List[TaskResult] = []
        
        try:
            # 使用拓扑排序确定执行顺序
            execution_order = self._topological_sort()
            
            # 统计并行任务组
            parallel_groups = self._group_parallel_tasks(execution_order)
            
            for group in parallel_groups:
                if not group:  # 跳过空组
                    continue
                
                # 检查组内任务是否可并行
                parallelable_tasks = []
                for task in group:
                    deps = self._task_graph.get(task.id, set())
                    if deps.issubset(self._completed_tasks):
                        if task.parallel:
                            parallelable_tasks.append(task)
                        else:
                            # 串行任务单独执行
                            result = await self._execute_task(task)
                            task_results.append(result)
                
                # 并行执行
                if parallelable_tasks:
                    await asyncio.gather(*[self._execute_task(t) for t in parallelable_tasks])
                
                # 收集结果
                for task in group:
                    if task.result:
                        task_results.append(task.result)
            
            # 计算状态
            failed_count = sum(1 for r in task_results if r.status == TaskStatus.FAILED)
            overall_status = PipelineStatus.COMPLETED if failed_count == 0 else PipelineStatus.FAILED
            
            # 收集最终输出
            overall_output = context or {}
            for r in task_results:
                if r.status == TaskStatus.SUCCESS:
                    overall_output[r.task_name] = r.output
            
        except Exception as e:
            logger.exception(f"Pipeline execution error: {e}")
            overall_status = PipelineStatus.FAILED
            task_results = []
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = PipelineResult(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            status=overall_status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            task_results=task_results,
            overall_output=overall_output,
            error=None if overall_status == PipelineStatus.COMPLETED else "Pipeline failed"
        )
        
        self._execution_history.append(result)
        logger.info(f"Pipeline '{pipeline_name}' completed with status: {overall_status.value}")
        
        return result
    
    def _topological_sort(self) -> List[Task]:
        """拓扑排序确定执行顺序"""
        visited = set()
        order = []
        
        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)
            
            for dep_id in self._task_graph.get(task_id, set()):
                visit(dep_id)
            
            if task_id in self._tasks:
                order.append(self._tasks[task_id])
        
        for task_id in self._tasks:
            visit(task_id)
        
        return order
    
    def _group_parallel_tasks(self, ordered_tasks: List[Task]) -> List[List[Task]]:
        """分组可并行的任务"""
        groups = []
        current_group = []
        completed = set()
        
        for task in ordered_tasks:
            deps = self._task_graph.get(task.id, set())
            if deps.issubset(completed):
                if task.parallel and task.dependencies:
                    # 依赖任务已完成，可以并行
                    current_group.append(task)
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = [task]
                    completed.add(task.id)
            else:
                current_group.append(task)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "total_tasks": len(self._tasks),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len(self._failed_tasks),
            "progress": self._get_completed_count()
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return [r.to_dict() for r in self._execution_history]
    
    async def cancel(self) -> None:
        """取消当前执行"""
        for task_id in list(self._running_tasks):
            self._running_tasks.discard(task_id)
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.CANCELLED
        
        logger.info("Pipeline execution cancelled")


# 便捷函数
def create_pipeline(
    name: str,
    task_configs: List[Dict[str, Any]],
    **kwargs
) -> PipelineEngine:
    """
    创建流水线引擎并添加任务
    
    Args:
        name: 流水线名称
        task_configs: 任务配置列表
        **kwargs: PipelineEngine其他参数
        
    Returns:
        配置好的流水线引擎实例
    """
    engine = PipelineEngine(**kwargs)
    
    for config in task_configs:
        task = Task(
            name=config["name"],
            func=config["func"],
            dependencies=config.get("dependencies", []),
            retry_count=config.get("retry_count", 3),
            timeout=config.get("timeout", 300),
            parallel=config.get("parallel", False),
            skip_if=config.get("skip_if"),
            fallback=config.get("fallback")
        )
        engine.add_task(task)
    
    return engine
