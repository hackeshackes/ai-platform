"""
Task Scheduler - AI Platform v6

训练任务调度器，支持任务队列、优先级调度和资源分配。
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
import uuid

from .trainer import TrainingTask, TrainingConfig, TaskStatus, get_distributed_trainer

logger = logging.getLogger(__name__)


class Priority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class SchedulerState(Enum):
    """调度器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class QueueConfig:
    """队列配置"""
    max_size: int = 100  # 最大队列长度
    max_concurrent_tasks: int = 10  # 最大并发任务数
    default_priority: Priority = Priority.NORMAL
    task_timeout: int = 3600  # 任务超时时间（秒）
    auto_retry: bool = True
    max_retries: int = 3


@dataclass
class ScheduledTask:
    """调度任务"""
    task_id: str
    training_task: TrainingTask
    priority: Priority
    state: TaskStatus = TaskStatus.PENDING
    submitted_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    retries: int = 0
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class TaskScheduler:
    """
    任务调度器
    
    管理训练任务的排队、调度和执行。
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        """
        初始化任务调度器
        
        Args:
            config: 队列配置
        """
        self.config = config or QueueConfig()
        self._queues: Dict[Priority, deque] = {
            Priority.LOW: deque(),
            Priority.NORMAL: deque(),
            Priority.HIGH: deque(),
            Priority.URGENT: deque()
        }
        self._running_tasks: Dict[str, ScheduledTask] = {}
        self._completed_tasks: Dict[str, ScheduledTask] = {}
        self._task_map: Dict[str, ScheduledTask] = {}  # task_id -> ScheduledTask
        self._state = SchedulerState.IDLE
        self._scheduler_task: Optional[asyncio.Task] = None
        self._trainer = get_distributed_trainer()
        
    async def start(self):
        """启动调度器"""
        if self._state != SchedulerState.IDLE:
            return
        
        self._state = SchedulerState.RUNNING
        self._scheduler_task = asyncio.create_task(self._schedule_loop())
        logger.info("任务调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        self._state = SchedulerState.STOPPED
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("任务调度器已停止")
    
    async def pause(self):
        """暂停调度器"""
        if self._state == SchedulerState.RUNNING:
            self._state = SchedulerState.PAUSED
            logger.info("任务调度器已暂停")
    
    async def resume(self):
        """恢复调度器"""
        if self._state == SchedulerState.PAUSED:
            self._state = SchedulerState.RUNNING
            logger.info("任务调度器已恢复")
    
    async def submit_task(self,
                         name: str,
                         config: TrainingConfig,
                         priority: Priority = Priority.NORMAL,
                         dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        提交训练任务
        
        Args:
            name: 任务名称
            config: 训练配置
            priority: 优先级
            dependencies: 依赖任务ID列表
            
        Returns:
            提交结果
        """
        # 检查队列是否已满
        total_queued = sum(len(q) for q in self._queues.values())
        if total_queued >= self.config.max_size:
            return {
                "success": False,
                "error": "任务队列已满"
            }
        
        # 检查依赖
        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self._task_map:
                    return {
                        "success": False,
                        "error": f"依赖任务不存在: {dep_id}"
                    }
        
        # 创建训练任务
        training_task = await self._trainer.create_task(name, config)
        task_id = training_task.task_id
        
        # 创建调度任务
        scheduled_task = ScheduledTask(
            task_id=task_id,
            training_task=training_task,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self._queues[priority].append(scheduled_task)
        self._task_map[task_id] = scheduled_task
        
        logger.info(f"任务已提交: {task_id} - {name} (优先级: {priority.name})")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "任务已加入调度队列",
            "queue_position": len(self._queues[priority]),
            "priority": priority.name
        }
    
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            取消结果
        """
        # 检查是否在队列中
        for priority, queue in self._queues.items():
            for i, task in enumerate(queue):
                if task.task_id == task_id:
                    queue.remove(task)
                    task.state = TaskStatus.CANCELLED
                    await self._trainer.cancel_task(task_id)
                    logger.info(f"任务已取消: {task_id}")
                    return {
                        "success": True,
                        "message": "任务已取消"
                    }
        
        # 检查是否在运行中
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            task.state = TaskStatus.CANCELLED
            await self._trainer.cancel_task(task_id)
            del self._running_tasks[task_id]
            logger.info(f"运行中的任务已取消: {task_id}")
            return {
                "success": True,
                "message": "任务已取消"
            }
        
        return {
            "success": False,
            "error": f"任务不存在: {task_id}"
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        # 检查是否在队列中
        for priority, queue in self._queues.items():
            for task in queue:
                if task.task_id == task_id:
                    queue_position = list(queue).index(task)
                    return {
                        "task_id": task_id,
                        "state": task.state.value,
                        "status": "queued",
                        "priority": priority.name,
                        "queue_position": queue_position,
                        "submitted_at": task.submitted_at.isoformat()
                    }
        
        # 检查是否在运行中
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            return {
                "task_id": task_id,
                "state": task.state.value,
                "status": "running",
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "retries": task.retries,
                "error": task.error
            }
        
        # 检查是否已完成
        if task_id in self._completed_tasks:
            task = self._completed_tasks[task_id]
            return {
                "task_id": task_id,
                "state": task.state.value,
                "status": "completed",
                "completed_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
                "retries": task.retries,
                "error": task.error
            }
        
        return None
    
    async def list_tasks(self, 
                        status_filter: Optional[List[str]] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出所有任务
        
        Args:
            status_filter: 状态过滤器
            limit: 返回数量限制
            
        Returns:
            任务列表
        """
        tasks = []
        
        # 从队列中获取
        for priority, queue in self._queues.items():
            for task in queue:
                if not status_filter or task.state.value in status_filter:
                    tasks.append({
                        "task_id": task.task_id,
                        "name": task.training_task.name,
                        "status": task.state.value,
                        "queue_status": "queued",
                        "priority": priority.name,
                        "submitted_at": task.submitted_at.isoformat()
                    })
        
        # 从运行中获取
        for task_id, task in self._running_tasks.items():
            if not status_filter or task.state.value in status_filter:
                tasks.append({
                    "task_id": task_id,
                    "name": task.training_task.name,
                    "status": task.state.value,
                    "queue_status": "running",
                    "priority": task.priority.name,
                    "started_at": task.started_at.isoformat() if task.started_at else None
                })
        
        # 从已完成获取
        for task_id, task in self._completed_tasks.items():
            if not status_filter or task.state.value in status_filter:
                tasks.append({
                    "task_id": task_id,
                    "name": task.training_task.name,
                    "status": task.state.value,
                    "queue_status": "completed",
                    "priority": task.priority.name,
                    "completed_at": task.scheduled_at.isoformat() if task.scheduled_at else None
                })
        
        return tasks[:limit]
    
    async def _schedule_loop(self):
        """调度循环"""
        while self._state == SchedulerState.RUNNING:
            try:
                # 按优先级从高到低检查队列
                for priority in [Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                    queue = self._queues[priority]
                    
                    # 检查并发任务数
                    if len(self._running_tasks) >= self.config.max_concurrent_tasks:
                        break
                    
                    # 获取队首任务
                    if not queue:
                        continue
                    
                    task = queue.popleft()
                    
                    # 检查依赖
                    deps_met = True
                    for dep_id in task.dependencies:
                        if dep_id in self._running_tasks:
                            deps_met = False
                            break
                        if dep_id in self._task_map and self._task_map[dep_id].state != TaskStatus.COMPLETED:
                            deps_met = False
                            break
                    
                    if not deps_met:
                        # 将任务放回队列末尾
                        queue.append(task)
                        continue
                    
                    # 启动任务
                    task.scheduled_at = datetime.now()
                    task.started_at = datetime.now()
                    task.state = TaskStatus.RUNNING
                    self._running_tasks[task.task_id] = task
                    
                    # 提交到训练器
                    asyncio.create_task(self._execute_task(task))
                    
                    logger.info(f"任务已调度: {task.task_id} - {task.training_task.name}")
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, scheduled_task: ScheduledTask):
        """
        执行任务
        
        Args:
            scheduled_task: 调度任务
        """
        try:
            task_id = scheduled_task.task_id
            
            # 提交到训练器
            result = await self._trainer.submit_task(task_id)
            
            if not result["success"]:
                scheduled_task.error = result.get("error", "提交失败")
                
                # 重试逻辑
                if self.config.auto_retry and scheduled_task.retries < self.config.max_retries:
                    scheduled_task.retries += 1
                    scheduled_task.state = TaskStatus.PENDING
                    self._queues[scheduled_task.priority].append(scheduled_task)
                else:
                    scheduled_task.state = TaskStatus.FAILED
                    self._completed_tasks[task_id] = scheduled_task
                    del self._running_tasks[task_id]
                
                return
            
            # 等待任务完成
            while True:
                status = await self._trainer.get_task_status(task_id)
                if status is None or status.status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
                    break
                await asyncio.sleep(5)
            
            # 更新状态
            if status and status.status == TaskStatus.COMPLETED.value:
                scheduled_task.state = TaskStatus.COMPLETED
            elif status and status.status == TaskStatus.FAILED.value:
                scheduled_task.state = TaskStatus.FAILED
                scheduled_task.error = status.get("error")
            else:
                scheduled_task.state = TaskStatus.FAILED
                scheduled_task.error = "未知错误"
            
            scheduled_task.scheduled_at = datetime.now()
            self._completed_tasks[task_id] = scheduled_task
            del self._running_tasks[task_id]
            
            logger.info(f"任务执行完成: {task_id} - 状态: {scheduled_task.state.value}")
            
        except Exception as e:
            scheduled_task.error = str(e)
            scheduled_task.state = TaskStatus.FAILED
            self._completed_tasks[scheduled_task.task_id] = scheduled_task
            del self._running_tasks[scheduled_task.task_id]
            logger.error(f"执行任务失败: {scheduled_task.task_id} - {e}")
    
    def get_scheduler_state(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            调度器状态
        """
        return {
            "state": self._state.value,
            "queued_tasks": {
                "urgent": len(self._queues[Priority.URGENT]),
                "high": len(self._queues[Priority.HIGH]),
                "normal": len(self._queues[Priority.NORMAL]),
                "low": len(self._queues[Priority.LOW]),
                "total": sum(len(q) for q in self._queues.values())
            },
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "max_concurrent": self.config.max_concurrent_tasks
        }


# 单例任务调度器
_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler(config: Optional[QueueConfig] = None) -> TaskScheduler:
    """获取任务调度器单例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler(config)
    return _scheduler