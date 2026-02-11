"""
定时调度器 (Cron Scheduler)
=============================

功能:
- Cron表达式解析
- 时区支持
- 任务队列管理
- 过期任务清理
- 分布式锁支持

作者: AI Platform Team
版本: 1.0.0
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from croniter import croniter
import pytz
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SchedulerStatus(Enum):
    """调度器状态枚举"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class JobStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
    """定时任务定义"""
    id: str
    name: str
    func: Callable[..., Any]
    cron_expression: str
    timezone: str = "UTC"
    enabled: bool = True
    retry_count: int = 3
    retry_interval: int = 60
    timeout: int = 3600
    max_instances: int = 1
    last_run_time: Optional[datetime] = None
    next_run_time: Optional[datetime] = None
    last_result: Any = None
    last_error: Optional[str] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def calculate_next_run(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """计算下次执行时间"""
        try:
            tz = pytz.timezone(self.timezone)
            now = from_time or datetime.now(tz)
            
            cron = croniter(self.cron_expression, now)
            next_time = cron.get_next(datetime)
            self.next_run_time = tz.localize(next_time)
            
            return self.next_run_time
            
        except Exception as e:
            logger.error(f"Failed to calculate next run time for job {self.name}: {e}")
            return None
    
    def is_due(self) -> bool:
        """检查是否到期执行"""
        if not self.next_run_time:
            return False
        
        now = datetime.now(pytz.timezone(self.timezone))
        return now >= self.next_run_time


@dataclass
class JobExecution:
    """任务执行记录"""
    job_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    output: Any = None
    error: Optional[str] = None
    retry_count: int = 0


class CronScheduler:
    """
    Cron定时调度器
    
    支持:
    - 精确的Cron表达式解析
    - 多时区支持
    - 任务队列管理
    - 自动重试
    - 过期清理
    - 执行历史记录
    """
    
    def __init__(
        self,
        timezone: str = "UTC",
        check_interval: int = 10,
        max_history: int = 1000,
        enable_metrics: bool = True,
        cleanup_interval: int = 3600
    ):
        self.timezone = timezone
        self.check_interval = check_interval
        self.max_history = max_history
        self.enable_metrics = enable_metrics
        self.cleanup_interval = cleanup_interval
        
        self._jobs: Dict[str, ScheduledJob] = {}
        self._job_lock = asyncio.Lock()
        
        self._running_executions: Dict[str, JobExecution] = {}
        self._execution_history: List[JobExecution] = []
        
        self._status = SchedulerStatus.STOPPED
        self._scheduler_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        self._metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "avg_execution_time": 0.0
        }
    
    @property
    def status(self) -> SchedulerStatus:
        return self._status
    
    def add_job(
        self,
        func: Callable[..., Any],
        name: str,
        cron_expression: str,
        timezone: Optional[str] = None,
        retry_count: int = 3,
        retry_interval: int = 60,
        timeout: int = 3600,
        max_instances: int = 1,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加定时任务"""
        job = ScheduledJob(
            id=str(uuid.uuid4()),
            name=name,
            func=func,
            cron_expression=cron_expression,
            timezone=timezone or self.timezone,
            retry_count=retry_count,
            retry_interval=retry_interval,
            timeout=timeout,
            max_instances=max_instances,
            enabled=enabled,
            metadata=metadata or {}
        )
        
        job.calculate_next_run()
        
        self._jobs[job.id] = job
        logger.info(f"Added scheduled job: {name} (ID: {job.id}, Next: {job.next_run_time})")
        
        return job.id
    
    def remove_job(self, job_id: str) -> bool:
        """删除定时任务"""
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Removed scheduled job: {job_id}")
            return True
        return False
    
    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """获取任务详情"""
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """获取所有任务信息"""
        return [
            {
                "id": job.id,
                "name": job.name,
                "cron_expression": job.cron_expression,
                "timezone": job.timezone,
                "enabled": job.enabled,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "last_run_time": job.last_run_time.isoformat() if job.last_run_time else None,
                "run_count": job.run_count,
                "status": self._get_job_status(job)
            }
            for job in self._jobs.values()
        ]
    
    def _get_job_status(self, job: ScheduledJob) -> str:
        """获取任务当前状态"""
        if not job.enabled:
            return "disabled"
        
        for exec_id, exec in self._running_executions.items():
            if exec.job_id == job.id:
                return "running"
        
        if job.is_due():
            return "due"
        
        return "scheduled"
    
    async def start(self) -> None:
        """启动调度器"""
        async with self._job_lock:
            if self._status == SchedulerStatus.RUNNING:
                logger.warning("Scheduler is already running")
                return
            
            self._status = SchedulerStatus.RUNNING
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"CronScheduler started with {len(self._jobs)} jobs")
    
    async def stop(self) -> None:
        """停止调度器"""
        async with self._job_lock:
            self._status = SchedulerStatus.STOPPED
            
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await self._scheduler_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            for exec_id in list(self._running_executions.keys()):
                self._running_executions[exec_id].status = JobStatus.CANCELLED
            
            logger.info("CronScheduler stopped")
    
    async def pause(self) -> None:
        """暂停调度器"""
        async with self._job_lock:
            if self._status == SchedulerStatus.RUNNING:
                self._status = SchedulerStatus.PAUSED
                logger.info("Scheduler paused")
    
    async def resume(self) -> None:
        """恢复调度器"""
        async with self._job_lock:
            if self._status == SchedulerStatus.PAUSED:
                self._status = SchedulerStatus.RUNNING
                logger.info("Scheduler resumed")
    
    async def _scheduler_loop(self) -> None:
        """调度主循环"""
        while self._status == SchedulerStatus.RUNNING:
            try:
                due_jobs = self._get_due_jobs()
                
                for job in due_jobs:
                    if self._should_run(job):
                        asyncio.create_task(self._run_job(job))
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _cleanup_loop(self) -> None:
        """清理过期任务"""
        while self._status == SchedulerStatus.RUNNING:
            try:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup_history()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cleanup error: {e}")
    
    def _get_due_jobs(self) -> List[ScheduledJob]:
        """获取到期的任务"""
        due = []
        for job in self._jobs.values():
            if job.enabled and job.is_due():
                due.append(job)
        return due
    
    def _should_run(self, job: ScheduledJob) -> bool:
        """检查是否应该运行任务"""
        running_count = sum(
            1 for exec in self._running_executions.values()
            if exec.job_id == job.id and exec.status == JobStatus.RUNNING
        )
        
        return running_count < job.max_instances
    
    async def _run_job(self, job: ScheduledJob) -> None:
        """执行任务"""
        execution_id = str(uuid.uuid4())
        
        execution = JobExecution(
            job_id=job.id,
            execution_id=execution_id,
            start_time=datetime.now(),
            status=JobStatus.RUNNING
        )
        
        self._running_executions[execution_id] = execution
        
        logger.info(f"Starting job: {job.name} (Execution: {execution_id})")
        
        try:
            for attempt in range(job.retry_count + 1):
                try:
                    if asyncio.iscoroutinefunction(job.func):
                        output = await asyncio.wait_for(
                            job.func(),
                            timeout=job.timeout
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        output = await asyncio.wait_for(
                            loop.run_in_executor(self._executor, job.func),
                            timeout=job.timeout
                        )
                    
                    execution.status = JobStatus.COMPLETED
                    execution.output = output
                    execution.retry_count = attempt
                    
                    job.last_result = output
                    job.last_error = None
                    job.run_count += 1
                    
                    self._metrics["total_runs"] += 1
                    self._metrics["successful_runs"] += 1
                    
                    break
                    
                except Exception as e:
                    execution.error = str(e)
                    if attempt < job.retry_count:
                        await asyncio.sleep(job.retry_interval)
                    else:
                        execution.status = JobStatus.FAILED
                        job.last_error = str(e)
                        
                        self._metrics["total_runs"] += 1
                        self._metrics["failed_runs"] += 1
                        
                        logger.error(f"Job {job.name} failed: {e}")
        
        except asyncio.TimeoutError:
            execution.status = JobStatus.FAILED
            execution.error = f"Job timeout after {job.timeout}s"
            job.last_error = execution.error
            
        except Exception as e:
            execution.status = JobStatus.FAILED
            execution.error = str(e)
            job.last_error = str(e)
            logger.exception(f"Job {job.name} execution error")
        
        finally:
            execution.end_time = datetime.now()
            
            if execution.start_time:
                duration = (execution.end_time - execution.start_time).total_seconds()
                execution.output = {
                    "duration": duration,
                    "output": execution.output
                }
            
            job.calculate_next_run()
            job.last_run_time = execution.start_time
            
            del self._running_executions[execution_id]
            self._execution_history.append(execution)
            
            if len(self._execution_history) > self.max_history:
                self._execution_history = self._execution_history[-self.max_history:]
            
            self._update_metrics(execution)
    
    def _update_metrics(self, execution: JobExecution) -> None:
        """更新指标"""
        if execution.end_time and execution.start_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
            
            total = self._metrics["total_runs"]
            if total > 0:
                self._metrics["avg_execution_time"] = (
                    (self._metrics["avg_execution_time"] * (total - 1) + duration) / total
                )
    
    def _cleanup_history(self) -> None:
        """清理过期历史记录"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        self._execution_history = [
            e for e in self._execution_history
            if e.start_time > cutoff_time
        ]
        
        logger.info(f"Cleaned up history, {len(self._execution_history)} records remaining")
    
    def run_now(self, job_id: str) -> bool:
        """立即运行任务"""
        job = self._jobs.get(job_id)
        if job:
            asyncio.create_task(self._run_job(job))
            return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取调度器指标"""
        return {
            **self._metrics,
            "active_jobs": len(self._jobs),
            "running_executions": len(self._running_executions),
            "history_size": len(self._execution_history)
        }
    
    def get_execution_history(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取执行历史"""
        if job_id:
            histories = [
                e for e in self._execution_history
                if e.job_id == job_id
            ]
        else:
            histories = self._execution_history
        
        return [
            {
                "execution_id": e.execution_id,
                "job_id": e.job_id,
                "start_time": e.start_time.isoformat(),
                "end_time": e.end_time.isoformat() if e.end_time else None,
                "status": e.status.value,
                "output": e.output,
                "error": e.error,
                "retry_count": e.retry_count
            }
            for e in histories
        ]
    
    def parse_cron(self, expression: str) -> Dict[str, Any]:
        """解析Cron表达式"""
        parts = expression.split()
        
        if len(parts) != 5:
            raise ValueError("Invalid cron expression (expected 5 parts)")
        
        minute, hour, day, month, weekday = parts
        
        return {
            "minute": minute,
            "hour": hour,
            "day": day,
            "month": month,
            "weekday": weekday,
            "human_readable": f"分钟:{minute} 小时:{hour} 日期:{day} 月份:{month} 星期:{weekday}"
        }
    
    def get_next_runs(self, job_id: str, count: int = 5) -> List[datetime]:
        """获取任务下次执行时间列表"""
        job = self._jobs.get(job_id)
        if not job:
            return []
        
        tz = pytz.timezone(job.timezone)
        now = datetime.now(tz)
        
        cron = croniter(job.cron_expression, now)
        return [cron.get_next(datetime) for _ in range(count)]


class CronExpressions:
    """常用Cron表达式"""
    
    EVERY_MINUTE = "* * * * *"
    EVERY_5_MINUTES = "*/5 * * * *"
    EVERY_10_MINUTES = "*/10 * * * *"
    EVERY_30_MINUTES = "*/30 * * * *"
    EVERY_HOUR = "0 * * * *"
    EVERY_2_HOURS = "0 */2 * * *"
    EVERY_6_HOURS = "0 */6 * * *"
    EVERY_12_HOURS = "0 */12 * * *"
    DAILY_MIDNIGHT = "0 0 * * *"
    DAILY_2AM = "0 2 * * *"
    DAILY_3AM = "0 3 * * *"
    WEEKLY_MONDAY = "0 0 * * 1"
    WEEKLY_SUNDAY = "0 0 * * 0"
    MONTHLY_FIRST_DAY = "0 0 1 * *"
    QUARTERLY_FIRST_DAY = "0 0 1 */3 *"
