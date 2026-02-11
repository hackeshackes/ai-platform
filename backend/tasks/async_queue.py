"""
异步任务队列 (轻量级实现)
"""
import asyncio
import uuid
import time
import threading
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """任务"""
    id: str
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

class AsyncQueue:
    """异步任务队列"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: Dict[str, Task] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._lock = threading.Lock()
    
    async def start(self):
        """启动任务处理器"""
        self._running = True
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_task(task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Task error: {e}")
    
    async def _process_task(self, task: Task):
        """处理任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: task.func(*task.args, **task.kwargs)
            )
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        
        task.completed_at = time.time()
    
    def submit(self, name: str, func: Callable, *args, **kwargs) -> Task:
        """提交任务"""
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        asyncio.create_task(self._queue.put(task))
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def list_tasks(self, status: TaskStatus = None) -> list:
        """列出任务"""
        with self._lock:
            tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def stop(self):
        """停止队列"""
        self._running = False
        self._executor.shutdown(wait=False)

# 全局队列实例
async_queue = AsyncQueue(max_workers=4)

# 启动队列
async def start_queue():
    await async_queue.start()

def background_task(func: Callable) -> Callable:
    """后台任务装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return async_queue.submit(func.__name__, func, *args, **kwargs)
    return wrapper

# 便捷函数
def submit_task(name: str, func: Callable, *args, **kwargs) -> Task:
    return async_queue.submit(name, func, *args, **kwargs)

def get_task_status(task_id: str) -> Optional[Task]:
    return async_queue.get_task(task_id)
