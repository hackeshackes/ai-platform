"""
资源调度器 - Phase 2
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from uuid import uuid4
import asyncio

class Priority(Enum):
    """任务优先级"""
    CRITICAL = 0  # 关键任务
    HIGH = 1       # 高优先级
    NORMAL = 2     # 普通
    LOW = 3        # 低优先级

@dataclass
class ResourceRequest:
    """资源请求"""
    gpu_count: int
    gpu_memory_gb: int
    cpu_count: int
    memory_gb: int
    duration_hours: float
    priority: Priority

@dataclass
class SchedulingResult:
    """调度结果"""
    scheduled: bool
    node_id: Optional[str] = None
    gpus: Optional[List[int]] = None
    wait_time_hours: Optional[float] = None
    reason: Optional[str] = None

class ResourceScheduler:
    """资源调度器"""
    
    def __init__(self):
        self.priority_queues: Dict[Priority, List] = {
            Priority.CRITICAL: [],
            Priority.HIGH: [],
            Priority.NORMAL: [],
            Priority.LOW: []
        }
    
    def request_resources(self, request: ResourceRequest) -> str:
        """
        申请资源
        
        返回任务ID
        """
        task_id = str(uuid4())
        
        # 加入对应优先级队列
        self.priority_queues[request.priority].append({
            "task_id": task_id,
            "request": request,
            "created_at": datetime.utcnow()
        })
        
        return task_id
    
    async def allocate,
        available_g_resources(
        selfpus: Dict[str, List[int]],
        gpu_memory: Dict[str, Dict[int, int]],
        total_gpu_memory: Dict[str, int]
    ) -> Tuple[bool, Optional[str], Optional[List[int]]]:
        """
        分配资源
        
        策略:
        1. 按优先级从高到低调度
        2. 优先分配给负载最低的节点
        3. GPU亲和性优化
        """
        # 按优先级处理
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            queue = self.priority_queues[priority]
            
            for task in queue:
                request = task["request"]
                
                # 查找合适的节点
                for node_id, node_gpus in available_gpus.items():
                    required_memory = request.gpu_memory_gb * 1024  # GB -> MB
                    
                    # 检查GPU内存
                    available_gpu_memory = [
                        gpu_id
                        for gpu_id in node_gpus
                        if gpu_memory[node_id][gpu_id] >= required_memory
                    ]
                    
                    if len(available_gpu_memory) >= request.gpu_count:
                        # 分配
                        allocated = available_gpu_memory[:request.gpu_count]
                        
                        # 更新状态
                        for gpu_id in allocated:
                            gpu_memory[node_id][gpu_id] -= required_memory
                        
                        # 移除任务
                        queue.remove(task)
                        
                        return True, node_id, allocated
        
        return False, None, None
    
    def get_wait_time(self, priority: Priority, gpu_count: int) -> float:
        """估算等待时间"""
        # 简化的等待时间计算
        ahead = sum(
            1 for p in list(Priority)[:list(Priority).index(priority)+1]
            for t in self.priority_queues[p]
            if t["request"].gpu_count <= gpu_count
        )
        
        # 假设每个GPU任务平均4小时
        return ahead * 4.0 / max(gpu_count, 1)
    
    def cancel_request(self, task_id: str) -> bool:
        """取消资源申请"""
        for queue in self.priority_queues.values():
            for task in queue:
                if task["task_id"] == task_id:
                    queue.remove(task)
                    return True
        return False
    
    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        return {
            "critical": len(self.priority_queues[Priority.CRITICAL]),
            "high": len(self.priority_queues[Priority.HIGH]),
            "normal": len(self.priority_queues[Priority.NORMAL]),
            "low": len(self.priority_queues[Priority.LOW]),
            "total": sum(len(q) for q in self.priority_queues.values())
        }

# 调度器实例
resource_scheduler = ResourceScheduler()
