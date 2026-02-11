"""
资源管理器 - 负责量子设备调度、队列管理和负载均衡
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import heapq
import logging
import random

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """后端类型枚举"""
    QUANTUM_SIMULATOR = "quantum_simulator"
    QUANTUM_DEVICE = "quantum_device"
    CLASSICAL_CPU = "classical_cpu"
    CLASSICAL_GPU = "classical_gpu"
    HYBRID = "hybrid"


class DeviceStatus(Enum):
    """设备状态枚举"""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class JobPriority(Enum):
    """作业优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QuantumDevice:
    """量子设备信息"""
    device_id: str
    name: str
    backend_type: BackendType
    qubits: int
    connectivity: List[Tuple[int, int]]  # 连接性
    status: DeviceStatus = DeviceStatus.AVAILABLE
    cost_per_shot: float = 0.01
    avg_execution_time: float = 1.0  # 秒
    accuracy: float = 0.99
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """检查设备是否可用"""
        return self.status == DeviceStatus.AVAILABLE
    
    def estimated_cost(self, shots: int) -> float:
        """估算执行成本"""
        return shots * self.cost_per_shot
    
    def estimated_time(self, shots: int, circuit_depth: int) -> float:
        """估算执行时间"""
        base_time = self.avg_execution_time
        return base_time * (shots / 1024) * (circuit_depth / 10)


@dataclass
class ComputeJob:
    """计算作业"""
    job_id: str
    job_type: BackendType
    priority: JobPriority
    qubits_required: int = 4
    shots: int = 1024
    circuit_depth: int = 10
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, queued, running, completed, failed, cancelled
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def waiting_time(self) -> float:
        """等待时间（秒）"""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return 0


class PriorityQueue:
    """优先级队列实现"""
    
    def __init__(self):
        self._queue: List[Tuple[int, datetime, ComputeJob]] = []
        self._counter = 0
    
    def enqueue(self, job: ComputeJob, priority: int = 2) -> None:
        """入队"""
        entry = (priority, self._counter, job)
        heapq.heappush(self._queue, entry)
        self._counter += 1
    
    def dequeue(self) -> Optional[ComputeJob]:
        """出队"""
        if self._queue:
            _, _, job = heapq.heappop(self._queue)
            return job
        return None
    
    def peek(self) -> Optional[ComputeJob]:
        """查看队首元素"""
        if self._queue:
            return self._queue[0][2]
        return None
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def is_empty(self) -> bool:
        return len(self._queue) == 0


class ResourceManager:
    """
    资源管理器核心类
    
    负责量子设备调度、队列管理、负载均衡和成本优化。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化资源管理器"""
        self.config = config or {}
        self.devices: Dict[str, QuantumDevice] = {}
        self.job_queues: Dict[BackendType, PriorityQueue] = {
            bt: PriorityQueue() for bt in BackendType
        }
        self.active_jobs: Dict[str, ComputeJob] = {}
        self.completed_jobs: Dict[str, ComputeJob] = {}
        
        # 负载均衡策略
        self.load_balancing_strategy = self.config.get('load_balancing', 'round_robin')
        
        # 成本优化
        self.cost_budget = self.config.get('cost_budget', 100.0)
        self.current_cost = 0.0
        
        # 初始化默认设备
        self._init_default_devices()
    
    def _init_default_devices(self) -> None:
        """初始化默认量子设备"""
        # 量子模拟器
        simulator = QuantumDevice(
            device_id="simulator_local",
            name="本地量子模拟器",
            backend_type=BackendType.QUANTUM_SIMULATOR,
            qubits=16,
            connectivity=[(i, i+1) for i in range(15)],
            cost_per_shot=0.001,
            avg_execution_time=0.1,
            accuracy=0.999,
            metadata={'type': 'statevector', 'noise_model': None}
        )
        self.devices[simulator.device_id] = simulator
        
        # 模拟真实量子设备
        ibm_like = QuantumDevice(
            device_id="ibm_heron",
            name="IBM Heron",
            backend_type=BackendType.QUANTUM_DEVICE,
            qubits=127,
            connectivity=[(i, (i+1)%127) for i in range(127)],
            cost_per_shot=0.1,
            avg_execution_time=2.0,
            accuracy=0.98,
            metadata={'type': 'superconducting', 'coherence_time': 100}
        )
        self.devices[ibm_like.device_id] = ibm_like
        
        # 经典CPU后端
        cpu = QuantumDevice(
            device_id="cpu_local",
            name="本地CPU",
            backend_type=BackendType.CLASSICAL_CPU,
            qubits=0,
            connectivity=[],
            cost_per_shot=0.0,
            avg_execution_time=0.01,
            accuracy=1.0,
            metadata={'cores': 8, 'memory': 32}
        )
        self.devices[cpu.device_id] = cpu
    
    def register_device(self, device: QuantumDevice) -> bool:
        """
        注册量子设备
        
        Args:
            device: 量子设备信息
            
        Returns:
            bool: 是否注册成功
        """
        if device.device_id in self.devices:
            logger.warning(f"设备 {device.device_id} 已存在，更新信息")
            self.devices[device.device_id] = device
        else:
            logger.info(f"设备 {device.name} ({device.device_id}) 已注册")
        
        return True
    
    def unregister_device(self, device_id: str) -> bool:
        """
        注销设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否注销成功
        """
        if device_id in self.devices:
            del self.devices[device_id]
            logger.info(f"设备 {device_id} 已注销")
            return True
        return False
    
    def submit_job(self, job: ComputeJob) -> str:
        """
        提交计算作业
        
        Args:
            job: 计算作业
            
        Returns:
            str: 作业ID
        """
        # 估算成本
        if job.job_type in [BackendType.QUANTUM_SIMULATOR, BackendType.QUANTUM_DEVICE]:
            for device in self.devices.values():
                if device.backend_type == job.job_type:
                    estimated_cost = device.estimated_cost(job.shots)
                    if self.current_cost + estimated_cost > self.cost_budget:
                        job.status = "cancelled"
                        job.error = "超出成本预算"
                        self.completed_jobs[job.job_id] = job
                        logger.warning(f"作业 {job.job_id} 因成本超限被取消")
                        return job.job_id
        
        # 入队
        priority = job.priority.value if isinstance(job.priority, JobPriority) else 2
        self.job_queues[job.job_type].enqueue(job, priority)
        job.status = "queued"
        
        logger.info(f"作业 {job.job_id} 已提交到 {job.job_type.value} 队列")
        
        # 尝试调度
        self._schedule_jobs()
        
        return job.job_id
    
    def cancel_job(self, job_id: str) -> bool:
        """取消作业"""
        # 检查活跃作业
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            job.completed_at = datetime.now()
            del self.active_jobs[job_id]
            self.completed_jobs[job_id] = job
            logger.info(f"作业 {job_id} 已取消")
            return True
        
        # 检查队列
        for queue in self.job_queues.values():
            for entry in queue._queue:
                if entry[2].job_id == job_id:
                    entry[2].status = "cancelled"
                    logger.info(f"作业 {job_id} 已从队列移除")
                    return True
        
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """获取作业状态"""
        # 检查活跃作业
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                'job_id': job_id,
                'status': job.status,
                'waiting_time': job.waiting_time
            }
        
        # 检查已完成作业
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                'job_id': job_id,
                'status': job.status,
                'completed': True
            }
        
        return None
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """获取作业结果"""
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return job.result
        return None
    
    def _schedule_jobs(self) -> None:
        """调度作业"""
        for backend_type in BackendType:
            queue = self.job_queues[backend_type]
            
            while not queue.is_empty():
                # 尝试找到可用设备
                device = self._find_available_device(backend_type)
                
                if device is None:
                    break  # 没有可用设备
                
                job = queue.dequeue()
                if job is None:
                    break
                
                # 分配设备并执行
                self._execute_job(device, job)
    
    def _find_available_device(self, backend_type: BackendType) -> Optional[QuantumDevice]:
        """查找可用设备"""
        candidates = [
            d for d in self.devices.values() 
            if d.backend_type == backend_type and d.is_available()
        ]
        
        if not candidates:
            return None
        
        # 应用负载均衡策略
        if self.load_balancing_strategy == 'round_robin':
            # 轮询：选择使用最少的设备
            return min(candidates, key=lambda d: d.last_used or datetime.min)
        
        elif self.load_balancing_strategy == 'least_connections':
            # 最小连接：选择活跃作业最少的设备
            return min(candidates, key=lambda d: len([
                j for j in self.active_jobs.values() 
                if j.submitted_at and d.last_used and j.submitted_at > d.last_used
            ]))
        
        elif self.load_balancing_strategy == 'shortest_queue':
            # 最短队列：选择队列最短的设备
            queue = self.job_queues.get(backend_type, PriorityQueue())
            return min(candidates, key=lambda d: len(queue))
        
        else:  # random
            return random.choice(candidates)
    
    def _execute_job(self, device: QuantumDevice, job: ComputeJob) -> None:
        """
        执行作业
        
        Args:
            device: 目标设备
            job: 计算作业
        """
        # 更新设备状态
        device.status = DeviceStatus.BUSY
        device.last_used = datetime.now()
        
        # 更新作业状态
        job.status = "running"
        job.started_at = datetime.now()
        job.submitted_at = job.submitted_at or datetime.now()
        
        self.active_jobs[job.job_id] = job
        
        logger.info(f"作业 {job.job_id} 开始在 {device.name} 上执行")
        
        # 模拟执行（实际会调用真实量子后端）
        try:
            result = self._run_job_on_device(device, job)
            job.result = result
            job.status = "completed"
        except Exception as e:
            job.error = str(e)
            job.status = "failed"
        finally:
            job.completed_at = datetime.now()
            device.status = DeviceStatus.AVAILABLE
            
            # 移动到已完成
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            self.completed_jobs[job.job_id] = job
            
            # 更新成本
            if job.job_type in [BackendType.QUANTUM_SIMULATOR, BackendType.QUANTUM_DEVICE]:
                self.current_cost += device.estimated_cost(job.shots)
            
            # 继续调度
            self._schedule_jobs()
    
    def _run_job_on_device(self, device: QuantumDevice, job: ComputeJob) -> Dict:
        """在设备上运行作业"""
        import random
        import time
        
        # 模拟执行时间
        execution_time = device.estimated_time(job.shots, job.circuit_depth)
        time.sleep(min(execution_time, 0.1))  # 限制实际等待时间
        
        # 生成模拟结果
        if device.backend_type == BackendType.QUANTUM_SIMULATOR:
            # 量子模拟器结果
            measurements = {}
            for _ in range(min(job.shots, 100)):
                outcome = ''.join(random.choice('01') for _ in range(job.qubits_required))
                measurements[outcome] = measurements.get(outcome, 0) + 1
            
            total = sum(measurements.values())
            probabilities = {k: v / total for k, v in measurements.items()}
            
            return {
                'device_id': device.device_id,
                'measurements': measurements,
                'probabilities': probabilities,
                'shots': job.shots,
                'qubits': job.qubits_required,
                'execution_time': execution_time
            }
        
        elif device.backend_type == BackendType.CLASSICAL_CPU:
            # 经典计算结果
            return {
                'device_id': device.device_id,
                'result': {'status': 'computed'},
                'execution_time': execution_time
            }
        
        else:
            return {
                'device_id': device.device_id,
                'status': 'executed',
                'execution_time': execution_time
            }
    
    def get_available_devices(self, backend_type: Optional[BackendType] = None) -> List[QuantumDevice]:
        """获取可用设备列表"""
        if backend_type:
            return [d for d in self.devices.values() 
                   if d.backend_type == backend_type and d.is_available()]
        return [d for d in self.devices.values() if d.is_available()]
    
    def get_device_stats(self) -> Dict[str, Any]:
        """获取设备统计信息"""
        stats = {
            'total_devices': len(self.devices),
            'available_devices': len(self.get_available_devices()),
            'active_jobs': len(self.active_jobs),
            'queued_jobs': sum(len(q) for q in self.job_queues.values()),
            'completed_jobs': len(self.completed_jobs),
            'total_cost': self.current_cost,
            'cost_budget': self.cost_budget,
            'devices': {}
        }
        
        for device_id, device in self.devices.items():
            stats['devices'][device_id] = {
                'name': device.name,
                'status': device.status.value,
                'qubits': device.qubits,
                'backend_type': device.backend_type.value,
                'accuracy': device.accuracy
            }
        
        return stats
    
    def optimize_cost(self, jobs: List[ComputeJob]) -> List[ComputeJob]:
        """
        优化作业成本
        
        Args:
            jobs: 作业列表
            
        Returns:
            List[ComputeJob]: 优化后的作业列表
        """
        # 按成本排序
        optimized = sorted(jobs, key=lambda j: (
            j.shots * min(
                d.cost_per_shot for d in self.devices.values() 
                if d.backend_type == j.job_type
            ) if j.job_type in [BackendType.QUANTUM_SIMULATOR, BackendType.QUANTUM_DEVICE] else 0
        ))
        
        return optimized
    
    def suggest_device(self, job: ComputeJob) -> Optional[QuantumDevice]:
        """
        为作业推荐最佳设备
        
        Args:
            job: 计算作业
            
        Returns:
            Optional[QuantumDevice]: 推荐的设备
        """
        candidates = [
            d for d in self.devices.values() 
            if d.backend_type == job.job_type and d.is_available()
            and d.qubits >= job.qubits_required
        ]
        
        if not candidates:
            return None
        
        # 综合评分
        scores = []
        for device in candidates:
            # 成本评分
            cost_score = 1.0 - (device.estimated_cost(job.shots) / self.cost_budget)
            
            # 性能评分
            time_score = 1.0 / (device.estimated_time(job.shots, job.circuit_depth) + 0.1)
            
            # 准确度评分
            accuracy_score = device.accuracy
            
            # 综合评分
            total_score = cost_score * 0.3 + time_score * 0.3 + accuracy_score * 0.4
            scores.append((device, total_score))
        
        # 返回最高分设备
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else None
    
    def get_queue_status(self, backend_type: BackendType) -> Dict[str, Any]:
        """获取队列状态"""
        queue = self.job_queues.get(backend_type, PriorityQueue())
        
        return {
            'backend_type': backend_type.value,
            'queue_length': len(queue),
            'is_empty': queue.is_empty(),
            'next_job': queue.peek().job_id if queue.peek() else None
        }
