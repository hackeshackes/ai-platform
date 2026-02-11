"""
Distributed Trainer - AI Platform v6

分布式训练器，支持数据并行和模型并行训练。
"""
import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TrainingStrategy(Enum):
    """训练策略"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """训练配置"""
    model_name: str = ""
    model_path: Optional[str] = None
    training_script: Optional[str] = None
    training_args: Dict[str, Any] = field(default_factory=dict)
    data_path: Optional[str] = None
    data_config: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None
    strategy: TrainingStrategy = TrainingStrategy.DATA_PARALLEL
    num_workers: int = 1
    num_cpus_per_worker: int = 4
    num_gpus_per_worker: int = 1
    memory_per_worker: int = 0
    object_store_memory: int = 10737418240  # 10GB
    checkpoint_interval: int = 1000
    max_training_steps: int = 0
    max_training_time: int = 0  # 秒
    env_vars: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    name: str
    config: TrainingConfig
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class DistributedTrainer:
    """
    分布式训练器
    
    管理分布式训练任务的提交、执行和监控。
    """
    
    def __init__(self, ray_address: Optional[str] = None):
        """
        初始化分布式训练器
        
        Args:
            ray_address: Ray集群地址
        """
        self.ray_address = ray_address
        self._tasks: Dict[str, TrainingTask] = {}
        self._task_results: Dict[str, Any] = {}
        self._ray_client = None
        
    async def initialize(self) -> bool:
        """
        初始化训练器
        
        Returns:
            是否初始化成功
        """
        try:
            from .ray import get_ray_client
            self._ray_client = get_ray_client(self.ray_address)
            return await self._ray_client.initialize()
        except Exception as e:
            logger.error(f"初始化分布式训练器失败: {e}")
            return False
    
    async def create_task(self, 
                         name: str, 
                         config: TrainingConfig,
                         description: str = "") -> TrainingTask:
        """
        创建训练任务
        
        Args:
            name: 任务名称
            config: 训练配置
            description: 任务描述
            
        Returns:
            创建的训练任务
        """
        task_id = str(uuid.uuid4())[:8]
        task = TrainingTask(
            task_id=task_id,
            name=name,
            config=config
        )
        
        self._tasks[task_id] = task
        logger.info(f"创建训练任务: {task_id} - {name}")
        
        return task
    
    async def submit_task(self, task_id: str) -> Dict[str, Any]:
        """
        提交训练任务到Ray集群
        
        Args:
            task_id: 任务ID
            
        Returns:
            提交结果
        """
        if task_id not in self._tasks:
            return {
                "success": False,
                "error": f"任务不存在: {task_id}"
            }
        
        task = self._tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # 异步执行训练任务
            asyncio.create_task(self._run_training(task))
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "训练任务已提交"
            }
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"提交训练任务失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_training(self, task: TrainingTask):
        """
        执行训练任务
        
        Args:
            task: 训练任务
        """
        try:
            task.logs.append(f"开始执行训练任务: {task.task_id}")
            
            # 构建训练脚本或使用预定义的训练函数
            if task.config.training_script:
                result = await self._execute_training_script(task)
            else:
                result = await self._execute_remote_training(task)
            
            if result["success"]:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.progress = 100.0
                task.logs.append("训练任务完成")
                task.metrics = result.get("metrics", {})
                task.checkpoint_path = result.get("checkpoint_path")
            else:
                task.status = TaskStatus.FAILED
                task.error = result.get("error", "未知错误")
                task.logs.append(f"训练任务失败: {task.error}")
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.logs.append(f"训练任务异常: {str(e)}")
            logger.error(f"执行训练任务失败: {e}")
    
    async def _execute_training_script(self, task: TrainingTask) -> Dict[str, Any]:
        """
        执行训练脚本
        
        Args:
            task: 训练任务
            
        Returns:
            执行结果
        """
        import subprocess
        
        cmd = ["python", task.config.training_script]
        
        # 添加训练参数
        for key, value in task.config.training_args.items():
            cmd.extend([f"--{key}", str(value)])
        
        # 添加数据路径
        if task.config.data_path:
            cmd.extend(["--data-path", task.config.data_path])
        
        # 添加输出路径
        if task.config.output_path:
            cmd.extend(["--output-path", task.config.output_path])
        
        env = os.environ.copy()
        env.update(task.config.env_vars)
        env["RAY_ADDRESS"] = self.ray_address or ""
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # 读取输出
            for line in iter(process.stdout.readline, ''):
                if line:
                    task.logs.append(line.strip())
            
            process.wait(timeout=task.config.max_training_time if task.config.max_training_time > 0 else 86400)
            
            if process.returncode == 0:
                return {
                    "success": True,
                    "message": "训练完成"
                }
            else:
                return {
                    "success": False,
                    "error": f"训练脚本返回非零状态: {process.returncode}"
                }
                
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "success": False,
                "error": "训练超时"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_remote_training(self, task: TrainingTask) -> Dict[str, Any]:
        """
        执行远程训练（使用Ray远程函数）
        
        Args:
            task: 训练任务
            
        Returns:
            执行结果
        """
        try:
            # 定义远程训练函数
            @ray.remote
            def distributed_training_function(config: Dict[str, Any]):
                import os
                import json
                
                # 在这里实现实际的分布式训练逻辑
                # 这是示例实现
                num_epochs = config.get("num_epochs", 1)
                batch_size = config.get("batch_size", 32)
                learning_rate = config.get("learning_rate", 0.001)
                
                # 模拟训练过程
                for epoch in range(num_epochs):
                    # 模拟每个epoch的训练
                    pass
                
                return {
                    "final_loss": 0.1,
                    "final_accuracy": 0.95,
                    "training_time": 100.0
                }
            
            # 提交远程任务
            if not self._ray_client._ray_available:
                return {
                    "success": False,
                    "error": "Ray客户端未初始化"
                }
            
            import ray
            
            # 转换配置为字典
            config_dict = {
                "num_epochs": task.config.training_args.get("num_epochs", 1),
                "batch_size": task.config.training_args.get("batch_size", 32),
                "learning_rate": task.config.training_args.get("learning_rate", 0.001),
                "data_path": task.config.data_path,
                "output_path": task.config.output_path
            }
            
            # 使用Ray分布式执行
            resources = {
                "CPU": task.config.num_cpus_per_worker,
                "GPU": task.config.num_gpus_per_worker
            }
            resources.update(task.config.resources)
            
            # 提交到Ray集群
            if task.config.strategy == TrainingStrategy.DATA_PARALLEL:
                # 数据并行
                results = []
                for i in range(task.config.num_workers):
                    future = distributed_training_function.options(
                        num_cpus=task.config.num_cpus_per_worker,
                        num_gpus=task.config.num_gpus_per_worker,
                        resources=resources
                    ).remote(config_dict)
                    results.append(future)
                
                ray.get(results)
            else:
                # 其他并行策略
                future = distributed_training_function.options(
                    num_cpus=task.config.num_cpus_per_worker * task.config.num_workers,
                    num_gpus=task.config.num_gpus_per_worker * task.config.num_workers,
                    resources=resources
                ).remote(config_dict)
                
                ray.get(future)
            
            return {
                "success": True,
                "message": "分布式训练完成"
            }
            
        except Exception as e:
            logger.error(f"远程训练执行失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        取消训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            取消结果
        """
        if task_id not in self._tasks:
            return {
                "success": False,
                "error": f"任务不存在: {task_id}"
            }
        
        task = self._tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return {
                "success": False,
                "error": f"任务已结束，无法取消"
            }
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        task.logs.append("任务已取消")
        
        return {
            "success": True,
            "message": "任务已取消"
        }
    
    async def get_task_status(self, task_id: str) -> Optional[TrainingTask]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象或None
        """
        return self._tasks.get(task_id)
    
    async def list_tasks(self, 
                        status_filter: Optional[List[TaskStatus]] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出任务
        
        Args:
            status_filter: 状态过滤器
            limit: 返回数量限制
            
        Returns:
            任务列表
        """
        tasks = []
        
        for task in self._tasks.values():
            if status_filter and task.status not in status_filter:
                continue
            
            tasks.append({
                "task_id": task.task_id,
                "name": task.name,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error": task.error
            })
        
        return tasks[:limit]
    
    async def get_task_logs(self, task_id: str, tail: int = 100) -> List[str]:
        """
        获取任务日志
        
        Args:
            task_id: 任务ID
            tail: 返回最后N行
            
        Returns:
            日志列表
        """
        if task_id not in self._tasks:
            return []
        
        logs = self._tasks[task_id].logs
        return logs[-tail:] if tail > 0 else logs
    
    async def cleanup(self):
        """清理资源"""
        # 取消所有运行中的任务
        for task in self._tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
        
        # 关闭Ray客户端
        if self._ray_client:
            await self._ray_client.shutdown()


# 单例分布式训练器
_trainer: Optional[DistributedTrainer] = None


def get_distributed_trainer(ray_address: Optional[str] = None) -> DistributedTrainer:
    """获取分布式训练器单例"""
    global _trainer
    if _trainer is None:
        _trainer = DistributedTrainer(ray_address)
    return _trainer
