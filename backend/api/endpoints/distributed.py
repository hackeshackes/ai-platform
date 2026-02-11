"""
Distributed Training API Endpoints - AI Platform v6

分布式训练API端点，提供Ray集群管理、训练任务调度和资源监控REST API。
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from backend.distributed.ray import (
    RayClusterManager,
    RayClusterConfig,
    get_ray_cluster_manager
)
from backend.distributed.trainer import (
    DistributedTrainer,
    TrainingConfig,
    TrainingStrategy,
    TaskStatus,
    get_distributed_trainer
)
from backend.distributed.scheduler import (
    TaskScheduler,
    QueueConfig,
    Priority,
    get_task_scheduler
)
from backend.distributed.monitor import (
    ResourceMonitor,
    get_resource_monitor
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============ 共享依赖 ============

def get_ray_manager() -> RayClusterManager:
    """获取Ray集群管理器"""
    return get_ray_cluster_manager()


def get_trainer() -> DistributedTrainer:
    """获取分布式训练器"""
    return get_distributed_trainer()


def get_scheduler() -> TaskScheduler:
    """获取任务调度器"""
    return get_task_scheduler()


def get_monitor() -> ResourceMonitor:
    """获取资源监控器"""
    return get_resource_monitor()


# ============ 请求/响应模型 ============

# Ray集群配置
class RayClusterStartRequest(BaseModel):
    """启动Ray集群请求"""
    head_ip: str = "0.0.0.0"
    head_port: int = 6379
    dashboard_port: int = 8265
    num_workers: int = 0
    object_store_memory: int = 107374182400  # 100GB
    env_vars: Dict[str, str] = Field(default_factory=dict)


class RayClusterScaleRequest(BaseModel):
    """Ray集群扩缩容请求"""
    num_nodes: int = Field(..., ge=1, description="目标节点数量")


# 训练配置
class TrainingArgs(BaseModel):
    """训练参数"""
    num_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, ge=0.0)
    max_steps: int = Field(default=0, ge=0)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    optimizer: str = "adam"
    scheduler: str = "linear"
    fp16: bool = False
    deepspeed: bool = False
    deepspeed_config: Optional[Dict[str, Any]] = None


class DataConfig(BaseModel):
    """数据配置"""
    data_path: str = Field(..., description="数据路径")
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4


class TrainingTaskRequest(BaseModel):
    """提交训练任务请求"""
    name: str = Field(..., min_length=1, max_length=255, description="任务名称")
    model_name: str = Field(..., description="模型名称或路径")
    model_path: Optional[str] = None
    training_script: Optional[str] = None
    training_args: TrainingArgs = Field(default_factory=TrainingArgs)
    data_path: Optional[str] = None
    data_config: Optional[DataConfig] = None
    output_path: str = Field(..., description="输出路径")
    strategy: str = Field(default="data_parallel", description="并行策略: data_parallel, model_parallel, hybrid_parallel")
    num_workers: int = Field(default=1, ge=1, description="工作节点数")
    num_cpus_per_worker: int = Field(default=4, ge=1)
    num_gpus_per_worker: int = Field(default=1, ge=0)
    memory_per_worker: int = Field(default=0, ge=0)
    object_store_memory: int = Field(default=10737418240, ge=0)
    checkpoint_interval: int = Field(default=1000, ge=1)
    priority: str = Field(default="normal", description="任务优先级: low, normal, high, urgent")
    dependencies: Optional[List[str]] = None
    env_vars: Dict[str, str] = Field(default_factory=dict)


# ============ Ray集群管理端点 ============

@router.post("/cluster/start")
async def start_ray_cluster(
    request: RayClusterStartRequest,
    background_tasks: BackgroundTasks,
    manager: RayClusterManager = Depends(get_ray_manager)
) -> Dict[str, Any]:
    """
    启动Ray集群
    
    启动一个新的Ray集群，包含head节点和可选的worker节点。
    """
    # 配置集群
    config = RayClusterConfig(
        head_ip=request.head_ip,
        head_port=request.head_port,
        dashboard_port=request.dashboard_port,
        num_workers=request.num_workers,
        object_store_memory=request.object_store_memory,
        env_vars=request.env_vars
    )
    manager.config = config
    
    # 启动集群
    result = await manager.start_cluster(head_only=request.num_workers == 0)
    
    if result["success"]:
        # 启动资源监控
        monitor = get_resource_monitor()
        await monitor.start_monitoring(interval=5)
        
        # 初始化训练器和调度器
        trainer = get_distributed_trainer()
        await trainer.initialize()
        
        scheduler = get_task_scheduler()
        await scheduler.start()
    
    return result


@router.post("/cluster/stop")
async def stop_ray_cluster(
    manager: RayClusterManager = Depends(get_ray_manager)
) -> Dict[str, Any]:
    """
    停止Ray集群
    
    停止当前运行的Ray集群。
    """
    # 停止监控
    monitor = get_resource_monitor()
    await monitor.stop_monitoring()
    
    # 停止调度器
    scheduler = get_task_scheduler()
    await scheduler.stop()
    
    # 停止集群
    result = await manager.stop_cluster()
    
    return result


@router.get("/cluster/status")
async def get_cluster_status(
    manager: RayClusterManager = Depends(get_ray_manager)
) -> Dict[str, Any]:
    """
    获取Ray集群状态
    
    返回当前Ray集群的运行状态和资源信息。
    """
    status = await manager.get_status()
    
    return {
        "is_running": status.is_running,
        "head_address": status.head_address,
        "dashboard_url": status.dashboard_url,
        "num_nodes": status.num_nodes,
        "uptime_seconds": status.uptime_seconds,
        "available_resources": status.available_resources,
        "used_resources": status.used_resources,
        "last_heartbeat": status.last_heartbeat.isoformat() if status.last_heartbeat else None
    }


@router.post("/cluster/scale")
async def scale_cluster(
    request: RayClusterScaleRequest,
    manager: RayClusterManager = Depends(get_ray_manager)
) -> Dict[str, Any]:
    """
    扩缩容Ray集群
    
    调整Ray集群的节点数量。
    """
    result = await manager.scale_cluster(request.num_nodes)
    
    return result


@router.get("/cluster/nodes")
async def list_cluster_nodes(
    manager: RayClusterManager = Depends(get_ray_manager)
) -> Dict[str, Any]:
    """
    列出集群节点
    
    返回Ray集群中所有节点的信息。
    """
    status = await manager.get_status()
    
    nodes = []
    # 简化实现：返回head节点信息
    if status.is_running:
        nodes.append({
            "node_id": "head",
            "type": "head",
            "address": status.head_address,
            "status": "alive"
        })
        
        for i in range(status.num_nodes - 1):
            nodes.append({
                "node_id": f"worker-{i}",
                "type": "worker",
                "address": f"{manager.config.head_ip}:{manager.config.head_port}",
                "status": "alive"
            })
    
    return {
        "nodes": nodes,
        "total": len(nodes)
    }


# ============ 训练任务端点 ============

@router.post("/train")
async def submit_training_task(
    request: TrainingTaskRequest,
    background_tasks: BackgroundTasks,
    scheduler: TaskScheduler = Depends(get_scheduler)
) -> Dict[str, Any]:
    """
    提交训练任务
    
    提交一个新的分布式训练任务到调度队列。
    """
    # 转换策略
    strategy_map = {
        "data_parallel": TrainingStrategy.DATA_PARALLEL,
        "model_parallel": TrainingStrategy.MODEL_PARALLEL,
        "hybrid_parallel": TrainingStrategy.HYBRID_PARALLEL,
        "pipeline_parallel": TrainingStrategy.PIPELINE_PARALLEL
    }
    strategy = strategy_map.get(request.strategy, TrainingStrategy.DATA_PARALLEL)
    
    # 转换优先级
    priority_map = {
        "low": Priority.LOW,
        "normal": Priority.NORMAL,
        "high": Priority.HIGH,
        "urgent": Priority.URGENT
    }
    priority = priority_map.get(request.priority, Priority.NORMAL)
    
    # 构建训练配置
    config = TrainingConfig(
        model_name=request.model_name,
        model_path=request.model_path,
        training_script=request.training_script,
        training_args=request.training_args.dict(),
        data_path=request.data_path,
        data_config=request.data_config.dict() if request.data_config else {},
        output_path=request.output_path,
        strategy=strategy,
        num_workers=request.num_workers,
        num_cpus_per_worker=request.num_cpus_per_worker,
        num_gpus_per_worker=request.num_gpus_per_worker,
        memory_per_worker=request.memory_per_worker,
        object_store_memory=request.object_store_memory,
        checkpoint_interval=request.checkpoint_interval,
        env_vars=request.env_vars
    )
    
    # 提交任务到调度器
    result = await scheduler.submit_task(
        name=request.name,
        config=config,
        priority=priority,
        dependencies=request.dependencies
    )
    
    return result


@router.get("/tasks")
async def list_training_tasks(
    status_filter: Optional[List[str]] = Query(None, description="状态过滤: pending, running, completed, failed, cancelled"),
    limit: int = Query(100, ge=1, le=1000),
    scheduler: TaskScheduler = Depends(get_scheduler)
) -> Dict[str, Any]:
    """
    列出训练任务
    
    返回所有训练任务的状态列表。
    """
    # 转换状态过滤
    valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
    if status_filter:
        status_filter = [s for s in status_filter if s in valid_statuses]
    
    tasks = await scheduler.list_tasks(status_filter=status_filter, limit=limit)
    
    return {
        "tasks": tasks,
        "total": len(tasks)
    }


@router.get("/tasks/{task_id}")
async def get_training_task(
    task_id: str,
    scheduler: TaskScheduler = Depends(get_scheduler)
) -> Dict[str, Any]:
    """
    获取训练任务详情
    
    返回指定训练任务的详细信息。
    """
    status = await scheduler.get_task_status(task_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    return status


@router.delete("/tasks/{task_id}")
async def cancel_training_task(
    task_id: str,
    scheduler: TaskScheduler = Depends(get_scheduler)
) -> Dict[str, Any]:
    """
    取消训练任务
    
    取消指定ID的训练任务。
    """
    result = await scheduler.cancel_task(task_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error", "任务不存在"))
    
    return result


@router.get("/tasks/{task_id}/logs")
async def get_training_logs(
    task_id: str,
    tail: int = Query(100, ge=0, le=10000),
    scheduler: TaskScheduler = Depends(get_scheduler)
) -> Dict[str, Any]:
    """
    获取训练任务日志
    
    返回指定训练任务的日志。
    """
    trainer = get_distributed_trainer()
    logs = await trainer.get_task_logs(task_id, tail=tail)
    
    if logs is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    return {
        "task_id": task_id,
        "logs": logs,
        "total_lines": len(logs)
    }


# ============ 资源监控端点 ============

@router.get("/resources")
async def get_resource_usage(
    monitor: ResourceMonitor = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    获取资源使用情况
    
    返回当前系统的资源使用情况。
    """
    return await monitor.get_resource_usage()


@router.get("/resources/history")
async def get_resource_history(
    minutes: int = Query(5, ge=1, le=60),
    monitor: ResourceMonitor = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    获取资源历史数据
    
    返回指定时间范围内的资源使用历史数据。
    """
    node_history = monitor.get_node_history(minutes=minutes)
    cluster_history = monitor.get_cluster_history(minutes=minutes)
    
    return {
        "time_range_minutes": minutes,
        "node_history": [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "memory_used": m.memory_used,
                "memory_total": m.memory_total,
                "gpu_count": m.gpu_count,
                "gpu_percent": m.gpu_percent,
                "gpu_memory_used": m.gpu_memory_used,
                "gpu_memory_total": m.gpu_memory_total,
                "network_sent": m.network_sent,
                "network_recv": m.network_recv
            }
            for m in node_history
        ],
        "cluster_history": [
            {
                "timestamp": m.timestamp.isoformat(),
                "total_nodes": m.total_nodes,
                "cpu_usage_percent": m.cpu_usage_percent,
                "memory_usage_percent": m.memory_usage_percent,
                "gpu_usage_percent": m.gpu_usage_percent,
                "available_cpus": m.available_cpus,
                "available_gpus": m.available_gpus,
                "available_memory": m.available_memory
            }
            for m in cluster_history
        ]
    }


@router.get("/monitoring/status")
async def get_monitoring_status(
    monitor: ResourceMonitor = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    获取监控状态
    
    返回详细的监控状态信息。
    """
    return await monitor.get_detailed_status()


@router.get("/scheduler/status")
async def get_scheduler_status(
    scheduler: TaskScheduler = Depends(get_scheduler)
) -> Dict[str, Any]:
    """
    获取调度器状态
    
    返回任务调度器的当前状态。
    """
    return scheduler.get_scheduler_state()


# ============ 健康检查端点 ============

@router.get("/health")
async def distributed_health_check() -> Dict[str, Any]:
    """
    分布式服务健康检查
    
    返回分布式训练服务的健康状态。
    """
    ray_manager = get_ray_cluster_manager()
    ray_status = await ray_manager.get_status()
    
    scheduler = get_task_scheduler()
    scheduler_state = scheduler.get_scheduler_state()
    
    return {
        "status": "healthy" if ray_status.is_running else "degraded",
        "ray_cluster": {
            "is_running": ray_status.is_running,
            "num_nodes": ray_status.num_nodes
        },
        "scheduler": {
            "state": scheduler_state["state"],
            "running_tasks": scheduler_state["running_tasks"],
            "queued_tasks": scheduler_state["queued_tasks"]["total"]
        }
    }    
