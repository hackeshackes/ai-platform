"""
Resource Monitor - AI Platform v6

资源监控模块，监控Ray集群和训练任务的资源使用情况。
"""
import asyncio
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from .ray import get_ray_cluster_manager, RayClusterStatus

logger = logging.getLogger(__name__)


@dataclass
class NodeMetrics:
    """节点指标"""
    node_id: str
    hostname: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used: int = 0
    memory_total: int = 0
    disk_percent: float = 0.0
    disk_used: int = 0
    disk_total: int = 0
    network_sent: int = 0
    network_recv: int = 0
    gpu_count: int = 0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskMetrics:
    """任务指标"""
    task_id: str
    name: str
    status: str
    progress: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: int = 0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    throughput: float = 0.0  # samples/sec
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    step: int = 0
    eta_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ClusterMetrics:
    """集群指标"""
    total_nodes: int = 0
    healthy_nodes: int = 0
    total_cpus: float = 0.0
    available_cpus: float = 0.0
    total_gpus: float = 0.0
    available_gpus: float = 0.0
    total_memory: int = 0
    available_memory: int = 0
    total_disk: int = 0
    available_disk: int = 0
    total_workers: int = 0
    running_tasks: int = 0
    queued_tasks: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ResourceMonitor:
    """
    资源监控器
    
    监控Ray集群和训练任务的资源使用情况。
    """
    
    def __init__(self, history_size: int = 100):
        """
        初始化资源监控器
        
        Args:
            history_size: 历史数据保留数量
        """
        self.history_size = history_size
        self._node_history: deque = deque(maxlen=history_size)
        self._task_history: Dict[str, deque] = {}
        self._cluster_history: deque = deque(maxlen=history_size)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self._ray_manager = get_ray_cluster_manager()
        self._last_network_stats: Dict[str, Any] = {}
        
    async def start_monitoring(self, interval: int = 5):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("资源监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("资源监控已停止")
    
    async def _monitor_loop(self, interval: int):
        """
        监控循环
        
        Args:
            interval: 监控间隔
        """
        while self._is_monitoring:
            try:
                # 收集节点指标
                node_metrics = await self._collect_node_metrics()
                self._node_history.append(node_metrics)
                
                # 收集集群指标
                cluster_metrics = await self._collect_cluster_metrics(node_metrics)
                self._cluster_history.append(cluster_metrics)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_node_metrics(self) -> List[NodeMetrics]:
        """
        收集节点指标
        
        Returns:
            节点指标列表
        """
        metrics_list = []
        
        try:
            # 获取本机指标
            node = await self._get_local_node_metrics()
            metrics_list.append(node)
            
            # 获取Ray集群节点指标
            ray_status = await self._ray_manager.get_status()
            if ray_status.is_running:
                # 这里可以扩展以获取Ray集群中其他节点的信息
                pass
            
        except Exception as e:
            logger.error(f"收集节点指标失败: {e}")
        
        return metrics_list
    
    async def _get_local_node_metrics(self) -> NodeMetrics:
        """
        获取本地节点指标
        
        Returns:
            本地节点指标
        """
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        
        # 网络信息
        network = psutil.net_io_counters()
        
        # GPU信息
        gpu_info = await self._get_gpu_metrics()
        
        # 网络速度计算
        network_speed = self._calculate_network_speed(network)
        
        node = NodeMetrics(
            node_id="local",
            hostname="localhost",
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used,
            memory_total=memory.total,
            disk_percent=disk.percent,
            disk_used=disk.used,
            disk_total=disk.total,
            network_sent=network_speed["sent"],
            network_recv=network_speed["recv"],
            gpu_count=gpu_info["count"],
            gpu_memory_used=gpu_info["memory_used"],
            gpu_memory_total=gpu_info["memory_total"],
            gpu_percent=gpu_info["utilization"]
        )
        
        return node
    
    async def _get_gpu_metrics(self) -> Dict[str, Any]:
        """
        获取GPU指标
        
        Returns:
            GPU指标
        """
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_memory_used = 0
                total_memory = 0
                gpu_utilization = []
                
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        total_memory_used += float(parts[1].strip())
                        total_memory += float(parts[2].strip())
                        gpu_utilization.append(float(parts[3].strip()))
                
                return {
                    "count": len(lines),
                    "memory_used": total_memory_used * 1024 * 1024,  # 转换为字节
                    "memory_total": total_memory * 1024 * 1024,
                    "utilization": sum(gpu_utilization) / len(gpu_utilization) if gpu_utilization else 0
                }
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return {
            "count": 0,
            "memory_used": 0,
            "memory_total": 0,
            "utilization": 0
        }
    
    def _calculate_network_speed(self, network: psutil.net_io_counters) -> Dict[str, int]:
        """
        计算网络速度
        
        Args:
            network: 网络IO计数
            
        Returns:
            网络速度
        """
        sent = network.bytes_sent
        recv = network.bytes_recv
        
        if self._last_network_stats:
            time_diff = 1  # 假设1秒间隔
            sent_speed = max(0, sent - self._last_network_stats.get("bytes_sent", 0)) // time_diff
            recv_speed = max(0, recv - self._last_network_stats.get("bytes_recv", 0)) // time_diff
        else:
            sent_speed = 0
            recv_speed = 0
        
        self._last_network_stats = {
            "bytes_sent": sent,
            "bytes_recv": recv,
            "timestamp": datetime.now()
        }
        
        return {
            "sent": sent_speed,
            "recv": recv_speed
        }
    
    async def _collect_cluster_metrics(self, 
                                       node_metrics: List[NodeMetrics],
                                       ray_status: Optional[RayClusterStatus] = None) -> ClusterMetrics:
        """
        收集集群指标
        
        Args:
            node_metrics: 节点指标列表
            ray_status: Ray集群状态
            
        Returns:
            集群指标
        """
        metrics = ClusterMetrics()
        
        if not node_metrics:
            return metrics
        
        # 计算汇总指标
        total_cpus = 0
        available_cpus = 0
        total_memory = 0
        available_memory = 0
        
        for node in node_metrics:
            metrics.total_cpus += psutil.cpu_count()
            metrics.available_cpus += psutil.cpu_count() * (100 - node.cpu_percent) / 100
            total_memory += node.memory_total
            available_memory += node.memory_total * (100 - node.memory_percent) / 100
            metrics.gpu_count += node.gpu_count
            metrics.available_gpus += node.gpu_count * (100 - node.gpu_percent) / 100
        
        metrics.total_nodes = len(node_metrics)
        metrics.healthy_nodes = len(node_metrics)
        metrics.total_memory = total_memory
        metrics.available_memory = int(available_memory)
        metrics.total_disk = node_metrics[0].disk_total if node_metrics else 0
        metrics.available_disk = node_metrics[0].disk_total - node_metrics[0].disk_used if node_metrics else 0
        
        # Ray集群状态
        if ray_status:
            metrics.total_nodes = ray_status.num_nodes
            metrics.running_tasks = ray_status.num_nodes  # 简化估算
        
        # 计算使用率
        if total_memory > 0:
            metrics.memory_usage_percent = 100 - (available_memory / total_memory * 100)
        
        metrics.cpu_usage_percent = sum(n.cpu_percent for n in node_metrics) / len(node_metrics)
        metrics.gpu_usage_percent = sum(n.gpu_percent for n in node_metrics) / len(node_metrics) if node_metrics else 0
        
        return metrics
    
    async def get_node_metrics(self, node_id: Optional[str] = None) -> Optional[NodeMetrics]:
        """
        获取节点指标
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点指标
        """
        if self._node_history:
            if node_id:
                for metrics in reversed(self._node_history):
                    if isinstance(metrics, list):
                        for m in metrics:
                            if m.node_id == node_id:
                                return m
                    elif metrics.node_id == node_id:
                        return metrics
            else:
                # 返回最新的本地节点指标
                for metrics in reversed(self._node_history):
                    if isinstance(metrics, list):
                        for m in metrics:
                            if m.node_id == "local":
                                return m
                    elif metrics.node_id == "local":
                        return metrics
        
        return None
    
    async def get_cluster_metrics(self) -> ClusterMetrics:
        """
        获取集群指标
        
        Returns:
            集群指标
        """
        if self._cluster_history:
            return self._cluster_history[-1]
        
        # 返回默认指标
        return ClusterMetrics()
    
    async def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """
        获取任务指标
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务指标
        """
        if task_id in self._task_history and self._task_history[task_id]:
            return self._task_history[task_id][-1]
        
        return None
    
    async def update_task_metrics(self, task_id: str, metrics: Dict[str, Any]):
        """
        更新任务指标
        
        Args:
            task_id: 任务ID
            metrics: 指标数据
        """
        if task_id not in self._task_history:
            self._task_history[task_id] = deque(maxlen=self.history_size)
        
        task_metrics = TaskMetrics(
            task_id=task_id,
            name=metrics.get("name", ""),
            status=metrics.get("status", "unknown"),
            progress=metrics.get("progress", 0.0),
            cpu_usage=metrics.get("cpu_usage", 0.0),
            memory_usage=metrics.get("memory_usage", 0),
            gpu_usage=metrics.get("gpu_usage", 0.0),
            gpu_memory=metrics.get("gpu_memory", 0.0),
            throughput=metrics.get("throughput", 0.0),
            loss=metrics.get("loss", 0.0),
            accuracy=metrics.get("accuracy", 0.0),
            learning_rate=metrics.get("learning_rate", 0.0),
            epoch=metrics.get("epoch", 0),
            step=metrics.get("step", 0),
            eta_seconds=metrics.get("eta_seconds", 0.0)
        )
        
        self._task_history[task_id].append(task_metrics)
    
    def get_node_history(self, minutes: int = 5) -> List[NodeMetrics]:
        """
        获取节点历史指标
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            历史指标列表
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        history = []
        for metrics in self._node_history:
            if isinstance(metrics, list):
                history.extend([m for m in metrics if m.timestamp >= cutoff])
            elif metrics.timestamp >= cutoff:
                history.append(metrics)
        
        return history
    
    def get_cluster_history(self, minutes: int = 5) -> List[ClusterMetrics]:
        """
        获取集群历史指标
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            历史指标列表
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        return [m for m in self._cluster_history if m.timestamp >= cutoff]
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取资源使用情况
        
        Returns:
            资源使用情况
        """
        cluster_metrics = await self.get_cluster_metrics()
        node_metrics = await self.get_node_metrics()
        
        return {
            "cpu": {
                "total": cluster_metrics.total_cpus,
                "available": cluster_metrics.available_cpus,
                "usage_percent": cluster_metrics.cpu_usage_percent
            },
            "memory": {
                "total_bytes": cluster_metrics.total_memory,
                "available_bytes": cluster_metrics.available_memory,
                "usage_percent": cluster_metrics.memory_usage_percent
            },
            "gpu": {
                "total": cluster_metrics.total_gpus,
                "available": cluster_metrics.available_gpus,
                "usage_percent": cluster_metrics.gpu_usage_percent
            },
            "disk": {
                "total_bytes": cluster_metrics.total_disk,
                "available_bytes": cluster_metrics.available_disk
            },
            "nodes": {
                "total": cluster_metrics.total_nodes,
                "healthy": cluster_metrics.healthy_nodes
            },
            "tasks": {
                "running": cluster_metrics.running_tasks,
                "queued": cluster_metrics.queued_tasks
            },
            "node_details": {
                "cpu_percent": node_metrics.cpu_percent if node_metrics else 0,
                "memory_percent": node_metrics.memory_percent if node_metrics else 0,
                "gpu_count": node_metrics.gpu_count if node_metrics else 0,
                "gpu_memory_used": node_metrics.gpu_memory_used if node_metrics else 0,
                "gpu_memory_total": node_metrics.gpu_memory_total if node_metrics else 0,
                "network_sent": node_metrics.network_sent if node_metrics else 0,
                "network_recv": node_metrics.network_recv if node_metrics else 0
            }
        }
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """
        获取详细状态
        
        Returns:
            详细状态信息
        """
        ray_status = await self._ray_manager.get_status()
        cluster_metrics = await self.get_cluster_metrics()
        
        return {
            "ray_cluster": {
                "is_running": ray_status.is_running,
                "head_address": ray_status.head_address,
                "dashboard_url": ray_status.dashboard_url,
                "num_nodes": ray_status.num_nodes,
                "uptime_seconds": ray_status.uptime_seconds,
                "available_resources": ray_status.available_resources,
                "used_resources": ray_status.used_resources
            },
            "resources": await self.get_resource_usage(),
            "cluster_metrics": {
                "total_nodes": cluster_metrics.total_nodes,
                "cpu_usage_percent": cluster_metrics.cpu_usage_percent,
                "memory_usage_percent": cluster_metrics.memory_usage_percent,
                "gpu_usage_percent": cluster_metrics.gpu_usage_percent
            }
        }


# 单例资源监控器
_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor(history_size: int = 100) -> ResourceMonitor:
    """获取资源监控器单例"""
    global _monitor
    if _monitor is None:
        _monitor = ResourceMonitor(history_size)
    return _monitor
