"""
Ray Cluster Manager - AI Platform v6

Ray集群管理器，提供集群启动、停止、状态查询等功能。
"""
import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class RayClusterConfig:
    """Ray集群配置"""
    head_ip: str = "0.0.0.0"
    head_port: int = 6379
    dashboard_port: int = 8265
    num_workers: int = 0
    worker_ip: str = ""
    worker_port: int = 6379
    object_store_memory: int = 107374182400  # 100GB
    memory: int = 0
    cpus: int = 0
    gpus: int = 0
    resources: Dict[str, float] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class RayClusterStatus:
    """Ray集群状态"""
    is_running: bool = False
    head_address: str = ""
    dashboard_url: str = ""
    num_nodes: int = 0
    available_resources: Dict[str, float] = field(default_factory=dict)
    used_resources: Dict[str, float] = field(default_factory=dict)
    uptime_seconds: int = 0
    last_heartbeat: Optional[datetime] = None


class RayClusterManager:
    """
    Ray集群管理器
    
    管理Ray集群的启动、停止、扩缩容和状态监控。
    """
    
    def __init__(self, config: Optional[RayClusterConfig] = None):
        """
        初始化Ray集群管理器
        
        Args:
            config: Ray集群配置
        """
        self.config = config or RayClusterConfig()
        self._process: Optional[subprocess.Popen] = None
        self._worker_processes: List[subprocess.Popen] = []
        self._start_time: Optional[datetime] = None
        
    async def start_cluster(self, head_only: bool = False) -> Dict[str, Any]:
        """
        启动Ray集群
        
        Args:
            head_only: 是否只启动head节点
            
        Returns:
            启动结果
        """
        try:
            # 检查Ray是否已安装
            result = subprocess.run(
                ["ray", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "Ray未安装，请先安装Ray: pip install ray"
                }
            
            # 构建ray start命令
            cmd = [
                "ray", "start",
                "--head",
                "--address", f"{self.config.head_ip}:{self.config.head_port}",
                "--dashboard-port", str(self.config.dashboard_port),
                "--object-store-memory", str(self.config.object_store_memory),
                "--include-dashboard", "true"
            ]
            
            # 添加环境变量
            env = os.environ.copy()
            for key, value in self.config.env_vars.items():
                env[key] = value
            
            # 启动head节点
            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 等待启动
            await asyncio.sleep(5)
            
            # 检查进程状态
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                return {
                    "success": False,
                    "error": f"Ray启动失败: {stderr.decode()}"
                }
            
            self._start_time = datetime.now()
            
            # 如果需要，启动worker节点
            if not head_only and self.config.num_workers > 0:
                await self.add_workers(self.config.num_workers)
            
            return {
                "success": True,
                "message": "Ray集群启动成功",
                "head_address": f"{self.config.head_ip}:{self.config.head_port}",
                "dashboard_url": f"http://{self.config.head_ip}:{self.config.dashboard_port}",
                "num_nodes": 1 + len(self._worker_processes)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Ray版本检查超时"
            }
        except Exception as e:
            logger.error(f"启动Ray集群失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_workers(self, num_workers: int) -> Dict[str, Any]:
        """
        添加Worker节点
        
        Args:
            num_workers: 添加的worker数量
            
        Returns:
            添加结果
        """
        if not self._process or self._process.poll() is not None:
            return {
                "success": False,
                "error": "Ray集群未运行"
            }
        
        try:
            for i in range(num_workers):
                cmd = [
                    "ray", "start",
                    "--address", f"{self.config.head_ip}:{self.config.head_port}",
                    "--object-store-memory", str(self.config.object_store_memory)
                ]
                
                env = os.environ.copy()
                for key, value in self.config.env_vars.items():
                    env[key] = value
                
                worker = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self._worker_processes.append(worker)
                await asyncio.sleep(2)
            
            return {
                "success": True,
                "message": f"成功添加 {num_workers} 个Worker节点",
                "total_workers": len(self._worker_processes)
            }
            
        except Exception as e:
            logger.error(f"添加Worker节点失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def remove_workers(self, num_workers: int) -> Dict[str, Any]:
        """
        移除Worker节点
        
        Args:
            num_workers: 移除的worker数量
            
        Returns:
            移除结果
        """
        try:
            removed = 0
            for _ in range(min(num_workers, len(self._worker_processes))):
                if self._worker_processes:
                    worker = self._worker_processes.pop()
                    worker.terminate()
                    removed += 1
                    await asyncio.sleep(1)
            
            return {
                "success": True,
                "message": f"成功移除 {removed} 个Worker节点",
                "remaining_workers": len(self._worker_processes)
            }
            
        except Exception as e:
            logger.error(f"移除Worker节点失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stop_cluster(self) -> Dict[str, Any]:
        """
        停止Ray集群
        
        Returns:
            停止结果
        """
        try:
            # 停止所有worker
            for worker in self._worker_processes:
                try:
                    worker.terminate()
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
            self._worker_processes.clear()
            
            # 停止head
            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None
            
            self._start_time = None
            
            return {
                "success": True,
                "message": "Ray集群已停止"
            }
            
        except Exception as e:
            logger.error(f"停止Ray集群失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status(self) -> RayClusterStatus:
        """
        获取Ray集群状态
        
        Returns:
            集群状态
        """
        status = RayClusterStatus()
        
        if not self._process or self._process.poll() is not None:
            return status
        
        # 检查head进程
        status.is_running = True
        status.head_address = f"{self.config.head_ip}:{self.config.head_port}"
        status.dashboard_url = f"http://{self.config.head_ip}:{self.config.dashboard_port}"
        status.num_nodes = 1 + len(self._worker_processes)
        
        if self._start_time:
            uptime = datetime.now() - self._start_time
            status.uptime_seconds = int(uptime.total_seconds())
        
        status.last_heartbeat = datetime.now()
        
        # 尝试获取资源信息
        try:
            import ray
            if ray.is_initialized():
                status.available_resources = ray.available_resources()
                status.used_resources = ray.cluster_resources()
        except Exception:
            pass
        
        return status
    
    async def scale_cluster(self, num_nodes: int) -> Dict[str, Any]:
        """
        扩缩容Ray集群
        
        Args:
            num_nodes: 目标节点数量
            
        Returns:
            扩缩容结果
        """
        current_nodes = 1 + len(self._worker_processes)
        
        if num_nodes > current_nodes:
            add_count = num_nodes - current_nodes
            return await self.add_workers(add_count)
        elif num_nodes < current_nodes:
            remove_count = current_nodes - num_nodes
            return await self.remove_workers(remove_count)
        else:
            return {
                "success": True,
                "message": "节点数量无变化",
                "current_nodes": current_nodes
            }
    
    async def execute_ray_command(self, command: str) -> Dict[str, Any]:
        """
        执行Ray命令
        
        Args:
            command: 要执行的命令
            
        Returns:
            命令执行结果
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "命令执行超时"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class RayClient:
    """
    Ray客户端
    
    提供与Ray集群交互的便捷方法。
    """
    
    def __init__(self, address: Optional[str] = None):
        """
        初始化Ray客户端
        
        Args:
            address: Ray集群地址
        """
        self.address = address
        self._ray_available = False
        
    async def initialize(self) -> bool:
        """
        初始化Ray客户端
        
        Returns:
            是否初始化成功
        """
        try:
            import ray
            if not ray.is_initialized():
                ray.init(address=self.address)
            self._ray_available = True
            return True
        except Exception as e:
            logger.error(f"初始化Ray客户端失败: {e}")
            return False
    
    async def submit_task(self, 
                         func: callable, 
                         *args, 
                         num_cpus: int = 1,
                         num_gpus: int = 0,
                         resources: Optional[Dict[str, float]] = None,
                         **kwargs) -> Any:
        """
        提交任务到Ray集群
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            num_cpus: 使用的CPU数量
            num_gpus: 使用的GPU数量
            resources: 使用的资源
            **kwargs: 关键字参数
            
        Returns:
            任务结果
        """
        if not self._ray_available:
            raise RuntimeError("Ray客户端未初始化")
        
        import ray
        future = ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            resources=resources or {}
        )(func).remote(*args, **kwargs)
        
        return ray.get(future)
    
    async def get_cluster_resources(self) -> Dict[str, Any]:
        """
        获取集群资源信息
        
        Returns:
            资源信息
        """
        try:
            import ray
            if not ray.is_initialized():
                return {"error": "Ray未初始化"}
            
            return {
                "available": ray.available_resources(),
                "total": ray.cluster_resources()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def list_actors(self) -> List[Dict[str, Any]]:
        """
        列出所有Actor
        
        Returns:
            Actor列表
        """
        try:
            import ray
            if not ray.is_initialized():
                return []
            
            actors = []
            for actor_class in ray.util.list_actors():
                actors.append({
                    "class": actor_class.class_name,
                    "status": actor_class.status,
                    "name": actor_class.name
                })
            
            return actors
        except Exception as e:
            logger.error(f"列出Actor失败: {e}")
            return []
    
    async def shutdown(self):
        """关闭Ray客户端"""
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
            self._ray_available = False
        except Exception as e:
            logger.error(f"关闭Ray客户端失败: {e}")


# 单例Ray集群管理器
_ray_manager: Optional[RayClusterManager] = None


def get_ray_cluster_manager() -> RayClusterManager:
    """获取Ray集群管理器单例"""
    global _ray_manager
    if _ray_manager is None:
        _ray_manager = RayClusterManager()
    return _ray_manager


def get_ray_client(address: Optional[str] = None) -> RayClient:
    """创建Ray客户端"""
    return RayClient(address)
