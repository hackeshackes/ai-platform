"""
分布式训练模块 v2.2
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import asyncio

@dataclass
class Worker:
    """工作节点"""
    worker_id: str
    hostname: str
    gpu_count: int
    gpu_info: List[Dict] = field(default_factory=list)
    status: str = "idle"  # idle, running, error
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TrainingJob:
    """分布式训练任务"""
    job_id: str
    name: str
    script: str
    world_size: int  # 总进程数
    rank: int = 0  # 当前进程rank
    status: str = "pending"  # pending, running, completed, failed
    workers: List[str] = field(default_factory=list)  # worker_ids
    metrics: Dict = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class ClusterConfig:
    """集群配置"""
    config_id: str
    name: str
    workers: List[str] = field(default_factory=list)
    scheduler: str = "ray"  # ray, slurm, k8s
    default_resources: Dict = field(default_factory=dict)

class DistributedTrainer:
    """分布式训练管理器"""
    
    def __init__(self):
        self.workers: Dict[str, Worker] = {}
        self.jobs: Dict[str, TrainingJob] = {}
        self.clusters: Dict[str, ClusterConfig] = {}
    
    # Worker管理
    async def register_worker(
        self,
        hostname: str,
        gpu_count: int,
        gpu_info: Optional[List[Dict]] = None
    ) -> Worker:
        """注册worker"""
        worker = Worker(
            worker_id=str(uuid4()),
            hostname=hostname,
            gpu_count=gpu_count,
            gpu_info=gpu_info or []
        )
        
        self.workers[worker.worker_id] = worker
        return worker
    
    def get_worker(self, worker_id: str) -> Optional[Worker]:
        """获取worker"""
        return self.workers.get(worker_id)
    
    def list_workers(self, status: Optional[str] = None) -> List[Worker]:
        """列出workers"""
        workers = list(self.workers.values())
        
        if status:
            workers = [w for w in workers if w.status == status]
        
        return workers
    
    def update_worker_status(self, worker_id: str, status: str) -> bool:
        """更新worker状态"""
        worker = self.workers.get(worker_id)
        if worker:
            worker.status = status
            worker.last_heartbeat = datetime.utcnow()
            return True
        return False
    
    # 任务管理
    async def create_job(
        self,
        name: str,
        script: str,
        world_size: int,
        worker_ids: Optional[List[str]] = None
    ) -> TrainingJob:
        """创建训练任务"""
        # 选择workers
        if not worker_ids:
            available = [w for w in self.workers.values() if w.status == "idle"]
            if len(available) < world_size:
                raise ValueError(f"Not enough workers: need {world_size}, have {len(available)}")
            worker_ids = [w.worker_id for w in available[:world_size]]
        
        job = TrainingJob(
            job_id=str(uuid4()),
            name=name,
            script=script,
            world_size=world_size,
            workers=worker_ids
        )
        
        self.jobs[job.job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """获取任务"""
        return self.jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[TrainingJob]:
        """列出任务"""
        jobs = list(self.jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return jobs[:limit]
    
    async def start_job(self, job_id: str) -> bool:
        """启动任务"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        # 更新workers状态
        for worker_id in job.workers:
            self.update_worker_status(worker_id, "running")
        
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.logs.append(f"[{datetime.utcnow().isoformat()}] Job started")
        
        # 模拟训练
        asyncio.create_task(self._simulate_training(job))
        
        return True
    
    async def _simulate_training(self, job: TrainingJob):
        """模拟训练过程"""
        for i in range(10):
            await asyncio.sleep(1)
            job.metrics[f"step_{i}"] = {
                "loss": 1.0 / (i + 1),
                "accuracy": 0.5 + 0.05 * i
            }
            job.logs.append(f"[{datetime.utcnow().isoformat()}] Step {i} completed")
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.logs.append(f"[{datetime.utcnow().isoformat()}] Job completed")
        
        # 释放workers
        for worker_id in job.workers:
            self.update_worker_status(worker_id, "idle")
    
    async def stop_job(self, job_id: str) -> bool:
        """停止任务"""
        job = self.jobs.get(job_id)
        if not job or job.status != "running":
            return False
        
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        job.logs.append(f"[{datetime.utcnow().isoformat()}] Job stopped")
        
        # 释放workers
        for worker_id in job.workers:
            self.update_worker_status(worker_id, "idle")
        
        return True
    
    def get_job_logs(self, job_id: str, tail: int = 100) -> List[str]:
        """获取任务日志"""
        job = self.jobs.get(job_id)
        if not job:
            return []
        
        return job.logs[-tail:]
    
    # 集群管理
    async def create_cluster(
        self,
        name: str,
        worker_ids: List[str],
        scheduler: str = "ray"
    ) -> ClusterConfig:
        """创建集群配置"""
        cluster = ClusterConfig(
            config_id=str(uuid4()),
            name=name,
            workers=worker_ids,
            scheduler=scheduler
        )
        
        self.clusters[cluster.config_id] = cluster
        return cluster
    
    def get_cluster(self, cluster_id: str) -> Optional[ClusterConfig]:
        """获取集群"""
        return self.clusters.get(cluster_id)
    
    def list_clusters(self) -> List[ClusterConfig]:
        """列出集群"""
        return list(self.clusters.values())
    
    # 资源统计
    def get_cluster_resources(self) -> Dict[str, Any]:
        """获取集群资源统计"""
        workers = list(self.workers.values())
        
        total_gpus = sum(w.gpu_count for w in workers)
        idle_gpus = sum(w.gpu_count for w in workers if w.status == "idle")
        running_gpus = sum(w.gpu_count for w in workers if w.status == "running")
        
        return {
            "total_workers": len(workers),
            "total_gpus": total_gpus,
            "idle_gpus": idle_gpus,
            "running_gpus": running_gpus,
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "hostname": w.hostname,
                    "gpus": w.gpu_count,
                    "status": w.status
                }
                for w in workers
            ]
        }
    
    def get_job_resources(self, job_id: str) -> Dict[str, Any]:
        """获取任务资源使用"""
        job = self.jobs.get(job_id)
        if not job:
            return {}
        
        return {
            "job_id": job.job_id,
            "world_size": job.world_size,
            "workers": job.workers,
            "status": job.status,
            "duration_seconds": (
                (job.completed_at or datetime.utcnow()) - job.started_at
            ).total_seconds() if job.started_at else None
        }

# DistributedTrainer实例
distributed_trainer = DistributedTrainer()
