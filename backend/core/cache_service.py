"""
缓存服务层 - 缓存集成示例
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from core.cache import cache
from core.decorators import cached, invalidate_cache, CacheKeys
from models import Project, Task, Dataset

class CacheService:
    """缓存服务"""
    
    # 项目缓存
    @staticmethod
    @cached(ttl_key="project_list", key_builder=lambda f, u: CacheKeys.project_list(u))
    async def get_user_projects_cached(db: Session, user_id: int) -> List[dict]:
        """获取用户项目列表（带缓存）"""
        projects = db.query(Project).filter(
            Project.owner_id == user_id
        ).all()
        return [p.to_dict() for p in projects]
    
    @staticmethod
    async def invalidate_user_projects(user_id: int):
        """清除用户项目缓存"""
        cache.delete(CacheKeys.project_list(user_id))
    
    # 任务缓存
    @staticmethod
    @cached(ttl_key="task_status", key_builder=lambda f, t: CacheKeys.task_detail(t))
    async def get_task_status_cached(db: Session, task_id: int) -> Optional[dict]:
        """获取任务状态（带缓存）"""
        task = db.query(Task).filter(Task.id == task_id).first()
        return task.to_dict() if task else None
    
    @staticmethod
    async def update_task_progress(task_id: int, progress: float):
        """更新任务进度缓存"""
        cache.set(
            CacheKeys.task_detail(task_id),
            {"task_id": task_id, "progress": progress},
            "task_status"
        )
    
    # GPU指标缓存
    @staticmethod
    async def set_gpu_metrics(metrics: dict):
        """设置GPU指标缓存"""
        cache.set("gpu:metrics", metrics, "gpu_metrics")
    
    @staticmethod
    async def get_gpu_metrics() -> Optional[dict]:
        """获取GPU指标缓存"""
        return cache.get("gpu:metrics")
    
    # 系统配置缓存
    @staticmethod
    @cached(ttl_key="system_config", key_builder=lambda f: CacheKeys.system_config())
    async def get_system_config_cached() -> dict:
        """获取系统配置（带缓存）"""
        # TODO: 从数据库或配置获取
        return {
            "site_name": "AI Platform",
            "version": "2.0.0",
            "features": {
                "gpu_monitoring": True,
                "distributed_training": False,
            }
        }
    
    @staticmethod
    async def invalidate_system_config():
        """清除系统配置缓存"""
        cache.delete(CacheKeys.system_config())

# 缓存服务实例
cache_service = CacheService()
