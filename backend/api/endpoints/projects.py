"""
Project endpoints v2.0 - 带缓存
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid

# v2.0 导入
from core.cache_service import cache_service
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from api.endpoints.auth import get_current_user
except ImportError:
    get_current_user = None

router = APIRouter()

# 模拟数据库
fake_projects_db = {
    1: {
        "id": 1,
        "name": "LLM Fine-tuning Demo",
        "description": "Llama 2 微调示例项目",
        "owner_id": 1,
        "status": "active",
        "settings": {"training_framework": "deepspeed"},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
}

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    settings: Optional[dict] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    settings: Optional[dict] = None

@router.get("")
async def list_projects(
    skip: int = 0,
    limit: int = 100,
    owner_id: Optional[int] = None,
    current_user = Depends(get_current_user)
):
    """
    获取项目列表
    
    v2.0: 使用缓存
    """
    # v2.0: 尝试从缓存获取
    cached_result = await cache_service.get_user_projects_cached(
        {"owner_id": owner_id} if owner_id else {},
        owner_id or current_user.id
    )
    
    if cached_result is not None:
        return {
            "total": len(cached_result),
            "projects": cached_result[skip:skip+limit],
            "cached": True
        }
    
    # 缓存未命中，从数据库获取
    projects = list(fake_projects_db.values())
    
    if owner_id:
        projects = [p for p in projects if p["owner_id"] == owner_id]
    
    return {
        "total": len(projects),
        "projects": projects[skip:skip+limit],
        "cached": False
    }

@router.post("")
async def create_project(
    project: ProjectCreate,
    current_user = Depends(get_current_user)
):
    """
    创建项目
    
    v2.0: 清除缓存
    """
    # 创建项目
    new_project = {
        "id": len(fake_projects_db) + 1,
        "name": project.name,
        "description": project.description,
        "owner_id": current_user.id,
        "status": "active",
        "settings": project.settings or {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    fake_projects_db[new_project["id"]] = new_project
    
    # v2.0: 清除项目列表缓存
    await cache_service.invalidate_user_projects(current_user.id)
    
    return new_project

@router.put("/{project_id}")
async def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    current_user = Depends(get_current_user)
):
    """
    更新项目
    
    v2.0: 清除缓存
    """
    if project_id not in fake_projects_db:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    project = fake_projects_db[project_id]
    
    # 更新字段
    if project_update.name is not None:
        project["name"] = project_update.name
    if project_update.description is not None:
        project["description"] = project_update.description
    if project_update.status is not None:
        project["status"] = project_update.status
    if project_update.settings is not None:
        project["settings"] = project_update.settings
    
    project["updated_at"] = datetime.utcnow()
    
    # v2.0: 清除缓存
    await cache_service.invalidate_user_projects(current_user.id)
    
    return project

@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    current_user = Depends(get_current_user)
):
    """
    删除项目
    
    v2.0: 清除缓存
    """
    if project_id not in fake_projects_db:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    del fake_projects_db[project_id]
    
    # v2.0: 清除缓存
    await cache_service.invalidate_user_projects(current_user.id)
    
    return {"message": "项目已删除"}

@router.get("/stats")
async def get_project_stats():
    """
    获取项目统计
    
    v2.0: 使用缓存
    """
    cached_stats = cache.get("project:stats")
    if cached_stats:
        return cached_stats
    
    stats = {
        "total_projects": len(fake_projects_db),
        "active_projects": sum(1 for p in fake_projects_db.values() if p["status"] == "active"),
        "completed_projects": sum(1 for p in fake_projects_db.values() if p["status"] == "completed"),
    }
    
    # 缓存5分钟
    cache.set("project:stats", stats, "project_list")
    
    return stats
