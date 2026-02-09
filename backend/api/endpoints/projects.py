"""Project endpoints"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid

router = APIRouter()

# 模拟数据库
fake_projects_db = {
    1: {
        "id": 1,
        "name": "LLM Fine-tuning Demo",
        "description": "Llama 2 微调示例项目",
        "owner_id": 1,
        "status": "active",
        "settings": {
            "training_framework": "deepspeed",
            "compute_type": "gpu"
        },
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
async def list_projects(skip: int = 0, limit: int = 100, owner_id: Optional[int] = None):
    """获取项目列表"""
    projects = list(fake_projects_db.values())
    
    if owner_id:
        projects = [p for p in projects if p["owner_id"] == owner_id]
    
    return {
        "total": len(projects),
        "projects": projects[skip:skip+limit]
    }

@router.get("/{project_id}")
async def get_project(project_id: int):
    """获取单个项目"""
    if project_id not in fake_projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    return fake_projects_db[project_id]

@router.post("", response_model=dict, status_code=201)
async def create_project(project_data: ProjectCreate):
    """创建项目"""
    project_id = len(fake_projects_db) + 1
    
    new_project = {
        "id": project_id,
        "name": project_data.name,
        "description": project_data.description,
        "owner_id": 1,  # 从认证中获取
        "status": "active",
        "settings": project_data.settings or {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    fake_projects_db[project_id] = new_project
    return new_project

@router.put("/{project_id}")
async def update_project(project_id: int, project_data: ProjectUpdate):
    """更新项目"""
    if project_id not in fake_projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    for field, value in project_data.dict(exclude_unset=True).items():
        fake_projects_db[project_id][field] = value
    
    fake_projects_db[project_id]["updated_at"] = datetime.utcnow()
    return fake_projects_db[project_id]

@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: int):
    """删除项目"""
    if project_id not in fake_projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    del fake_projects_db[project_id]
    return None
